"""Central Weights & Biases logging helpers for the CryoEM pipeline.

All W&B interactions go through this module so that:

* W&B can be disabled globally via ``cfg.wandb.enabled = false``.
* Every stage uses a consistent run schema (project, entity, tags, job_type).
* The rest of the codebase stays free of ``import wandb`` boilerplate.

Typical usage::

    from src.utils.wandb_logger import WandbRun

    with WandbRun(cfg, job_type="train", run_name="vista2d", tags=["segmentation"]) as run:
        run.log({"train_loss": 0.42})
        run.log_figure(path_to_png, caption="UMAP")
        run.log_segmentation_table(df)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import pandas as pd

if TYPE_CHECKING:
    import wandb as _wandb

logger = logging.getLogger(__name__)


class WandbRun:
    """Context-manager wrapper around a single ``wandb.Run``.

    Parameters
    ----------
    cfg:
        Root ``PipelineConfig`` (reads ``cfg.wandb``).
    job_type:
        W&B job type string (e.g. ``"train"``, ``"eval"``, ``"embedding_extraction"``).
    run_name:
        Human-readable run name shown in the W&B UI.
    tags:
        List of string tags for grouping / filtering.
    extra_config:
        Additional key-value pairs merged into the W&B run config.
    """

    def __init__(
        self,
        cfg: Any,
        job_type: str,
        run_name: str,
        tags: list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self._enabled: bool = getattr(getattr(cfg, "wandb", None), "enabled", False)
        self._log_figures: bool = getattr(getattr(cfg, "wandb", None), "log_figures", True)
        self._run: Any = None

        if not self._enabled:
            logger.info("W&B disabled — skipping wandb.init (job_type=%s)", job_type)
            return

        try:
            import wandb  # noqa: PLC0415

            wb_cfg = cfg.wandb  # type: ignore[attr-defined]
            init_config: dict[str, Any] = {}
            if extra_config:
                init_config.update(extra_config)

            self._run = wandb.init(
                project=wb_cfg.project,
                entity=wb_cfg.entity or None,
                job_type=job_type,
                name=run_name,
                tags=tags or [],
                config=init_config,
                reinit=True,
            )
            logger.info(
                "W&B run started — project=%s  name=%s  url=%s",
                wb_cfg.project,
                run_name,
                self._run.url if self._run else "N/A",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise W&B (will continue without it): %s", exc)
            self._run = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "WandbRun":
        return self

    def __exit__(self, *_: object) -> None:
        self.finish()

    # ── Core logging helpers ──────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalar metrics to W&B."""
        if self._run is None:
            return
        kwargs: dict[str, Any] = {}
        if step is not None:
            kwargs["step"] = step
        self._run.log(metrics, **kwargs)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    # ── Figure logging ────────────────────────────────────────────────────────

    def log_figure(self, path: str | Path, caption: str = "") -> None:
        """Upload a PNG/image file as a ``wandb.Image``.

        Respects ``cfg.wandb.log_figures``.  No-ops if W&B is disabled or the
        file does not exist.
        """
        if self._run is None or not self._log_figures:
            return
        p = Path(path)
        if not p.exists():
            logger.warning("W&B figure not found, skipping: %s", p)
            return
        try:
            import wandb  # noqa: PLC0415

            self._run.log({caption or p.stem: wandb.Image(str(p), caption=caption)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log figure %s to W&B: %s", p, exc)

    def log_figures_dir(
        self,
        directory: str | Path,
        captions: dict[str, str] | None = None,
    ) -> None:
        """Upload all PNG files in *directory* as W&B images."""
        if self._run is None or not self._log_figures:
            return
        d = Path(directory)
        for png in sorted(d.glob("*.png")):
            caption = (captions or {}).get(png.name, png.stem)
            self.log_figure(png, caption=caption)

    # ── Segmentation metrics ──────────────────────────────────────────────────

    def log_segmentation_metrics(
        self,
        df: pd.DataFrame,
        model_name: str,
    ) -> None:
        """Log per-image segmentation metrics and aggregate scalars.

        Parameters
        ----------
        df:
            DataFrame with columns ``name``, ``dice``, ``iou``,
            and optionally ``hausdorff_95``.
        model_name:
            ``"classical"`` or ``"vista2d"`` — used as a config tag so W&B can
            group runs for the comparison chart.
        """
        if self._run is None:
            return
        try:
            import wandb  # noqa: PLC0415

            cols = [c for c in ["name", "dice", "iou", "hausdorff_95"] if c in df.columns]
            table = wandb.Table(dataframe=df[cols])
            self._run.log({"segmentation/per_image_metrics": table})

            agg: dict[str, float] = {}
            for col in ["dice", "iou", "hausdorff_95"]:
                if col in df.columns:
                    agg[f"segmentation/{col}_mean"] = float(df[col].mean())
            agg["segmentation/n_images"] = float(len(df))
            self._run.log(agg)

            self._run.config.update({"seg_model": model_name}, allow_val_change=True)
            logger.info("W&B: logged segmentation metrics for model=%s", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log segmentation metrics to W&B: %s", exc)

    # ── SSL embedding extraction ──────────────────────────────────────────────

    def log_ssl_metrics(
        self,
        model_name: str,
        n_cells: int,
        embedding_dim: int,
        extraction_time_s: float,
        embeddings_path: str | Path | None = None,
    ) -> None:
        """Log SSL embedding extraction statistics and optionally save as artifact.

        Parameters
        ----------
        model_name:
            ``"mae"``, ``"cryo-ief"``, or ``"rotnet"``.
        n_cells:
            Number of cell crops processed.
        embedding_dim:
            Dimensionality of the output embedding vector.
        extraction_time_s:
            Wall-clock extraction time in seconds.
        embeddings_path:
            Path to the saved ``embeddings.npy`` file; if provided, it is
            uploaded as a W&B Artifact.
        """
        if self._run is None:
            return
        try:
            import wandb  # noqa: PLC0415

            self._run.log({
                "ssl/n_cells": n_cells,
                "ssl/embedding_dim": embedding_dim,
                "ssl/extraction_time_s": extraction_time_s,
            })
            self._run.config.update({"ssl_model": model_name}, allow_val_change=True)

            if embeddings_path is not None:
                p = Path(embeddings_path)
                if p.exists():
                    artifact = wandb.Artifact(
                        name=f"embeddings-{model_name}",
                        type="embeddings",
                        description=(
                            f"SSL embeddings from {model_name} "
                            f"({n_cells} cells, dim={embedding_dim})"
                        ),
                        metadata={"model": model_name, "n_cells": n_cells, "dim": embedding_dim},
                    )
                    artifact.add_file(str(p))
                    self._run.log_artifact(artifact)

            logger.info(
                "W&B: logged SSL metrics — model=%s  n_cells=%d  dim=%d  t=%.1fs",
                model_name, n_cells, embedding_dim, extraction_time_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log SSL metrics to W&B: %s", exc)

    # ── Cluster quality ───────────────────────────────────────────────────────

    def log_cluster_metrics(
        self,
        quality: dict[str, float],
        stability: float,
        n_clusters: int,
        n_cells: int,
        noise_fraction: float,
        ssl_model: str | None = None,
    ) -> None:
        """Log HDBSCAN cluster quality scalars.

        Parameters
        ----------
        quality:
            Dict with keys ``silhouette`` and ``davies_bouldin``.
        stability:
            Bootstrap ARI stability score.
        n_clusters:
            Number of discovered clusters (excluding noise).
        n_cells:
            Total number of cells.
        noise_fraction:
            Fraction of cells labelled as noise (label=-1).
        ssl_model:
            Name of the SSL model used to produce the embeddings; written
            to run config for cross-model comparison.
        """
        if self._run is None:
            return
        try:
            metrics: dict[str, Any] = {
                "cluster/silhouette": quality.get("silhouette", float("nan")),
                "cluster/davies_bouldin": quality.get("davies_bouldin", float("nan")),
                "cluster/bootstrap_stability": stability,
                "cluster/n_clusters": n_clusters,
                "cluster/n_cells": n_cells,
                "cluster/noise_fraction": noise_fraction,
            }
            self._run.log(metrics)
            if ssl_model:
                self._run.config.update({"ssl_model": ssl_model}, allow_val_change=True)
            logger.info("W&B: logged cluster quality metrics")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log cluster metrics to W&B: %s", exc)

    # ── ANOVA table ───────────────────────────────────────────────────────────

    def log_anova_table(self, anova_df: pd.DataFrame) -> None:
        """Log the ANOVA results DataFrame as a W&B Table."""
        if self._run is None or anova_df.empty:
            return
        try:
            import wandb  # noqa: PLC0415

            table = wandb.Table(dataframe=anova_df)
            n_sig = int(anova_df["significant"].sum()) if "significant" in anova_df.columns else 0
            self._run.log({
                "anova/results_table": table,
                "anova/n_significant_features": n_sig,
            })
            logger.info("W&B: logged ANOVA table (%d significant features)", n_sig)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log ANOVA table to W&B: %s", exc)

    # ── Checkpoint artifact ───────────────────────────────────────────────────

    def log_checkpoint_artifact(
        self,
        ckpt_path: str | Path,
        name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload a Lightning checkpoint file as a W&B Artifact."""
        if self._run is None:
            return
        try:
            import wandb  # noqa: PLC0415

            p = Path(ckpt_path)
            if not p.exists():
                logger.warning("Checkpoint not found, skipping artifact upload: %s", p)
                return
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata or {},
            )
            artifact.add_file(str(p))
            self._run.log_artifact(artifact)
            logger.info("W&B: uploaded checkpoint artifact %s → %s", p.name, name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to upload checkpoint artifact to W&B: %s", exc)


# ── Convenience context manager ───────────────────────────────────────────────


@contextmanager
def wandb_run(
    cfg: Any,
    job_type: str,
    run_name: str,
    tags: list[str] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Generator[WandbRun, None, None]:
    """Context manager that yields a :class:`WandbRun` and calls ``finish()`` on exit.

    Example::

        with wandb_run(cfg, "eval", "seg-classical", tags=["segmentation"]) as run:
            run.log_segmentation_metrics(df, model_name="classical")
    """
    run = WandbRun(
        cfg,
        job_type=job_type,
        run_name=run_name,
        tags=tags,
        extra_config=extra_config,
    )
    try:
        yield run
    finally:
        run.finish()
