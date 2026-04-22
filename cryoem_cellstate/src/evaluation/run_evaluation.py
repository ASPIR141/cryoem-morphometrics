"""Stage 6 pipeline driver: evaluation metrics and figure generation.

Usage::

    python -m src.evaluation.run_evaluation --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything

from .figures import (
    figure3_size_entropy_histograms,
    figure7_diffusion_map,
)
from .metrics import anova_per_feature, plot_significance_heatmap

logger = logging.getLogger(__name__)


def run_evaluation(cfg: object) -> dict[str, object]:
    """Generate evaluation metrics and all figure outputs.

    Expects the following pre-computed artifacts to exist:

    * ``<results_dir>/embeddings.npy`` — SSL embedding matrix
    * ``<morphometrics.output_path>`` — morphometrics parquet
    * ``<clustering.results_dir>/umap_2d.npy`` — 2-D UMAP coordinates
    * ``<clustering.results_dir>/hdbscan_labels.npy`` — cluster labels

    Parameters
    ----------
    cfg:
        Root pipeline config.

    Returns
    -------
    results:
        Dict with keys ``quality``, ``stability``, ``anova``, and ``state_map``.
    """
    from src.clustering.state_analysis import (
        compute_cluster_quality,
        compute_cluster_stability,
        interpret_states,
        plot_morphometric_boxplots,
        plot_umap,
        summarize_clusters,
    )

    figs_dir = Path(cfg.evaluation.figures_dir)  # type: ignore[attr-defined]
    figs_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = Path(cfg.clustering.results_dir)  # type: ignore[attr-defined]
    results_dir = Path(cfg.project.results_dir)  # type: ignore[attr-defined]

    # ── Load artifacts ─────────────────────────────────────────────────────────
    embeddings_path = results_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found at {embeddings_path}")
    embeddings = np.load(embeddings_path)

    umap_path = cluster_dir / "umap_2d.npy"
    if not umap_path.exists():
        raise FileNotFoundError(
            f"umap_2d.npy not found at {umap_path}. Run clustering stage first."
        )
    umap_2d = np.load(umap_path)

    labels_path = cluster_dir / "hdbscan_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"hdbscan_labels.npy not found at {labels_path}. Run clustering stage first."
        )
    hdb_labels = np.load(labels_path)

    morph_path = Path(cfg.morphometrics.output_path)  # type: ignore[attr-defined]
    if not morph_path.exists():
        raise FileNotFoundError(
            f"morphometrics.parquet not found at {morph_path}. Run morphometrics stage first."
        )
    morph_df = pd.read_parquet(morph_path)

    # Align lengths
    n_min = min(len(embeddings), len(hdb_labels), len(morph_df))
    if len(set([len(embeddings), len(hdb_labels), len(morph_df)])) > 1:
        logger.warning(
            "Length mismatch — embeddings=%d  labels=%d  morph=%d; truncating to %d",
            len(embeddings),
            len(hdb_labels),
            len(morph_df),
            n_min,
        )
    embeddings = embeddings[:n_min]
    hdb_labels = hdb_labels[:n_min]
    morph_df = morph_df.iloc[:n_min].reset_index(drop=True)
    umap_2d = umap_2d[:n_min]

    # ── Cluster quality metrics ────────────────────────────────────────────────
    quality = compute_cluster_quality(embeddings, hdb_labels)
    stability = compute_cluster_stability(
        embeddings,
        hdb_labels,
        n_iterations=cfg.clustering.bootstrap_iterations,  # type: ignore[attr-defined]
    )
    logger.info(
        "Cluster quality — silhouette=%.4f  db=%.4f  stability=%.4f",
        quality.get("silhouette", float("nan")),
        quality.get("davies_bouldin", float("nan")),
        stability,
    )

    # ── Figures ────────────────────────────────────────────────────────────────
    figure3_size_entropy_histograms(morph_df, figs_dir / "fig3_histograms.png")

    plot_umap(umap_2d, hdb_labels, figs_dir / "fig4_umap_hdbscan.png", title="UMAP — HDBSCAN clusters")

    key_features = ["area", "circularity", "eccentricity", "intensity_entropy", "glcm_contrast_mean"]
    key_features = [f for f in key_features if f in morph_df.columns]
    plot_morphometric_boxplots(morph_df, hdb_labels, key_features, figs_dir / "fig5_boxplots.png")

    # ── Statistical testing ────────────────────────────────────────────────────
    anova_df = anova_per_feature(morph_df, hdb_labels, alpha=cfg.evaluation.significance_alpha)  # type: ignore[attr-defined]
    plot_significance_heatmap(anova_df, figs_dir / "fig6_significance.png")

    anova_out = figs_dir / "anova_results.csv"
    anova_df.to_csv(anova_out, index=False)
    logger.info("ANOVA results written to %s", anova_out)

    # ── Optional diffusion map ─────────────────────────────────────────────────
    figure7_diffusion_map(embeddings, hdb_labels, figs_dir / "fig7_diffusion.png")

    # ── State interpretation ───────────────────────────────────────────────────
    summary = summarize_clusters(morph_df, hdb_labels)
    state_map = interpret_states(summary)
    logger.info("Discovered states: %s", state_map)

    # ── Metrics summary CSV ────────────────────────────────────────────────────
    metrics_summary = {
        "silhouette": quality.get("silhouette"),
        "davies_bouldin": quality.get("davies_bouldin"),
        "bootstrap_stability": stability,
        "n_clusters": int((hdb_labels >= 0).astype(int).max()) + 1 if (hdb_labels >= 0).any() else 0,
        "n_cells": n_min,
        "noise_fraction": float((hdb_labels == -1).mean()),
    }
    pd.DataFrame([metrics_summary]).to_csv(figs_dir / "eval_summary.csv", index=False)
    logger.info("Stage 6 complete — figures saved to %s", figs_dir)

    return {"quality": quality, "stability": stability, "anova": anova_df, "state_map": state_map}


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--seg-eval",
    is_flag=True,
    help="Also evaluate segmentation masks against CVAT ground truth.",
)
@click.option(
    "--gt-dir",
    default=None,
    help="CVAT ground-truth directory for segmentation evaluation.",
)
def main(config: str | None, seg_eval: bool, gt_dir: str | None) -> None:
    """Run Stage 6: evaluation metrics and figure generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    if seg_eval:
        from src.segmentation.evaluate_segmentation import evaluate_masks

        pred_dir = Path(cfg.data.masks_dir)  # type: ignore[attr-defined]
        gt_path = Path(gt_dir or cfg.data.cvat_gt_dir)  # type: ignore[attr-defined]
        results_dir = Path(cfg.project.results_dir) / "stage2"  # type: ignore[attr-defined]

        if not gt_path.exists():
            logger.warning("CVAT GT dir not found: %s — skipping segmentation eval", gt_path)
        else:
            seg_df = evaluate_masks(pred_dir, gt_path, results_dir)
            if not seg_df.empty:
                logger.info(
                    "Segmentation — mean Dice=%.4f  mean IoU=%.4f",
                    seg_df["dice"].mean(),
                    seg_df["iou"].mean(),
                )

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
