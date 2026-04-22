"""Stage 5 pipeline driver: UMAP dimensionality reduction, HDBSCAN clustering,
and per-cluster state analysis.

Usage::

    python -m src.clustering.run_clustering --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything

from .cluster import cluster_embeddings
from .state_analysis import (
    compute_cluster_quality,
    compute_cluster_stability,
    interpret_states,
    plot_morphometric_boxplots,
    plot_umap,
    summarize_clusters,
)
from .umap_reduce import reduce_embeddings

logger = logging.getLogger(__name__)


def run_clustering(cfg: object) -> dict[str, object]:
    """Run PCA → UMAP → HDBSCAN and generate cluster summaries.

    Parameters
    ----------
    cfg:
        Root pipeline config.

    Returns
    -------
    results:
        Dict with keys ``umap_2d``, ``labels``, ``quality``, ``stability``,
        ``state_map``, and ``summary``.
    """
    embeddings_path = Path(cfg.project.results_dir) / "embeddings.npy"  # type: ignore[attr-defined]
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"embeddings.npy not found at {embeddings_path}. Run ssl stage first."
        )

    morph_path = Path(cfg.morphometrics.output_path)  # type: ignore[attr-defined]
    morph_df: pd.DataFrame | None = None
    if morph_path.exists():
        morph_df = pd.read_parquet(morph_path)
        logger.info("Loaded %d morphometric rows", len(morph_df))
    else:
        logger.warning("morphometrics.parquet not found — skipping morphometric analysis")

    cluster_results_dir = Path(cfg.clustering.results_dir)  # type: ignore[attr-defined]
    cluster_results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(cfg.evaluation.figures_dir)  # type: ignore[attr-defined]
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ── Dimensionality reduction ───────────────────────────────────────────────
    logger.info("Running PCA → UMAP reduction")
    umap_2d, umap_cluster = reduce_embeddings(embeddings_path, cfg, cluster_results_dir)

    # ── Clustering ────────────────────────────────────────────────────────────
    logger.info("Running HDBSCAN clustering")
    label_dict = cluster_embeddings(umap_cluster, cfg)
    hdb_labels = label_dict["hdbscan"]
    np.save(cluster_results_dir / "hdbscan_labels.npy", hdb_labels)

    # ── Cluster quality ───────────────────────────────────────────────────────
    embeddings = np.load(embeddings_path)
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

    # ── UMAP scatter plot ─────────────────────────────────────────────────────
    plot_umap(umap_2d, hdb_labels, figs_dir / "fig4_umap_hdbscan.png", title="UMAP — HDBSCAN clusters")

    # ── Morphometric summaries ────────────────────────────────────────────────
    summary: pd.DataFrame | None = None
    state_map: dict[int, str] = {}
    if morph_df is not None:
        n_min = min(len(morph_df), len(hdb_labels))
        if len(morph_df) != len(hdb_labels):
            logger.warning(
                "Morphometrics rows (%d) ≠ embedding count (%d); truncating to %d",
                len(morph_df),
                len(hdb_labels),
                n_min,
            )
        morph_df = morph_df.iloc[:n_min].reset_index(drop=True)
        aligned_labels = hdb_labels[:n_min]

        summary = summarize_clusters(morph_df, aligned_labels)
        state_map = interpret_states(summary)

        key_features = ["area", "circularity", "eccentricity", "intensity_entropy", "glcm_contrast_mean"]
        key_features = [f for f in key_features if f in morph_df.columns]
        plot_morphometric_boxplots(morph_df, aligned_labels, key_features, figs_dir / "fig5_boxplots.png")

        cluster_size_path = cluster_results_dir / "cluster_sizes.csv"
        size_df = (
            pd.Series(aligned_labels, name="cluster")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "cluster", "cluster": "count"})
        )
        size_df.to_csv(cluster_size_path, index=False)
        logger.info("Cluster sizes written to %s", cluster_size_path)

        summary.to_parquet(cluster_results_dir / "cluster_summary.parquet", index=False)
        logger.info("Discovered states: %s", state_map)

    logger.info("Stage 5 complete")
    return {
        "umap_2d": umap_2d,
        "labels": hdb_labels,
        "quality": quality,
        "stability": stability,
        "state_map": state_map,
        "summary": summary,
    }


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--embeddings",
    default=None,
    help="Override path to embeddings.npy.",
)
def main(config: str | None, embeddings: str | None) -> None:
    """Run Stage 5: UMAP reduction, HDBSCAN clustering, and state analysis."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    if embeddings:
        import shutil
        target = Path(cfg.project.results_dir) / "embeddings.npy"  # type: ignore[attr-defined]
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(embeddings).resolve() != target.resolve():
            shutil.copy(embeddings, target)
            logger.info("Copied embeddings from %s to %s", embeddings, target)

    results = run_clustering(cfg)
    quality = results["quality"]
    logger.info(
        "Final: silhouette=%.4f  stability=%.4f  states=%s",
        quality.get("silhouette", float("nan")),
        results["stability"],
        results["state_map"],
    )


if __name__ == "__main__":
    main()
