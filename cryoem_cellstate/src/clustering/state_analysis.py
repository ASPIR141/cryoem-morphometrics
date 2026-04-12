"""Per-cluster morphometric summaries and quality metrics.

Plotting functions have been moved to ``src.utils.plots`` and are re-exported
here for convenience.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.utils.plots import plot_morphometric_boxplots, plot_umap  # re-export

logger = logging.getLogger(__name__)

_STATE_LABELS = {
    "dense_shrunken": "Dense / Shrunken",
    "intermediate": "Intermediate",
    "expanded_irregular": "Expanded / Irregular",
    "recovered": "Recovered / Normal",
}

__all__ = [
    "compute_cluster_quality",
    "compute_cluster_stability",
    "summarize_clusters",
    "interpret_states",
    "plot_umap",
    "plot_morphometric_boxplots",
]


# ── Quality metrics ────────────────────────────────────────────────────────────


def compute_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute silhouette score and Davies–Bouldin index.

    Noise points (label == -1) are excluded.

    Parameters
    ----------
    embeddings:
        (N × D) embedding array.
    labels:
        (N,) cluster label array.

    Returns
    -------
    metrics:
        Dict with ``silhouette`` and ``davies_bouldin``.
    """
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        logger.warning("Too few valid clusters to compute quality metrics")
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}

    sil = float(silhouette_score(embeddings[valid], labels[valid]))
    db = float(davies_bouldin_score(embeddings[valid], labels[valid]))
    logger.info("Cluster quality — silhouette=%.4f  davies_bouldin=%.4f", sil, db)
    return {"silhouette": sil, "davies_bouldin": db}


def compute_cluster_stability(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 100,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> float:
    """Estimate cluster label stability via bootstrap resampling.

    Parameters
    ----------
    embeddings:
        (N × D) embedding array.
    labels:
        Reference cluster labels (N,).
    n_iterations:
        Number of bootstrap iterations.
    sample_fraction:
        Fraction of data to sample per iteration.
    random_state:
        Base random seed.

    Returns
    -------
    mean_ari:
        Mean Adjusted Rand Index over all bootstrap iterations.
    """
    import hdbscan
    from sklearn.metrics import adjusted_rand_score

    n_clusters = len(set(labels[labels != -1]))
    if n_clusters < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    aris: list[float] = []
    n = len(embeddings)

    for _ in range(n_iterations):
        idx = rng.choice(n, size=int(n * sample_fraction), replace=False)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_clusters), min_samples=1)
        boot_labels = clusterer.fit_predict(embeddings[idx])
        aris.append(float(adjusted_rand_score(labels[idx], boot_labels)))

    mean_ari = float(np.mean(aris))
    logger.info("Cluster stability (bootstrap ARI): %.4f", mean_ari)
    return mean_ari


# ── Per-cluster morphometric summary ──────────────────────────────────────────


def summarize_clusters(
    morphometrics_df: pd.DataFrame,
    labels: np.ndarray,
    label_col: str = "cluster",
) -> pd.DataFrame:
    """Add cluster labels to morphometrics and compute per-cluster summary stats.

    Parameters
    ----------
    morphometrics_df:
        DataFrame from Stage 4 with one row per cell.
    labels:
        (N,) cluster label array aligned with *morphometrics_df*.
    label_col:
        Column name for the cluster labels.

    Returns
    -------
    summary_df:
        Per-cluster mean ± std for all numeric features.
    """
    df = morphometrics_df.copy()
    df[label_col] = labels
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != label_col]
    return df.groupby(label_col)[numeric_cols].agg(["mean", "std"])


def interpret_states(summary: pd.DataFrame) -> dict[int, str]:
    """Assign a candidate state name to each cluster based on area and circularity.

    Parameters
    ----------
    summary:
        Per-cluster summary DataFrame from :func:`summarize_clusters`.

    Returns
    -------
    state_map:
        Dict mapping cluster label → state name string.
    """
    state_map: dict[int, str] = {}
    try:
        area_mean = summary[("area", "mean")]
        circ_mean = summary[("circularity", "mean")]
        area_med = area_mean.median()
        circ_med = circ_mean.median()

        for cluster in area_mean.index:
            a = area_mean[cluster]
            c = circ_mean[cluster]
            if c >= circ_med * 1.1:
                state_map[int(cluster)] = _STATE_LABELS["recovered"]
            elif a < area_med and c < circ_med:
                state_map[int(cluster)] = _STATE_LABELS["dense_shrunken"]
            elif a >= area_med and c < circ_med:
                state_map[int(cluster)] = _STATE_LABELS["expanded_irregular"]
            else:
                state_map[int(cluster)] = _STATE_LABELS["intermediate"]
    except KeyError:
        logger.warning("Could not compute state interpretation (missing area/circularity)")

    return state_map
