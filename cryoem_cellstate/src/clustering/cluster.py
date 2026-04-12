"""HDBSCAN clustering on PCA → UMAP reduced embeddings.

:class:`HDBSCANCluster` is a ``monai.transforms.Transform`` subclass so it
slots into MONAI ``Compose`` pipelines alongside :class:`~src.clustering.umap_reduce.PCAReducer`
and :class:`~src.clustering.umap_reduce.UMAPReducer`.
"""

from __future__ import annotations

import logging

import numpy as np
from monai.transforms import Transform

logger = logging.getLogger(__name__)


class HDBSCANCluster(Transform):
    """Cluster embeddings with HDBSCAN.

    Parameters
    ----------
    min_cluster_size:
        Minimum number of samples to form a cluster.
    min_samples:
        Minimum samples in a neighbourhood for core point definition.
    metric:
        Distance metric.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and return cluster labels.

        Parameters
        ----------
        data:
            (N × D) reduced embeddings.

        Returns
        -------
        labels:
            Integer cluster labels (N,).  ``-1`` denotes noise points.
        """
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        labels: np.ndarray = clusterer.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = (labels == -1).mean()
        logger.info(
            "HDBSCAN: %d clusters  noise_fraction=%.3f", n_clusters, noise_frac
        )
        return labels


# ── Convenience helpers ────────────────────────────────────────────────────────


def run_hdbscan(
    data: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster with HDBSCAN.  Thin wrapper around :class:`HDBSCANCluster`."""
    return HDBSCANCluster(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )(data)


def cluster_embeddings(
    umap_cluster: np.ndarray,
    cfg: object,
) -> dict[str, np.ndarray]:
    """Run HDBSCAN and return label array.

    Parameters
    ----------
    umap_cluster:
        (N × D) cluster-space embeddings (output of PCA → UMAP).
    cfg:
        Root pipeline config.

    Returns
    -------
    labels:
        Dict with key ``"hdbscan"`` → (N,) label array.
    """
    cluster_cfg = cfg.clustering  # type: ignore[attr-defined]
    hdb_labels = HDBSCANCluster(
        min_cluster_size=cluster_cfg.hdbscan.min_cluster_size,
        min_samples=cluster_cfg.hdbscan.min_samples,
        metric=cluster_cfg.hdbscan.metric,
    )(umap_cluster)
    return {"hdbscan": hdb_labels}
