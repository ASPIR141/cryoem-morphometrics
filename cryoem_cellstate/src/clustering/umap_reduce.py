"""Dimensionality reduction pipeline for SSL embeddings.

The full pipeline is:  **PCA → UMAP** so that UMAP operates on a compact,
noise-reduced representation rather than raw high-dimensional features.

Every step is a ``monai.transforms.Transform`` subclass and can be composed
with ``monai.transforms.Compose``.

Typical usage::

    from monai.transforms import Compose
    from src.clustering.umap_reduce import PCAReducer, UMAPReducer

    pipeline = Compose([
        PCAReducer(n_components=50),
        UMAPReducer(n_components=2),
    ])
    umap_2d = pipeline(embeddings)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from monai.transforms import Compose, Transform

logger = logging.getLogger(__name__)


# ── PCA ───────────────────────────────────────────────────────────────────────


class PCAReducer(Transform):
    """PCA dimensionality reduction via scikit-learn.

    Fits PCA on the input array and stores the fitted model in
    ``self.pca`` for later :meth:`transform` calls on new data.

    Parameters
    ----------
    n_components:
        Number of principal components to retain.  ``None`` keeps all.
    whiten:
        Divide components by their standard deviation (unit variance output).
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int | None = 50,
        whiten: bool = False,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca: object = None  # populated on first call

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA and project *embeddings*.

        Parameters
        ----------
        embeddings:
            (N × D) array.

        Returns
        -------
        reduced:
            (N × n_components) array.
        """
        from sklearn.decomposition import PCA

        n_comp = (
            min(self.n_components, embeddings.shape[1], embeddings.shape[0])
            if self.n_components is not None
            else None
        )
        self.pca = PCA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        reduced: np.ndarray = self.pca.fit_transform(embeddings)

        explained = (
            float(self.pca.explained_variance_ratio_.sum())  # type: ignore[union-attr]
            if n_comp is not None
            else 1.0
        )
        logger.info(
            "PCA: %d→%d  explained_variance=%.3f",
            embeddings.shape[1],
            reduced.shape[1],
            explained,
        )
        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project new data with the already-fitted PCA model.

        Raises
        ------
        RuntimeError
            If called before :meth:`__call__` has fitted the model.
        """
        if self.pca is None:
            raise RuntimeError("Fit PCAReducer first by calling it on training data.")
        return self.pca.transform(embeddings)  # type: ignore[union-attr]


# ── UMAP ──────────────────────────────────────────────────────────────────────


class UMAPReducer(Transform):
    """UMAP dimensionality reduction.

    The fitted UMAP object is stored as ``self.reducer`` after the first call
    and can be reused via :meth:`transform`.

    Parameters
    ----------
    n_components:
        Number of UMAP output dimensions.
    n_neighbors:
        UMAP ``n_neighbors`` parameter.
    min_dist:
        UMAP ``min_dist`` parameter.
    metric:
        Distance metric.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer: object = None

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP and return reduced coordinates.

        Parameters
        ----------
        embeddings:
            (N × D) array.

        Returns
        -------
        reduced:
            (N × n_components) array.
        """
        import umap

        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        reduced: np.ndarray = self.reducer.fit_transform(embeddings)
        logger.info(
            "UMAP: %d→%d  n_neighbors=%d  min_dist=%.2f",
            embeddings.shape[1],
            self.n_components,
            self.n_neighbors,
            self.min_dist,
        )
        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project new data with the already-fitted reducer.

        Raises
        ------
        RuntimeError
            If called before :meth:`__call__`.
        """
        if self.reducer is None:
            raise RuntimeError("Fit UMAPReducer first by calling it on training data.")
        return self.reducer.transform(embeddings)  # type: ignore[union-attr]


# ── Convenience helpers ────────────────────────────────────────────────────────


def build_reduction_pipeline(cfg: object) -> Compose:
    """Build a ``monai.transforms.Compose([PCAReducer, UMAPReducer])`` from config.

    Parameters
    ----------
    cfg:
        Root pipeline config (uses ``cfg.clustering.pca`` and
        ``cfg.clustering.umap``).

    Returns
    -------
    pipeline:
        Composed PCA → UMAP transform.
    """
    pca_cfg = cfg.clustering.pca  # type: ignore[attr-defined]
    umap_cfg = cfg.clustering.umap  # type: ignore[attr-defined]
    return Compose([
        PCAReducer(
            n_components=pca_cfg.n_components,
            whiten=pca_cfg.whiten,
            random_state=pca_cfg.random_state,
        ),
        UMAPReducer(
            n_components=umap_cfg.n_components_cluster,
            n_neighbors=umap_cfg.n_neighbors,
            min_dist=umap_cfg.min_dist,
            metric=umap_cfg.metric,
            random_state=umap_cfg.random_state,
        ),
    ])


def reduce_embeddings(
    embeddings_path: str | Path,
    cfg: object,
    results_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings, run PCA → UMAP to 2-D and to cluster-space dim.

    Pipeline for each output:

    * **2-D visualisation**: PCA(n=pca_components) → UMAP(n=2)
    * **Cluster space**: PCA(n=pca_components) → UMAP(n=n_components_cluster)

    Parameters
    ----------
    embeddings_path:
        Path to ``embeddings.npy``.
    cfg:
        Root pipeline config.
    results_dir:
        Directory where outputs are saved.

    Returns
    -------
    umap_2d:
        (N × 2) visualisation array.
    umap_cluster:
        (N × n_components_cluster) clustering array.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embeddings_path)
    pca_cfg = cfg.clustering.pca  # type: ignore[attr-defined]
    umap_cfg = cfg.clustering.umap  # type: ignore[attr-defined]

    # PCA is shared — fit once
    pca = PCAReducer(
        n_components=pca_cfg.n_components,
        whiten=pca_cfg.whiten,
        random_state=pca_cfg.random_state,
    )
    pca_reduced = pca(embeddings)
    np.save(results_dir / "pca_reduced.npy", pca_reduced)

    # 2-D UMAP for visualisation
    umap_2d_reducer = UMAPReducer(
        n_components=umap_cfg.n_components_2d,
        n_neighbors=umap_cfg.n_neighbors,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
        random_state=umap_cfg.random_state,
    )
    umap_2d = umap_2d_reducer(pca_reduced)

    # Higher-dim UMAP for clustering
    umap_cluster_reducer = UMAPReducer(
        n_components=umap_cfg.n_components_cluster,
        n_neighbors=umap_cfg.n_neighbors,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
        random_state=umap_cfg.random_state,
    )
    umap_cluster = umap_cluster_reducer(pca_reduced)

    np.save(results_dir / "umap_2d.npy", umap_2d)
    np.save(results_dir / "umap_cluster.npy", umap_cluster)
    logger.info("Reduction outputs saved to %s", results_dir)
    return umap_2d, umap_cluster
