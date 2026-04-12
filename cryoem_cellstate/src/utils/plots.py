"""Centralised plotting utilities for the CryoEM cell-state pipeline.

All visualisation functions live here so that stage modules stay free of
matplotlib logic.  Every function writes a PNG to disk and closes the figure.

Public API
----------
Noise / preprocessing
    plot_psd            – radially averaged power-spectral-density curve
    plot_snr_histogram  – intensity histogram with fitted Gaussian
    plot_qa_gallery     – before/after preprocessing grid

Segmentation
    plot_segmentation_overlay  – prediction + GT blended with monai.visualize

Morphometrics
    plot_size_entropy_histograms – cell area and entropy distributions

Clustering
    plot_umap                   – 2-D UMAP scatter coloured by cluster
    plot_morphometric_boxplots  – per-cluster feature box plots
    plot_diffusion_map          – pseudo-temporal diffusion / spectral map

Evaluation
    plot_significance_heatmap   – ANOVA -log10(p) bar chart

Internal
    _save                       – shared figure-save-and-close helper
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Internal helper ────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save *fig* to *path* and close it.

    Parameters
    ----------
    fig:
        Matplotlib figure to persist.
    path:
        Output file path (PNG).
    dpi:
        Resolution.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Figure saved → %s", out)


# ── Noise / preprocessing ──────────────────────────────────────────────────────


def plot_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    image_name: str,
    output_path: str | Path,
) -> None:
    """Radially averaged Power Spectral Density curve.

    Parameters
    ----------
    frequencies:
        Normalised spatial frequencies (0–0.5).
    psd:
        Mean power at each frequency bin.
    image_name:
        Title label.
    output_path:
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(frequencies, psd)
    ax.set_xlabel("Normalised spatial frequency")
    ax.set_ylabel("Power (log)")
    ax.set_title(f"PSD — {image_name}")
    fig.tight_layout()
    _save(fig, output_path, dpi=100)


def plot_snr_histogram(
    image: np.ndarray,
    image_name: str,
    snr: float,
    noise_params: dict[str, float],
    output_path: str | Path,
) -> None:
    """Intensity histogram with fitted Gaussian noise model overlay.

    Parameters
    ----------
    image:
        2-D float image.
    image_name:
        Title label.
    snr:
        Pre-computed SNR value shown in the title.
    noise_params:
        Dict from :class:`~src.preprocessing.noise_stats.FitNoiseModel` with
        keys ``gaussian_mu``, ``gaussian_sigma``, ``gaussian_amp``.
    output_path:
        Output PNG path.
    """
    from src.preprocessing.noise_stats import _gaussian

    flat = image.flatten()
    fig, ax = plt.subplots(figsize=(6, 4))
    _, bins_, _ = ax.hist(flat, bins=256, density=True, alpha=0.6, label="histogram")
    bin_c = 0.5 * (bins_[:-1] + bins_[1:])
    fit = _gaussian(bin_c, noise_params["gaussian_mu"], noise_params["gaussian_sigma"], noise_params["gaussian_amp"])
    ax.plot(bin_c, fit, "r-", lw=2, label="Gaussian fit")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Intensity histogram — {image_name}  (SNR={snr:.2f})")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path, dpi=100)


def plot_qa_gallery(
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: str | Path,
    ncols: int = 4,
) -> None:
    """Before / after preprocessing grid for QA.

    Parameters
    ----------
    pairs:
        List of ``(raw_image, processed_image, name)`` triples.
    output_path:
        Output PNG path.
    ncols:
        Image-pair columns per row.
    """
    n = len(pairs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 3, nrows * 6))

    if axes.ndim == 1:
        axes = axes.reshape(-1, ncols)

    for idx, (before, after, name) in enumerate(pairs):
        row_before = (idx // ncols) * 2
        row_after = row_before + 1
        col = idx % ncols
        axes[row_before, col].imshow(before, cmap="gray")
        axes[row_before, col].set_title(f"{name}\nBefore", fontsize=7)
        axes[row_before, col].axis("off")
        axes[row_after, col].imshow(after, cmap="gray")
        axes[row_after, col].set_title("After", fontsize=7)
        axes[row_after, col].axis("off")

    for extra in range(n * 2, nrows * 2 * ncols):
        r, c = divmod(extra, ncols)
        if r < axes.shape[0]:
            axes[r, c].axis("off")

    fig.suptitle("Preprocessing QA — before (top) / after (bottom)", fontsize=10)
    fig.tight_layout()
    _save(fig, output_path, dpi=100)


# ── Segmentation ───────────────────────────────────────────────────────────────


def plot_segmentation_overlay(
    images: list[np.ndarray],
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray] | None,
    names: list[str],
    output_path: str | Path,
    max_cols: int = 4,
) -> None:
    """Segmentation examples with prediction and optional GT overlay.

    Uses ``monai.visualize.blend_images`` to composite masks over images.

    Parameters
    ----------
    images:
        List of 2-D grayscale images.
    pred_masks:
        Predicted binary masks.
    gt_masks:
        CVAT ground-truth masks, or *None*.
    names:
        Image names for titles.
    output_path:
        Output PNG path.
    max_cols:
        Maximum subplot columns.
    """
    import torch
    from monai.visualize import blend_images

    n = min(len(images), max_cols * 2)
    ncols = min(n, max_cols)
    n_sub = 2 if gt_masks is not None else 1
    nrows = ((n + ncols - 1) // ncols) * n_sub

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if axes.ndim == 1:
        axes = axes.reshape(-1, ncols)

    for idx in range(n):
        row_pred = (idx // ncols) * n_sub
        col = idx % ncols

        img = images[idx].astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)

        img_t = torch.from_numpy(img).unsqueeze(0)
        pred_t = torch.from_numpy((pred_masks[idx] > 0).astype(np.float32)).unsqueeze(0)

        blended_pred = blend_images(image=img_t, label=pred_t, alpha=0.35, cmap="cool")
        axes[row_pred, col].imshow(blended_pred.permute(1, 2, 0).numpy())
        axes[row_pred, col].set_title(f"{names[idx]}\nPrediction", fontsize=7)
        axes[row_pred, col].axis("off")

        if gt_masks is not None and gt_masks[idx] is not None:
            gt_t = torch.from_numpy((gt_masks[idx] > 0).astype(np.float32)).unsqueeze(0)
            blended_gt = blend_images(image=img_t, label=gt_t, alpha=0.35, cmap="autumn")
            axes[row_pred + 1, col].imshow(blended_gt.permute(1, 2, 0).numpy())
            axes[row_pred + 1, col].set_title("Ground Truth", fontsize=7)
            axes[row_pred + 1, col].axis("off")

    for extra in range(n * n_sub, nrows * ncols):
        r, c = divmod(extra, ncols)
        if r < axes.shape[0]:
            axes[r, c].axis("off")

    fig.suptitle("Segmentation overlays (MONAI blend_images)")
    fig.tight_layout()
    _save(fig, output_path)


# ── Morphometrics ──────────────────────────────────────────────────────────────


def plot_size_entropy_histograms(
    morphometrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Cell area and intensity-entropy histograms.

    Parameters
    ----------
    morphometrics_df:
        Per-cell morphometrics DataFrame from Stage 4.
    output_path:
        Output PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if "area" in morphometrics_df.columns:
        ax1.hist(morphometrics_df["area"].dropna(), bins=50, color="steelblue", edgecolor="white")
        ax1.set_xlabel("Cell area (pixels)")
        ax1.set_ylabel("Count")
        ax1.set_title("Cell size distribution")

    if "intensity_entropy" in morphometrics_df.columns:
        ax2.hist(morphometrics_df["intensity_entropy"].dropna(), bins=50, color="coral", edgecolor="white")
        ax2.set_xlabel("Shannon entropy (bits)")
        ax2.set_ylabel("Count")
        ax2.set_title("Intensity entropy distribution")

    fig.tight_layout()
    _save(fig, output_path)


# ── Clustering ─────────────────────────────────────────────────────────────────


def plot_umap(
    umap_2d: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "UMAP coloured by cluster",
) -> None:
    """Scatter plot of 2-D UMAP projection coloured by cluster label.

    Parameters
    ----------
    umap_2d:
        (N × 2) UMAP coordinates.
    labels:
        (N,) integer cluster labels (``-1`` = noise).
    output_path:
        Output PNG path.
    title:
        Figure title.
    """
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        colour = "lightgray" if lbl == -1 else cmap(i)
        label_str = "noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=[colour], label=label_str, s=10, alpha=0.7, linewidths=0)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, output_path)


def plot_morphometric_boxplots(
    morphometrics_df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    output_path: str | Path,
) -> None:
    """Box plots of selected morphometric features per cluster.

    Parameters
    ----------
    morphometrics_df:
        Per-cell morphometrics DataFrame.
    labels:
        Cluster labels aligned with *morphometrics_df* rows.
    features:
        Feature column names to plot.
    output_path:
        Output PNG path.
    """
    df = morphometrics_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]

    n_feat = len(features)
    ncols = min(4, n_feat)
    nrows = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes_flat = np.array(axes).flatten() if n_feat > 1 else [axes]

    for i, feat in enumerate(features):
        if feat not in df.columns:
            continue
        groups = [df[df["cluster"] == c][feat].dropna().values for c in sorted(df["cluster"].unique())]
        axes_flat[i].boxplot(groups, labels=[f"C{c}" for c in sorted(df["cluster"].unique())])
        axes_flat[i].set_title(feat, fontsize=9)
        axes_flat[i].set_xlabel("Cluster")

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Per-cluster morphometric distributions", fontsize=11)
    fig.tight_layout()
    _save(fig, output_path)


def plot_diffusion_map(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Pseudo-temporal trajectory via spectral embedding / diffusion map.

    Parameters
    ----------
    embeddings:
        (N × D) embedding array.
    labels:
        Cluster labels for colouring.
    output_path:
        Output PNG path.
    """
    try:
        from sklearn.manifold import SpectralEmbedding

        coords = SpectralEmbedding(n_components=2, affinity="rbf", random_state=42).fit_transform(embeddings)
    except Exception:
        logger.warning("Diffusion map failed; skipping", exc_info=True)
        return

    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], label=f"Cluster {lbl}" if lbl != -1 else "noise", s=12, alpha=0.7, linewidths=0)

    ax.set_title("Pseudo-temporal trajectory (diffusion map)")
    ax.set_xlabel("DC-1")
    ax.set_ylabel("DC-2")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


# ── Evaluation ─────────────────────────────────────────────────────────────────


def plot_significance_heatmap(
    anova_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of -log10(p_corrected) for the top ANOVA features.

    Parameters
    ----------
    anova_df:
        Results DataFrame from :func:`~src.evaluation.metrics.anova_per_feature`.
    output_path:
        Output PNG path.
    top_n:
        Maximum number of features to display.
    """
    if anova_df.empty:
        logger.warning("ANOVA results empty; skipping significance heatmap")
        return

    df = anova_df.sort_values("p_corrected").head(top_n)
    values = -np.log10(df["p_corrected"].clip(lower=1e-300).values)

    fig, ax = plt.subplots(figsize=(4, max(4, len(df) * 0.4)))
    ax.barh(df["feature"], values, color="steelblue")
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    ax.set_xlabel("-log₁₀(p corrected)")
    ax.set_title("ANOVA significance per morphometric feature")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)
