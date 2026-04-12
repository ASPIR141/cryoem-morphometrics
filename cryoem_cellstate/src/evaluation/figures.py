"""Backward-compatibility re-exports for evaluation figures.

All plotting functions now live in :mod:`src.utils.plots`.
Import from there directly in new code.
"""

from src.utils.plots import (
    plot_diffusion_map as figure7_diffusion_map,
    plot_qa_gallery as figure1_preprocessing_grid,
    plot_segmentation_overlay as figure2_segmentation_overlay,
    plot_significance_heatmap,
    plot_size_entropy_histograms as figure3_size_entropy_histograms,
)

__all__ = [
    "figure1_preprocessing_grid",
    "figure2_segmentation_overlay",
    "figure3_size_entropy_histograms",
    "figure7_diffusion_map",
    "plot_significance_heatmap",
]
