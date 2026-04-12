"""Reporting utilities for CryoEM preprocessing.

Combines noise statistics with plot generation from :mod:`src.utils.plots`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.preprocessing.noise_stats import EstimateSNR, FitNoiseModel, RadialPSD
from src.utils.plots import plot_psd, plot_snr_histogram

logger = logging.getLogger(__name__)


def save_noise_report(
    image: np.ndarray,
    image_name: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Compute noise statistics and save PSD + histogram plots for one image.

    Parameters
    ----------
    image:
        2-D float array.
    image_name:
        Base name used for output files and plot titles.
    output_dir:
        Directory where plots are saved.

    Returns
    -------
    metrics:
        Dict with ``snr``, ``gaussian_mu``, ``gaussian_sigma``,
        ``gaussian_amp``, and ``poisson_lambda``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snr = EstimateSNR()(image)
    noise_params = FitNoiseModel()(image)
    freqs, psd = RadialPSD()(image)

    plot_psd(freqs, psd, image_name, out / f"{image_name}_psd.png")
    plot_snr_histogram(image, image_name, snr, noise_params, out / f"{image_name}_snr_hist.png")

    logger.info("Noise report saved for %s  (SNR=%.2f)", image_name, snr)
    return {"snr": snr, **noise_params}
