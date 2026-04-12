"""Noise characterisation for CryoEM images.

Computes:
- Power Spectral Density (PSD) via ``torch.fft``
- Per-image SNR histograms
- Simple Gaussian + Poisson noise model fit

All classes inherit from ``monai.transforms.Transform`` so they can be used
inside a ``monai.transforms.Compose`` pipeline alongside built-in MONAI
transforms.  The analysis classes (``RadialPSD``, ``EstimateSNR``,
``FitNoiseModel``) return statistics rather than images — they share the same
base for API consistency but are not typically placed inside image pipelines.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from monai.transforms import Transform
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian helper used by :class:`FitNoiseModel`."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class PowerSpectrum2D(Transform):
    """Compute the 2-D power spectrum (|FFT|²) of an image.

    Inherits ``monai.transforms.Transform``; returns a numpy array of the
    same spatial shape rather than a transformed image tensor.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the centred power spectrum with the same spatial shape."""
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {image.shape}")
        img_t = torch.from_numpy(image.astype(np.float64))
        fft = torch.fft.fftshift(torch.fft.fft2(img_t))
        return (fft.abs() ** 2).numpy()


class RadialPSD(Transform):
    """Compute the radially averaged Power Spectral Density of a 2-D image.

    Inherits ``monai.transforms.Transform``; returns ``(frequencies, psd)``
    rather than a transformed image.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ps = PowerSpectrum2D()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequencies, psd)`` arrays.

        ``frequencies`` are normalised spatial frequencies in [0, 0.5].
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {image.shape}")

        ps = self._ps(image)
        h, w = ps.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy).astype(int)
        max_r = min(cy, cx)
        psd = np.zeros(max_r)
        for i in range(max_r):
            ring = ps[r == i]
            if ring.size > 0:
                psd[i] = ring.mean()
        frequencies = np.linspace(0.0, 0.5, max_r)
        return frequencies, psd


class EstimateSNR(Transform):
    """Estimate a rough image SNR.

    Signal = mean of pixels above *signal_percentile*; noise = std of all pixels.

    Inherits ``monai.transforms.Transform``; returns a scalar float rather than
    a transformed image.

    Parameters
    ----------
    signal_percentile:
        Intensity threshold for "signal" region.
    """

    def __init__(self, signal_percentile: float = 75.0) -> None:
        super().__init__()
        self.signal_percentile = signal_percentile

    def __call__(self, image: np.ndarray) -> float:
        """Return the signal-to-noise ratio (linear scale)."""
        threshold = np.percentile(image, self.signal_percentile)
        signal_mean = image[image >= threshold].mean()
        noise_std = image.std()
        if noise_std == 0:
            return float("inf")
        return float(signal_mean / noise_std)


class FitNoiseModel(Transform):
    """Fit a Gaussian + Poisson noise model to the intensity histogram.

    Inherits ``monai.transforms.Transform``; returns a statistics dict rather
    than a transformed image.

    Parameters
    ----------
    bins:
        Number of histogram bins.
    """

    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        self.bins = bins

    def __call__(self, image: np.ndarray) -> dict[str, float]:
        """Return dict with ``gaussian_mu``, ``gaussian_sigma``, ``gaussian_amp``, ``poisson_lambda``."""
        flat = image.flatten().astype(np.float64)
        counts, bin_edges = np.histogram(flat, bins=self.bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mu0, sigma0 = flat.mean(), flat.std()
        amp0 = counts.max()
        try:
            popt, _ = curve_fit(
                _gaussian,
                bin_centers,
                counts,
                p0=[mu0, sigma0, amp0],
                maxfev=5000,
            )
            g_mu, g_sigma, g_amp = float(popt[0]), abs(float(popt[1])), float(popt[2])
        except RuntimeError:
            g_mu, g_sigma, g_amp = mu0, sigma0, float(amp0)

        nonneg = flat[flat >= 0]
        p_lambda = float(nonneg.mean()) if len(nonneg) > 0 else 0.0

        return {
            "gaussian_mu": g_mu,
            "gaussian_sigma": g_sigma,
            "gaussian_amp": g_amp,
            "poisson_lambda": p_lambda,
        }
