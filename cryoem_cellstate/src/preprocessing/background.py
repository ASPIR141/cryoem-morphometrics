"""Background subtraction / illumination flattening for CryoEM images.

All classes inherit from ``monai.transforms.Transform`` so they compose
directly with ``monai.transforms.Compose`` alongside built-in MONAI transforms.
"""

from __future__ import annotations

import logging

import numpy as np
from monai.transforms import Transform

logger = logging.getLogger(__name__)


class GaussianBackgroundSubtraction(Transform):
    """Flatten uneven illumination by subtracting a large-sigma Gaussian blur.

    Parameters
    ----------
    sigma:
        Standard deviation of the Gaussian kernel used to estimate the
        low-frequency background.
    """

    def __init__(self, sigma: float = 50.0) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the background-subtracted image."""
        from scipy.ndimage import gaussian_filter

        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {image.shape}")

        background = gaussian_filter(image.astype(np.float64), sigma=self.sigma)
        corrected = image.astype(np.float64) - background
        logger.debug("Gaussian background subtracted (sigma=%.1f)", self.sigma)
        return corrected


class TopHatBackgroundSubtraction(Transform):
    """Apply morphological white top-hat to remove slow-varying background.

    Parameters
    ----------
    radius:
        Radius of the disk-shaped structuring element in pixels.
    """

    def __init__(self, radius: int = 40) -> None:
        super().__init__()
        self.radius = radius

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the top-hat filtered image (foreground signal only)."""
        from skimage.morphology import disk, white_tophat

        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {image.shape}")

        img_norm = image.astype(np.float64)
        img_min, img_max = img_norm.min(), img_norm.max()
        if img_max > img_min:
            img_01 = (img_norm - img_min) / (img_max - img_min)
        else:
            img_01 = img_norm

        corrected_01 = white_tophat(img_01, footprint=disk(self.radius))
        corrected = corrected_01 * (img_max - img_min)
        logger.debug("Top-hat background subtracted (radius=%d)", self.radius)
        return corrected


class BackgroundSubtraction(Transform):
    """Dispatch to the chosen background subtraction method.

    Wraps either :class:`GaussianBackgroundSubtraction` or
    :class:`TopHatBackgroundSubtraction` based on *method*.

    Parameters
    ----------
    method:
        ``"gaussian"`` (default) or ``"tophat"``.
    sigma:
        Gaussian sigma (used only when *method* == ``"gaussian"``).
    tophat_radius:
        Structuring element radius (used only when *method* == ``"tophat"``).
    """

    def __init__(
        self,
        method: str = "gaussian",
        sigma: float = 50.0,
        tophat_radius: int = 40,
    ) -> None:
        super().__init__()
        if method == "gaussian":
            self._transform: Transform = GaussianBackgroundSubtraction(sigma=sigma)
        elif method == "tophat":
            self._transform = TopHatBackgroundSubtraction(radius=tophat_radius)
        else:
            raise ValueError(
                f"Unknown background method: '{method}'. Choose 'gaussian' or 'tophat'."
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the background-corrected image."""
        return self._transform(image)
