"""CLAHE normalisation for CryoEM images.

Uses ``monai.transforms.NormalizeIntensity`` for the final per-image z-score
normalisation step.

All classes inherit from ``monai.transforms.Transform`` so they compose
directly with ``monai.transforms.Compose`` alongside built-in MONAI transforms.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from monai.transforms import Transform

logger = logging.getLogger(__name__)


class CLAHE(Transform):
    """Apply Contrast-Limited Adaptive Histogram Equalisation.

    The image is converted to uint8 for OpenCV, CLAHE is applied, then the
    result is returned as float32 in [0, 1].

    Parameters
    ----------
    clip_limit:
        CLAHE contrast-limit threshold.
    tile_grid_size:
        (cols, rows) tile grid for local histogram equalisation.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a float32 image in [0, 1]."""
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {image.shape}")

        img_f = image.astype(np.float64)
        img_min, img_max = img_f.min(), img_f.max()

        if img_max > img_min:
            img_u8 = ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img_f, dtype=np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=tuple(int(x) for x in self.tile_grid_size),
        )
        enhanced_u8 = clahe.apply(img_u8)
        enhanced = enhanced_u8.astype(np.float32) / 255.0
        logger.debug(
            "CLAHE applied (clip_limit=%.2f, tile_grid=%s)",
            self.clip_limit,
            self.tile_grid_size,
        )
        return enhanced


class ZScoreNormalize(Transform):
    """Per-image z-score normalisation via ``monai.transforms.NormalizeIntensity``.

    Uses MONAI's ``NormalizeIntensity`` with ``nonzero=False`` so mean and std
    are computed over all pixels.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a float32 array with zero mean and unit standard deviation.

        Returns the input unchanged if std == 0.
        """
        from monai.transforms import NormalizeIntensity

        if image.std() == 0:
            logger.warning("Image std is 0; skipping z-score normalisation")
            return image.astype(np.float32)

        normalizer = NormalizeIntensity(nonzero=False, channel_wise=True)
        img_chw = image.astype(np.float32)[np.newaxis]  # 1 × H × W
        normalised_chw = normalizer(img_chw)
        return np.asarray(normalised_chw[0])  # H × W


class CLAHEAndNormalize(Transform):
    """Apply CLAHE followed by MONAI z-score normalisation.

    Composed of :class:`CLAHE` and :class:`ZScoreNormalize` — both are
    ``monai.transforms.Transform`` subclasses, so this class itself is also
    usable inside a ``monai.transforms.Compose``.

    Parameters
    ----------
    clip_limit:
        CLAHE clip limit.
    tile_grid_size:
        CLAHE tile grid size.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.clahe = CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        self.zscore = ZScoreNormalize()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the processed float32 image."""
        return self.zscore(self.clahe(image))
