"""Classical segmentation baseline: threshold → morphological cleanup → watershed.

Every step is a ``monai.transforms.Transform`` subclass so the full pipeline
can be composed with ``monai.transforms.Compose``.
"""

from __future__ import annotations

import logging

import numpy as np
from monai.transforms import Compose, Transform
from skimage import morphology
from skimage.filters import threshold_local, threshold_otsu

logger = logging.getLogger(__name__)


class OtsuThreshold(Transform):
    """Binarise an image using Otsu's global threshold.

    Parameters
    ----------
    None — threshold is computed per-call from the image statistics.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a uint8 binary mask (0/1) via Otsu thresholding.

        Parameters
        ----------
        image:
            2-D float image, normalised to [0, 1].

        Returns
        -------
        mask:
            uint8 binary array (0 = background, 1 = foreground).
        """
        thresh = threshold_otsu(image)
        return (image > thresh).astype(np.uint8)


class AdaptiveThreshold(Transform):
    """Binarise an image using a local (adaptive) Gaussian threshold.

    Parameters
    ----------
    block_size:
        Size of the local neighbourhood window (must be odd; will be forced
        odd if even).
    """

    def __init__(self, block_size: int = 51) -> None:
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a uint8 binary mask via adaptive thresholding.

        Parameters
        ----------
        image:
            2-D float image, normalised to [0, 1].

        Returns
        -------
        mask:
            uint8 binary array.
        """
        thresh = threshold_local(image, block_size=self.block_size, method="gaussian")
        return (image > thresh).astype(np.uint8)


class MorphologicalCleanup(Transform):
    """Remove small objects/holes and close gaps; discard over-large blobs.

    Parameters
    ----------
    min_cell_area:
        Minimum connected-component area in pixels to retain.
    max_cell_area:
        Components larger than this are discarded.
    """

    def __init__(self, min_cell_area: int = 500, max_cell_area: int = 50_000) -> None:
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area

    def __call__(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological cleanup to a binary mask.

        Parameters
        ----------
        binary:
            uint8 or bool binary mask.

        Returns
        -------
        cleaned:
            Cleaned boolean array.
        """
        from skimage.measure import label, regionprops

        cleaned = morphology.remove_small_objects(
            binary.astype(bool), min_size=self.min_cell_area
        )
        cleaned = morphology.remove_small_holes(
            cleaned, area_threshold=self.min_cell_area // 2
        )
        cleaned = morphology.binary_closing(cleaned, footprint=morphology.disk(3))

        # Remove blobs that are too large
        labeled = label(cleaned)
        for prop in regionprops(labeled):
            if prop.area > self.max_cell_area:
                cleaned[labeled == prop.label] = False

        return cleaned


class WatershedSeparation(Transform):
    """Separate touching cells using distance-transform watershed.

    Parameters
    ----------
    footprint_size:
        Side length of the square footprint used for peak detection.
    """

    def __init__(self, footprint_size: int = 5) -> None:
        self.footprint_size = footprint_size

    def __call__(self, cleaned: np.ndarray) -> np.ndarray:
        """Apply watershed separation to a cleaned binary mask.

        Parameters
        ----------
        cleaned:
            Bool or uint8 binary mask.

        Returns
        -------
        separated:
            Boolean mask with separated cells.
        """
        from scipy.ndimage import distance_transform_edt
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        distance = distance_transform_edt(cleaned)
        coords = peak_local_max(
            distance,
            footprint=np.ones((self.footprint_size, self.footprint_size)),
            labels=cleaned,
        )
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        ws_labels = watershed(-distance, markers, mask=cleaned)
        return ws_labels > 0


class ClassicalSegmentation(Transform):
    """Full classical segmentation pipeline: threshold → cleanup → watershed.

    Internally builds a ``monai.transforms.Compose`` from the constituent
    :class:`Transform` steps, so each step is independently replaceable.

    Parameters
    ----------
    method:
        ``"otsu"`` or ``"adaptive"``.
    min_cell_area:
        Minimum component area to retain.
    max_cell_area:
        Maximum component area to retain.
    apply_watershed:
        Whether to apply :class:`WatershedSeparation`.
    """

    def __init__(
        self,
        method: str = "otsu",
        min_cell_area: int = 500,
        max_cell_area: int = 50_000,
        apply_watershed: bool = True,
    ) -> None:
        if method == "otsu":
            threshold_step: Transform = OtsuThreshold()
        elif method == "adaptive":
            threshold_step = AdaptiveThreshold()
        else:
            raise ValueError(f"Unknown segmentation method '{method}'")

        steps: list[Transform] = [
            threshold_step,
            MorphologicalCleanup(min_cell_area=min_cell_area, max_cell_area=max_cell_area),
        ]
        if apply_watershed:
            steps.append(WatershedSeparation())

        self._pipeline = Compose(steps)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run the full segmentation pipeline on a single image.

        Parameters
        ----------
        image:
            2-D float image (should be preprocessed / normalised).

        Returns
        -------
        mask:
            uint8 binary mask (0 = background, 255 = cell).
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D image, got {image.shape}")

        # Normalise to [0, 1] before thresholding
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        cleaned = self._pipeline(img)
        mask = (np.asarray(cleaned) * 255).astype(np.uint8)
        logger.debug(
            "Classical segmentation complete. Non-zero pixels: %d", int((mask > 0).sum())
        )
        return mask


# ---------------------------------------------------------------------------
# Backward-compatible function shim
# ---------------------------------------------------------------------------

def segment_classical(
    image: np.ndarray,
    method: str = "otsu",
    min_cell_area: int = 500,
    max_cell_area: int = 50_000,
    apply_watershed: bool = True,
) -> np.ndarray:
    """Produce a binary cell mask using classical image processing.

    Thin wrapper around :class:`ClassicalSegmentation` kept for backward
    compatibility.  Prefer instantiating the class directly.

    Parameters
    ----------
    image:
        2-D float image (should be preprocessed / normalised).
    method:
        ``"otsu"`` or ``"adaptive"``.
    min_cell_area:
        Minimum connected component area in pixels to retain.
    max_cell_area:
        Maximum area; larger components are discarded.
    apply_watershed:
        Whether to apply watershed separation of touching cells.

    Returns
    -------
    mask:
        uint8 binary mask (0 = background, 255 = cell).
    """
    return ClassicalSegmentation(
        method=method,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        apply_watershed=apply_watershed,
    )(image)
