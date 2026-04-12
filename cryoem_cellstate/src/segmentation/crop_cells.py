"""Extract individual cell crops from segmentation masks.

Produces per-cell image crops (padded to square) and a ``cells.parquet``
catalogue with metadata.

Steps are implemented as ``monai.transforms.Transform`` subclasses so they
can be composed into larger MONAI pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from monai.transforms import Transform
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


class PadToSquare(Transform):
    """Resize and pad an image patch to a fixed square size.

    Parameters
    ----------
    size:
        Target square side length in pixels.
    """

    def __init__(self, size: int = 128) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Resize *image* to a (size × size) float32 array.

        Parameters
        ----------
        image:
            2-D crop of any size.

        Returns
        -------
        padded:
            Float32 array of shape ``(size, size)``.
        """
        from skimage.transform import resize

        return resize(
            image.astype(np.float32),
            (self.size, self.size),
            anti_aliasing=True,
        )


class CellCropper(Transform):
    """Extract, filter, and save individual cell crops from an image/mask pair.

    Each accepted connected component is resized to a square by
    :class:`PadToSquare` and saved as ``.npy`` under *crops_dir*.

    Parameters
    ----------
    crops_dir:
        Root output directory; sub-directories ``images/`` and ``masks/``
        are created automatically.
    min_area:
        Minimum component area (pixels) to keep.
    max_aspect_ratio:
        Bounding-box aspect ratio above which a cell is discarded.
    crop_size:
        Side length of the output square crop.
    """

    def __init__(
        self,
        crops_dir: str | Path,
        min_area: int = 500,
        max_aspect_ratio: float = 4.0,
        crop_size: int = 128,
    ) -> None:
        self.crops_dir = Path(crops_dir)
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self._pad = PadToSquare(size=crop_size)

    def __call__(
        self,
        data: dict[str, object],
    ) -> list[dict[str, object]]:
        """Extract crops for all valid cells in a single image.

        Parameters
        ----------
        data:
            Dict with keys:

            * ``"image"`` — 2-D float preprocessed image.
            * ``"mask"``  — 2-D uint8/bool binary mask (same H × W).
            * ``"source"`` — string identifier for the source image.

        Returns
        -------
        records:
            List of metadata dicts with keys ``cell_id``, ``source``,
            ``bbox``, ``area``, ``crop_path``, ``mask_path``.
        """
        image: np.ndarray = data["image"]  # type: ignore[assignment]
        mask: np.ndarray = data["mask"]    # type: ignore[assignment]
        source_name: str = data["source"]  # type: ignore[assignment]

        crop_img_dir = self.crops_dir / "images"
        crop_mask_dir = self.crops_dir / "masks"
        crop_img_dir.mkdir(parents=True, exist_ok=True)
        crop_mask_dir.mkdir(parents=True, exist_ok=True)

        binary = (mask > 0).astype(np.uint8)
        labeled = label(binary)
        props = regionprops(labeled, intensity_image=image)

        records: list[dict[str, object]] = []
        kept = 0

        for prop in props:
            if prop.area < self.min_area:
                continue
            min_row, min_col, max_row, max_col = prop.bbox
            h = max_row - min_row
            w = max_col - min_col
            aspect = max(h, w) / max(min(h, w), 1)
            if aspect > self.max_aspect_ratio:
                continue

            pad = 4
            r0 = max(0, min_row - pad)
            r1 = min(image.shape[0], max_row + pad)
            c0 = max(0, min_col - pad)
            c1 = min(image.shape[1], max_col + pad)

            img_crop = self._pad(image[r0:r1, c0:c1])
            mask_crop = self._pad(
                (labeled[r0:r1, c0:c1] == prop.label).astype(np.float32)
            )

            cell_id = f"{source_name}_cell{prop.label:04d}"
            crop_path = crop_img_dir / f"{cell_id}.npy"
            mask_path = crop_mask_dir / f"{cell_id}.npy"

            np.save(crop_path, img_crop)
            np.save(mask_path, mask_crop)

            records.append(
                {
                    "cell_id": cell_id,
                    "source": source_name,
                    "bbox": f"{r0},{c0},{r1},{c1}",
                    "area": prop.area,
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                }
            )
            kept += 1

        logger.info(
            "%s: extracted %d crops (from %d components)",
            source_name,
            kept,
            len(props),
        )
        return records


# ---------------------------------------------------------------------------
# Backward-compatible function shims
# ---------------------------------------------------------------------------

def crop_cells(
    image: np.ndarray,
    mask: np.ndarray,
    source_name: str,
    crops_dir: str | Path,
    min_area: int = 500,
    max_aspect_ratio: float = 4.0,
    crop_size: int = 128,
) -> list[dict[str, object]]:
    """Extract and save individual cell crops from an image+mask pair.

    Thin wrapper around :class:`CellCropper`.  Prefer the class directly.
    """
    return CellCropper(
        crops_dir=crops_dir,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        crop_size=crop_size,
    )({"image": image, "mask": mask, "source": source_name})


def build_cells_parquet(
    records: list[dict[str, object]],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write the cells catalogue to a Parquet file.

    Parameters
    ----------
    records:
        List of record dicts from :func:`crop_cells` / :class:`CellCropper`.
    output_path:
        Destination ``.parquet`` path.

    Returns
    -------
    df:
        The catalogue DataFrame.
    """
    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("cells.parquet written to %s  (%d rows)", output_path, len(df))
    return df
