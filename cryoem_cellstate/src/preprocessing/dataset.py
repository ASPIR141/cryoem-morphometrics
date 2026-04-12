"""MONAI dataset for raw CryoEM images.

:class:`CryoEMRawDataset` wraps a directory of raw images (JPEG, TIFF, PNG)
and exposes them as a ``monai.data.Dataset`` so they integrate naturally with
``monai.data.DataLoader`` and the rest of the MONAI pipeline.

Each item is a dict::

    {
        "image": np.ndarray,   # float64, shape (H, W)
        "name":  str,           # file stem, e.g. "Sucrose400mM_trial2_new0000"
        "path":  Path,          # absolute path to the source file
    }

Usage in preprocessing::

    from src.preprocessing.dataset import CryoEMRawDataset

    dataset = CryoEMRawDataset("data/raw")
    for item in dataset:
        raw   = item["image"]   # np.ndarray float64
        name  = item["name"]
        ...

With a DataLoader (batch_size=1 required for variable-size images)::

    from monai.data import DataLoader
    loader = DataLoader(dataset, batch_size=1,
                        collate_fn=CryoEMRawDataset.single_collate)
    for batch in loader:
        raw  = batch["image"][0]   # np.ndarray float64
        name = batch["name"][0]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from monai.data import Dataset
from monai.transforms import Compose, MapTransform, Transform

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS: set[str] = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ── Custom MONAI transforms ───────────────────────────────────────────────────


class LoadCryoEMImaged(MapTransform):
    """Load a CryoEM image file as a float64 grayscale numpy array.

    Supports JPEG, TIFF, and PNG via ``skimage.io``.  The ``"image"`` key
    in the data dict is replaced with the loaded array; the ``"path"`` key
    is updated to a resolved :class:`~pathlib.Path`.

    Parameters
    ----------
    keys:
        Keys whose values are file path strings to load.
    """

    def __init__(self, keys: tuple[str, ...] = ("image",)) -> None:
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        from skimage import io as skio

        d = dict(data)
        for key in self.key_iterator(d):
            path = Path(str(d[key]))
            img = skio.imread(str(path), as_gray=True).astype(np.float64)
            d[key] = img
            d["path"] = path
        return d


# ── Dataset ───────────────────────────────────────────────────────────────────


class CryoEMRawDataset(Dataset):
    """MONAI Dataset for a directory of raw CryoEM images.

    Discovers all supported image files in *raw_dir*, loads them as float64
    grayscale arrays via :class:`LoadCryoEMImaged`, and optionally applies
    additional transforms.

    Parameters
    ----------
    raw_dir:
        Directory containing raw images (``data/raw/`` by default).
    extensions:
        File extensions to include.  Defaults to
        ``{".tif", ".tiff", ".png", ".jpg", ".jpeg"}``.
    transform:
        Optional additional ``monai.transforms.Transform`` applied *after*
        loading.  Pass your preprocessing ``Compose`` here to fuse loading
        and preprocessing into a single dataset pass.

    Examples
    --------
    Load only::

        ds = CryoEMRawDataset("data/raw")
        item = ds[0]   # {"image": ndarray, "name": str, "path": Path}

    Load + preprocess in one step::

        from src.preprocessing.run_preprocess import build_pipeline
        from src.utils.config import load_config

        cfg = load_config()
        ds  = CryoEMRawDataset("data/raw", transform=build_pipeline(cfg))
    """

    def __init__(
        self,
        raw_dir: str | Path,
        extensions: set[str] | list[str] | None = None,
        transform: Transform | None = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

        exts = set(extensions) if extensions is not None else _DEFAULT_EXTENSIONS
        paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in exts)

        if not paths:
            logger.warning("No images found in %s with extensions %s", raw_dir, exts)

        data = [{"image": str(p), "name": p.stem, "path": p} for p in paths]

        # Always load first; then apply user-supplied transform if any
        load_t: Transform = LoadCryoEMImaged(keys=("image",))
        full_transform = Compose([load_t, transform]) if transform is not None else load_t

        super().__init__(data=data, transform=full_transform)

        logger.info(
            "CryoEMRawDataset: %d images in %s", len(paths), raw_dir
        )

    # ── DataLoader helpers ────────────────────────────────────────────────────

    @staticmethod
    def single_collate(batch: list[dict]) -> dict:
        """Collate function for ``DataLoader(batch_size=1)``.

        Returns the single-element batch as a plain dict so callers can write
        ``batch["image"]`` instead of ``batch["image"][0]``.

        Usage::

            from monai.data import DataLoader
            loader = DataLoader(
                dataset, batch_size=1,
                collate_fn=CryoEMRawDataset.single_collate,
            )
            for item in loader:
                raw = item["image"]   # np.ndarray, not a list
        """
        assert len(batch) == 1, "single_collate is only for batch_size=1"
        return batch[0]

    @staticmethod
    def variable_size_collate(batch: list[dict]) -> list[dict]:
        """Collate function for variable-size images.

        Returns the batch as a list of dicts so spatial arrays are never
        stacked.  Use this when images have different spatial dimensions.

        Usage::

            loader = DataLoader(
                dataset, batch_size=4,
                collate_fn=CryoEMRawDataset.variable_size_collate,
            )
            for items in loader:
                for item in items:
                    raw = item["image"]
        """
        return batch
