"""Extract backbone embeddings from all cell crops using a trained SimCLR encoder.

Saves ``embeddings.npy`` aligned row-for-row with ``cells.parquet``.

Usage::

    python -m src.ssl.extract_embeddings --config configs/default.yaml \\
        --checkpoint results/ssl_checkpoints/best_simclr.pth
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, NormalizeIntensity
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

from .encoders import Encoder
from .simclr import SimCLR

logger = logging.getLogger(__name__)

# Deterministic loading pipeline for inference (no augmentation).
_INFERENCE_TRANSFORM = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),  # (H,W) → (1,H,W)
    NormalizeIntensity(nonzero=True),
    EnsureType(dtype=torch.float32),
])


class CropInferenceDataset(Dataset):
    """Ordered dataset of cell crops for embedding extraction.

    Each crop is loaded from ``.npy``, normalised, and converted to a
    ``(1, H, W)`` float tensor via a MONAI ``Compose`` pipeline so that
    inference uses the same normalisation as training.

    Parameters
    ----------
    cell_ids:
        Ordered list of cell IDs (aligned with *crop_paths*).
    crop_paths:
        Ordered list of ``.npy`` crop paths.
    """

    def __init__(self, cell_ids: list[str], crop_paths: list[str]) -> None:
        self.cell_ids = cell_ids
        self.crop_paths = crop_paths

    def __len__(self) -> int:
        return len(self.crop_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.load(self.crop_paths[idx]).astype(np.float32)
        return _INFERENCE_TRANSFORM(img)   # (1, H, W) normalised tensor


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to SimCLR checkpoint. Defaults to <checkpoints_dir>/best_simclr.pth.",
)
@click.option(
    "--output",
    default=None,
    help="Path for output embeddings.npy. Defaults to <results_dir>/embeddings.npy.",
)
def main(config: str | None, checkpoint: str | None, output: str | None) -> None:
    """Extract and save backbone embeddings for all cells."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    ssl_cfg = cfg.ssl
    ckpt_path = Path(checkpoint or (Path(ssl_cfg.checkpoints_dir) / "best_simclr.pth"))
    out_path = Path(output or (Path(cfg.project.results_dir) / "embeddings.npy"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cells catalogue
    cells_parquet = Path(cfg.data.crops_dir) / "cells.parquet"
    if not cells_parquet.exists():
        cells_parquet = Path(cfg.project.results_dir) / "cells.parquet"
    cells_df = pd.read_parquet(cells_parquet)

    dataset = CropInferenceDataset(
        cell_ids=cells_df["cell_id"].tolist(),
        crop_paths=cells_df["crop_path"].tolist(),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    # Build model and load weights
    encoder = Encoder(
        backbone_name=ssl_cfg.backbone,
        pretrained=False,
        in_channels=1,
        embedding_dim=ssl_cfg.embedding_dim,
    )
    model = SimCLR(encoder, projection_dim=ssl_cfg.simclr.projection_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint from %s", ckpt_path)

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model.encoder.get_features(batch).cpu().numpy()
            embeddings.append(feats)

    all_embeddings = np.concatenate(embeddings, axis=0)
    np.save(out_path, all_embeddings)
    logger.info(
        "Saved embeddings %s → %s", all_embeddings.shape, out_path
    )


if __name__ == "__main__":
    main()
