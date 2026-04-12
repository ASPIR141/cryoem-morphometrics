"""Train SimCLR on unlabeled cell crops.

Usage::

    python -m src.ssl.train_simclr --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, EnsureType
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

from .encoders import Encoder
from .simclr import NTXentLoss, SimCLR, build_simclr_transform

logger = logging.getLogger(__name__)

# Pre-augmentation loading pipeline: add channel dim, normalise, convert to tensor.
_LOAD_TRANSFORM = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),  # (H,W) → (1,H,W)
    NormalizeIntensity(nonzero=True),
    EnsureType(dtype=torch.float32),
])


# ── Dataset ───────────────────────────────────────────────────────────────────


class TwoViewDataset(Dataset):
    """Returns two augmented views of each cell crop for contrastive training.

    Each crop is first loaded and normalised via a MONAI ``Compose`` pipeline
    (channel-first + intensity normalisation), then the *augmentation*
    transform is applied independently twice to produce two views.

    Parameters
    ----------
    crops_dir:
        Directory containing per-cell ``.npy`` image crops.
    transform:
        Single-view augmentation transform (e.g. from
        :func:`~src.ssl.simclr.build_simclr_transform`); applied independently
        twice per sample.
    """

    def __init__(self, crops_dir: Path, transform: object) -> None:
        self.paths = sorted((crops_dir / "images").glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy crops found in {crops_dir / 'images'}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.paths[idx]).astype(np.float32)
        t = _LOAD_TRANSFORM(img)   # (1, H, W) normalised tensor
        v1 = self.transform(t)
        v2 = self.transform(t)
        return v1, v2


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(
    model: SimCLR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NTXentLoss,
    device: torch.device,
) -> float:
    """Run one epoch and return average NT-Xent loss."""
    model.train()
    total = 0.0
    for v1, v2 in loader:
        v1, v2 = v1.to(device), v2.to(device)
        z1 = model(v1)
        z2 = model(v2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
def main(config: str | None) -> None:
    """Train SimCLR on cell crops."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    ssl_cfg = cfg.ssl
    simclr_cfg = ssl_cfg.simclr
    ckpt_dir = Path(ssl_cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = simclr_cfg.augmentation
    transform = build_simclr_transform(
        crop_scale=list(aug_cfg.crop_scale),
        flip=aug_cfg.flip,
        rotation_degrees=aug_cfg.rotation_degrees,
        gaussian_blur_kernel=aug_cfg.gaussian_blur_kernel,
        gaussian_noise_std=aug_cfg.gaussian_noise_std,
        image_size=cfg.segmentation.crop_cells.crop_size,
    )

    dataset = TwoViewDataset(Path(cfg.data.crops_dir), transform)
    loader = DataLoader(
        dataset,
        batch_size=simclr_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    encoder = Encoder(
        backbone_name=ssl_cfg.backbone,
        pretrained=False,  # SSL: train from scratch on domain data
        in_channels=1,
        embedding_dim=ssl_cfg.embedding_dim,
    )
    model = SimCLR(encoder, projection_dim=simclr_cfg.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=simclr_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=simclr_cfg.epochs
    )
    criterion = NTXentLoss(temperature=simclr_cfg.temperature)

    best_loss = float("inf")
    for epoch in range(1, simclr_cfg.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        logger.info("Epoch %3d/%d  loss=%.4f", epoch, simclr_cfg.epochs, loss)

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "loss": loss},
                ckpt_dir / "best_simclr.pth",
            )
            logger.info("Checkpoint saved (loss=%.4f)", loss)

    logger.info("SimCLR training complete. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
