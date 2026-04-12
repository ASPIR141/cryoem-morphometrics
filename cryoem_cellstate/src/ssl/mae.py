"""Masked Autoencoder (MAE) training via HuggingFace ViT-MAE.

Uses ``facebook/vit-mae-base`` as starting point, fine-tuned on CryoEM crops.

Usage::

    python -m src.ssl.train_mae --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


class CropDataset(Dataset):
    """Single-view dataset of cell crops for MAE training.

    Parameters
    ----------
    crops_dir:
        Directory containing per-cell ``.npy`` image crops.
    image_size:
        Resize crops to this square size for ViT compatibility.
    """

    def __init__(self, crops_dir: Path, image_size: int = 224) -> None:
        self.paths = sorted((crops_dir / "images").glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy crops found in {crops_dir / 'images'}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        img = np.load(self.paths[idx]).astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0)  # 1 × H × W
        # ViT-MAE expects 3-channel input; replicate grayscale
        t3 = t.repeat(3, 1, 1)
        t3 = TF.resize(t3, [self.image_size, self.image_size], antialias=True)
        return t3


def _psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Peak signal-to-noise ratio between two tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def _ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Simplified SSIM (single-value) using scikit-image."""
    from skimage.metrics import structural_similarity as sk_ssim

    o = original.cpu().numpy().mean(axis=0)   # H × W
    r = reconstructed.cpu().numpy().mean(axis=0)
    score: float = sk_ssim(o, r, data_range=float(o.max() - o.min()))
    return score


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
def main(config: str | None) -> None:
    """Fine-tune ViT-MAE on CryoEM crops."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    mae_cfg = cfg.ssl.mae
    ckpt_dir = Path(cfg.ssl.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from transformers import ViTMAEConfig, ViTMAEForPreTraining

    mae_config = ViTMAEConfig(
        mask_ratio=mae_cfg.mask_ratio,
        num_channels=3,
    )
    model = ViTMAEForPreTraining(mae_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mae_cfg.lr)

    dataset = CropDataset(Path(cfg.data.crops_dir))
    loader = DataLoader(dataset, batch_size=mae_cfg.batch_size, shuffle=True)

    best_loss = float("inf")
    for epoch in range(1, mae_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            outputs = model(pixel_values=batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Log reconstruction quality on first batch
        model.eval()
        with torch.no_grad():
            sample = next(iter(loader)).to(device)
            out = model(pixel_values=sample)
            # Reconstruct: MAE patches back to image
            recon = model.unpatchify(out.logits)
            psnr = _psnr(sample[:1], recon[:1])
            ssim = _ssim(sample[0], recon[0])

        logger.info(
            "MAE Epoch %3d/%d  loss=%.4f  PSNR=%.2f  SSIM=%.4f",
            epoch,
            mae_cfg.epochs,
            avg_loss,
            psnr,
            ssim,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best_mae.pth")

    logger.info("MAE training complete. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
