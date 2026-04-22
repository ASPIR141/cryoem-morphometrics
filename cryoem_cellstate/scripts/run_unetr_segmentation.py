"""Run Swin UNETR inference on all images in ``data/processed/`` (or a custom dir).

Loads a trained Swin UNETR Lightning checkpoint, applies the same deterministic
MONAI transform pipeline used during validation, and writes binary masks to
``data/masks/``.

Usage::

    # From repo root
    python cryoem_cellstate/scripts/run_unetr_segmentation.py \\
        --config cryoem_cellstate/configs/default.yaml \\
        --checkpoint results/seg_checkpoints/best_swin_unetr.ckpt

    # Override input/output dirs
    python cryoem_cellstate/scripts/run_unetr_segmentation.py \\
        --input-dir data/processed \\
        --output-dir data/masks \\
        --checkpoint results/seg_checkpoints/best_swin_unetr.ckpt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
)

from src.preprocessing.dataset import CryoEMRawDataset
from src.segmentation.train_unet_lightning import SwinUNETRLightningModule
from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

_CROP_SIZE = 256  # must match training CROP_SIZE (divisible by 32)


def _inference_transform(img_size: int) -> Compose:
    """Deterministic MONAI dict-transform for inference (matches val pipeline)."""
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(img_size, img_size)),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    """Run Swin UNETR inference with sliding-window inferer and save binary masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sigmoid = Activations(sigmoid=True)
    binarise = AsDiscrete(threshold=threshold)
    inferer = SlidingWindowInferer(
        roi_size=(img_size, img_size),
        sw_batch_size=4,
        overlap=0.25,
        mode="gaussian",
    )

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images: torch.Tensor = batch["image"].to(device)
            names: list[str] = batch["name"]

            logits = inferer(inputs=images, network=model)
            preds = binarise(sigmoid(logits))   # (B, 1, H, W)

            for pred, name in zip(preds, names):
                mask = pred.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                out_path = output_dir / f"{name}_mask.npy"
                np.save(out_path, mask)
                logger.info("Saved mask → %s  (non-zero: %d px)", out_path, int((mask > 0).sum()))


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--checkpoint",
    required=True,
    help="Path to Swin UNETR Lightning checkpoint (.ckpt).",
)
@click.option("--input-dir", default=None, help="Images to segment. Defaults to data.processed_dir.")
@click.option("--output-dir", default=None, help="Output mask directory. Defaults to data.masks_dir.")
@click.option("--batch-size", default=4, show_default=True, help="Inference batch size.")
@click.option("--threshold", default=0.5, show_default=True, help="Sigmoid binarisation threshold.")
def main(
    config: str | None,
    checkpoint: str,
    input_dir: str | None,
    output_dir: str | None,
    batch_size: int,
    threshold: float,
) -> None:
    """Segment all images with a trained Swin UNETR and write masks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    in_path = Path(input_dir or cfg.data.processed_dir)  # type: ignore[attr-defined]
    out_path = Path(output_dir or cfg.data.masks_dir)  # type: ignore[attr-defined]
    ckpt_path = Path(checkpoint)

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    module = SwinUNETRLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, crop_size=_CROP_SIZE
    )
    model = module.model.to(device)
    logger.info("Loaded Swin UNETR checkpoint from %s", ckpt_path)

    dataset = CryoEMRawDataset(
        raw_dir=in_path,
        transform=_inference_transform(_CROP_SIZE),
    )
    if len(dataset) == 0:
        logger.error("No images found in %s", in_path)
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    run_inference(model=model, loader=loader, device=device,
                  img_size=_CROP_SIZE, output_dir=out_path, threshold=threshold)
    logger.info("Swin UNETR segmentation complete — masks written to %s", out_path)


if __name__ == "__main__":
    main()
