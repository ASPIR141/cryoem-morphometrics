"""Run UNETR inference on all images in ``data/processed/`` (or a custom dir).

Loads a trained UNETR checkpoint, applies the same deterministic MONAI
transform pipeline used during validation, and writes binary masks to
``data/masks/``.

Usage::

    # From repo root
    python cryoem_cellstate/scripts/run_unetr_segmentation.py \\
        --config cryoem_cellstate/configs/default.yaml \\
        --checkpoint results/seg_checkpoints/best_unetr.pth

    # Override input/output dirs
    python cryoem_cellstate/scripts/run_unetr_segmentation.py \\
        --input-dir data/processed \\
        --output-dir data/masks \\
        --checkpoint results/seg_checkpoints/best_unetr.pth
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the package root is importable when running from the repo root
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
from src.segmentation.unet import build_unetr
from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _inference_transform(img_size: int) -> Compose:
    """MONAI dict-transform pipeline applied to each loaded image before inference.

    Adds a channel dimension, normalises intensity, pads/crops to *img_size*,
    and ensures float32 tensors — matching the validation pipeline in training.
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(img_size, img_size)),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    """Run UNETR inference and save binary masks.

    Uses ``monai.inferers.SlidingWindowInferer`` to handle images larger
    than the model's training size gracefully.

    Parameters
    ----------
    model:
        Trained UNETR model in eval mode.
    loader:
        DataLoader over :class:`~src.preprocessing.dataset.CryoEMRawDataset`.
    device:
        Torch device.
    img_size:
        Tile size for the sliding-window inferer.
    output_dir:
        Directory where ``<stem>_mask.npy`` files are written.
    threshold:
        Sigmoid threshold for binarising the logit output.
    """
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
            preds = binarise(sigmoid(logits))   # (B, 1, H, W) float {0, 1}

            for pred, name in zip(preds, names):
                mask = pred.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                out_path = output_dir / f"{name}_mask.npy"
                np.save(out_path, mask)
                logger.info("Saved mask → %s  (non-zero: %d px)", out_path, int((mask > 0).sum()))


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to UNETR checkpoint (.pth). Defaults to results/seg_checkpoints/best_unetr.pth.",
)
@click.option(
    "--input-dir",
    default=None,
    help="Directory of images to segment. Defaults to data.processed_dir from config.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory for output masks. Defaults to data.masks_dir from config.",
)
@click.option(
    "--batch-size",
    default=4,
    show_default=True,
    help="Inference batch size.",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Sigmoid threshold for mask binarisation.",
)
def main(
    config: str | None,
    checkpoint: str | None,
    input_dir: str | None,
    output_dir: str | None,
    batch_size: int,
    threshold: float,
) -> None:
    """Segment all images in *input-dir* with a trained UNETR and write masks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    seg_cfg = cfg.segmentation.unet
    unetr_cfg = seg_cfg.unetr
    img_size: int = 256  # CROP_SIZE used during training; must be divisible by 16

    # Resolve paths
    in_path = Path(input_dir or cfg.data.processed_dir)
    out_path = Path(output_dir or cfg.data.masks_dir)
    ckpt_path = Path(
        checkpoint
        or (Path(cfg.project.results_dir) / "seg_checkpoints" / "best_unetr.pth")
    )

    # Build model
    model = build_unetr(
        img_size=img_size,
        in_channels=seg_cfg.in_channels,
        out_channels=seg_cfg.num_classes,
        feature_size=unetr_cfg.feature_size,
        hidden_size=unetr_cfg.hidden_size,
        mlp_dim=unetr_cfg.mlp_dim,
        num_heads=unetr_cfg.num_heads,
        pos_embed=unetr_cfg.pos_embed,
        dropout_rate=unetr_cfg.dropout_rate,
    ).to(device)

    # Load checkpoint
    if not ckpt_path.exists():
        logger.error(
            "Checkpoint not found: %s — run train_unet first.", ckpt_path
        )
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    # Build dataset / loader using CryoEMRawDataset with inference transforms
    dataset = CryoEMRawDataset(
        raw_dir=in_path,
        transform=_inference_transform(img_size),
    )
    if len(dataset) == 0:
        logger.error("No images found in %s", in_path)
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Inference
    run_inference(
        model=model,
        loader=loader,
        device=device,
        img_size=img_size,
        output_dir=out_path,
        threshold=threshold,
    )
    logger.info("UNETR segmentation complete — masks written to %s", out_path)


if __name__ == "__main__":
    main()
