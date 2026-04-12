"""Train the MONAI UNETR on classical pseudo-masks with CVAT validation.

Uses:
- ``monai.transforms`` for the augmentation pipeline
- ``monai.data.CacheDataset`` for fast repeated access
- ``monai.metrics.DiceMetric`` and ``monai.metrics.MeanIoU`` for evaluation
- ``monai.losses.DiceCELoss`` as the training objective

Usage::

    python -m src.segmentation.train_unet --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
)

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

from .unet import build_loss, build_unetr

logger = logging.getLogger(__name__)

CROP_SIZE = 256  # must be divisible by 16 (UNETR patch size)


# ── Data helpers ──────────────────────────────────────────────────────────────


def _collect_pairs(
    image_dir: Path, mask_dir: Path
) -> list[dict[str, str]]:
    """Build list of {image, label} path dicts for MONAI LoadImaged."""
    pairs: list[dict[str, str]] = []
    for img_path in sorted(image_dir.glob("*.npy")):
        mask_npy = mask_dir / f"{img_path.stem}.npy"
        mask_png = mask_dir / f"{img_path.stem}.png"
        if mask_npy.exists():
            pairs.append({"image": str(img_path), "label": str(mask_npy)})
        elif mask_png.exists():
            pairs.append({"image": str(img_path), "label": str(mask_png)})
    return pairs


def _build_transforms(train: bool = True, crop_size: int = CROP_SIZE) -> Compose:
    """Build MONAI transform pipeline for training or validation.

    Parameters
    ----------
    train:
        If *True* include stochastic augmentations; otherwise deterministic only.
    crop_size:
        Spatial crop size (square).

    Returns
    -------
    transforms:
        Composed MONAI transform.
    """
    keys = ["image", "label"]

    # Use a plain list[Any] so mypy doesn't fight over Transform vs MapTransform
    base: list = [
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        ScaleIntensityd(keys=["label"], minv=0.0, maxv=1.0),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ResizeWithPadOrCropd(keys=keys, spatial_size=(crop_size, crop_size)),
        EnsureTyped(keys=keys, dtype=torch.float32),
    ]

    if train:
        augmentations: list = [
            RandSpatialCropd(keys=keys, roi_size=(crop_size, crop_size), random_size=False),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandAffined(
                keys=keys,
                prob=0.3,
                rotate_range=(0.26,),   # ~15°
                shear_range=None,
                scale_range=(0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="reflection",
            ),
            RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
        ]
        return Compose(augmentations + base)

    return Compose(base)


# ── Training & evaluation ─────────────────────────────────────────────────────


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model; return (mean_loss, mean_dice, mean_iou).

    Uses ``monai.metrics.DiceMetric`` and ``monai.metrics.MeanIoU`` with
    per-batch ``decollate_batch`` for correct MONAI metric accumulation.
    """
    from monai.transforms import Activations, AsDiscrete

    sigmoid = Activations(sigmoid=True)
    threshold = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()

            # Ensure outputs are tensors before passing to MONAI metrics
            outputs = [
                threshold(sigmoid(i)) if isinstance(i, torch.Tensor) else i
                for i in decollate_batch(logits)
            ]
            targets = [
                t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                for t in decollate_batch(labels)
            ]
            dice_metric(y_pred=outputs, y=targets)
            iou_metric(y_pred=outputs, y=targets)

    dice = float(dice_metric.aggregate())   # aggregate() returns Tensor with get_not_nans=False
    iou = float(iou_metric.aggregate())
    dice_metric.reset()
    iou_metric.reset()
    return total_loss / max(len(loader), 1), dice, iou


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
def main(config: str | None) -> None:
    """Train the MONAI UNETR segmentation model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)
    device = get_device(cfg)

    seg_cfg = cfg.segmentation.unet
    unetr_cfg = seg_cfg.unetr
    processed_dir = Path(cfg.data.processed_dir)
    masks_dir = Path(cfg.data.masks_dir)
    ckpt_dir = Path(cfg.project.results_dir) / "seg_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = _collect_pairs(processed_dir, masks_dir)
    if not all_pairs:
        logger.error("No image/mask pairs found — run Stage 1 & classical segmentation first.")
        return

    val_size = max(1, int(len(all_pairs) * seg_cfg.val_fraction))
    train_pairs = all_pairs[:-val_size]
    val_pairs = all_pairs[-val_size:]

    train_ds = CacheDataset(
        data=train_pairs,
        transform=_build_transforms(train=True, crop_size=CROP_SIZE),
        cache_rate=1.0,
        num_workers=4,
    )
    val_ds = CacheDataset(
        data=val_pairs,
        transform=_build_transforms(train=False, crop_size=CROP_SIZE),
        cache_rate=1.0,
        num_workers=4,
    )

    train_loader = DataLoader(train_ds, batch_size=seg_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=seg_cfg.batch_size)

    model = build_unetr(
        img_size=CROP_SIZE,
        in_channels=seg_cfg.in_channels,
        out_channels=seg_cfg.num_classes,
        feature_size=unetr_cfg.feature_size,
        hidden_size=unetr_cfg.hidden_size,
        mlp_dim=unetr_cfg.mlp_dim,
        num_heads=unetr_cfg.num_heads,
        pos_embed=unetr_cfg.pos_embed,
        dropout_rate=unetr_cfg.dropout_rate,
    ).to(device)

    criterion = build_loss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=seg_cfg.epochs)

    best_dice = 0.0
    for epoch in range(1, seg_cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_dice=%.4f  val_iou=%.4f",
            epoch, seg_cfg.epochs, train_loss, val_loss, val_dice, val_iou,
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_dir / "best_unetr.pth")
            logger.info("New best model saved (dice=%.4f)", best_dice)

    logger.info("Training complete. Best val Dice: %.4f", best_dice)


if __name__ == "__main__":
    main()
