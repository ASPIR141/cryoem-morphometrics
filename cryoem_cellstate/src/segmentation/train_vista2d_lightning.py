"""PyTorch Lightning training for MONAI VISTA2D cell segmentation.

Fine-tunes the pretrained VISTA2D foundation model (SAM ViT-B +
MONAI adapters) on our CryoEM cell dataset.  The resulting checkpoint is
consumed by ``run_segmentation.py --use-vista2d`` for inference.

Usage::

    python -m src.segmentation.train_vista2d_lightning --config configs/default.yaml
    python -m src.segmentation.train_vista2d_lightning --config configs/default.yaml \\
        --max-epochs 50 --accelerator gpu --devices 1

Checkpoint is saved to ``results/seg_checkpoints/best_vista2d.ckpt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
from monai.data import CacheDataset
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    Activations,
    AsDiscrete,
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
    decollate_batch,
)
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.wandb_logger import WandbRun

from .vista2d import build_loss, build_vista2d

logger = logging.getLogger(__name__)

CROP_SIZE = 256  # sliding-window ROI size (must be divisible by 32)


# ── Data helpers ──────────────────────────────────────────────────────────────


def _collect_pairs(image_dir: Path, mask_dir: Path) -> list[dict[str, str]]:
    """Return a list of ``{"image": ..., "label": ...}`` path dicts."""
    pairs: list[dict[str, str]] = []
    for img_path in sorted(image_dir.glob("*.npy")):
        mask_npy = mask_dir / f"{img_path.stem}.npy"
        mask_png = mask_dir / f"{img_path.stem}.png"
        if mask_npy.exists():
            pairs.append({"image": str(img_path), "label": str(mask_npy)})
        elif mask_png.exists():
            pairs.append({"image": str(img_path), "label": str(mask_png)})
    return pairs


def _build_transforms(train: bool, crop_size: int = CROP_SIZE) -> Compose:
    """Build MONAI transform pipeline for training or validation."""
    keys = ["image", "label"]
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
                rotate_range=(0.26,),
                scale_range=(0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="reflection",
            ),
            RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
        ]
        return Compose(augmentations + base)
    return Compose(base)


# ── Lightning DataModule ──────────────────────────────────────────────────────


class Vista2DDataModule(L.LightningDataModule):
    """MONAI CacheDataset-backed DataModule for VISTA2D fine-tuning.

    Parameters
    ----------
    processed_dir:
        Directory of preprocessed ``.npy`` images (Stage 1 output).
    masks_dir:
        Directory of pseudo-mask ``.npy`` files (classical segmentation output).
    batch_size:
        Training and validation batch size.
    val_fraction:
        Fraction of pairs to hold out for validation.
    crop_size:
        Spatial crop size fed to the sliding-window inferer ROI.
    num_workers:
        DataLoader worker count.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        masks_dir: str | Path,
        batch_size: int = 4,
        val_fraction: float = 0.15,
        crop_size: int = CROP_SIZE,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.masks_dir = Path(masks_dir)
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.crop_size = crop_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        all_pairs = _collect_pairs(self.processed_dir, self.masks_dir)
        if not all_pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.processed_dir} / {self.masks_dir}. "
                "Run preprocessing and classical segmentation first."
            )
        val_size = max(1, int(len(all_pairs) * self.val_fraction))
        train_pairs = all_pairs[:-val_size]
        val_pairs = all_pairs[-val_size:]

        self._train_ds = CacheDataset(
            data=train_pairs,
            transform=_build_transforms(train=True, crop_size=self.crop_size),
            cache_rate=1.0,
            num_workers=self.num_workers,
        )
        self._val_ds = CacheDataset(
            data=val_pairs,
            transform=_build_transforms(train=False, crop_size=self.crop_size),
            cache_rate=1.0,
            num_workers=self.num_workers,
        )
        logger.info(
            "DataModule: %d train  %d val  (crop_size=%d)",
            len(train_pairs), len(val_pairs), self.crop_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ── Lightning Module ──────────────────────────────────────────────────────────


class Vista2DLightningModule(L.LightningModule):
    """Lightning wrapper around the MONAI VISTA2D cell-segmentation model.

    Loads pretrained SAM ViT-B + VISTA2D fine-tuned weights from HuggingFace
    Hub on first instantiation, then fine-tunes on the CryoEM dataset.

    Logs ``train_loss``, ``val_loss``, ``val_dice``, and ``val_iou`` every epoch.

    Parameters
    ----------
    cfg:
        Root pipeline config (reads ``segmentation.segmentation_model`` and
        ``segmentation.segmentation_model.vista2d``).
    crop_size:
        Input spatial size for the sliding-window ROI.
    """

    def __init__(
        self,
        cfg: object,
        crop_size: int = CROP_SIZE,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])

        seg_cfg = cfg.segmentation.segmentation_model  # type: ignore[attr-defined]
        vista_cfg = seg_cfg.vista2d

        self.model: nn.Module = build_vista2d(
            hf_repo=vista_cfg.hf_repo,
            hf_revision=vista_cfg.hf_revision,
            cache_dir=vista_cfg.cache_dir,
        )
        self.criterion = build_loss(sigmoid=True)
        self._lr: float = float(seg_cfg.lr)
        self._epochs: int = int(seg_cfg.epochs)

        self._sigmoid = Activations(sigmoid=True)
        self._threshold = AsDiscrete(threshold=0.5)
        self._dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self._iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)

    def training_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        logits = self.model(batch["image"])
        loss = self.criterion(logits, batch["label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int = 0) -> None:  # noqa: ARG002
        logits = self.model(batch["image"])
        loss = self.criterion(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = [self._threshold(self._sigmoid(t)) for t in decollate_batch(logits)]
        targets = list(decollate_batch(batch["label"]))
        self._dice_metric(y_pred=preds, y=targets)
        self._iou_metric(y_pred=preds, y=targets)

    def on_validation_epoch_end(self) -> None:
        dice = float(self._dice_metric.aggregate())
        iou = float(self._iou_metric.aggregate())
        self._dice_metric.reset()
        self._iou_metric.reset()
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_iou", iou)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option("--max-epochs", default=None, type=int, help="Override segmentation.segmentation_model.epochs.")
@click.option("--accelerator", default="auto", show_default=True, help="Lightning accelerator.")
@click.option("--devices", default=1, show_default=True, type=int, help="Number of devices.")
@click.option(
    "--ckpt-dir",
    default=None,
    help="Override checkpoint output directory (default: results/seg_checkpoints).",
)
def main(
    config: str | None,
    max_epochs: int | None,
    accelerator: str,
    devices: int,
    ckpt_dir: str | None,
) -> None:
    """Fine-tune MONAI VISTA2D on CryoEM data with PyTorch Lightning."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    seg_cfg = cfg.segmentation.segmentation_model  # type: ignore[attr-defined]
    epochs = max_epochs or seg_cfg.epochs
    ckpt_output = Path(ckpt_dir or (Path(cfg.project.results_dir) / "seg_checkpoints"))  # type: ignore[attr-defined]
    ckpt_output.mkdir(parents=True, exist_ok=True)

    datamodule = Vista2DDataModule(
        processed_dir=cfg.data.processed_dir,  # type: ignore[attr-defined]
        masks_dir=cfg.data.masks_dir,  # type: ignore[attr-defined]
        batch_size=seg_cfg.batch_size,
        val_fraction=seg_cfg.val_fraction,
        crop_size=CROP_SIZE,
    )

    module = Vista2DLightningModule(cfg=cfg, crop_size=CROP_SIZE)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_output),
            filename="best_vista2d",
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        EarlyStopping(monitor="val_dice", mode="max", patience=15, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── W&B logger ────────────────────────────────────────────────────────────
    wb_cfg = cfg.wandb  # type: ignore[attr-defined]
    loggers: list = []
    if wb_cfg.enabled:
        try:
            wandb_logger = WandbLogger(
                project=wb_cfg.project,
                entity=wb_cfg.entity or None,
                name="vista2d-finetune",
                tags=["segmentation", "vista2d"],
                job_type="train",
                log_model=False,  # we handle artifact upload manually below
                config={
                    "epochs": epochs,
                    "lr": seg_cfg.lr,
                    "batch_size": seg_cfg.batch_size,
                    "crop_size": CROP_SIZE,
                    "backbone": "SAM-ViT-B+VISTA2D",
                    "seg_model": "vista2d",
                },
            )
            loggers.append(wandb_logger)
            logger.info("W&B training logger initialised")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise WandbLogger (training will continue): %s", exc)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        loggers=loggers if loggers else True,  # True = default TensorBoard/CSV logger
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=datamodule)
    logger.info("Training complete. Best checkpoint saved to %s", ckpt_output)

    # ── Upload best checkpoint as W&B artifact ────────────────────────────────
    if wb_cfg.enabled and loggers:
        best_ckpt = ckpt_output / "best_vista2d.ckpt"
        wb_run = WandbRun.__new__(WandbRun)
        wb_run._enabled = True  # noqa: SLF001
        wb_run._log_figures = wb_cfg.log_figures  # noqa: SLF001
        # Re-use the already active wandb run created by WandbLogger
        try:
            import wandb  # noqa: PLC0415

            wb_run._run = wandb.run  # noqa: SLF001
            wb_run.log_checkpoint_artifact(
                best_ckpt,
                name="vista2d-best-checkpoint",
                artifact_type="model",
                metadata={
                    "monitor": "val_dice",
                    "epochs": epochs,
                    "lr": seg_cfg.lr,
                },
            )
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not upload checkpoint artifact to W&B: %s", exc)


if __name__ == "__main__":
    main()
