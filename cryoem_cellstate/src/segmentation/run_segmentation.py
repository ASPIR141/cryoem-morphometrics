"""Stage 2 inference driver: segmentation and cell cropping.

Produces binary masks and cell crops using either:

* **Classical** (default) — Otsu/adaptive threshold → morphological cleanup → watershed
* **VISTA2D** (``--use-vista2d``) — loads a fine-tuned VISTA2D checkpoint
  produced by ``train_vista2d_lightning.py`` and runs sliding-window inference

VISTA2D is a MONAI generalist cell-segmentation foundation model built on
SAM ViT-B and fine-tuned on ~15 K public microscopy images.

Training is handled separately by ``train_vista2d_lightning.py``.

Usage::

    # Classical segmentation (no trained model required):
    python -m src.segmentation.run_segmentation --config configs/default.yaml

    # VISTA2D inference with a fine-tuned checkpoint:
    python -m src.segmentation.run_segmentation --config configs/default.yaml \\
        --use-vista2d --checkpoint results/seg_checkpoints/best_vista2d.ckpt

    # Also evaluate against CVAT ground truth:
    python -m src.segmentation.run_segmentation --config configs/default.yaml \\
        --use-vista2d --checkpoint results/seg_checkpoints/best_vista2d.ckpt \\
        --evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

from .classical import segment_classical
from .crop_cells import build_cells_parquet, crop_cells

logger = logging.getLogger(__name__)

_VISTA2D_CROP_SIZE = 256  # sliding-window ROI size (must be divisible by 32)


# ── Segmentation backends ─────────────────────────────────────────────────────


def _segment_classical_all(cfg: object) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Run classical segmentation on all processed images."""
    processed_path = Path(cfg.data.processed_dir)  # type: ignore[attr-defined]
    masks_path = Path(cfg.data.masks_dir)  # type: ignore[attr-defined]
    masks_path.mkdir(parents=True, exist_ok=True)

    cl_cfg = cfg.segmentation.classical  # type: ignore[attr-defined]
    image_files = sorted(processed_path.glob("*.npy")) + sorted(processed_path.glob("*.pt"))
    if not image_files:
        logger.warning("No processed images found in %s", processed_path)
        return []

    triples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for img_path in tqdm(image_files, desc="Classical segmentation"):
        image = torch.load(img_path).numpy() if img_path.suffix == ".pt" else np.load(img_path)
        mask = segment_classical(
            image,
            method=cl_cfg.method,
            min_cell_area=cl_cfg.min_cell_area,
            max_cell_area=cl_cfg.max_cell_area,
            apply_watershed=cl_cfg.watershed,
        )
        np.save(masks_path / f"{img_path.stem}.npy", mask)
        triples.append((img_path.stem, image, mask))

    logger.info("Classical segmentation: %d images processed", len(triples))
    return triples


def _segment_vista2d_all(
    cfg: object,
    checkpoint: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Run VISTA2D inference on all processed images using a fine-tuned checkpoint.

    Loads the Lightning checkpoint produced by ``train_vista2d_lightning.py``
    and applies sliding-window inference via MONAI ``SlidingWindowInferer``.
    """
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import Activations, AsDiscrete

    from .train_vista2d_lightning import Vista2DLightningModule

    device = get_device(cfg)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VISTA2D checkpoint not found: {ckpt_path}")

    vista_cfg = cfg.segmentation.segmentation_model.vista2d  # type: ignore[attr-defined]
    module = Vista2DLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, crop_size=_VISTA2D_CROP_SIZE
    )
    model = module.model.to(device).eval()
    logger.info("Loaded VISTA2D checkpoint from %s", ckpt_path)

    sigmoid = Activations(sigmoid=True)
    threshold = AsDiscrete(threshold=0.5)
    inferer = SlidingWindowInferer(
        roi_size=(vista_cfg.roi_size[0], vista_cfg.roi_size[1]),
        sw_batch_size=vista_cfg.sw_batch_size,
        overlap=vista_cfg.overlap,
        mode="gaussian",
    )

    processed_path = Path(cfg.data.processed_dir)  # type: ignore[attr-defined]
    masks_path = Path(cfg.data.masks_dir)  # type: ignore[attr-defined]
    masks_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(processed_path.glob("*.npy")) + sorted(processed_path.glob("*.pt"))
    if not image_files:
        logger.warning("No processed images found in %s", processed_path)
        return []

    triples: list[tuple[str, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="VISTA2D inference"):
            image = torch.load(img_path).numpy() if img_path.suffix == ".pt" else np.load(img_path)
            t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            logits = inferer(t, model)
            pred = threshold(sigmoid(logits.squeeze(0)))   # (1, H, W)
            mask = (pred.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            np.save(masks_path / f"{img_path.stem}.npy", mask)
            triples.append((img_path.stem, image, mask))

    logger.info("VISTA2D inference: %d images processed", len(triples))
    return triples


# ── Cell cropping ─────────────────────────────────────────────────────────────


def _crop_all(
    triples: list[tuple[str, np.ndarray, np.ndarray]],
    cfg: object,
) -> pd.DataFrame:
    """Extract cell crops from all (stem, image, mask) triples."""
    crops_dir = Path(cfg.data.crops_dir)  # type: ignore[attr-defined]
    crop_cfg = cfg.segmentation.crop_cells  # type: ignore[attr-defined]

    all_records: list[dict] = []
    for stem, image, mask in tqdm(triples, desc="Cropping cells"):
        records = crop_cells(
            image, mask,
            source_name=stem,
            crops_dir=crops_dir,
            min_area=crop_cfg.min_area,
            max_aspect_ratio=crop_cfg.max_aspect_ratio,
            crop_size=crop_cfg.crop_size,
        )
        all_records.extend(records)

    cells_df = build_cells_parquet(all_records, crops_dir / "cells.parquet")
    logger.info("%d cells extracted from %d images", len(cells_df), len(triples))
    return cells_df


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--use-vista2d",
    is_flag=True,
    help="Use fine-tuned VISTA2D for inference instead of classical segmentation.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="VISTA2D Lightning checkpoint path (required with --use-vista2d).",
)
@click.option("--evaluate", is_flag=True, help="Evaluate predicted masks against CVAT GT.")
@click.option("--gt-dir", default=None, help="Override CVAT ground-truth directory.")
def main(
    config: str | None,
    use_vista2d: bool,
    checkpoint: str | None,
    evaluate: bool,
    gt_dir: str | None,
) -> None:
    """Run Stage 2: segmentation inference and cell cropping."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    if use_vista2d:
        if not checkpoint:
            raise click.UsageError("--checkpoint is required when using --use-vista2d.")
        triples = _segment_vista2d_all(cfg, checkpoint)
    else:
        triples = _segment_classical_all(cfg)

    if triples:
        _crop_all(triples, cfg)

    if evaluate:
        from .evaluate_segmentation import evaluate_masks

        pred_dir = Path(cfg.data.masks_dir)  # type: ignore[attr-defined]
        gt_path = Path(gt_dir or cfg.data.cvat_gt_dir)  # type: ignore[attr-defined]
        results_dir = Path(cfg.project.results_dir) / "stage2"  # type: ignore[attr-defined]

        if not gt_path.exists():
            logger.warning("CVAT GT dir not found: %s — skipping evaluation", gt_path)
        else:
            df = evaluate_masks(pred_dir, gt_path, results_dir)
            if not df.empty:
                logger.info(
                    "Segmentation eval — mean Dice=%.4f  mean IoU=%.4f",
                    df["dice"].mean(), df["iou"].mean(),
                )


if __name__ == "__main__":
    main()
