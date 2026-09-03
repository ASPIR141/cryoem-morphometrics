"""Evaluate predicted masks against CVAT ground truth using MONAI metrics.

Uses ``monai.metrics.DiceMetric``, ``monai.metrics.MeanIoU``, and
``monai.metrics.HausdorffDistanceMetric`` for a thorough segmentation report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

from src.utils.wandb_logger import wandb_run as wb_run

logger = logging.getLogger(__name__)


def _load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from .npy or image file."""
    if path.suffix == ".npy":
        return (np.load(path) > 0).astype(np.float32)
    from skimage import io as skio

    return (skio.imread(str(path), as_gray=True) > 0).astype(np.float32)


def evaluate_masks(
    pred_dir: str | Path,
    gt_dir: str | Path,
    results_dir: str | Path,
    compute_hausdorff: bool = True,
    model_name: str = "classical",
    cfg: object | None = None,
) -> pd.DataFrame:
    """Compare predicted masks against CVAT ground-truth masks.

    Uses MONAI ``DiceMetric``, ``MeanIoU``, and ``HausdorffDistanceMetric``
    for per-image and aggregate evaluation.

    Parameters
    ----------
    pred_dir:
        Directory containing predicted binary masks (.npy or .png).
    gt_dir:
        Directory containing CVAT ground-truth masks.
    results_dir:
        Output directory for ``segmentation_metrics.csv``.
    compute_hausdorff:
        Whether to include Hausdorff distance (slower; requires non-empty masks).
    model_name:
        Name of the segmentation model (``"classical"`` or ``"vista2d"``); used
        as a W&B config tag so runs can be compared across models.
    cfg:
        Root pipeline config. When provided and ``cfg.wandb.enabled`` is true,
        results are logged to Weights & Biases.

    Returns
    -------
    df:
        DataFrame with columns ``name``, ``dice``, ``iou``,
        and optionally ``hausdorff_95``.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*"))
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files found in {gt_dir}")

    # MONAI metrics accumulate over individual samples
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="none", get_not_nans=False)
    if compute_hausdorff:
        hd_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="none",
            get_not_nans=False,
        )

    rows: list[dict[str, object]] = []

    for gt_path in gt_files:
        stem = gt_path.stem
        pred_npy = pred_dir / f"{stem}.npy"
        pred_png = pred_dir / f"{stem}.png"
        pred_path = pred_npy if pred_npy.exists() else (pred_png if pred_png.exists() else None)

        if pred_path is None:
            logger.warning("No prediction found for %s, skipping", stem)
            continue

        pred = _load_mask(pred_path)
        gt = _load_mask(gt_path)

        if pred.shape != gt.shape:
            logger.warning("Shape mismatch for %s: pred=%s gt=%s", stem, pred.shape, gt.shape)
            continue

        # MONAI expects (B, C, H, W) float tensors
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)   # 1×1×H×W
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

        dice_metric(y_pred=pred_t, y=gt_t)
        iou_metric(y_pred=pred_t, y=gt_t)

        dice_val = float(dice_metric.aggregate().squeeze())
        iou_val = float(iou_metric.aggregate().squeeze())
        dice_metric.reset()
        iou_metric.reset()

        row: dict[str, object] = {"name": stem, "dice": dice_val, "iou": iou_val}

        if compute_hausdorff and pred.any() and gt.any():
            hd_metric(y_pred=pred_t, y=gt_t)
            row["hausdorff_95"] = float(hd_metric.aggregate().squeeze())
            hd_metric.reset()

        rows.append(row)
        logger.debug(
            "%s — dice=%.4f  iou=%.4f", stem, dice_val, iou_val
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        num_cols = [c for c in ["dice", "iou", "hausdorff_95"] if c in df.columns]
        summary = df[num_cols].mean()
        logger.info("Segmentation evaluation (mean): %s", summary.to_dict())
        df.to_csv(out_dir / "segmentation_metrics.csv", index=False)

    # ── Log to W&B ────────────────────────────────────────────────────────────
    if cfg is not None and not df.empty:
        with wb_run(
            cfg,
            job_type="eval",
            run_name=f"seg-eval-{model_name}",
            tags=["segmentation", "evaluation", model_name],
            extra_config={"seg_model": model_name},
        ) as run:
            run.log_segmentation_metrics(df, model_name=model_name)

    return df
