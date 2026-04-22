"""Swin UNETR model for cell segmentation using MONAI.

Uses ``monai.networks.nets.SwinUNETR`` (Swin Transformer encoder + CNN decoder)
and ``monai.losses.DiceCELoss`` as the training objective.

Swin UNETR advantages over plain UNETR:
- Hierarchical shifted-window attention → better local-global feature balance
- Lower memory footprint at the same spatial resolution
- Stronger inductive bias for dense prediction tasks

Reference: Tang et al., "Self-Supervised Pre-Training of Swin Transformers
for 3D Medical Image Analysis", CVPR 2022.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)


def build_swin_unetr(
    img_size: int = 256,
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    depths: tuple[int, ...] = (2, 2, 2, 2),
    num_heads: tuple[int, ...] = (3, 6, 12, 24),
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    use_checkpoint: bool = False,
    weights_path: str | Path | None = None,
) -> nn.Module:
    """Construct a 2-D MONAI SwinUNETR for binary cell segmentation.

    Parameters
    ----------
    img_size:
        Square spatial size of the input image (must be divisible by
        ``32 = 2^5`` for the 4-stage Swin hierarchy).
    in_channels:
        Input channels (1 for grayscale CryoEM images).
    out_channels:
        Output channels (1 for binary segmentation; raw logits).
    feature_size:
        Base feature map channel count in the CNN decoder.
        Swin-Tiny: 48, Swin-Small: 96, Swin-Base: 128.
    depths:
        Number of Swin Transformer blocks per stage.
    num_heads:
        Number of attention heads per stage (must be consistent with
        ``feature_size * 2^stage`` being divisible by the head count).
    drop_rate:
        Dropout probability in the Swin encoder.
    attn_drop_rate:
        Attention dropout probability.
    use_checkpoint:
        Enable gradient checkpointing to reduce GPU memory at the cost of
        slightly slower training.
    weights_path:
        Optional path to MONAI SSL pretrained weights (``.pt`` file from the
        MONAI Model Zoo).  When provided, ``model.load_from(weights)`` is
        called before returning, initialising the Swin encoder from
        self-supervised pretraining.

    Returns
    -------
    model:
        ``torch.nn.Module`` ready for training (outputs raw logits).

    Raises
    ------
    ValueError
        If *img_size* is not divisible by 32.
    """
    if img_size % 32 != 0:
        raise ValueError(
            f"img_size must be divisible by 32 (Swin hierarchy), got {img_size}"
        )

    model = SwinUNETR(
        img_size=(img_size, img_size),
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        norm_name="instance",
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=2,
    )

    if weights_path is not None:
        w = torch.load(str(weights_path), map_location="cpu")
        model.load_from(w)
        logger.info("SwinUNETR: loaded pretrained SSL weights from %s", weights_path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "SwinUNETR: img_size=%d  in_ch=%d  out_ch=%d  feature_size=%d  params=%d",
        img_size, in_channels, out_channels, feature_size, n_params,
    )
    return model


def build_loss(sigmoid: bool = True) -> DiceCELoss:
    """Build MONAI DiceCELoss for binary segmentation.

    Parameters
    ----------
    sigmoid:
        Apply sigmoid to logits before computing Dice (set *True* when the
        model outputs raw logits with ``out_channels=1``).

    Returns
    -------
    criterion:
        ``monai.losses.DiceCELoss`` instance.
    """
    return DiceCELoss(sigmoid=sigmoid, lambda_dice=0.5, lambda_ce=0.5)
