"""UNETR model for cell segmentation using MONAI.

Uses ``monai.networks.nets.UNETR`` (ViT encoder + CNN decoder) and
``monai.losses.DiceCELoss`` as the training objective.

Reference: Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image
Segmentation", WACV 2022.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR

logger = logging.getLogger(__name__)


def build_unetr(
    img_size: int = 256,
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    pos_embed: str = "conv",
    dropout_rate: float = 0.0,
) -> nn.Module:
    """Construct a 2-D MONAI UNETR for binary cell segmentation.

    UNETR uses a Vision Transformer as the encoder and a CNN decoder with
    skip connections.  The input spatial size must be divisible by the ViT
    patch size (16 by default).

    Parameters
    ----------
    img_size:
        Square spatial size of the input image.  Must be divisible by 16.
    in_channels:
        Input channels (1 for grayscale CryoEM images).
    out_channels:
        Output channels (1 for binary segmentation; raw logits).
    feature_size:
        Feature map size in the CNN decoder.
    hidden_size:
        ViT encoder hidden / embedding dimension.
    mlp_dim:
        ViT encoder MLP intermediate dimension.
    num_heads:
        Number of self-attention heads (must divide *hidden_size*).
    pos_embed:
        Positional embedding type: ``"conv"`` (default) or ``"perceptron"``.
    dropout_rate:
        Dropout probability applied in the ViT encoder.

    Returns
    -------
    model:
        ``torch.nn.Module`` ready for training (outputs raw logits).

    Raises
    ------
    ValueError
        If *img_size* is not divisible by 16.
    """
    if img_size % 16 != 0:
        raise ValueError(
            f"img_size must be divisible by 16 (ViT patch size), got {img_size}"
        )

    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=(img_size, img_size),
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed=pos_embed,
        norm_name="instance",
        dropout_rate=dropout_rate,
        spatial_dims=2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "UNETR: img_size=%d  in_ch=%d  out_ch=%d  hidden=%d  heads=%d  params=%d",
        img_size,
        in_channels,
        out_channels,
        hidden_size,
        num_heads,
        n_params,
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
