"""SimCLR self-supervised learning components.

Implements:
- CryoEM-adapted two-view augmentation pipeline via **MONAI transforms**
  (no colour jitter; Gaussian noise simulates CryoEM detector noise)
- NT-Xent contrastive loss (wraps ``monai.losses.ContrastiveLoss``)
- SimCLR model wrapper
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

logger = logging.getLogger(__name__)


# ── Augmentation pipeline ─────────────────────────────────────────────────────


def build_simclr_transform(
    crop_scale: tuple[float, float] = (0.2, 1.0),
    flip: bool = True,
    rotation_degrees: float = 15.0,
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0),
    gaussian_noise_std: float = 0.05,
    image_size: int = 128,
) -> Compose:
    """Build a CryoEM-adapted two-view augmentation pipeline using MONAI transforms.

    All transforms operate on tensors with a channel dimension (C × H × W).
    No colour jitter is applied; instead Gaussian noise models detector noise.

    Parameters
    ----------
    crop_scale:
        (min, max) area fraction for random zoom (simulates random crop).
    flip:
        Whether to include random horizontal and vertical flips.
    rotation_degrees:
        Max rotation angle in degrees for ``RandRotate``.
    gaussian_blur_sigma:
        (min, max) sigma range for ``RandGaussianSmooth``.
    gaussian_noise_std:
        Std of additive Gaussian noise for ``RandGaussianNoise``.
    image_size:
        Spatial size to pad/crop to (not used directly here; caller resizes).

    Returns
    -------
    transform:
        ``monai.transforms.Compose`` pipeline producing a single augmented view.
        Call twice independently to get two views.
    """
    rotation_radians = rotation_degrees * (3.14159265 / 180.0)
    zoom_min = crop_scale[0] ** 0.5   # area → linear scale
    zoom_max = crop_scale[1] ** 0.5

    transforms: list = [
        RandZoom(
            prob=1.0,
            min_zoom=zoom_min,
            max_zoom=zoom_max,
            mode="bilinear",
            align_corners=False,
        ),
    ]

    if flip:
        transforms += [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
        ]

    transforms += [
        RandRotate(
            prob=0.8,
            range_x=(-rotation_radians, rotation_radians),
            mode="bilinear",
            align_corners=False,
            padding_mode="reflection",
        ),
        RandGaussianSmooth(
            prob=0.3,
            sigma_x=gaussian_blur_sigma,
            sigma_y=gaussian_blur_sigma,
        ),
        RandGaussianNoise(prob=0.5, std=gaussian_noise_std),
        ScaleIntensity(),   # re-normalise to [0, 1] after noise
    ]

    return Compose(transforms)


# ── NT-Xent Loss (wrapping MONAI ContrastiveLoss) ────────────────────────────


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss for SimCLR.

    Internally delegates to ``monai.losses.ContrastiveLoss`` which implements
    the same InfoNCE objective, adding temperature scaling on top.

    Parameters
    ----------
    temperature:
        Softmax temperature τ.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        # MONAI ContrastiveLoss: temperature=1 → we scale logits manually
        from monai.losses import ContrastiveLoss

        self._monai_loss = ContrastiveLoss(temperature=temperature, batch_size=-1)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss for a batch of L2-normalised projection pairs.

        Parameters
        ----------
        z1:
            Projections of view 1, shape (N × D).
        z2:
            Projections of view 2, shape (N × D).

        Returns
        -------
        loss:
            Scalar contrastive loss tensor.
        """
        z1_n = F.normalize(z1, dim=1)
        z2_n = F.normalize(z2, dim=1)
        return self._monai_loss(z1_n, z2_n)


# ── SimCLR model ──────────────────────────────────────────────────────────────


class SimCLR(nn.Module):
    """SimCLR wrapper: encoder + MLP projection head.

    Parameters
    ----------
    encoder:
        Backbone :class:`~src.ssl.encoders.Encoder` instance.
    projection_dim:
        Output dimension of the projection head.
    """

    def __init__(self, encoder: nn.Module, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder

        with torch.no_grad():
            dummy = torch.zeros(2, 1, 128, 128)
            feat_dim: int = encoder.get_features(dummy).shape[1]

        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim),
        )

        logger.info("SimCLR: feat_dim=%d  projection_dim=%d", feat_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projection of *x*.

        Parameters
        ----------
        x:
            Input batch (B × C × H × W).

        Returns
        -------
        z:
            (B × projection_dim) L2-normalised projections.
        """
        features = self.encoder.get_features(x)
        z = self.projection_head(features)
        return F.normalize(z, dim=1)
