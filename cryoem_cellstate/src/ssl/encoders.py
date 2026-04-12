"""Encoder factory wrapping timm backbones for SSL training.

Supports:
- ResNet18 (default)
- ConvNeXt-Tiny
- ViT-Small/16
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_SUPPORTED_BACKBONES = {
    "resnet18",
    "convnext_tiny",
    "vit_small_patch16_224",
}


class Encoder(nn.Module):
    """Thin wrapper around a timm backbone that strips the classification head.

    Parameters
    ----------
    backbone_name:
        timm model name (one of the supported backbones).
    pretrained:
        Load ImageNet weights.
    in_channels:
        Input channels (1 for grayscale).
    embedding_dim:
        Size of the output feature vector.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        in_channels: int = 1,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        import timm

        if backbone_name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {sorted(_SUPPORTED_BACKBONES)}"
            )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,        # remove classifier head
            global_pool="avg",    # global average pool → 1-D feature vector
            in_chans=in_channels,
        )

        # Probe feature dimensionality
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            feat_dim: int = self.backbone(dummy).shape[1]

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )

        logger.info(
            "Encoder: backbone=%s  feat_dim=%d  embedding_dim=%d",
            backbone_name,
            feat_dim,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected embedding.

        Parameters
        ----------
        x:
            Input tensor (B × C × H × W).

        Returns
        -------
        embedding:
            (B × embedding_dim) float tensor.
        """
        features = self.backbone(x)
        return self.projector(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone features (before projection head).

        Parameters
        ----------
        x:
            Input tensor (B × C × H × W).

        Returns
        -------
        features:
            (B × feat_dim) float tensor.
        """
        return self.backbone(x)
