"""Pretrained DINO ViT-Base feature extractor (hf_hub:timm/vit_base_patch16_224.dino).

Loads the DINO self-supervised ViT-Base from the timm HuggingFace Hub.
No rotation-prediction training required.  Use ``run_ssl.py --model rotnet``
to extract CLS-token embeddings.

Typical usage::

    from src.ssl.rotnet import DinoViT

    model = DinoViT()
    imgs  = torch.zeros(4, 1, 224, 224)
    feats = model.get_features(imgs)  # (4, 768) — CLS token
"""

from __future__ import annotations

import logging

from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

_DEFAULT_BACKBONE = "hf_hub:timm/vit_base_patch16_224.dino"


class DinoViT(nn.Module):
    """CLS-token feature extractor backed by the timm DINO-pretrained ViT-Base.

    Loads ``hf_hub:timm/vit_base_patch16_224.dino`` with pretrained weights.
    No training required.

    Parameters
    ----------
    backbone:
        timm model name.  Defaults to DINO ViT-Base.
    """

    def __init__(self, backbone: str = _DEFAULT_BACKBONE) -> None:
        super().__init__()
        import timm

        self.encoder: Any = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",   # keep all tokens so CLS is at index 0
            in_chans=3,
        )
        self.embed_dim: int = self.encoder.embed_dim
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DinoViT: backbone=%s  embed_dim=%d  params=%d",
            backbone, self.embed_dim, n_params,
        )

    def get_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embedding.

        Parameters
        ----------
        imgs:
            ``(B, 1, H, W)`` or ``(B, 3, H, W)`` tensor.

        Returns
        -------
        features:
            ``(B, embed_dim)`` float tensor.
        """
        if imgs.shape[1] == 1:
            imgs = imgs.expand(-1, 3, -1, -1)
        out = self.encoder.forward_features(imgs)  # (B, N+1, D)
        return out[:, 0, :]  # CLS token

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.get_features(imgs)
