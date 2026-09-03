"""Pretrained DINOv3 ViT-Large feature extractor (facebook/dinov3-vitl16-pretrain-lvd1689m).

Loads the DINOv3 self-supervised ViT-Large from the HuggingFace Hub via the
``transformers`` library (≥ 4.56.0 required).  No rotation-prediction training
required.  Use ``run_ssl.py --model rotnet`` to extract CLS-token embeddings.

Typical usage::

    from src.ssl.rotnet import DinoV3ViT

    model = DinoV3ViT()
    imgs  = torch.zeros(4, 1, 224, 224)
    feats = model.get_features(imgs)  # (4, 1024) — CLS token
"""

from __future__ import annotations

import logging

from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class DinoV3ViT(nn.Module):
    """CLS-token feature extractor backed by the DINOv3-pretrained ViT-Large.

    Loads ``facebook/dinov3-vitl16-pretrain-lvd1689m`` (or any compatible
    DINOv3 model ID) with pretrained weights via HuggingFace ``transformers``
    (≥ 4.56.0).  No training required.

    The model processes images as 201 tokens:
    1 CLS token + 4 register tokens + 196 patch tokens (14×14 patches at 16px
    each for a 224×224 input).  CLS token is always at position 0 of
    ``last_hidden_state``.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to DINOv3 ViT-Large.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        super().__init__()
        from transformers import AutoModel  # type: ignore[import]

        self.encoder: Any = AutoModel.from_pretrained(model_name)
        # DINOv3 ViT-Large hidden size is 1024
        self.embed_dim: int = self.encoder.config.hidden_size
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DinoV3ViT: model=%s  embed_dim=%d  params=%d",
            model_name, self.embed_dim, n_params,
        )

    def get_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embedding.

        Parameters
        ----------
        imgs:
            ``(B, 1, H, W)`` or ``(B, 3, H, W)`` tensor.  Values should be in
            ``[0, 1]`` or normalised — the model will normalise internally via
            the image processor if you use the full pipeline; when calling
            directly here we pass pixel values as-is and rely on the model's
            LayerNorm robustness.

        Returns
        -------
        features:
            ``(B, embed_dim)`` float tensor (CLS token).
        """
        if imgs.shape[1] == 1:
            imgs = imgs.expand(-1, 3, -1, -1)
        outputs = self.encoder(pixel_values=imgs)  # type: ignore[call-arg]
        # last_hidden_state: (B, 1 + 4_registers + N_patches, D)
        # CLS token is always at index 0
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.get_features(imgs)


# ---------------------------------------------------------------------------
# Backward-compat alias so any code that imported ``DinoViT`` still works
# ---------------------------------------------------------------------------
DinoViT = DinoV3ViT
