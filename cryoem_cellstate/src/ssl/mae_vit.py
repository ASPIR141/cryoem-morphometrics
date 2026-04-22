"""Pretrained MAE-ViT feature extractor (hf_hub:timm/vit_base_patch16_224.mae_in1k).

Loads the ImageNet MAE-pretrained ViT-Base from the timm HuggingFace Hub.
No custom training or decoder — use directly for embedding extraction via
``run_ssl.py --model mae``.

Typical usage::

    from src.ssl.mae_vit import MAEViT, CropDataset

    model = MAEViT()
    imgs  = torch.zeros(4, 1, 224, 224)
    feats = model.get_features(imgs)  # (4, 768) — CLS token
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_DEFAULT_BACKBONE = "hf_hub:timm/vit_base_patch16_224.mae_in1k"


class CropDataset(Dataset):
    """Single-view dataset of cell crops for SSL feature extraction.

    Parameters
    ----------
    crops_dir:
        Root directory of crops; images are read from ``crops_dir/images/``.
    image_size:
        Square resize target (must match the ViT's expected input, e.g. 224).
    """

    def __init__(self, crops_dir: str | Path, image_size: int = 224) -> None:
        self.paths = sorted(Path(crops_dir, "images").glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy crops found in {Path(crops_dir) / 'images'}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        img = np.load(self.paths[idx]).astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        t = TF.resize(t, [self.image_size, self.image_size], antialias=True)
        return t  # (1, H, W)


class MAEViT(nn.Module):
    """CLS-token feature extractor backed by the timm MAE-pretrained ViT-Base.

    Loads ``hf_hub:timm/vit_base_patch16_224.mae_in1k`` with pretrained ImageNet
    weights.  No custom training required.

    Parameters
    ----------
    backbone:
        timm model name.  Defaults to the MAE-pretrained ViT-Base.
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
            "MAEViT: backbone=%s  embed_dim=%d  params=%d",
            backbone, self.embed_dim, n_params,
        )

    def get_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embedding (no masking at inference).

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
