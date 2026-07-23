"""MONAI VISTA2D model for cell segmentation.

Downloads and wraps the ``MONAI/vista2d`` pretrained checkpoint from
HuggingFace Hub.  VISTA2D is a **generalist cell-segmentation foundation
model** built on SAM ViT-B and fine-tuned on ~15 K public microscopy images,
covering diverse cell types and imaging modalities.

Architecture
------------
- **Backbone**: SAM ViT-B (Segment Anything — ``sam_vit_b_01ec64.pth``)
- **Adapter**: VISTA2D fine-tuned weights (``model.pt``)
- **Inference**: MONAI ``SlidingWindowInferer`` over 256×256 ROIs
- **Output**: binary cell segmentation masks (logits → sigmoid → threshold)

Reference: NVIDIA / MONAI — "Advancing Cell Segmentation and Morphology
Analysis with NVIDIA AI Foundation Model VISTA-2D", 2024.
https://huggingface.co/MONAI/vista2d
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

logger = logging.getLogger(__name__)

# HuggingFace filenames published under MONAI/vista2d
_SAM_CKPT_FILENAME = "sam_vit_b_01ec64.pth"
_VISTA2D_CKPT_FILENAME = "model.pt"


def _download_vista2d_weights(
    hf_repo: str,
    hf_revision: str,
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    """Download SAM + VISTA2D weights from HuggingFace Hub if not cached.

    Parameters
    ----------
    hf_repo:
        HuggingFace model repository ID (e.g. ``"MONAI/vista2d"``).
    hf_revision:
        Git revision / tag to pin (e.g. ``"0.4.0"``).
    cache_dir:
        Local directory to cache the downloaded weights.

    Returns
    -------
    sam_path, vista2d_path:
        Absolute paths to the two downloaded checkpoint files.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download VISTA2D weights. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sam_path = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=_SAM_CKPT_FILENAME,
            revision=hf_revision,
            local_dir=str(cache_dir),
        )
    )
    vista2d_path = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=_VISTA2D_CKPT_FILENAME,
            revision=hf_revision,
            local_dir=str(cache_dir),
        )
    )
    logger.info(
        "VISTA2D weights ready — SAM: %s  model: %s", sam_path, vista2d_path
    )
    return sam_path, vista2d_path


def build_vista2d(
    hf_repo: str = "MONAI/vista2d",
    hf_revision: str = "0.4.0",
    cache_dir: str | Path = "~/.cache/monai/vista2d",
) -> nn.Module:
    """Build a MONAI VISTA2D cell-segmentation model.

    Downloads SAM ViT-B backbone + VISTA2D fine-tuned adapter weights from
    HuggingFace Hub on first call; subsequent calls use the local cache.

    Parameters
    ----------
    hf_repo:
        HuggingFace repository ID for the VISTA2D bundle.
    hf_revision:
        Git revision / tag to use (pinned to ``"0.4.0"`` by default).
    cache_dir:
        Local cache directory for downloaded weights.

    Returns
    -------
    model:
        ``torch.nn.Module`` ready for fine-tuning or inference.
        Accepts ``(B, 1, H, W)`` or ``(B, 3, H, W)`` float tensors and
        returns raw logits of shape ``(B, 1, H, W)``.

    Raises
    ------
    ImportError
        If ``segment_anything`` is not installed.
    RuntimeError
        If the downloaded checkpoint cannot be loaded.
    """
    try:
        from segment_anything import build_sam_vit_b
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required for VISTA2D. "
            "Install it with: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    sam_path, vista2d_path = _download_vista2d_weights(
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        cache_dir=cache_dir,
    )

    # Build SAM ViT-B and load fine-tuned VISTA2D weights on top
    sam = build_sam_vit_b(checkpoint=str(sam_path))
    model = _Vista2DWrapper(sam)

    state = torch.load(str(vista2d_path), map_location="cpu")
    # The VISTA2D bundle stores weights under "model" or directly as state_dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("VISTA2D: %d missing keys (expected for partial load)", len(missing))
    if unexpected:
        logger.warning("VISTA2D: %d unexpected keys", len(unexpected))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("VISTA2D: loaded from %s  trainable_params=%d", vista2d_path, n_params)
    return model


class _Vista2DWrapper(nn.Module):
    """Thin wrapper around the SAM ViT-B encoder/decoder for binary cell segmentation.

    Exposes the same forward signature as the previous SwinUNETR:
    ``(B, C, H, W) → (B, 1, H, W)`` raw logits.

    The SAM image encoder extracts multi-scale features; a lightweight
    mask decoder produces the binary segmentation output.
    """

    def __init__(self, sam: nn.Module) -> None:
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw binary-segmentation logits.

        Parameters
        ----------
        imgs:
            ``(B, 1, H, W)`` or ``(B, 3, H, W)`` float tensor (any resolution).

        Returns
        -------
        logits:
            ``(B, 1, H, W)`` raw logits (apply sigmoid for probabilities).
        """
        # SAM expects 3-channel input at 1024×1024; we resize via interpolation
        if imgs.shape[1] == 1:
            imgs = imgs.expand(-1, 3, -1, -1)

        import torch.nn.functional as F

        # Resize to SAM's native resolution (1024×1024)
        h_orig, w_orig = imgs.shape[-2], imgs.shape[-1]
        imgs_1024 = F.interpolate(
            imgs, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        # Normalise to SAM pixel stats
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=imgs.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=imgs.device).view(1, 3, 1, 1)
        imgs_norm = (imgs_1024 * 255.0 - pixel_mean) / pixel_std

        image_embeddings = self.image_encoder(imgs_norm)  # (B, 256, 64, 64)

        # Empty prompt → predict foreground cells
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        # Broadcast to batch dimension
        sparse_embeddings = sparse_embeddings.expand(imgs.shape[0], -1, -1)
        dense_embeddings = dense_embeddings.expand(imgs.shape[0], -1, -1, -1)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # (B, 1, 256, 256)

        # Upsample back to input resolution
        logits = F.interpolate(
            low_res_masks, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )
        return logits  # (B, 1, H, W)


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
