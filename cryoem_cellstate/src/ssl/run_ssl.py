"""Stage 3 inference driver: extract SSL embeddings using pretrained models.

Supports three model families, selected via ``--model``:

* **mae** (default) — pretrained ``hf_hub:timm/vit_base_patch16_224.mae_in1k``;
  CLS-token embedding; no local training required.
* **cryo-ief** — zero-shot ``westlake-repl/Cryo-IEF`` from HuggingFace Hub;
  no local training required.
* **rotnet** — pretrained ``hf_hub:timm/vit_base_patch16_224.dino``;
  CLS-token embedding; no local training required.

Usage::

    python -m src.ssl.run_ssl --model mae
    python -m src.ssl.run_ssl --model cryo-ief
    python -m src.ssl.run_ssl --model rotnet
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, NormalizeIntensity
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_device, load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# ── Shared inference transform (grayscale → (1, H, W) normalised tensor) ─────
_INFERENCE_TRANSFORM = Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    NormalizeIntensity(nonzero=True),
    EnsureType(dtype=torch.float32),
])


class CropInferenceDataset(Dataset):
    """Ordered dataset of cell crops for embedding extraction.

    Parameters
    ----------
    cell_ids:
        Ordered list of cell IDs aligned with *crop_paths*.
    crop_paths:
        Ordered list of ``.npy`` crop file paths.
    """

    def __init__(self, cell_ids: list[str], crop_paths: list[str]) -> None:
        self.cell_ids = cell_ids
        self.crop_paths = crop_paths

    def __len__(self) -> int:
        return len(self.crop_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.load(self.crop_paths[idx]).astype(np.float32)
        return _INFERENCE_TRANSFORM(img)  # (1, H, W)


# ── Per-model extractors ──────────────────────────────────────────────────────


def _extract_mae(
    cfg: object,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract CLS-token embeddings using the pretrained MAE ViT-Base."""
    import torch.nn.functional as F

    from .mae_vit import MAEViT

    mae_cfg = cfg.ssl.mae  # type: ignore[attr-defined]
    model = MAEViT(backbone=mae_cfg.backbone).to(device).eval()
    img_size: int = mae_cfg.image_size

    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imgs = F.interpolate(
                batch.to(device), size=(img_size, img_size),
                mode="bilinear", align_corners=False,
            )
            parts.append(model.get_features(imgs).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _extract_cryo_ief(
    cfg: object,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract embeddings using the Cryo-IEF foundation model from HuggingFace.

    Loads ``westlake-repl/Cryo-IEF`` via the HuggingFace ``transformers``
    library.  The model is a vision-language foundation model trained on
    cryo-EM images; we extract the visual CLS embedding from the vision
    encoder.
    """
    from transformers import AutoModel, AutoProcessor

    cryo_cfg = cfg.ssl.cryo_ief  # type: ignore[attr-defined]
    model_name: str = cryo_cfg.model_name
    img_size: int = cryo_cfg.image_size

    logger.info("Loading Cryo-IEF model: %s", model_name)
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    hf_model = hf_model.to(device).eval()

    # Try to load a processor; fall back to manual resize if unavailable
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        use_processor = True
    except Exception:
        use_processor = False
        logger.warning("No AutoProcessor found for %s — using manual resize", model_name)

    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            # batch: (B, 1, H, W) grayscale
            imgs_rgb = batch.expand(-1, 3, -1, -1)  # (B, 3, H, W)

            if use_processor:
                # processor expects PIL images or numpy arrays — convert
                import torchvision.transforms.functional as TF
                pil_imgs = [TF.to_pil_image(imgs_rgb[i].cpu()) for i in range(imgs_rgb.shape[0])]
                inputs = processor(images=pil_imgs, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = hf_model(**inputs)
            else:
                import torch.nn.functional as F
                imgs_resized = F.interpolate(imgs_rgb.to(device), size=(img_size, img_size),
                                             mode="bilinear", align_corners=False)
                outputs = hf_model(pixel_values=imgs_resized)

            # Extract the CLS token (first position of last_hidden_state)
            if hasattr(outputs, "last_hidden_state"):
                feats = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "pooler_output"):
                feats = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                feats = outputs.image_embeds
            else:
                raise RuntimeError(f"Cannot extract embeddings from Cryo-IEF output: {type(outputs)}")

            parts.append(feats.cpu().numpy())

    return np.concatenate(parts, axis=0)


def _extract_rotnet(
    cfg: object,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract CLS-token embeddings using the pretrained DINO ViT-Base."""
    import torch.nn.functional as F

    from .rotnet import DinoViT

    rotnet_cfg = cfg.ssl.rotnet  # type: ignore[attr-defined]
    model = DinoViT(backbone=rotnet_cfg.backbone).to(device).eval()
    img_size: int = rotnet_cfg.image_size

    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imgs = F.interpolate(
                batch.to(device), size=(img_size, img_size),
                mode="bilinear", align_corners=False,
            )
            parts.append(model.get_features(imgs).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main extractor ────────────────────────────────────────────────────────────


def extract_embeddings(
    cfg: object,
    model: str,
) -> np.ndarray:
    """Dispatch to the appropriate model extractor and save ``embeddings.npy``.

    Parameters
    ----------
    cfg:
        Root pipeline config.
    model:
        One of ``"mae"``, ``"cryo-ief"``, ``"rotnet"``.

    Returns
    -------
    embeddings:
        ``(N, D)`` float32 array; also written to ``<results_dir>/embeddings.npy``.
    """
    device = get_device(cfg)

    cells_parquet = Path(cfg.data.crops_dir) / "cells.parquet"  # type: ignore[attr-defined]
    if not cells_parquet.exists():
        raise FileNotFoundError(
            f"cells.parquet not found at {cells_parquet}. Run segmentation first."
        )
    cells_df = pd.read_parquet(cells_parquet)
    logger.info("Extracting embeddings for %d cells using model=%s", len(cells_df), model)

    dataset = CropInferenceDataset(
        cell_ids=cells_df["cell_id"].tolist(),
        crop_paths=cells_df["crop_path"].tolist(),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    if model == "mae":
        embeddings = _extract_mae(cfg, loader, device)
    elif model == "cryo-ief":
        embeddings = _extract_cryo_ief(cfg, loader, device)
    elif model == "rotnet":
        embeddings = _extract_rotnet(cfg, loader, device)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    out_path = Path(cfg.project.results_dir) / "embeddings.npy"  # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    logger.info("Saved embeddings %s → %s", embeddings.shape, out_path)
    return embeddings


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--model",
    type=click.Choice(["mae", "cryo-ief", "rotnet"]),
    default="mae",
    show_default=True,
    help="SSL model family to use for embedding extraction.",
)
def main(config: str | None, model: str) -> None:
    """Run Stage 3: extract SSL embeddings and save to results/embeddings.npy."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    embeddings = extract_embeddings(cfg, model=model)
    logger.info("Stage 3 complete — embeddings shape: %s", embeddings.shape)


if __name__ == "__main__":
    main()
