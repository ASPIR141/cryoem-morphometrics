"""Stage 1 pipeline driver: preprocess a directory of raw CryoEM images.

Usage::

    python -m src.preprocessing.run_preprocess --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
from monai.transforms import Compose
from tqdm import tqdm

from src.utils.config import PipelineConfig, load_config
from src.utils.plots import plot_qa_gallery
from src.utils.reporting import save_noise_report
from src.utils.seed import seed_everything

from .background import BackgroundSubtraction
from .clahe import CLAHEAndNormalize
from .dataset import CryoEMRawDataset
from .fft_filter import FFTFilter
from .metrics import compute_filter_metrics

logger = logging.getLogger(__name__)


def build_pipeline(cfg: PipelineConfig) -> Compose:
    """Build the Stage 1 preprocessing pipeline as a ``monai.transforms.Compose``.

    Returns a single composable object containing all three preprocessing steps
    (FFT filter → background subtraction → CLAHE + z-score).  Because every
    component inherits from ``monai.transforms.Transform``, the resulting
    ``Compose`` is also usable inside any larger MONAI pipeline.

    Parameters
    ----------
    cfg:
        Root pipeline config.

    Returns
    -------
    pipeline:
        ``monai.transforms.Compose([FFTFilter, BackgroundSubtraction, CLAHEAndNormalize])``.
    """
    pp = cfg.preprocessing
    return Compose([
        FFTFilter(
            low_cutoff=pp.fft_filter.low_cutoff,
            high_cutoff=pp.fft_filter.high_cutoff,
            high_pass_only=pp.fft_filter.high_pass_only,
        ),
        BackgroundSubtraction(
            method=pp.background.method,
            sigma=pp.background.sigma,
            tophat_radius=pp.background.tophat_radius,
        ),
        CLAHEAndNormalize(
            clip_limit=pp.clahe.clip_limit,
            tile_grid_size=list(pp.clahe.tile_grid_size),
        ),
    ])


def preprocess_image(
    image: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the full Stage 1 preprocessing pipeline to a single image.

    The FFT filter step is run first so that ``compute_filter_metrics`` can
    compare the raw image against the frequency-filtered version before
    background subtraction and normalisation are applied.

    Parameters
    ----------
    image:
        Raw 2-D float64 image.
    cfg:
        Root pipeline config.

    Returns
    -------
    cleaned:
        Preprocessed float64 image.
    metrics:
        Dictionary of quality metrics (sharpness and entropy before/after
        FFT filtering).
    """
    pp = cfg.preprocessing

    fft = FFTFilter(
        low_cutoff=pp.fft_filter.low_cutoff,
        high_cutoff=pp.fft_filter.high_cutoff,
        high_pass_only=pp.fft_filter.high_pass_only,
    )
    filtered = fft(image)
    filter_metrics = compute_filter_metrics(image, filtered)

    post_fft = Compose([
        BackgroundSubtraction(
            method=pp.background.method,
            sigma=pp.background.sigma,
            tophat_radius=pp.background.tophat_radius,
        ),
        CLAHEAndNormalize(
            clip_limit=pp.clahe.clip_limit,
            tile_grid_size=list(pp.clahe.tile_grid_size),
        ),
    ])
    cleaned = post_fft(filtered)

    return cleaned, filter_metrics


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--raw-dir",
    default=None,
    help="Override data.raw_dir from config.",
)
def main(config: str | None, raw_dir: str | None) -> None:
    """Run Stage 1 preprocessing on all images in the raw directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    raw_path = Path(raw_dir or cfg.data.raw_dir)
    processed_path = Path(cfg.data.processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    noise_dir = Path(cfg.preprocessing.noise_stats.results_dir)
    noise_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = CryoEMRawDataset(
        raw_dir=raw_path,
        extensions=set(cfg.data.image_extensions),
    )
    if len(dataset) == 0:
        logger.warning(
            "No images found in %s with extensions %s",
            raw_path,
            cfg.data.image_extensions,
        )
        return

    # ── Process each image ────────────────────────────────────────────────────
    qa_pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
    all_metrics: list[dict[str, float]] = []

    for item in tqdm(dataset, desc="Preprocessing"):
        name: str = item["name"]
        raw: np.ndarray = item["image"]          # float64 (H, W)

        logger.info("Processing %s", name)

        # Noise statistics on the raw image
        noise_metrics = save_noise_report(raw, name, noise_dir)

        # Full preprocessing
        cleaned, filter_metrics = preprocess_image(raw, cfg)

        # Save processed tensor
        if cfg.preprocessing.output_format == "pt":
            import torch
            torch.save(
                torch.from_numpy(cleaned).float(),
                processed_path / f"{name}.pt",
            )
        else:
            np.save(processed_path / f"{name}.npy", cleaned)

        qa_pairs.append((raw, cleaned, name))
        all_metrics.append({"name": name, **noise_metrics, **filter_metrics})

        logger.info("Saved processed tensor for %s", name)

    # ── Save QA gallery & metrics CSV ─────────────────────────────────────────
    qa_path = Path(cfg.project.results_dir) / "stage1" / "qa_gallery.png"
    plot_qa_gallery(qa_pairs, qa_path)

    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(noise_dir / "noise_metrics.csv", index=False)
    logger.info("Stage 1 complete — processed %d images", len(dataset))


if __name__ == "__main__":
    main()
