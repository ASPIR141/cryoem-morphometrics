"""YAML config loading backed by Pydantic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


# ── Section models ────────────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    name: str = "cryoem_cellstate"
    seed: int = 42
    device: str = "auto"
    results_dir: str = "results"


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    masks_dir: str = "data/masks"
    crops_dir: str = "data/crops"
    cvat_gt_dir: str = "data/cvat_gt"
    image_extensions: list[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


# ── Stage 1 ───────────────────────────────────────────────────────────────────


class FFTFilterConfig(BaseModel):
    low_cutoff: float = 0.02
    high_cutoff: float = 0.4
    high_pass_only: bool = False


class BackgroundConfig(BaseModel):
    method: str = "gaussian"
    sigma: float = 50.0
    tophat_radius: int = 40


class CLAHEConfig(BaseModel):
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


class NoiseStatsConfig(BaseModel):
    results_dir: str = "results/stage1/noise"


class PreprocessingConfig(BaseModel):
    output_format: str = "npy"
    fft_filter: FFTFilterConfig = FFTFilterConfig()
    background: BackgroundConfig = BackgroundConfig()
    clahe: CLAHEConfig = CLAHEConfig()
    noise_stats: NoiseStatsConfig = NoiseStatsConfig()


# ── Stage 2 ───────────────────────────────────────────────────────────────────


class ClassicalSegConfig(BaseModel):
    method: str = "otsu"
    min_cell_area: int = 500
    max_cell_area: int = 50000
    watershed: bool = True


class Vista2DConfig(BaseModel):
    hf_repo: str = "MONAI/vista2d"
    hf_revision: str = "0.4.0"
    cache_dir: str = "~/.cache/monai/vista2d"
    roi_size: list[int] = [256, 256]
    sw_batch_size: int = 4
    overlap: float = 0.25


class SegmentationModelConfig(BaseModel):
    in_channels: int = 1
    num_classes: int = 1
    lr: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    val_fraction: float = 0.15
    vista2d: Vista2DConfig = Vista2DConfig()


class CropCellsConfig(BaseModel):
    min_area: int = 500
    max_aspect_ratio: float = 4.0
    crop_size: int = 128
    results_dir: str = "results/stage2"


class SegmentationConfig(BaseModel):
    classical: ClassicalSegConfig = ClassicalSegConfig()
    segmentation_model: SegmentationModelConfig = SegmentationModelConfig()
    crop_cells: CropCellsConfig = CropCellsConfig()


# ── Stage 3 ───────────────────────────────────────────────────────────────────


class AugmentationConfig(BaseModel):
    crop_scale: list[float] = [0.2, 1.0]
    flip: bool = True
    rotation_degrees: int = 15
    gaussian_blur_kernel: int = 23
    gaussian_noise_std: float = 0.05


class MAEConfig(BaseModel):
    backbone: str = "hf_hub:timm/vit_base_patch16_224.mae_in1k"
    image_size: int = 224
    batch_size: int = 64


class RotNetConfig(BaseModel):
    backbone: str = "hf_hub:timm/vit_base_patch16_224.dino"
    image_size: int = 224
    batch_size: int = 64


class CryoIEFConfig(BaseModel):
    model_name: str = "westlake-repl/Cryo-IEF"
    image_size: int = 224
    batch_size: int = 64


class SSLConfig(BaseModel):
    active_model: str = "mae"
    checkpoints_dir: str = "results/ssl_checkpoints"
    mae: MAEConfig = MAEConfig()
    rotnet: RotNetConfig = RotNetConfig()
    cryo_ief: CryoIEFConfig = CryoIEFConfig()


# ── Stage 4 ───────────────────────────────────────────────────────────────────


class MorphometricsConfig(BaseModel):
    glcm_distances: list[int] = [1, 3, 5]
    glcm_angles: list[float] = [0, 0.785, 1.571, 2.356]
    output_path: str = "results/morphometrics.parquet"


# ── Stage 5 ───────────────────────────────────────────────────────────────────


class UMAPConfig(BaseModel):
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components_2d: int = 2
    n_components_cluster: int = 10
    metric: str = "euclidean"
    random_state: int = 42


class PCAConfig(BaseModel):
    n_components: int = 50   # retained principal components fed to UMAP
    whiten: bool = False
    random_state: int = 42


class HDBSCANConfig(BaseModel):
    min_cluster_size: int = 15
    min_samples: int = 5
    metric: str = "euclidean"


class ClusteringConfig(BaseModel):
    pca: PCAConfig = PCAConfig()
    umap: UMAPConfig = UMAPConfig()
    hdbscan: HDBSCANConfig = HDBSCANConfig()
    bootstrap_iterations: int = 100
    results_dir: str = "results/clustering"


# ── Stage 6 ───────────────────────────────────────────────────────────────────


class EvaluationConfig(BaseModel):
    significance_alpha: float = 0.05
    figures_dir: str = "results/figures"
    report_path: str = "results/report.md"


# ── Root config ───────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    ssl: SSLConfig = SSLConfig()
    morphometrics: MorphometricsConfig = MorphometricsConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# ── Loader ────────────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load pipeline config, merging an optional override file on top of defaults.

    Parameters
    ----------
    path:
        Path to a YAML override file.  If *None*, built-in defaults are used.

    Returns
    -------
    PipelineConfig
        Validated Pydantic configuration object.
    """
    if path is not None:
        with open(Path(path)) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = PipelineConfig(**overrides)
        logger.info("Config loaded: overrides from %s", path)
    else:
        # Load from default YAML to support any values not in model defaults
        if _DEFAULT_CONFIG.exists():
            with open(_DEFAULT_CONFIG) as f:
                data = yaml.safe_load(f) or {}
            cfg = PipelineConfig(**data)
        else:
            cfg = PipelineConfig()
        logger.info("Config loaded: defaults only")
    return cfg


def get_device(cfg: PipelineConfig) -> Any:
    """Resolve the torch device from config.

    Parameters
    ----------
    cfg:
        Root pipeline config.

    Returns
    -------
    torch.device
    """
    import torch

    setting = cfg.project.device
    if setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(setting)
    logger.info("Using device: %s", device)
    return device
