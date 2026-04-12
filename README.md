# CryoEM Cell-State Discovery Pipeline

Automated discovery and quantification of morphological cell states in unlabeled CryoEM images of cells exposed to 400 mM sucrose.

**Core hypothesis:** Cells exposed to 400 mM sucrose exhibit distinct morphological states that can be discovered without labels using SSL embeddings + morphometric analysis.

---

## Setup

`pyproject.toml` and the virtual environment live in the **repo root** (`CryoEM/`). All commands below are run from the repo root.

```bash
uv venv --python 3.11
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

---

## Data Layout

```
cryoem_cellstate/data/
├── raw/          ← Input CryoEM images (.tif, .tiff, .png, .jpg, .jpeg)
├── processed/    ← Stage 1 output: preprocessed .npy tensors
├── masks/        ← Stage 2 output: binary segmentation masks
├── crops/
│   ├── images/   ← Per-cell crops (.npy, 128×128)
│   └── masks/    ← Corresponding cell masks (.npy)
└── cvat_gt/      ← Optional CVAT ground-truth masks for evaluation
```

Place raw CryoEM images into `cryoem_cellstate/data/raw/`. JPEG, TIFF, and PNG are all supported.

---

## Architecture Overview

### Stage 1 — Preprocessing (`src/preprocessing/`)

| Component | File | Description |
|-----------|------|-------------|
| Dataset | `dataset.py` | `CryoEMRawDataset` — MONAI `Dataset` that discovers and loads all images from `data/raw/`. Each item: `{"image": float64 ndarray, "name": str, "path": Path}`. Accepts an optional `transform` for fused load+preprocess. |
| FFT filter | `fft_filter.py` | `FFTFilter(Transform)` — radial band-pass via `torch.fft` |
| Background | `background.py` | `BackgroundSubtraction(Transform)` — Gaussian blur or top-hat |
| CLAHE | `clahe.py` | `CLAHEAndNormalize(Transform)` — OpenCV CLAHE + MONAI `NormalizeIntensity` z-score |
| Noise stats | `noise_stats.py` | `EstimateSNR`, `RadialPSD`, `FitNoiseModel` (all `Transform`) |
| Driver | `run_preprocess.py` | `build_pipeline(cfg)` returns a full MONAI `Compose`; iterates `CryoEMRawDataset` |

Every preprocessing step is a `monai.transforms.Transform` subclass, so the full pipeline is a single `monai.transforms.Compose` that can be passed directly to any MONAI dataset or loader.

### Stage 2 — Segmentation (`src/segmentation/`)

| Component | File | Description |
|-----------|------|-------------|
| Classical | `classical.py` | `ClassicalSegmentation(Transform)` — internally a `Compose([OtsuThreshold, MorphologicalCleanup, WatershedSeparation])` |
| UNETR model | `unet.py` | `build_unetr()` — `monai.networks.nets.UNETR` (ViT encoder + CNN decoder, 2-D, 256×256 input) |
| Training | `train_unet.py` | `CacheDataset` + MONAI augmentation pipeline; `DiceMetric`/`MeanIoU` evaluation; saves `best_unetr.pth` |
| Inference | `scripts/run_unetr_segmentation.py` | Runs `CryoEMRawDataset` through a trained UNETR with `SlidingWindowInferer`; writes masks to `data/masks/` |
| Evaluation | `evaluate_segmentation.py` | `DiceMetric`, `MeanIoU`, `HausdorffDistanceMetric` vs CVAT ground truth |
| Cropper | `crop_cells.py` | `CellCropper(Transform)` — extracts per-cell crops; `PadToSquare(Transform)` |

### Stage 3 — Self-Supervised Learning (`src/ssl/`)

SimCLR with a timm backbone (ResNet18 default). Augmentation pipeline is a MONAI `Compose` adapted for grayscale CryoEM (no colour jitter; Gaussian noise models detector noise). `NTXentLoss` wraps `monai.losses.ContrastiveLoss`.

Both `TwoViewDataset` (training) and `CropInferenceDataset` (embedding extraction) apply a MONAI normalisation `Compose` on load.

### Stage 4 — Morphometrics (`src/morphometrics/`)

`MorphometricExtractor(Transform)` accepts `{"image", "mask", "cell_id"}` dicts and returns a flat feature dict. Six feature groups:

| Group | Features |
|-------|----------|
| Shape | area, circularity, eccentricity, solidity, extent, boundary roughness |
| Electron density | `mean_intensity`, `std_intensity`, `total_electron_density`, `density_cv` (CV), MONAI-normalised counterparts |
| Volume shrinkage | `volume_shrinkage` — computed population-wide by `compute_volume_shrinkage()` after extraction |
| Nucleoid condensation | `nucleoid_condensed_fraction`, `nucleoid_intensity_cv`, `nucleoid_n_foci` (local maxima) |
| Boundary / LEDS | `leds_fft_roughness` (Fourier descriptor energy ratio), `leds_protrusion_count` (convex-hull defects), `leds_hull_defect_fraction` |
| GLCM texture | contrast, dissimilarity, homogeneity, energy, correlation (mean + range) |

### Stage 5 — Clustering (`src/clustering/`)

Pipeline: **PCA → UMAP → HDBSCAN** (KMeans removed).

| Component | File | Description |
|-----------|------|-------------|
| PCA | `umap_reduce.py` | `PCAReducer(Transform)` — scikit-learn PCA, stores fitted model in `self.pca` |
| UMAP | `umap_reduce.py` | `UMAPReducer(Transform)` — two runs: 2-D (visualisation) and n-D (clustering) |
| Composed pipeline | `umap_reduce.py` | `build_reduction_pipeline(cfg)` returns `Compose([PCAReducer, UMAPReducer])` |
| HDBSCAN | `cluster.py` | `HDBSCANCluster(Transform)` |

### Stage 6 — Evaluation (`src/evaluation/`)

ANOVA + Kolmogorov–Smirnov tests per morphometric feature with Benjamini–Hochberg correction. All plotting delegates to `src/utils/plots.py`.

---

## Running Each Stage

All commands are run from the **repo root** (`CryoEM/`).

### Stage 1 — Preprocessing
```bash
uv run --directory cryoem_cellstate -m src.preprocessing.run_preprocess
# Override raw image directory:
uv run --directory cryoem_cellstate -m src.preprocessing.run_preprocess --raw-dir /path/to/images
```

### Stage 2a — Classical segmentation (runs inside full pipeline automatically)

### Stage 2b — Train UNETR
```bash
uv run --directory cryoem_cellstate -m src.segmentation.train_unet
```
Saves checkpoint to `results/seg_checkpoints/best_unetr.pth`.

### Stage 2c — Run UNETR inference on images
```bash
# On processed images (default)
uv run --directory cryoem_cellstate cryoem_cellstate/scripts/run_unetr_segmentation.py \
    --checkpoint results/seg_checkpoints/best_unetr.pth

# On a custom directory (e.g. raw images)
uv run --directory cryoem_cellstate cryoem_cellstate/scripts/run_unetr_segmentation.py \
    --input-dir data/raw \
    --output-dir data/masks \
    --checkpoint results/seg_checkpoints/best_unetr.pth
```

### Stage 3a — Train SimCLR
```bash
uv run --directory cryoem_cellstate -m src.ssl.train_simclr
```

### Stage 3b — Extract embeddings
```bash
uv run --directory cryoem_cellstate -m src.ssl.extract_embeddings
```

### Full end-to-end pipeline
```bash
uv run --directory cryoem_cellstate cryoem_cellstate/scripts/run_full_pipeline.py
```

| Flag | Effect |
|------|--------|
| `--skip-stage1` | Skip preprocessing (use existing `data/processed/`) |
| `--skip-stage2` | Skip segmentation (use existing `data/crops/cells.parquet`) |
| `--skip-stage3` | Skip SSL training (use existing `results/embeddings.npy`) |

---

## Running Tests

```bash
# All tests
pytest

# Single module
pytest cryoem_cellstate/tests/test_preprocessing.py -v

# With coverage
pytest --cov=cryoem_cellstate/src --cov-report=term-missing
```

---

## Configuration

All parameters live in `cryoem_cellstate/configs/default.yaml`. Pass `--config` to any script to override.

Key parameters:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `data` | `image_extensions` | `[.tif,.tiff,.png,.jpg,.jpeg]` | Supported raw image formats |
| `preprocessing.fft_filter` | `low_cutoff` / `high_cutoff` | 0.02 / 0.40 | Band-pass cutoffs (Nyquist fraction) |
| `segmentation.unet.unetr` | `hidden_size` | 768 | UNETR ViT encoder dimension |
| `segmentation.unet.unetr` | `num_heads` | 12 | UNETR self-attention heads |
| `segmentation.unet` | `batch_size` | 4 | Training batch size |
| `ssl.simclr` | `temperature` | 0.07 | NT-Xent temperature |
| `ssl.simclr` | `epochs` | 200 | SimCLR training epochs |
| `clustering.pca` | `n_components` | 50 | PCA components fed to UMAP |
| `clustering.umap` | `n_neighbors` | 15 | UMAP neighbourhood size |
| `clustering.hdbscan` | `min_cluster_size` | 15 | Minimum HDBSCAN cluster size |

---

## Results

After a full run, `cryoem_cellstate/results/` contains:

```
results/
├── stage1/
│   ├── noise/               ← PSD plots, SNR histograms, noise_metrics.csv
│   └── qa_gallery.png       ← Before/after preprocessing grid
├── seg_checkpoints/
│   └── best_unetr.pth       ← Best UNETR checkpoint
├── ssl_checkpoints/
│   └── best_simclr.pth
├── morphometrics.parquet    ← Per-cell features incl. volume_shrinkage, LEDS, nucleoid
├── embeddings.npy
├── clustering/
│   ├── pca_reduced.npy
│   ├── umap_2d.npy
│   └── umap_cluster.npy
├── figures/
│   ├── fig3_histograms.png
│   ├── fig4_umap_hdbscan.png
│   ├── fig5_boxplots.png
│   ├── fig6_significance.png
│   └── fig7_diffusion.png
└── report.md                ← Final summary with cluster quality & state interpretation
```
