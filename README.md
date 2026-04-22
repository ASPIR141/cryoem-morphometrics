# CryoEM Cell-State Discovery Pipeline

Automated discovery and quantification of morphological cell states in unlabeled CryoEM images of cells exposed to 400 mM sucrose.

**Core hypothesis:** Cells exposed to 400 mM sucrose exhibit distinct morphological states discoverable without labels using SSL embeddings + morphometric analysis.

---

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

All commands are run from the **repo root** (`CryoEM/`).

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

Place raw CryoEM images into `cryoem_cellstate/data/raw/`.

---

## Architecture Overview

### Stage 1 — Preprocessing (`src/preprocessing/`)

| Component | File | Description |
|-----------|------|-------------|
| Dataset | `dataset.py` | `CryoEMRawDataset` — discovers and loads raw images |
| FFT filter | `fft_filter.py` | `FFTFilter(Transform)` — radial band-pass via `torch.fft` |
| Background | `background.py` | `BackgroundSubtraction(Transform)` — Gaussian blur or top-hat |
| CLAHE | `clahe.py` | `CLAHEAndNormalize(Transform)` — OpenCV CLAHE + z-score |
| Noise stats | `noise_stats.py` | PSD, SNR histograms, Gaussian+Poisson noise model |
| Driver | `run_preprocess.py` | Iterates raw images, saves `.npy` tensors + QA gallery |

### Stage 2 — Segmentation (`src/segmentation/`)

| Component | File | Description |
|-----------|------|-------------|
| Classical | `classical.py` | `ClassicalSegmentation(Transform)` — Otsu/adaptive → cleanup → watershed |
| Swin UNETR model | `unet.py` | `build_swin_unetr()` — MONAI `SwinUNETR` (Swin-Transformer encoder + CNN decoder, 2-D, 256×256) |
| Training | `train_unet_lightning.py` | `SwinUNETRLightningModule` + `SwinUNETRDataModule`; saves `best_swin_unetr.ckpt` |
| Inference driver | `run_segmentation.py` | Classical (default) or Swin UNETR (`--use-swin-unetr`) + cell cropping |
| Inference script | `scripts/run_unetr_segmentation.py` | Standalone Swin UNETR inference with `SlidingWindowInferer` |
| Evaluation | `evaluate_segmentation.py` | `DiceMetric`, `MeanIoU`, `HausdorffDistanceMetric` vs CVAT GT |
| Cropper | `crop_cells.py` | `CellCropper(Transform)` — per-cell crops + `cells.parquet` catalogue |

### Stage 3 — Self-Supervised Learning (`src/ssl/`)

Three interchangeable SSL model families, selectable via `--model` in `run_ssl.py`.
**All models use pretrained weights — no local training required.**

| Model | Files | Description |
|-------|-------|-------------|
| **MAE-ViT** (primary) | `mae_vit.py` | `hf_hub:timm/vit_base_patch16_224.mae_in1k` — ImageNet MAE-pretrained ViT-Base; CLS-token embedding |
| **Cryo-IEF** (zero-shot) | `run_ssl.py` | `westlake-repl/Cryo-IEF` from HuggingFace; CLS embedding from vision encoder |
| **DINO ViT** (auxiliary) | `rotnet.py` | `hf_hub:timm/vit_base_patch16_224.dino` — DINO self-supervised ViT-Base; CLS-token embedding |

### Stage 4 — Morphometrics (`src/morphometrics/`)

`MorphometricExtractor(Transform)` accepts `{"image", "mask", "cell_id"}` dicts and returns a flat feature dict. Six feature groups:

| Group | Features |
|-------|----------|
| Shape | area, circularity, eccentricity, solidity, extent, boundary roughness |
| Electron density | mean/std intensity, total electron density, density CV |
| Volume shrinkage | `volume_shrinkage` — population-relative area ratio |
| Nucleoid condensation | `nucleoid_condensed_fraction`, `nucleoid_n_foci` |
| Boundary / LEDS | `leds_fft_roughness`, `leds_protrusion_count`, `leds_hull_defect_fraction` |
| GLCM texture | contrast, dissimilarity, homogeneity, energy, correlation |

### Stage 5 — Clustering (`src/clustering/`)

Pipeline: **PCA → UMAP → HDBSCAN**

| Component | File | Description |
|-----------|------|-------------|
| PCA | `umap_reduce.py` | `PCAReducer(Transform)` — scikit-learn PCA |
| UMAP | `umap_reduce.py` | Two runs: 2-D (visualisation) + n-D (clustering) |
| HDBSCAN | `cluster.py` | `HDBSCANCluster(Transform)` |
| State analysis | `state_analysis.py` | Per-cluster summaries, silhouette, Davies–Bouldin, bootstrap stability |

### Stage 6 — Evaluation (`src/evaluation/`)

ANOVA + Kolmogorov–Smirnov tests per morphometric feature with Benjamini–Hochberg correction. All plotting in `src/utils/plots.py`.

---

## Running Each Stage

### Stage 1 — Preprocessing
```bash
uv run --directory cryoem_cellstate -m src.preprocessing.run_preprocess
# Override raw image directory:
uv run --directory cryoem_cellstate -m src.preprocessing.run_preprocess --raw-dir /path/to/images
```

### Stage 2a — Classical segmentation + crop cells
```bash
uv run --directory cryoem_cellstate -m src.segmentation.run_segmentation
```

### Stage 2b — Train Swin UNETR
```bash
# Fine-tune from scratch:
uv run --directory cryoem_cellstate -m src.segmentation.train_unet_lightning

# Fine-tune from MONAI SSL pretrained weights:
uv run --directory cryoem_cellstate -m src.segmentation.train_unet_lightning \
    --pretrained-weights /path/to/swin_unetr_ssl.pt

# With GPU and custom epochs:
uv run --directory cryoem_cellstate -m src.segmentation.train_unet_lightning \
    --max-epochs 100 --accelerator gpu --devices 1
```
Saves checkpoint to `results/seg_checkpoints/best_swin_unetr.ckpt`.

> **Pretrained SSL weights** for MONAI SwinUNETR are available from the
> [MONAI Model Zoo](https://github.com/Project-MONAI/MONAI-extra-test-data/releases).
> Pass the downloaded `.pt` file via `--pretrained-weights`.

### Stage 2c — Swin UNETR inference
```bash
# Via run_segmentation (also crops cells):
uv run --directory cryoem_cellstate -m src.segmentation.run_segmentation \
    --use-swin-unetr --checkpoint results/seg_checkpoints/best_swin_unetr.ckpt

# Standalone inference script:
python cryoem_cellstate/scripts/run_unetr_segmentation.py \
    --checkpoint results/seg_checkpoints/best_swin_unetr.ckpt
```

### Stage 3 — Extract embeddings (pretrained, no training needed)
```bash
# MAE-ViT (hf_hub:timm/vit_base_patch16_224.mae_in1k):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model mae

# Cryo-IEF (westlake-repl/Cryo-IEF):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model cryo-ief

# DINO ViT-Base (hf_hub:timm/vit_base_patch16_224.dino):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model rotnet
```

### Stage 4 — Morphometric features
```bash
uv run --directory cryoem_cellstate -m src.morphometrics.run_morphometrics
```

### Stage 5 — Clustering
```bash
uv run --directory cryoem_cellstate -m src.clustering.run_clustering
```

### Stage 6 — Evaluation & figures
```bash
uv run --directory cryoem_cellstate -m src.evaluation.run_evaluation
```

### Full end-to-end pipeline
```bash
uv run --directory cryoem_cellstate cryoem_cellstate/scripts/run_full_pipeline.py

# Skip stages with existing artifacts:
#   --skip-stage1  processed/ already exists
#   --skip-stage2  masks/ and crops/ already exist
#   --skip-stage3  embeddings.npy already exists (skips pretrained model download + inference)
uv run --directory cryoem_cellstate cryoem_cellstate/scripts/run_full_pipeline.py \
    --skip-stage1 --skip-stage2 --skip-stage3
```

> Stage 3 uses pretrained HuggingFace Hub models — an internet connection is
> required on the first run. Subsequent runs use the local timm/HF cache.

---

## Running Tests

```bash
pytest
pytest cryoem_cellstate/tests/test_preprocessing.py -v
pytest --cov=cryoem_cellstate/src --cov-report=term-missing
```

---

## Configuration

All parameters live in `cryoem_cellstate/configs/default.yaml`. Pass `--config` to any script to override.

Key parameters:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `preprocessing.fft_filter` | `low_cutoff` / `high_cutoff` | 0.02 / 0.40 | Band-pass cutoffs (Nyquist fraction) |
| `segmentation.unet.swin_unetr` | `feature_size` | 48 | Swin-Tiny base channels (48/96/128) |
| `segmentation.unet.swin_unetr` | `depths` | [2,2,2,2] | Transformer blocks per stage |
| `ssl.active_model` | — | `mae` | Default model for `run_ssl.py` |
| `ssl.mae` | `backbone` | `hf_hub:timm/vit_base_patch16_224.mae_in1k` | Pretrained MAE ViT-Base from timm Hub |
| `ssl.rotnet` | `backbone` | `hf_hub:timm/vit_base_patch16_224.dino` | Pretrained DINO ViT-Base from timm Hub |
| `ssl.cryo_ief` | `model_name` | `westlake-repl/Cryo-IEF` | HuggingFace model ID |
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
│   └── best_swin_unetr.ckpt ← Fine-tuned Swin UNETR checkpoint (Stage 2b)
├── embeddings.npy           ← SSL embedding matrix (N × D) from Stage 3
├── morphometrics.parquet    ← Per-cell morphometric features from Stage 4
├── clustering/
│   ├── pca_reduced.npy
│   ├── umap_2d.npy
│   ├── umap_cluster.npy
│   └── hdbscan_labels.npy
├── figures/
│   ├── fig3_histograms.png
│   ├── fig4_umap_hdbscan.png
│   ├── fig5_boxplots.png
│   ├── fig6_significance.png
│   └── fig7_diffusion.png
└── report.md
```
