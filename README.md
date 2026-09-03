# Self-Supervised Morphological Phenotype Discovery in Cryo-EM Images

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
| VISTA2D model | `vista2d.py` | `build_vista2d()` — pretrained MONAI VISTA2D (SAM ViT-B + fine-tuned adapter, 2-D, 256×256 sliding window) |
| Training | `train_vista2d_lightning.py` | `Vista2DLightningModule` + `Vista2DDataModule`; saves `best_vista2d.ckpt` |
| Inference driver | `run_segmentation.py` | Classical (default) or VISTA2D (`--use-vista2d`) + cell cropping |
| Inference script | `scripts/run_vista2d_segmentation.py` | Standalone VISTA2D inference with `SlidingWindowInferer` |
| Evaluation | `evaluate_segmentation.py` | `DiceMetric`, `MeanIoU`, `HausdorffDistanceMetric` vs CVAT GT |
| Cropper | `crop_cells.py` | `CellCropper(Transform)` — per-cell crops + `cells.parquet` catalogue |

### Stage 3 — Self-Supervised Learning (`src/ssl/`)

Three interchangeable SSL model families, selectable via `--model` in `run_ssl.py`.
**All models use pretrained weights — no local training required.**

| Model | Files | Description |
|-------|-------|-------------|
| **MAE-ViT** (primary) | `mae_vit.py` | `hf_hub:timm/vit_base_patch16_224.mae_in1k` — ImageNet MAE-pretrained ViT-Base; CLS-token embedding |
| **Cryo-IEF** (zero-shot) | `run_ssl.py` | `westlake-repl/Cryo-IEF` from HuggingFace; CLS embedding from vision encoder |
| **DINOv3 ViT** (auxiliary) | `rotnet.py` | `facebook/dinov3-vitl16-pretrain-lvd1689m` — DINOv3 self-supervised ViT-Large (1024-D CLS token); requires `transformers ≥ 4.56.0` |

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

### Stage 2b — Fine-tune VISTA2D
```bash
# Fine-tune on your CryoEM dataset (downloads SAM + VISTA2D weights from HuggingFace on first run):
uv run --directory cryoem_cellstate -m src.segmentation.train_vista2d_lightning

# With GPU and custom epochs:
uv run --directory cryoem_cellstate -m src.segmentation.train_vista2d_lightning \
    --max-epochs 50 --accelerator gpu --devices 1
```
Saves checkpoint to `results/seg_checkpoints/best_vista2d.ckpt`.

> **Pretrained weights** are downloaded automatically from
> [`MONAI/vista2d @ 0.4.0`](https://huggingface.co/MONAI/vista2d/tree/0.4.0)
> on first run and cached in `~/.cache/monai/vista2d`.
> An internet connection is required only on the first run.

### Stage 2c — VISTA2D inference
```bash
# Via run_segmentation (also crops cells):
uv run --directory cryoem_cellstate -m src.segmentation.run_segmentation \
    --use-vista2d --checkpoint results/seg_checkpoints/best_vista2d.ckpt

# Standalone inference script:
python cryoem_cellstate/scripts/run_vista2d_segmentation.py \
    --checkpoint results/seg_checkpoints/best_vista2d.ckpt
```

### Stage 3 — Extract embeddings (pretrained, no training needed)
```bash
# MAE-ViT (hf_hub:timm/vit_base_patch16_224.mae_in1k):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model mae

# Cryo-IEF (westlake-repl/Cryo-IEF):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model cryo-ief

# DINOv3 ViT-Large (facebook/dinov3-vitl16-pretrain-lvd1689m):
uv run --directory cryoem_cellstate -m src.ssl.run_ssl --model rotnet
```

> **DINOv3 requirement**: `transformers ≥ 4.56.0` is required for the
> `facebook/dinov3-vitl16-pretrain-lvd1689m` model.  Install or upgrade with
> `uv pip install "transformers>=4.56.0"`.

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
| `segmentation.segmentation_model.vista2d` | `hf_repo` | `MONAI/vista2d` | HuggingFace repository for VISTA2D weights |
| `segmentation.segmentation_model.vista2d` | `hf_revision` | `0.4.0` | Pinned model version |
| `segmentation.segmentation_model.vista2d` | `roi_size` | `[256, 256]` | Sliding-window tile size |
| `ssl.active_model` | — | `mae` | Default model for `run_ssl.py` |
| `ssl.mae` | `backbone` | `hf_hub:timm/vit_base_patch16_224.mae_in1k` | Pretrained MAE ViT-Base from timm Hub |
| `ssl.rotnet` | `model_name` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 ViT-Large from HuggingFace (`transformers ≥ 4.56.0`) |
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
│   └── best_vista2d.ckpt    ← Fine-tuned VISTA2D checkpoint (Stage 2b)
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

# TODO
Stage 1: Advance the Noise Modeling (noise_stats.py). Don't just compute a generic Gaussian+Poisson model. Specifically implement a Shot-Noise (Poisson) simulation framework that models photon-starved regimes. Calculate the exact Signal-to-Noise Ratio (SNR) transition threshold where classical Otsu thresholding breaks down, forcing the system to rely on semantic segmentation.

Stage 1: Document the 2D FFT Band-Pass (fft_filter.py). Be ready to explain how you handle structural background artifacts (like the grid carbon edges or ice thickness variation in CryoEM) using frequency-domain masks. This perfectly mirrors how Spore.Bio isolates bacterial signatures from multi-modal optical backgrounds.

Stage 2: Expand the Classical Baseline (classical.py). Make sure you can write a clean, native OpenCV/NumPy pipeline for morphological operations (using custom structuring elements for dilation, erosion, and opening). This proves you have the core computer vision engineering depth required for edge deployment. 

Stage 3: Information Density & Tokenization Filtering. Treat your unlabeled cell crops as your "pretraining token database." Implement an upstream statistical filter (using your Stage 1 image entropy metrics or Stage 5 clustering densities) to prune redundant, blank, or highly artifacted crops before they ever hit the MAE-ViT encoder. This maps directly to the core responsibilities of the Gemini Data Pretraining team, showing you treat data curation as an active algorithmic science. 