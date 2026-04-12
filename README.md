# CryoEM Cell-State Discovery Pipeline

Automated discovery and quantification of morphological cell states in unlabeled CryoEM images of cells exposed to 400 mM sucrose.

**Core hypothesis:** Cells exposed to 400 mM sucrose exhibit distinct morphological states that can be discovered without labels using SSL embeddings + morphometric analysis.

---

## Setup

`pyproject.toml` and the virtual environment live in the **repo root** (`CryoEM/`). All commands below are run from the repo root.

```bash
# Create venv and install all dependencies
uv venv --python 3.11
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

---

## Data Layout

```
cryoem_cellstate/data/
├── raw/          ← Input CryoEM images (.tif, .tiff, .png)
├── processed/    ← Stage 1 output: preprocessed .npy tensors
├── masks/        ← Stage 2 output: binary segmentation masks
├── crops/
│   ├── images/   ← Per-cell crops (.npy, 128×128)
│   └── masks/    ← Corresponding cell masks (.npy)
└── cvat_gt/      ← Optional CVAT ground-truth masks for evaluation
```

Place raw CryoEM images into `cryoem_cellstate/data/raw/`.

---

## Running Each Stage

All commands are run **from the repo root** (`CryoEM/`). Pass `--config` to override defaults in `cryoem_cellstate/configs/default.yaml`.

### Stage 1 — Preprocessing
```bash
PYTHONPATH=cryoem_cellstate python -m src.preprocessing.run_preprocess
```
Outputs cleaned tensors to `cryoem_cellstate/data/processed/`, QA gallery to `cryoem_cellstate/results/stage1/`.

### Stage 2b — Train U-Net (optional, improves over classical baseline)
```bash
PYTHONPATH=cryoem_cellstate python -m src.segmentation.train_unet
```

### Stage 3a — Train SimCLR
```bash
PYTHONPATH=cryoem_cellstate python -m src.ssl.train_simclr
```

### Stage 3b — Extract embeddings
```bash
PYTHONPATH=cryoem_cellstate python -m src.ssl.extract_embeddings
```

### Stage 3c — Optional: MAE or RotNet
```bash
PYTHONPATH=cryoem_cellstate python -m src.ssl.mae
PYTHONPATH=cryoem_cellstate python -m src.ssl.rotnet
```

### Full end-to-end pipeline
```bash
PYTHONPATH=cryoem_cellstate python cryoem_cellstate/scripts/run_full_pipeline.py
```

Skippable stages:

| Flag | Skips |
|------|-------|
| `--skip-stage1` | Preprocessing (use existing `data/processed/`) |
| `--skip-stage2` | Segmentation (use existing `data/crops/cells.parquet`) |
| `--skip-stage3` | SSL training (use existing `results/embeddings.npy`) |

---

## Running Tests

Tests are run from the repo root — pytest picks up `cryoem_cellstate/tests/` automatically via `pyproject.toml`.

```bash
# All tests
pytest

# Single stage
pytest cryoem_cellstate/tests/test_preprocessing.py -v

# With coverage
pytest --cov=cryoem_cellstate/src --cov-report=term-missing
```

---

## Configuration

All parameters live in `cryoem_cellstate/configs/default.yaml`. Override per-run:

```bash
PYTHONPATH=cryoem_cellstate python -m src.preprocessing.run_preprocess \
    --config cryoem_cellstate/configs/my_overrides.yaml
```

Key parameters:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `preprocessing.fft_filter` | `low_cutoff` | 0.02 | Low freq cutoff (Nyquist fraction) |
| `preprocessing.fft_filter` | `high_cutoff` | 0.40 | High freq cutoff |
| `segmentation.classical` | `min_cell_area` | 500 | Min cell area (pixels) |
| `ssl.simclr` | `epochs` | 200 | SimCLR training epochs |
| `ssl.simclr` | `temperature` | 0.07 | NT-Xent temperature |
| `clustering.hdbscan` | `min_cluster_size` | 15 | Minimum cluster size |
| `clustering.kmeans` | `n_clusters` | 4 | KMeans cluster count |

---

## Results

After a full run, `cryoem_cellstate/results/` contains:

```
results/
├── stage1/
│   ├── noise/           ← PSD plots, SNR histograms, noise_metrics.csv
│   └── qa_gallery.png   ← Before/after preprocessing grid
├── stage2/
│   └── segmentation_metrics.csv
├── morphometrics.parquet
├── embeddings.npy
├── clustering/
│   ├── umap_2d.npy
│   └── umap_cluster.npy
├── figures/
│   ├── fig3_histograms.png
│   ├── fig4_umap_*.png
│   ├── fig5_boxplots.png
│   ├── fig6_significance.png
│   └── fig7_diffusion.png
└── report.md            ← Final summary report
```
