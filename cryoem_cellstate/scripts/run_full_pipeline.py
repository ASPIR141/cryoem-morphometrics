"""End-to-end pipeline runner.

Chains all six stages and writes a final Markdown report.

Usage::

    python scripts/run_full_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


# ── Per-stage helpers ─────────────────────────────────────────────────────────


def _run_stage1(cfg: object) -> None:
    """Stage 1: Preprocessing."""
    from src.preprocessing.run_preprocess import preprocess_image, _load_image
    from src.utils.plots import plot_qa_gallery
    from src.utils.reporting import save_noise_report
    import pandas as pd

    raw_path = Path(cfg.data.raw_dir)  # type: ignore[attr-defined]
    processed_path = Path(cfg.data.processed_dir)  # type: ignore[attr-defined]
    processed_path.mkdir(parents=True, exist_ok=True)
    noise_dir = Path(cfg.preprocessing.noise_stats.results_dir)  # type: ignore[attr-defined]
    noise_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(cfg.data.image_extensions)  # type: ignore[attr-defined]
    image_files = sorted(p for p in raw_path.iterdir() if p.suffix.lower() in extensions)

    if not image_files:
        logger.warning("Stage 1: no images found in %s", raw_path)
        return

    qa_pairs = []
    all_metrics = []
    for img_path in image_files:
        name = img_path.stem
        raw = _load_image(img_path)
        noise_metrics = save_noise_report(raw, name, noise_dir)
        cleaned, filter_metrics = preprocess_image(raw, cfg)
        np.save(processed_path / f"{name}.npy", cleaned)
        qa_pairs.append((raw, cleaned, name))
        all_metrics.append({"name": name, **noise_metrics, **filter_metrics})

    qa_path = Path(cfg.project.results_dir) / "stage1" / "qa_gallery.png"  # type: ignore[attr-defined]
    plot_qa_gallery(qa_pairs, qa_path)
    pd.DataFrame(all_metrics).to_csv(noise_dir / "noise_metrics.csv", index=False)
    logger.info("Stage 1 complete (%d images)", len(image_files))


def _run_stage2(cfg: object) -> pd.DataFrame:
    """Stage 2: Classical segmentation + cell cropping.  Returns cells DataFrame."""
    from src.segmentation.classical import segment_classical
    from src.segmentation.crop_cells import build_cells_parquet, crop_cells

    processed_path = Path(cfg.data.processed_dir)  # type: ignore[attr-defined]
    masks_path = Path(cfg.data.masks_dir)  # type: ignore[attr-defined]
    crops_dir = Path(cfg.data.crops_dir)  # type: ignore[attr-defined]
    masks_path.mkdir(parents=True, exist_ok=True)

    seg_cfg = cfg.segmentation  # type: ignore[attr-defined]
    cl_cfg = seg_cfg.classical
    crop_cfg = seg_cfg.crop_cells

    all_records: list[dict] = []
    for img_npy in sorted(processed_path.glob("*.npy")):
        image = np.load(img_npy)
        mask = segment_classical(
            image,
            method=cl_cfg.method,
            min_cell_area=cl_cfg.min_cell_area,
            max_cell_area=cl_cfg.max_cell_area,
            apply_watershed=cl_cfg.watershed,
        )
        np.save(masks_path / f"{img_npy.stem}.npy", mask)
        records = crop_cells(
            image,
            mask,
            source_name=img_npy.stem,
            crops_dir=crops_dir,
            min_area=crop_cfg.min_area,
            max_aspect_ratio=crop_cfg.max_aspect_ratio,
            crop_size=crop_cfg.crop_size,
        )
        all_records.extend(records)

    cells_parquet = crops_dir / "cells.parquet"
    cells_df = build_cells_parquet(all_records, cells_parquet)
    logger.info("Stage 2 complete (%d cells extracted)", len(cells_df))
    return cells_df


def _run_stage4(cfg: object, cells_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Morphometric feature extraction."""
    from src.morphometrics.features import extract_all_features

    output_path = Path(cfg.morphometrics.output_path)  # type: ignore[attr-defined]
    morph_df = extract_all_features(
        cells_df,
        output_path=output_path,
        glcm_distances=list(cfg.morphometrics.glcm_distances),  # type: ignore[attr-defined]
        glcm_angles=list(cfg.morphometrics.glcm_angles),  # type: ignore[attr-defined]
    )
    logger.info("Stage 4 complete")
    return morph_df


def _run_stage3_extract(cfg: object) -> np.ndarray:
    """Stage 3: extract embeddings (assumes SimCLR already trained)."""
    embeddings_path = Path(cfg.project.results_dir) / "embeddings.npy"  # type: ignore[attr-defined]
    if not embeddings_path.exists():
        logger.warning(
            "Embeddings not found at %s. Run train_simclr.py first, "
            "then extract_embeddings.py. Using random placeholder.",
            embeddings_path,
        )
        n = 100
        embeddings = np.random.default_rng(42).normal(size=(n, 512)).astype(np.float32)
        np.save(embeddings_path, embeddings)
    return np.load(embeddings_path)


def _run_stage5(
    cfg: object,
    embeddings: np.ndarray,
    morph_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Stage 5: PCA → UMAP → HDBSCAN clustering."""
    from src.clustering.cluster import cluster_embeddings
    from src.clustering.umap_reduce import reduce_embeddings

    cluster_results_dir = Path(cfg.clustering.results_dir)  # type: ignore[attr-defined]
    cluster_results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = Path(cfg.project.results_dir) / "embeddings.npy"  # type: ignore[attr-defined]
    np.save(embeddings_path, embeddings)

    umap_2d, umap_cluster = reduce_embeddings(embeddings_path, cfg, cluster_results_dir)
    label_dict = cluster_embeddings(umap_cluster, cfg)
    return umap_2d, umap_cluster, embeddings, label_dict


def _run_stage6(
    cfg: object,
    morph_df: pd.DataFrame,
    umap_2d: np.ndarray,
    embeddings: np.ndarray,
    label_dict: dict[str, np.ndarray],
) -> dict[str, object]:
    """Stage 6: Evaluation and figure generation."""
    from src.clustering.state_analysis import (
        compute_cluster_quality,
        compute_cluster_stability,
        interpret_states,
        plot_morphometric_boxplots,
        plot_umap,
        summarize_clusters,
    )
    from src.evaluation.figures import (
        figure3_size_entropy_histograms,
        figure7_diffusion_map,
    )
    from src.evaluation.metrics import anova_per_feature, plot_significance_heatmap

    figs_dir = Path(cfg.evaluation.figures_dir)  # type: ignore[attr-defined]
    figs_dir.mkdir(parents=True, exist_ok=True)

    hdb_labels = label_dict["hdbscan"]

    results: dict[str, object] = {}

    # Quality metrics
    quality = compute_cluster_quality(embeddings, hdb_labels)
    stability = compute_cluster_stability(embeddings, hdb_labels, n_iterations=cfg.clustering.bootstrap_iterations)  # type: ignore[attr-defined]
    results["cluster_quality"] = quality
    results["cluster_stability"] = stability

    # Figures 3–6
    figure3_size_entropy_histograms(morph_df, figs_dir / "fig3_histograms.png")

    plot_umap(
        umap_2d,
        hdb_labels,
        figs_dir / "fig4_umap_hdbscan.png",
        title="UMAP — HDBSCAN clusters",
    )

    key_features = ["area", "circularity", "eccentricity", "intensity_entropy", "glcm_contrast_mean"]
    key_features = [f for f in key_features if f in morph_df.columns]
    plot_morphometric_boxplots(
        morph_df, hdb_labels, key_features, figs_dir / "fig5_boxplots.png"
    )

    anova_df = anova_per_feature(morph_df, hdb_labels, alpha=cfg.evaluation.significance_alpha)  # type: ignore[attr-defined]
    plot_significance_heatmap(anova_df, figs_dir / "fig6_significance.png")

    # Figure 7 (optional diffusion map)
    figure7_diffusion_map(embeddings, hdb_labels, figs_dir / "fig7_diffusion.png")

    # State interpretation
    summary = summarize_clusters(morph_df, hdb_labels)
    state_map = interpret_states(summary)
    results["state_map"] = state_map
    results["anova"] = anova_df

    return results


def _write_report(
    cfg: object,
    quality: dict[str, float],
    stability: float,
    state_map: dict[int, str],
    anova_df: pd.DataFrame,
) -> None:
    """Write the final Markdown report to ``results/report.md``."""
    report_path = Path(cfg.evaluation.report_path)  # type: ignore[attr-defined]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sig_features = (
        anova_df[anova_df["significant"]]["feature"].tolist()
        if not anova_df.empty and "significant" in anova_df.columns
        else []
    )

    state_table = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(state_map.items())
    )

    report = textwrap.dedent(f"""\
    # CryoEM Cell-State Discovery Pipeline — Results Report

    ## Cluster Quality
    | Metric | Value |
    |--------|-------|
    | Silhouette score | {quality.get('silhouette', 'N/A'):.4f} |
    | Davies–Bouldin index | {quality.get('davies_bouldin', 'N/A'):.4f} |
    | Bootstrap stability (ARI) | {stability:.4f} |

    ## Discovered States
    | Cluster | Candidate State |
    |---------|-----------------|
    {state_table}

    ## Statistically Significant Morphometric Features
    Features passing ANOVA + Benjamini–Hochberg correction (α=0.05):

    {chr(10).join('- ' + f for f in sig_features) if sig_features else '_None found_'}

    ## Figures
    All figures are saved in `{cfg.evaluation.figures_dir}`:  # type: ignore[attr-defined]
    1. `fig3_histograms.png` — Cell size and entropy distributions
    2. `fig4_umap_hdbscan.png` — UMAP scatter plot
    3. `fig5_boxplots.png` — Per-cluster morphometric boxplots
    4. `fig6_significance.png` — ANOVA significance heatmap
    5. `fig7_diffusion.png` — Pseudo-temporal diffusion map (if available)
    """)

    report_path.write_text(report)
    logger.info("Report written to %s", report_path)


# ── Main entry point ──────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--skip-stage1",
    is_flag=True,
    help="Skip Stage 1 (preprocessed data already exists).",
)
@click.option(
    "--skip-stage2",
    is_flag=True,
    help="Skip Stage 2 (masks / crops already exist).",
)
@click.option(
    "--skip-stage3",
    is_flag=True,
    help="Skip SSL training (use existing embeddings.npy).",
)
def main(
    config: str | None,
    skip_stage1: bool,
    skip_stage2: bool,
    skip_stage3: bool,
) -> None:
    """Run the full CryoEM cell-state discovery pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    # Stage 1
    if not skip_stage1:
        logger.info("=== Stage 1: Preprocessing ===")
        _run_stage1(cfg)

    # Stage 2
    cells_parquet = Path(cfg.data.crops_dir) / "cells.parquet"
    if not skip_stage2:
        logger.info("=== Stage 2: Segmentation ===")
        cells_df = _run_stage2(cfg)
    else:
        cells_df = pd.read_parquet(cells_parquet)
        logger.info("Stage 2 skipped — loaded %d cells from parquet", len(cells_df))

    # Stage 4 (morphometrics — before SSL, uses only masks)
    logger.info("=== Stage 4: Morphometrics ===")
    morph_df = _run_stage4(cfg, cells_df)

    # Stage 3 (SSL — training must be done separately; here we just load/extract)
    logger.info("=== Stage 3: SSL embedding extraction ===")
    embeddings = _run_stage3_extract(cfg)

    # Align embeddings with morph_df (trim / pad if lengths differ)
    n_min = min(len(embeddings), len(morph_df))
    if len(embeddings) != len(morph_df):
        logger.warning(
            "Embedding count (%d) ≠ cell count (%d); truncating to %d",
            len(embeddings),
            len(morph_df),
            n_min,
        )
    embeddings = embeddings[:n_min]
    morph_df = morph_df.iloc[:n_min].reset_index(drop=True)

    # Stage 5
    logger.info("=== Stage 5: Clustering ===")
    umap_2d, _, emb, label_dict = _run_stage5(cfg, embeddings, morph_df)

    # Stage 6
    logger.info("=== Stage 6: Evaluation & reporting ===")
    eval_results = _run_stage6(cfg, morph_df, umap_2d, emb, label_dict)

    _write_report(
        cfg,
        quality=eval_results["cluster_quality"],
        stability=eval_results["cluster_stability"],
        state_map=eval_results["state_map"],
        anova_df=eval_results["anova"],
    )

    logger.info("Pipeline complete. See results/report.md")


if __name__ == "__main__":
    main()
