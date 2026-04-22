"""Stage 4 pipeline driver: morphometric feature extraction.

Usage::

    python -m src.morphometrics.run_morphometrics --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import seed_everything

from .features import extract_all_features

logger = logging.getLogger(__name__)


def run_morphometrics(cfg: object) -> pd.DataFrame:
    """Extract morphometric features for all cells in ``cells.parquet``.

    Parameters
    ----------
    cfg:
        Root pipeline config.

    Returns
    -------
    morph_df:
        Per-cell morphometrics DataFrame saved to ``morphometrics.parquet``.
    """
    cells_parquet = Path(cfg.data.crops_dir) / "cells.parquet"  # type: ignore[attr-defined]
    if not cells_parquet.exists():
        raise FileNotFoundError(
            f"cells.parquet not found at {cells_parquet}. Run segmentation first."
        )

    cells_df = pd.read_parquet(cells_parquet)
    logger.info("Loaded %d cells from %s", len(cells_df), cells_parquet)

    output_path = Path(cfg.morphometrics.output_path)  # type: ignore[attr-defined]
    morph_df = extract_all_features(
        cells_df,
        output_path=output_path,
        glcm_distances=list(cfg.morphometrics.glcm_distances),  # type: ignore[attr-defined]
        glcm_angles=list(cfg.morphometrics.glcm_angles),  # type: ignore[attr-defined]
    )

    logger.info("Stage 4 complete — %d feature rows written to %s", len(morph_df), output_path)
    return morph_df


@click.command()
@click.option("--config", default=None, help="Path to YAML config override.")
@click.option(
    "--cells-parquet",
    default=None,
    help="Override path to cells.parquet catalogue.",
)
@click.option(
    "--output",
    default=None,
    help="Override output path for morphometrics.parquet.",
)
def main(config: str | None, cells_parquet: str | None, output: str | None) -> None:
    """Run Stage 4: morphometric feature extraction on all cell crops."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    seed_everything(cfg.project.seed)

    if cells_parquet or output:
        import dataclasses

        # Apply CLI overrides by patching the config object attributes in-place.
        if cells_parquet:
            # Write the override parquet path into the expected crops_dir location
            # by loading directly rather than going through cfg.
            import pandas as _pd
            from .features import extract_all_features as _extract

            output_path = Path(output or cfg.morphometrics.output_path)  # type: ignore[attr-defined]
            cells_df = _pd.read_parquet(cells_parquet)
            morph_df = _extract(
                cells_df,
                output_path=output_path,
                glcm_distances=list(cfg.morphometrics.glcm_distances),  # type: ignore[attr-defined]
                glcm_angles=list(cfg.morphometrics.glcm_angles),  # type: ignore[attr-defined]
            )
            logger.info("Stage 4 complete — %d rows → %s", len(morph_df), output_path)
            return

    run_morphometrics(cfg)


if __name__ == "__main__":
    main()
