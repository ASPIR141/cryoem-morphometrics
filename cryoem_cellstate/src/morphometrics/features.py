"""Per-cell morphometric feature extraction for CryoEM cell-state analysis.

Features are grouped into six categories:

1. **Shape** — area, diameter, perimeter, circularity, eccentricity, solidity, extent
2. **Intensity / Electron Density** — mean/std intensity, total electron density, density
   heterogeneity (coefficient of variation)
3. **Volume Shrinkage** — per-cell shrinkage index relative to a reference area; call
   :func:`compute_volume_shrinkage` on the aggregated DataFrame to populate this column
4. **Nucleoid Condensation** — condensed-intensity fraction, intensity CV, number of
   discrete high-density foci; proxies for nucleoid condensation under osmotic stress
5. **Boundary Roughness / LEDS** — Fourier-descriptor-based boundary complexity
   (proxy for Large Extracellular Density Structure formation), convexity-defect
   protrusion count, boundary roughness ratio
6. **GLCM Texture** — contrast, dissimilarity, homogeneity, energy, correlation

:class:`MorphometricExtractor` is a ``monai.transforms.Transform`` that accepts a
``{"image": ..., "mask": ..., "cell_id": ...}`` dict and returns a flat feature dict,
so it composes naturally with any MONAI dict-transform pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from monai.transforms import NormalizeIntensity, Transform
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.measure import find_contours, label, regionprops

logger = logging.getLogger(__name__)

# Shared intensity normaliser — makes density metrics comparable across cells.
_NORMALISE = NormalizeIntensity(nonzero=True)


# ── 1. Shape features ─────────────────────────────────────────────────────────


def _shape_features(prop: object) -> dict[str, float]:
    """Extract standard shape features from a skimage RegionProperties object."""
    area: float = float(prop.area)  # type: ignore[attr-defined]
    perimeter: float = float(prop.perimeter)  # type: ignore[attr-defined]
    convex_perimeter: float = float(prop.perimeter_crofton)  # type: ignore[attr-defined]

    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    boundary_roughness = (perimeter / convex_perimeter) if convex_perimeter > 0 else 1.0

    return {
        "area": area,
        "equivalent_diameter": float(prop.equivalent_diameter_area),  # type: ignore[attr-defined]
        "perimeter": perimeter,
        "circularity": circularity,
        "eccentricity": float(prop.eccentricity),  # type: ignore[attr-defined]
        "solidity": float(prop.solidity),  # type: ignore[attr-defined]
        "extent": float(prop.extent),  # type: ignore[attr-defined]
        "boundary_roughness": boundary_roughness,
    }


# ── 2. Electron density features ─────────────────────────────────────────────


def _density_features(prop: object, norm_crop: np.ndarray) -> dict[str, float]:
    """Compute electron-density proxy features for a CryoEM cell crop.

    CryoEM pixel intensity is proportional to projected electron density.
    Features are computed on both raw (from RegionProperties) and
    MONAI-normalised intensity so they are comparable across images.

    Parameters
    ----------
    prop:
        skimage RegionProperties for the cell.
    norm_crop:
        MONAI-normalised intensity crop (same spatial extent as prop.image_intensity).

    Returns
    -------
    features:
        ``mean_intensity``, ``std_intensity``, ``total_electron_density``,
        ``density_cv`` (coefficient of variation — heterogeneity proxy).
    """
    raw = prop.image_intensity.flatten()  # type: ignore[attr-defined]
    norm = norm_crop.flatten()

    mean_i = float(raw.mean())
    std_i = float(raw.std())
    total_density = float(raw.sum())
    density_cv = float(std_i / max(abs(mean_i), 1e-10))

    return {
        "mean_intensity": mean_i,
        "std_intensity": std_i,
        "total_electron_density": total_density,
        "density_cv": density_cv,
        "norm_mean_intensity": float(norm.mean()),
        "norm_std_intensity": float(norm.std()),
    }


# ── 3. Nucleoid condensation features ────────────────────────────────────────


def _nucleoid_condensation_features(
    prop: object,
    foci_min_distance: int = 5,
    foci_threshold_rel: float = 0.5,
) -> dict[str, float]:
    """Estimate nucleoid condensation state from intensity distribution.

    Under osmotic stress the nucleoid compacts into discrete high-density
    regions (foci).  Three complementary signals are computed:

    * **condensed_fraction** — fraction of pixels exceeding mean + 1 σ;
      higher values indicate denser chromatin packing.
    * **intensity_cv** — coefficient of variation; high CV signals
      heterogeneous, condensed-patch nucleoid distribution.
    * **n_foci** — count of local intensity maxima (discrete foci) detected
      via ``skimage.feature.peak_local_max`` on the masked intensity patch.

    Parameters
    ----------
    prop:
        skimage RegionProperties.
    foci_min_distance:
        Minimum pixel separation between detected foci.
    foci_threshold_rel:
        Relative intensity threshold passed to ``peak_local_max``.

    Returns
    -------
    features:
        ``nucleoid_condensed_fraction``, ``nucleoid_intensity_cv``,
        ``nucleoid_n_foci``.
    """
    intensity = prop.image_intensity  # type: ignore[attr-defined]
    flat = intensity.flatten()
    threshold = flat.mean() + flat.std()
    condensed_fraction = float((flat > threshold).mean())
    intensity_cv = float(flat.std() / max(abs(flat.mean()), 1e-10))

    # Local maxima = discrete foci (nucleoid condensation hubs)
    coords = peak_local_max(
        intensity,
        min_distance=foci_min_distance,
        threshold_rel=foci_threshold_rel,
    )
    n_foci = int(len(coords))

    return {
        "nucleoid_condensed_fraction": condensed_fraction,
        "nucleoid_intensity_cv": intensity_cv,
        "nucleoid_n_foci": n_foci,
    }


# ── 4. Boundary roughness / LEDS formation ───────────────────────────────────


def _leds_boundary_features(binary_mask: np.ndarray) -> dict[str, float]:
    """Quantify boundary irregularity as a proxy for LEDS formation.

    Large Extracellular Density Structures (LEDS) appear in CryoEM as
    membrane protrusions / irregular boundary thickenings.  Two independent
    signals are computed:

    * **Fourier boundary complexity** — the boundary contour is parameterised
      as a complex curve and decomposed by FFT.  The ratio of high-frequency to
      low-frequency Fourier descriptor energy captures fine-scale boundary
      roughness (LEDS-like spikes raise this value).
    * **Protrusion count** — the number of connected defect regions between
      the cell mask and its convex hull.  Each defect corresponds to a concave
      indentation; the complementary convex protrusions drive LEDS scores.
    * **Convex hull defect area fraction** — fractional area of convex-hull
      regions not covered by the cell mask.

    Parameters
    ----------
    binary_mask:
        2-D boolean or uint8 mask for a single cell.

    Returns
    -------
    features:
        ``leds_fft_roughness``, ``leds_protrusion_count``,
        ``leds_hull_defect_fraction``.
    """
    from skimage.morphology import convex_hull_image

    binary = (binary_mask > 0).astype(np.uint8)

    # --- Fourier descriptor boundary complexity ---
    contours = find_contours(binary.astype(float), level=0.5)
    leds_fft_roughness = 0.0
    if contours:
        # Use the longest contour
        contour = max(contours, key=len)
        # Centre the contour
        coords = contour - contour.mean(axis=0)
        # Represent as complex numbers
        complex_contour = coords[:, 0] + 1j * coords[:, 1]
        descriptors = np.fft.fft(complex_contour)
        magnitudes = np.abs(descriptors[1:])   # skip DC
        n = len(magnitudes)
        if n >= 8:
            low = magnitudes[: n // 8].sum()
            high = magnitudes[n // 8 :].sum()
            leds_fft_roughness = float(high / max(low, 1e-10))

    # --- Convex hull defects (protrusions / concavities) ---
    leds_protrusion_count = 0
    leds_hull_defect_fraction = 0.0
    if binary.any():
        try:
            hull = convex_hull_image(binary.astype(bool))
            # Defect map: convex hull minus cell mask
            defects = hull & ~binary.astype(bool)
            defect_area = int(defects.sum())
            hull_area = max(int(hull.sum()), 1)
            leds_hull_defect_fraction = defect_area / hull_area

            # Count connected defect components
            defect_labeled = label(defects)
            leds_protrusion_count = int(defect_labeled.max())
        except Exception:
            pass  # convex_hull_image raises on degenerate masks

    return {
        "leds_fft_roughness": leds_fft_roughness,
        "leds_protrusion_count": leds_protrusion_count,
        "leds_hull_defect_fraction": leds_hull_defect_fraction,
    }


# ── 5. Shannon entropy ────────────────────────────────────────────────────────


def _intensity_entropy(intensity_crop: np.ndarray, bins: int = 64) -> float:
    """Compute Shannon entropy of the intensity distribution within a cell mask."""
    flat = intensity_crop.flatten()
    counts, _ = np.histogram(flat, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    prob = counts[counts > 0] / total
    return float(-np.sum(prob * np.log2(prob)))


# ── 6. GLCM texture features ─────────────────────────────────────────────────


def _glcm_features(
    intensity_crop: np.ndarray,
    distances: list[int],
    angles: list[float],
) -> dict[str, float]:
    """Compute GLCM-based texture features for a single intensity crop."""
    crop = intensity_crop.astype(np.float64)
    c_min, c_max = crop.min(), crop.max()
    crop_u8 = (
        ((crop - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        if c_max > c_min
        else np.zeros_like(crop, dtype=np.uint8)
    )

    glcm = graycomatrix(
        crop_u8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats: dict[str, float] = {}
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, prop_name).flatten()
        feats[f"glcm_{prop_name}_mean"] = float(values.mean())
        feats[f"glcm_{prop_name}_range"] = float(values.max() - values.min())
    return feats


# ── Transform ─────────────────────────────────────────────────────────────────


class MorphometricExtractor(Transform):
    """Extract all CryoEM morphometric features for a single cell crop.

    Accepts a dict with ``"image"``, ``"mask"``, and ``"cell_id"`` keys and
    returns a flat feature dict, making it composable in MONAI dict-transform
    pipelines.

    Parameters
    ----------
    glcm_distances:
        GLCM distances (default: [1, 3, 5]).
    glcm_angles:
        GLCM angles in radians (default: 0°, 45°, 90°, 135°).
    foci_min_distance:
        Minimum pixel distance between detected nucleoid foci.
    foci_threshold_rel:
        Relative intensity threshold for foci detection.
    """

    def __init__(
        self,
        glcm_distances: list[int] | None = None,
        glcm_angles: list[float] | None = None,
        foci_min_distance: int = 5,
        foci_threshold_rel: float = 0.5,
    ) -> None:
        self.glcm_distances = glcm_distances or [1, 3, 5]
        self.glcm_angles = glcm_angles or [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.foci_min_distance = foci_min_distance
        self.foci_threshold_rel = foci_threshold_rel

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Extract features from a single cell dict.

        Parameters
        ----------
        data:
            Dict with ``"image"`` (2-D float32 crop), ``"mask"`` (2-D binary
            crop), and ``"cell_id"`` (str).

        Returns
        -------
        features:
            Flat dict of all morphometric features plus ``"cell_id"``.
        """
        image_crop: np.ndarray = np.asarray(data["image"], dtype=np.float32)
        mask_crop: np.ndarray = np.asarray(data["mask"])
        cell_id: str = str(data["cell_id"])

        binary = (mask_crop > 0).astype(np.uint8)
        labeled = label(binary)
        props = regionprops(labeled, intensity_image=image_crop)

        if not props:
            logger.warning("No region found in mask for cell %s", cell_id)
            return {"cell_id": cell_id}

        prop = max(props, key=lambda p: p.area)

        # MONAI-normalised version of the intensity crop for density features
        norm_crop: np.ndarray = np.asarray(
            _NORMALISE(image_crop[np.newaxis])[0]  # add/remove channel dim
        )

        shape_feats = _shape_features(prop)
        density_feats = _density_features(prop, norm_crop)
        nucleoid_feats = _nucleoid_condensation_features(
            prop,
            foci_min_distance=self.foci_min_distance,
            foci_threshold_rel=self.foci_threshold_rel,
        )
        leds_feats = _leds_boundary_features(binary)
        entropy = _intensity_entropy(prop.image_intensity)  # type: ignore[attr-defined]
        glcm_feats = _glcm_features(
            prop.image_intensity,  # type: ignore[attr-defined]
            self.glcm_distances,
            self.glcm_angles,
        )

        return {
            "cell_id": cell_id,
            "intensity_entropy": entropy,
            **shape_feats,
            **density_feats,
            **nucleoid_feats,
            **leds_feats,
            **glcm_feats,
        }


# ── Population-level volume shrinkage ─────────────────────────────────────────


def compute_volume_shrinkage(
    features_df: pd.DataFrame,
    reference_area: float | None = None,
    area_col: str = "area",
) -> pd.DataFrame:
    """Add a ``volume_shrinkage`` column to *features_df*.

    Volume shrinkage is the fractional decrease in cell area relative to a
    reference (population median by default).  Positive values indicate
    shrinkage (cells smaller than reference); negative values indicate
    swelling.

    .. math::
        \\text{shrinkage} = 1 - \\frac{\\text{area}}{\\text{reference\\_area}}

    Parameters
    ----------
    features_df:
        DataFrame from :func:`extract_all_features`.
    reference_area:
        Reference cell area in pixels.  Defaults to the population median.
    area_col:
        Name of the area column in *features_df*.

    Returns
    -------
    features_df:
        Input DataFrame with an added ``volume_shrinkage`` column (in-place).
    """
    if area_col not in features_df.columns:
        logger.warning("Column '%s' not found; skipping volume shrinkage", area_col)
        return features_df
    ref = reference_area if reference_area is not None else float(features_df[area_col].median())
    features_df["volume_shrinkage"] = 1.0 - features_df[area_col] / max(ref, 1e-10)
    logger.info(
        "Volume shrinkage computed (reference_area=%.1f  median_shrinkage=%.4f)",
        ref,
        float(features_df["volume_shrinkage"].median()),
    )
    return features_df


# ── Convenience functions ─────────────────────────────────────────────────────


def extract_features_for_cell(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    cell_id: str,
    glcm_distances: list[int] | None = None,
    glcm_angles: list[float] | None = None,
) -> dict[str, object]:
    """Extract all morphometric features for a single cell crop.

    Thin wrapper around :class:`MorphometricExtractor`.
    """
    return MorphometricExtractor(
        glcm_distances=glcm_distances,
        glcm_angles=glcm_angles,
    )({"image": image_crop, "mask": mask_crop, "cell_id": cell_id})


def extract_all_features(
    cells_df: pd.DataFrame,
    output_path: str | Path,
    glcm_distances: list[int] | None = None,
    glcm_angles: list[float] | None = None,
    reference_area: float | None = None,
) -> pd.DataFrame:
    """Extract morphometric features for all cells and write to Parquet.

    Parameters
    ----------
    cells_df:
        ``cells.parquet`` DataFrame with ``cell_id``, ``crop_path``,
        ``mask_path`` columns.
    output_path:
        Destination ``morphometrics.parquet`` path.
    glcm_distances:
        GLCM pixel distances.
    glcm_angles:
        GLCM angles in radians.
    reference_area:
        Reference area for :func:`compute_volume_shrinkage`.  Defaults to
        the population median.

    Returns
    -------
    features_df:
        DataFrame with one row per cell and all morphometric columns,
        including ``volume_shrinkage``.
    """
    from tqdm import tqdm

    extractor = MorphometricExtractor(
        glcm_distances=glcm_distances,
        glcm_angles=glcm_angles,
    )

    rows: list[dict[str, object]] = []
    for _, row in tqdm(cells_df.iterrows(), total=len(cells_df), desc="Morphometrics"):
        try:
            feats = extractor({
                "image": np.load(row["crop_path"]),
                "mask": np.load(row["mask_path"]),
                "cell_id": row["cell_id"],
            })
            rows.append(feats)
        except Exception:
            logger.exception("Failed to extract features for cell %s", row["cell_id"])

    df = pd.DataFrame(rows)
    df = compute_volume_shrinkage(df, reference_area=reference_area)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Morphometrics saved to %s  (%d cells)", output_path, len(df))
    return df
