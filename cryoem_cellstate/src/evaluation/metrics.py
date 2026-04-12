"""Statistical evaluation of morphometrics across discovered clusters.

Includes:
- One-way ANOVA per feature
- Kolmogorov–Smirnov tests between cluster pairs
- Benjamini–Hochberg multiple-testing correction

Plotting is delegated to :mod:`src.utils.plots`.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.utils.plots import plot_significance_heatmap  # re-export

logger = logging.getLogger(__name__)


def anova_per_feature(
    morphometrics_df: pd.DataFrame,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One-way ANOVA for each numeric morphometric feature across clusters.

    Parameters
    ----------
    morphometrics_df:
        Per-cell morphometrics DataFrame.
    labels:
        Cluster labels aligned with *morphometrics_df* (noise=-1 excluded).
    alpha:
        Significance threshold (after Benjamini–Hochberg correction).

    Returns
    -------
    results_df:
        DataFrame with columns ``feature``, ``f_stat``, ``p_value``,
        ``p_corrected``, ``significant``.
    """
    df = morphometrics_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster"]

    rows: list[dict[str, object]] = []
    for feat in numeric_cols:
        groups = [
            df[df["cluster"] == c][feat].dropna().values
            for c in sorted(df["cluster"].unique())
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        rows.append({"feature": feat, "f_stat": float(f_stat), "p_value": float(p_val)})

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    _, p_corr, _, _ = multipletests(results["p_value"].values, method="fdr_bh", alpha=alpha)
    results["p_corrected"] = p_corr
    results["significant"] = results["p_corrected"] < alpha
    return results


def ks_pairwise(
    morphometrics_df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Kolmogorov–Smirnov tests between all cluster pairs for key features.

    Parameters
    ----------
    morphometrics_df:
        Per-cell morphometrics DataFrame.
    labels:
        Cluster labels.
    features:
        Features to test.  Defaults to all numeric columns.
    alpha:
        Significance threshold after BH correction.

    Returns
    -------
    results_df:
        Long-format DataFrame with columns
        ``feature``, ``cluster_a``, ``cluster_b``, ``ks_stat``,
        ``p_value``, ``p_corrected``, ``significant``.
    """
    df = morphometrics_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]
    unique_clusters = sorted(df["cluster"].unique())

    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()
        features = [f for f in features if f != "cluster"]

    rows: list[dict[str, object]] = []
    for feat in features:
        for ca, cb in combinations(unique_clusters, 2):
            ga = df[df["cluster"] == ca][feat].dropna().values
            gb = df[df["cluster"] == cb][feat].dropna().values
            if len(ga) < 2 or len(gb) < 2:
                continue
            ks, p = stats.ks_2samp(ga, gb)
            rows.append({
                "feature": feat,
                "cluster_a": int(ca),
                "cluster_b": int(cb),
                "ks_stat": float(ks),
                "p_value": float(p),
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    _, p_corr, _, _ = multipletests(results["p_value"].values, method="fdr_bh", alpha=alpha)
    results["p_corrected"] = p_corr
    results["significant"] = results["p_corrected"] < alpha
    return results


