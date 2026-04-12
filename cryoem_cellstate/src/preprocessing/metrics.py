"""Quality metrics for CryoEM image preprocessing."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


def edge_sharpness(image: np.ndarray) -> float:
    """Estimate edge sharpness as the variance of the Laplacian.

    Parameters
    ----------
    image:
        2-D float array.

    Returns
    -------
    sharpness:
        Variance of Laplacian (higher = sharper).
    """
    from scipy.ndimage import laplace

    return float(laplace(image.astype(np.float64)).var())


def image_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Compute Shannon entropy of the intensity histogram.

    Parameters
    ----------
    image:
        2-D float array.
    bins:
        Number of histogram bins.

    Returns
    -------
    entropy:
        Shannon entropy in nats.
    """
    counts, _ = np.histogram(image.flatten(), bins=bins)
    prob = counts / counts.sum()
    prob = prob[prob > 0]
    return float(scipy_entropy(prob))


def compute_filter_metrics(
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, float]:
    """Report quality metrics before and after filtering.

    Parameters
    ----------
    before:
        Original image.
    after:
        Filtered image.

    Returns
    -------
    metrics:
        Dict with keys ``sharpness_before``, ``sharpness_after``,
        ``entropy_before``, ``entropy_after``, ``entropy_change``.
    """
    sb = edge_sharpness(before)
    sa = edge_sharpness(after)
    eb = image_entropy(before)
    ea = image_entropy(after)
    logger.debug("Filter metrics — sharpness: %.3f→%.3f  entropy: %.3f→%.3f", sb, sa, eb, ea)
    return {
        "sharpness_before": sb,
        "sharpness_after": sa,
        "entropy_before": eb,
        "entropy_after": ea,
        "entropy_change": ea - eb,
    }
