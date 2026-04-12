"""Deterministic seeding utility for reproducible experiments."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for full reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value to use across all RNG sources.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seeded with %d", seed)
    except ImportError:
        logger.debug("PyTorch not available; skipping torch seeding")

    logger.info("All RNG sources seeded with %d", seed)
