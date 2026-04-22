"""No-op stub: MAE-ViT uses pretrained timm weights — no training needed.

The pretrained backbone (``hf_hub:timm/vit_base_patch16_224.mae_in1k``) is
downloaded and used automatically by ``run_ssl.py --model mae``.

To extract embeddings run::

    python -m src.ssl.run_ssl --model mae
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.info(
    "train_mae_lightning: no training needed — MAEViT loads pretrained "
    "weights from hf_hub:timm/vit_base_patch16_224.mae_in1k automatically. "
    "Run 'python -m src.ssl.run_ssl --model mae' to extract embeddings."
)
