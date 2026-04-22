"""No-op stub: DINO ViT-Base uses pretrained timm weights — no training needed.

The pretrained backbone (``hf_hub:timm/vit_base_patch16_224.dino``) is
downloaded and used automatically by ``run_ssl.py --model rotnet``.

To extract embeddings run::

    python -m src.ssl.run_ssl --model rotnet
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.info(
    "train_rotnet_lightning: no training needed — DinoViT loads pretrained "
    "weights from hf_hub:timm/vit_base_patch16_224.dino automatically. "
    "Run 'python -m src.ssl.run_ssl --model rotnet' to extract embeddings."
)
