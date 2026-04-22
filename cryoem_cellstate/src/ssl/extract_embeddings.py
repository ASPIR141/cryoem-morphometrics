"""Thin re-export kept for backward compatibility.

Embedding extraction has moved to ``run_ssl.py``.  Use that module directly::

    python -m src.ssl.run_ssl --model mae --checkpoint <ckpt>
    python -m src.ssl.run_ssl --model cryo-ief
    python -m src.ssl.run_ssl --model rotnet --checkpoint <ckpt>

:class:`CropInferenceDataset` is re-exported here so any code that imports
it from this module continues to work without changes.
"""

from .run_ssl import CropInferenceDataset  # noqa: F401

__all__ = ["CropInferenceDataset"]
