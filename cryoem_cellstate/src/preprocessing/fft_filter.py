"""Frequency-domain filtering for CryoEM images.

All classes inherit from ``monai.transforms.Transform`` so they compose
directly with ``monai.transforms.Compose`` alongside built-in MONAI transforms.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import Transform


class RadialBandpassMask(Transform):
    """Create a 2-D radial band-pass mask in the frequency domain.

    Parameters
    ----------
    low_cutoff:
        Low spatial frequency cut-off as a fraction of the Nyquist frequency (0–1).
        Frequencies *below* this are suppressed.
    high_cutoff:
        High spatial frequency cut-off (0–1). Frequencies *above* this are suppressed.
    """

    def __init__(self, low_cutoff: float, high_cutoff: float) -> None:
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def __call__(self, shape: tuple[int, int]) -> torch.Tensor:
        """Return a float32 mask of the given *shape* with values in [0, 1]."""
        h, w = shape
        cy, cx = h // 2, w // 2
        y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(w, dtype=torch.float32).unsqueeze(0)
        r = torch.hypot((x - cx) / w, (y - cy) / h)
        return ((r >= self.low_cutoff) & (r <= self.high_cutoff)).float()


class RadialHighpassMask(Transform):
    """Create a high-pass radial mask (passes frequencies above *low_cutoff*).

    Parameters
    ----------
    low_cutoff:
        Minimum normalised spatial frequency to retain.
    """

    def __init__(self, low_cutoff: float) -> None:
        super().__init__()
        self.low_cutoff = low_cutoff

    def __call__(self, shape: tuple[int, int]) -> torch.Tensor:
        """Return a float32 mask of the given *shape*."""
        return RadialBandpassMask(self.low_cutoff, high_cutoff=0.5)(shape)


class FFTFilter(Transform):
    """Apply a radial band-pass (or high-pass) filter in the frequency domain.

    Uses ``torch.fft`` for the forward/inverse transforms.

    Inherits ``monai.transforms.Transform`` — can be placed directly inside
    a ``monai.transforms.Compose`` pipeline.

    Parameters
    ----------
    low_cutoff:
        Low frequency cut-off (fraction of Nyquist).
    high_cutoff:
        High frequency cut-off (fraction of Nyquist). Ignored when *high_pass_only*.
    high_pass_only:
        If *True*, only suppress frequencies below *low_cutoff*.
    """

    def __init__(
        self,
        low_cutoff: float = 0.02,
        high_cutoff: float = 0.4,
        high_pass_only: bool = False,
    ) -> None:
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.high_pass_only = high_pass_only

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Filter *image* and return an array with the same shape and dtype."""
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {image.shape}")

        orig_dtype = image.dtype
        img_t = torch.from_numpy(image.astype(np.float64))

        fft = torch.fft.fft2(img_t)
        fft_shift = torch.fft.fftshift(fft)

        if self.high_pass_only:
            mask = RadialHighpassMask(self.low_cutoff)(image.shape)
        else:
            mask = RadialBandpassMask(self.low_cutoff, self.high_cutoff)(image.shape)

        fft_filtered = fft_shift * mask.to(fft_shift.dtype)
        filtered = torch.fft.ifft2(torch.fft.ifftshift(fft_filtered)).real
        return filtered.numpy().astype(orig_dtype)
