"""Unit tests for Stage 1 preprocessing transforms."""

from __future__ import annotations

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_image() -> np.ndarray:
    """512×512 synthetic image: Gaussian blob on a gradient background + Poisson noise."""
    rng = np.random.default_rng(42)
    h, w = 512, 512
    # Background gradient
    y, x = np.mgrid[:h, :w]
    bg = (x / w + y / h) * 0.3
    # Signal blob
    cy, cx = h // 2, w // 2
    signal = 0.6 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 80**2))
    img = bg + signal + rng.normal(0, 0.05, (h, w))
    return img.astype(np.float64)


@pytest.fixture()
def small_image() -> np.ndarray:
    """64×64 image of ones for shape-contract tests."""
    return np.ones((64, 64), dtype=np.float64)


# ── noise_stats ───────────────────────────────────────────────────────────────


class TestNoiseStats:
    def test_compute_psd_shape(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.noise_stats import RadialPSD

        freqs, psd = RadialPSD()(synthetic_image)
        assert freqs.shape == psd.shape
        assert len(freqs) > 0
        assert freqs[0] == pytest.approx(0.0)
        assert freqs[-1] == pytest.approx(0.5)

    def test_compute_psd_rejects_3d(self) -> None:
        from src.preprocessing.noise_stats import RadialPSD

        with pytest.raises(ValueError):
            RadialPSD()(np.ones((4, 4, 4)))

    def test_estimate_snr_positive(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.noise_stats import EstimateSNR

        snr = EstimateSNR()(synthetic_image)
        assert snr > 0

    def test_fit_noise_model_keys(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.noise_stats import FitNoiseModel

        params = FitNoiseModel()(synthetic_image)
        assert "gaussian_mu" in params
        assert "gaussian_sigma" in params
        assert "poisson_lambda" in params
        assert params["gaussian_sigma"] > 0

    def test_save_noise_report(self, synthetic_image: np.ndarray, tmp_path) -> None:
        from src.utils.reporting import save_noise_report

        metrics = save_noise_report(synthetic_image, "test", tmp_path)
        assert "snr" in metrics
        assert (tmp_path / "test_psd.png").exists()
        assert (tmp_path / "test_snr_hist.png").exists()


# ── fft_filter ────────────────────────────────────────────────────────────────


class TestFftFilter:
    def test_output_shape_preserved(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.fft_filter import FFTFilter

        out = FFTFilter()(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_dtype_preserved(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.fft_filter import FFTFilter

        out = FFTFilter()(synthetic_image.astype(np.float32))
        assert out.dtype == np.float32

    def test_rejects_3d(self) -> None:
        from src.preprocessing.fft_filter import FFTFilter

        with pytest.raises(ValueError):
            FFTFilter()(np.ones((4, 4, 4)))

    def test_high_pass_only_mode(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.fft_filter import FFTFilter

        out = FFTFilter(high_pass_only=True)(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_radial_bandpass_mask_range(self) -> None:
        from src.preprocessing.fft_filter import RadialBandpassMask

        mask = RadialBandpassMask(0.05, 0.4)((64, 64))
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0
        assert tuple(mask.shape) == (64, 64)

    def test_radial_highpass_mask_range(self) -> None:
        from src.preprocessing.fft_filter import RadialHighpassMask

        mask = RadialHighpassMask(0.05)((64, 64))
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0

    def test_compute_filter_metrics_keys(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.fft_filter import FFTFilter
        from src.preprocessing.metrics import compute_filter_metrics

        filtered = FFTFilter()(synthetic_image)
        metrics = compute_filter_metrics(synthetic_image, filtered)
        assert set(metrics.keys()) == {
            "sharpness_before",
            "sharpness_after",
            "entropy_before",
            "entropy_after",
            "entropy_change",
        }


# ── background ────────────────────────────────────────────────────────────────


class TestBackground:
    def test_gaussian_output_shape(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.background import GaussianBackgroundSubtraction

        out = GaussianBackgroundSubtraction(sigma=50)(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_tophat_output_shape(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.background import TopHatBackgroundSubtraction

        out = TopHatBackgroundSubtraction(radius=20)(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_dispatch_gaussian(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.background import BackgroundSubtraction

        out = BackgroundSubtraction(method="gaussian")(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_dispatch_tophat(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.background import BackgroundSubtraction

        out = BackgroundSubtraction(method="tophat")(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_unknown_method_raises(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.background import BackgroundSubtraction

        with pytest.raises(ValueError):
            BackgroundSubtraction(method="unknown")

    def test_gaussian_reduces_gradient(self) -> None:
        """After subtraction the gradient should be smaller than in the original."""
        from src.preprocessing.background import GaussianBackgroundSubtraction

        y, x = np.mgrid[:128, :128]
        gradient_img = (x / 128.0).astype(np.float64)
        corrected = GaussianBackgroundSubtraction(sigma=30)(gradient_img)
        assert corrected.std() < gradient_img.std()


# ── clahe ─────────────────────────────────────────────────────────────────────


class TestClahe:
    def test_apply_clahe_range(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.clahe import CLAHE

        out = CLAHE()(synthetic_image)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_apply_clahe_shape(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.clahe import CLAHE

        out = CLAHE()(synthetic_image)
        assert out.shape == synthetic_image.shape

    def test_zscore_zero_mean(self, synthetic_image: np.ndarray) -> None:
        """MONAI NormalizeIntensity produces ~zero mean (tolerance 1e-5)."""
        from src.preprocessing.clahe import ZScoreNormalize

        out = ZScoreNormalize()(synthetic_image)
        assert abs(out.mean()) < 1e-5

    def test_zscore_unit_std(self, synthetic_image: np.ndarray) -> None:
        """MONAI NormalizeIntensity produces ~unit std (tolerance 1e-5)."""
        from src.preprocessing.clahe import ZScoreNormalize

        out = ZScoreNormalize()(synthetic_image)
        assert abs(out.std() - 1.0) < 1e-5

    def test_zscore_constant_image(self, small_image: np.ndarray) -> None:
        from src.preprocessing.clahe import ZScoreNormalize

        # Should not raise; returns unchanged image
        out = ZScoreNormalize()(small_image)
        assert out.shape == small_image.shape

    def test_clahe_and_normalize_shape(self, synthetic_image: np.ndarray) -> None:
        from src.preprocessing.clahe import CLAHEAndNormalize

        out = CLAHEAndNormalize()(synthetic_image)
        assert out.shape == synthetic_image.shape
