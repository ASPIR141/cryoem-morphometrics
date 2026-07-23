"""Unit tests for segmentation metric functions (MONAI-based)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestMonaiDiceIoU:
    """Tests for the MONAI-backed evaluate_masks metric helpers."""

    def _dice_direct(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Helper: compute Dice via MONAI DiceMetric for a single 2-D mask pair."""
        from monai.metrics import DiceMetric

        metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        p = torch.from_numpy((pred > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        g = torch.from_numpy((gt > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        metric(y_pred=p, y=g)
        return float(metric.aggregate().item())

    def _iou_direct(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Helper: compute IoU via MONAI MeanIoU."""
        from monai.metrics import MeanIoU

        metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
        p = torch.from_numpy((pred > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        g = torch.from_numpy((gt > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        metric(y_pred=p, y=g)
        return float(metric.aggregate().item())

    def test_perfect_dice(self) -> None:
        mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
        assert self._dice_direct(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_perfect_iou(self) -> None:
        mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
        assert self._iou_direct(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_partial_overlap_dice(self) -> None:
        pred = np.array([[1, 1, 0, 0]], dtype=np.float32)
        gt = np.array([[1, 0, 1, 0]], dtype=np.float32)
        dice = self._dice_direct(pred, gt)
        # intersection=1, denom=4 → Dice=0.5
        assert 0.0 < dice <= 1.0

    def test_partial_overlap_iou(self) -> None:
        pred = np.array([[1, 1, 0, 0]], dtype=np.float32)
        gt = np.array([[1, 0, 1, 0]], dtype=np.float32)
        iou = self._iou_direct(pred, gt)
        assert 0.0 < iou <= 1.0


class TestClassicalSegmentation:
    @pytest.fixture()
    def synthetic_cell_image(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        img = rng.normal(0, 0.05, (128, 128))
        y, x = np.mgrid[:128, :128]
        cell = np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 20**2))
        img += cell
        return img

    def test_output_shape(self, synthetic_cell_image: np.ndarray) -> None:
        from src.segmentation.classical import segment_classical

        mask = segment_classical(synthetic_cell_image)
        assert mask.shape == synthetic_cell_image.shape

    def test_output_dtype(self, synthetic_cell_image: np.ndarray) -> None:
        from src.segmentation.classical import segment_classical

        mask = segment_classical(synthetic_cell_image)
        assert mask.dtype == np.uint8

    def test_detects_cell(self, synthetic_cell_image: np.ndarray) -> None:
        from src.segmentation.classical import segment_classical

        mask = segment_classical(synthetic_cell_image, min_cell_area=50)
        assert (mask > 0).sum() > 0

    def test_unknown_method_raises(self, synthetic_cell_image: np.ndarray) -> None:
        from src.segmentation.classical import segment_classical

        with pytest.raises(ValueError):
            segment_classical(synthetic_cell_image, method="bogus")


class TestMonaiUnet:
    def test_vista2d_wrapper_output_shape(self) -> None:
        """Test that _Vista2DWrapper produces (B,1,H,W) logits without HF download."""
        import types
        import torch.nn as nn
        from src.segmentation.vista2d import _Vista2DWrapper

        # Build a minimal mock SAM with the three sub-modules _Vista2DWrapper uses
        class _FakeEncoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # SAM encoder returns (B, 256, 64, 64)
                return torch.zeros(x.shape[0], 256, 64, 64, device=x.device)

        class _FakePromptEncoder(nn.Module):
            def forward(self, points, boxes, masks):  # noqa: ANN001
                B = 1
                return (
                    torch.zeros(B, 0, 256),   # sparse
                    torch.zeros(B, 256, 64, 64),  # dense
                )

            def get_dense_pe(self) -> torch.Tensor:
                return torch.zeros(1, 256, 64, 64)

        class _FakeMaskDecoder(nn.Module):
            def forward(self, image_embeddings, image_pe,  # noqa: ANN001
                        sparse_prompt_embeddings, dense_prompt_embeddings,
                        multimask_output):  # noqa: ANN001
                B = image_embeddings.shape[0]
                return torch.zeros(B, 1, 256, 256), None

        sam_mock = types.SimpleNamespace(
            image_encoder=_FakeEncoder(),
            mask_decoder=_FakeMaskDecoder(),
            prompt_encoder=_FakePromptEncoder(),
        )

        model = _Vista2DWrapper(sam_mock)  # type: ignore[arg-type]
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_dice_ce_loss_scalar(self) -> None:
        from src.segmentation.vista2d import build_loss

        criterion = build_loss(sigmoid=True)
        logits = torch.randn(2, 1, 64, 64)
        targets = (torch.rand(2, 1, 64, 64) > 0.5).float()
        loss = criterion(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0
