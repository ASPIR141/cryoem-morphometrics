"""Shape-contract tests for SSL forward passes (MONAI transforms + ContrastiveLoss)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture()
def batch() -> torch.Tensor:
    """Minimal batch: 4 grayscale 128×128 crops (C × H × W format)."""
    return torch.randn(4, 1, 128, 128)


class TestEncoder:
    def test_forward_shape(self, batch: torch.Tensor) -> None:
        from src.ssl.encoders import Encoder

        enc = Encoder(backbone_name="resnet18", pretrained=False, in_channels=1, embedding_dim=64)
        out = enc(batch)
        assert out.shape == (4, 64)

    def test_get_features_shape(self, batch: torch.Tensor) -> None:
        from src.ssl.encoders import Encoder

        enc = Encoder(backbone_name="resnet18", pretrained=False, in_channels=1, embedding_dim=64)
        feats = enc.get_features(batch)
        assert feats.ndim == 2
        assert feats.shape[0] == 4

    def test_unsupported_backbone_raises(self) -> None:
        from src.ssl.encoders import Encoder

        with pytest.raises(ValueError):
            Encoder(backbone_name="vgg16_not_supported")


class TestSimCLR:
    def test_forward_shape(self, batch: torch.Tensor) -> None:
        from src.ssl.encoders import Encoder
        from src.ssl.simclr import SimCLR

        enc = Encoder(backbone_name="resnet18", pretrained=False, in_channels=1, embedding_dim=64)
        model = SimCLR(enc, projection_dim=32)
        z = model(batch)
        assert z.shape == (4, 32)

    def test_output_l2_normalised(self, batch: torch.Tensor) -> None:
        import torch.nn.functional as F

        from src.ssl.encoders import Encoder
        from src.ssl.simclr import SimCLR

        enc = Encoder(backbone_name="resnet18", pretrained=False, in_channels=1, embedding_dim=64)
        model = SimCLR(enc, projection_dim=32)
        z = model(batch)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNTXentLoss:
    """NTXentLoss wraps MONAI ContrastiveLoss — test the combined interface."""

    def test_loss_is_scalar(self) -> None:
        from src.ssl.simclr import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        loss = loss_fn(z1, z2)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_is_finite(self) -> None:
        from src.ssl.simclr import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.5)
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        assert torch.isfinite(loss_fn(z1, z2))


class TestMonaiSimCLRTransform:
    """Verify MONAI-based two-view augmentation pipeline shapes and types."""

    def test_single_view_shape(self) -> None:
        from src.ssl.simclr import build_simclr_transform

        transform = build_simclr_transform(image_size=128)
        img = np.random.rand(1, 128, 128).astype(np.float32)
        out = transform(img)
        assert out.shape[0] == 1   # channel dim preserved

    def test_two_views_are_different(self) -> None:
        from src.ssl.simclr import build_simclr_transform

        transform = build_simclr_transform(image_size=128)
        img = np.random.rand(1, 128, 128).astype(np.float32)
        v1 = np.array(transform(img))
        v2 = np.array(transform(img))
        # With random augmentations they should differ
        assert not np.allclose(v1, v2)


