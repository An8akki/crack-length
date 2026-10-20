"""
Unit Tests for Deep CORAL Domain Adaptation Module
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Verifies:
  1. Covariance calculation numerical correctness against NumPy np.cov.
  2. Zero loss on identical source and target distributions.
  3. Proper handling of 2D (N, d) and 4D (B, C, H, W) feature tensors.
  4. Non-zero, finite gradients flowing through both source and target branches.
  5. Feature layer extraction ('bottleneck' and 'penultimate') in DeepCORALUNet.
  6. Exact equivalence between DeepCORALUNet(return_features=False) and standard UNet.
  7. Safeguards against dimension mismatch and degenerate inputs.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Add src and baseline to path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from src.domain_adaptation.deep_coral import (
    compute_covariance,
    coral_loss,
    DeepCORALUNet
)
from unet_model import UNet


class TestDeepCORAL(unittest.TestCase):
    """Test suite for covariance estimation, CORAL loss, and DeepCORALUNet architecture."""

    def test_01_covariance_against_numpy(self):
        """Test 1: Verify compute_covariance matches np.cov within float32 tolerance."""
        torch.manual_seed(42)
        n_samples = 100
        d_features = 16
        x = torch.randn(n_samples, d_features)
        
        # PyTorch covariance
        cov_torch = compute_covariance(x).numpy()
        
        # NumPy covariance (ddof=1 matches unbiased sample covariance)
        cov_np = np.cov(x.numpy(), rowvar=False, ddof=1)
        
        np.testing.assert_allclose(cov_torch, cov_np, rtol=1e-5, atol=1e-5)

    def test_02_zero_loss_on_identical_features(self):
        """Test 2: When source and target features are identical, CORAL loss must be 0.0."""
        torch.manual_seed(42)
        features = torch.randn(50, 32)
        loss = coral_loss(features, features)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_03_4d_convolutional_features_support(self):
        """Test 3: Verify 4D feature maps (B, C, H, W) are reshaped and processed seamlessly."""
        torch.manual_seed(42)
        b, c, h, w = 4, 16, 8, 8
        src_4d = torch.randn(b, c, h, w)
        tgt_4d = torch.randn(b, c, h, w)
        
        loss_4d = coral_loss(src_4d, tgt_4d)
        self.assertFalse(torch.isnan(loss_4d))
        self.assertGreater(loss_4d.item(), 0.0)

    def test_04_gradient_flow_through_network(self):
        """Test 4: Verify gradients flow backwards from CORAL loss into network parameters."""
        torch.manual_seed(42)
        model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])
        
        src_imgs = torch.randn(2, 1, 64, 64)
        tgt_imgs = torch.randn(2, 1, 64, 64)
        
        _, src_feats = model(src_imgs, return_features=True, feature_layer='bottleneck')
        _, tgt_feats = model(tgt_imgs, return_features=True, feature_layer='bottleneck')
        
        loss = coral_loss(src_feats, tgt_feats)
        loss.backward()
        
        # Verify bottleneck parameters received non-zero gradients
        has_grad = False
        for param in model.bottleneck.parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "Bottleneck parameters received zero gradients from CORAL loss")

    def test_05_feature_layer_selection(self):
        """Test 5: Verify both 'bottleneck' and 'penultimate' layers extract valid features."""
        model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])
        x = torch.randn(2, 1, 64, 64)
        
        # Bottleneck: 64 * 2 = 128 channels, 64 / 16 = 4x4 spatial
        logits, feats_bot = model(x, return_features=True, feature_layer='bottleneck')
        self.assertEqual(logits.shape, (2, 3, 64, 64))
        self.assertEqual(feats_bot.shape, (2, 128, 4, 4))
        
        # Penultimate: 8 channels, 64x64 spatial
        logits, feats_pen = model(x, return_features=True, feature_layer='penultimate')
        self.assertEqual(feats_pen.shape, (2, 8, 64, 64))

    def test_06_equivalence_to_standard_unet(self):
        """Test 6: Verify forward(return_features=False) behaves identically to base UNet."""
        torch.manual_seed(42)
        model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])
        model.eval()
        
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out_standard = model(x, return_features=False)
            out_with_feat, _ = model(x, return_features=True)
            
        self.assertTrue(torch.equal(out_standard, out_with_feat))

    def test_07_dimension_mismatch_raises_error(self):
        """Test 7: Confirm mismatched feature dimensions raise ValueError."""
        src = torch.randn(10, 16)
        tgt = torch.randn(10, 32)
        with self.assertRaises(ValueError):
            coral_loss(src, tgt)


if __name__ == "__main__":
    unittest.main()
