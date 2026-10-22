"""
Unit Tests for Deep CORAL Experiment Runner & Adaptation Pipeline
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Verifies:
  1. UnlabeledNormalizedPatchDataset interface strictly yields only images (no masks).
  2. Dataset patch shapes, data types, and normalization values.
  3. Single-step forward & backward pass through Deep CORAL loss (L_seg + 0.5 * L_CORAL).
  4. Palette colorization utility produces correct RGB output.
  5. Checkpoint weight interchangeability between DeepCORALUNet and UNet.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from run_coral_experiments import (
    UnlabeledNormalizedPatchDataset,
    colorize_prediction_map,
    DATA_DIR
)
from src.domain_adaptation.deep_coral import DeepCORALUNet, coral_loss
from unet_model import UNet


class TestCoralRunner(unittest.TestCase):
    """Test suite for Deep CORAL data loading, loss formulation, and components."""

    def test_01_unlabeled_target_dataset_interface(self):
        """Test 1: Verify UnlabeledNormalizedPatchDataset returns only 'image' and 'patch_id'."""
        if not os.path.exists(DATA_DIR):
            self.skipTest(f"Data directory not found: {DATA_DIR}")

        ds = UnlabeledNormalizedPatchDataset(DATA_DIR, fold_id="fold_1", mean_val=0.45, std_val=0.15)
        self.assertGreater(len(ds), 0)

        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("patch_id", sample)
        self.assertNotIn("mask", sample)
        self.assertNotIn("label", sample)

        # Confirm tensor properties
        self.assertEqual(sample["image"].shape, (1, 256, 256))
        self.assertEqual(sample["image"].dtype, torch.float32)

        # Confirm KeyError on mask access
        with self.assertRaises(KeyError):
            _ = sample["mask"]

    def test_02_adaptation_training_step(self):
        """Test 2: Verify a single end-to-end forward/backward step with Deep CORAL."""
        torch.manual_seed(42)
        model = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Synthetic source and target batches
        src_imgs = torch.randn(4, 1, 64, 64)
        src_masks = torch.randint(0, 3, (4, 64, 64))
        tgt_imgs = torch.randn(4, 1, 64, 64)

        optimizer.zero_grad()
        src_logits, src_feats = model(src_imgs, return_features=True, feature_layer="bottleneck")
        _, tgt_feats = model(tgt_imgs, return_features=True, feature_layer="bottleneck")

        loss_seg = criterion(src_logits, src_masks)
        loss_coral = coral_loss(src_feats, tgt_feats)
        total_loss = loss_seg + 0.5 * loss_coral

        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()
        optimizer.step()

        # Confirm parameter update
        has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
        self.assertTrue(has_grad)

    def test_03_colorization_palette(self):
        """Test 3: Verify colorize_prediction_map maps classes to exact specified RGB values."""
        pred = np.array([
            [0, 1],
            [2, 255]
        ], dtype=np.uint8)

        colorized = colorize_prediction_map(pred)
        self.assertEqual(colorized.shape, (2, 2, 3))

        # Class 0: Sound -> Gray [128, 128, 128]
        np.testing.assert_array_equal(colorized[0, 0], [128, 128, 128])
        # Class 1: Class 2 Delam -> Green [0, 255, 0]
        np.testing.assert_array_equal(colorized[0, 1], [0, 255, 0])
        # Class 2: Class 3 Delam -> Red [255, 0, 0]
        np.testing.assert_array_equal(colorized[1, 0], [255, 0, 0])
        # Excluded: 255 -> White [255, 255, 255]
        np.testing.assert_array_equal(colorized[1, 1], [255, 255, 255])

    def test_04_checkpoint_compatibility(self):
        """Test 4: Verify UNet can load state_dict directly from trained DeepCORALUNet."""
        m_coral = DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])
        m_base = UNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])

        state_coral = m_coral.state_dict()
        m_base.load_state_dict(state_coral)

        x = torch.randn(2, 1, 64, 64)
        m_coral.eval()
        m_base.eval()
        with torch.no_grad():
            out_c = m_coral(x, return_features=False)
            out_b = m_base(x)

        np.testing.assert_allclose(out_c.numpy(), out_b.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
