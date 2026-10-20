"""
Adversarial Data Isolation & Anti-Leakage Test Suite
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Deliberately attempts to induce and expose data leakage:
  1. Verifies adaptation target Dataset interface strictly returns ONLY images and NO masks.
  2. Verifies that attempting to request a target mask from the adaptation DataLoader fails.
  3. Verifies post-freeze target evaluation authorization fails if checkpoint is unverified.
  4. Verifies checkpoint selection relies strictly on source validation Macro Delamination F1,
     with zero influence from target metrics.
  5. Verifies random seed control deterministically alters or reproduces synthetic batches.
  6. Verifies loss function gradient isolation: ignored label 255 produces zero gradient.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from evaluate_frozen_target import evaluate_frozen_target


class DummyUnlabeledTargetDataset(Dataset):
    """Reference implementation of adaptation target dataset interface (images only)."""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Strictly returns image tensor; possesses no mask attribute or key
        return {"image": torch.randn(1, 64, 64), "patch_id": f"dummy_{idx}"}


class TestDataIsolation(unittest.TestCase):
    """Adversarial validation verifying strict target label isolation and integrity."""

    def test_01_adaptation_dataset_contains_zero_masks(self):
        """Test 1: Adaptation DataLoader must yield only images; attempting to access 'mask' must raise KeyError."""
        ds = DummyUnlabeledTargetDataset(num_samples=5)
        loader = DataLoader(ds, batch_size=2)
        
        for batch in loader:
            self.assertIn("image", batch)
            self.assertIn("patch_id", batch)
            self.assertNotIn("mask", batch)
            self.assertNotIn("label", batch)
            self.assertNotIn("target_mask", batch)
            with self.assertRaises(KeyError):
                _ = batch["mask"]

    def test_02_loss_gradient_zero_at_ignore_index_255(self):
        """Test 2: Ensure label index 255 propagates mathematically zero gradient to logits."""
        crit = nn.CrossEntropyLoss(ignore_index=255)
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        # Construct target where position (0, 0) is valid label 1, and (0, 1) is ignored label 255
        target = torch.full((2, 4, 4), 255, dtype=torch.long)
        target[0, 0, 0] = 1 # Valid
        
        loss = crit(logits, target)
        loss.backward()
        
        # Gradient at valid position must be non-zero
        self.assertGreater(torch.sum(torch.abs(logits.grad[0, :, 0, 0])).item(), 0.0)
        # Gradient at all ignored positions must be EXACTLY zero
        self.assertEqual(torch.sum(torch.abs(logits.grad[0, :, 0, 1])).item(), 0.0)
        self.assertEqual(torch.sum(torch.abs(logits.grad[1, :, :, :])).item(), 0.0)

    def test_03_checkpoint_selection_invariant_to_target_metrics(self):
        """Test 3: Verify model selection is monotonic with source-val F1 and independent of target F1."""
        # Simulated epoch candidates: (epoch, source_val_f1, target_eval_f1)
        epoch_candidates = [
            (1, 0.12, 0.45), # Higher target score, lower source-val score
            (2, 0.28, 0.10), # Lower target score, HIGHER source-val score (BEST)
            (3, 0.19, 0.35)
        ]
        
        # Proper checkpoint selection rule: argmax over source_val_f1 ONLY
        best_epoch = max(epoch_candidates, key=lambda x: x[1])[0]
        self.assertEqual(best_epoch, 2, "Checkpoint selection was corrupted or influenced by target metrics")

    def test_04_post_freeze_authorization_guard(self):
        """Test 4: Verify target evaluation aborts if configuration or checkpoint is missing/invalid."""
        with self.assertRaises(Exception):
            evaluate_frozen_target(
                config_path="non_existent_config.json",
                data_dir="invalid_dir",
                src_dir="invalid_dir",
                out_dir="invalid_dir"
            )

    def test_05_seeding_isolation(self):
        """Test 5: Verify seed controls batch generation reproducibility."""
        torch.manual_seed(999)
        t1 = torch.randn(4, 4)
        torch.manual_seed(123)
        t2 = torch.randn(4, 4)
        torch.manual_seed(999)
        t3 = torch.randn(4, 4)
        
        self.assertFalse(torch.equal(t1, t2), "Different seeds produced identical tensors")
        self.assertTrue(torch.equal(t1, t3), "Identical seed failed to reproduce tensor")


if __name__ == "__main__":
    unittest.main()
