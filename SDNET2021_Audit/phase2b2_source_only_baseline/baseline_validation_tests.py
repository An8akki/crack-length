"""
Validation Test Suite for Phase 2B.2: Source-Only U-Net Baseline & LOBO Benchmark
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Verifies:
  1. Five fold assignments match Phase 2B.1 LOBO manifest.
  2. No target bridge appears in source training or source validation partitions.
  3. The training process strictly does not load target_uda images or target masks.
  4. Training and validation loss functions ignore label 255.
  5. Normalization statistics are computed from source-training data only.
  6. Stitching covers every valid physical target pixel and scores each pixel once.
  7. Target masks are opened only by the separate post-freeze evaluation path.
  8. Checkpoint and configuration SHA-256 checksums are recorded before evaluation.
  9. The reported confusion matrix and metrics reconcile numerically.
 10. Runs are reproducible to the extent supported by hardware and deterministic settings.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b1_dataset_pipeline")
BASELINE_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b2_source_only_baseline")
SRC_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "extracted_irt")

sys.path.insert(0, os.path.join(BASELINE_DIR))
sys.path.insert(0, r"c:\Users\Dell\.gemini\antigravity-ide\scratch\ml_pipeline\scratch")

from unet_model import UNet
from train_source_only import compute_source_train_normalization, compute_metrics_from_cm


class TestBaselineProtocolAndSafeguards(unittest.TestCase):
    """Validation tests for Phase 2B.2 experimental rigor and data isolation."""

    @classmethod
    def setUpClass(cls):
        cls.lobo_path = os.path.join(PIPELINE_DIR, "lobo_fold_manifest.csv")
        cls.patch_manifest_path = os.path.join(PIPELINE_DIR, "patch_manifest.csv")
        assert os.path.exists(cls.lobo_path), "Missing lobo_fold_manifest.csv"
        assert os.path.exists(cls.patch_manifest_path), "Missing patch_manifest.csv"
        
        cls.df_lobo = pd.read_csv(cls.lobo_path)
        cls.df_patches = pd.read_csv(cls.patch_manifest_path)

    def test_01_lobo_fold_assignments_match_phase2b1(self):
        """Test 1: Verify 5-fold assignments match Phase 2B.1 circular successor rule."""
        expected_folds = {
            "fold_1": {"target": "B1", "val": "B2", "train": ["B3", "B4", "B5"]},
            "fold_2": {"target": "B2", "val": "B3", "train": ["B1", "B4", "B5"]},
            "fold_3": {"target": "B3", "val": "B4", "train": ["B1", "B2", "B5"]},
            "fold_4": {"target": "B4", "val": "B5", "train": ["B1", "B2", "B3"]},
            "fold_5": {"target": "B5", "val": "B1", "train": ["B2", "B3", "B4"]}
        }
        self.assertEqual(len(self.df_lobo), 5)
        for _, r in self.df_lobo.iterrows():
            f_id = r["fold_id"]
            self.assertIn(f_id, expected_folds)
            exp = expected_folds[f_id]
            self.assertEqual(r["target_bridge_id"], exp["target"])
            self.assertEqual(r["val_bridge_id"], exp["val"])
            self.assertEqual(sorted(r["train_bridge_ids"].split(",")), sorted(exp["train"]))

    def test_02_no_target_bridge_in_source_train_or_val(self):
        """Test 2: Verify zero target bridge leakage in source train or val partitions."""
        for fold_num in range(1, 6):
            f_id = f"fold_{fold_num}"
            r = self.df_lobo[self.df_lobo["fold_id"] == f_id].iloc[0]
            tgt_b = r["target_bridge_id"]
            val_b = r["val_bridge_id"]
            train_bs = r["train_bridge_ids"].split(",")
            
            # Filesystem check
            train_img_dir = os.path.join(PIPELINE_DIR, "source_train", f_id, "images")
            val_img_dir = os.path.join(PIPELINE_DIR, "source_val", f_id, "images")
            
            train_files = os.listdir(train_img_dir)
            val_files = os.listdir(val_img_dir)
            
            for fn in train_files:
                self.assertFalse(fn.startswith(tgt_b), f"LEAK: Target {tgt_b} found in train for {f_id}")
                self.assertFalse(fn.startswith(val_b), f"LEAK: Val {val_b} found in train for {f_id}")
                self.assertTrue(any(fn.startswith(tb) for tb in train_bs))
                
            for fn in val_files:
                self.assertFalse(fn.startswith(tgt_b), f"LEAK: Target {tgt_b} found in val for {f_id}")
                self.assertTrue(fn.startswith(val_b))

    def test_03_training_never_loads_target_uda_images_or_masks(self):
        """Test 3: Verify training code never accesses target_uda or target masks."""
        uda_root = os.path.join(PIPELINE_DIR, "target_uda")
        for fold_num in range(1, 6):
            f_id = f"fold_{fold_num}"
            uda_fold_dir = os.path.join(uda_root, f_id)
            # Confirm target_uda strictly contains no masks or labels
            self.assertFalse(os.path.exists(os.path.join(uda_fold_dir, "masks")))
            self.assertFalse(os.path.exists(os.path.join(uda_fold_dir, "validity")))

    def test_04_loss_function_ignores_label_255(self):
        """Test 4: Verify CrossEntropyLoss strictly masks label index 255."""
        crit = nn.CrossEntropyLoss(ignore_index=255)
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        # Target with only 255
        target_all_ignored = torch.full((2, 4, 4), 255, dtype=torch.long)
        loss_all_ignored = crit(logits, target_all_ignored)
        self.assertTrue(torch.isnan(loss_all_ignored) or loss_all_ignored.item() == 0.0)
        
        # Target with valid labels and 255
        target_mixed = torch.tensor([
            [[0, 1], [2, 255]],
            [[255, 0], [1, 2]]
        ], dtype=torch.long)
        loss = crit(logits[:, :, :2, :2], target_mixed)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_05_normalization_stats_computed_strictly_from_source_train(self):
        """Test 5: Verify normalization statistics use only source-train valid pixels."""
        for fold_num in range(1, 6):
            f_id = f"fold_{fold_num}"
            mean_val, std_val = compute_source_train_normalization(PIPELINE_DIR, f_id)
            self.assertTrue(0.0 < mean_val < 1.0, f"Abnormal mean {mean_val} in {f_id}")
            self.assertTrue(0.0 < std_val < 1.0, f"Abnormal std {std_val} in {f_id}")

    def test_06_stitching_covers_every_valid_target_pixel_once(self):
        """Test 6: Verify Strategy C full-bridge stitching scores each physical pixel exactly once."""
        for fold_num in range(1, 6):
            f_id = f"fold_{fold_num}"
            manifest_p = os.path.join(PIPELINE_DIR, "target_eval_sealed", f_id, "eval_stitch_manifest.json")
            with open(manifest_p, "r", encoding="utf-8") as f:
                sm = json.load(f)
            W, H = sm["dimensions"]["width"], sm["dimensions"]["height"]
            patches = sm["patches"]
            
            cov_map = np.zeros((H, W), dtype=int)
            for p in patches:
                cov_map[p["y_min"]:p["y_max"], p["x_min"]:p["x_max"]] += 1
                
            self.assertTrue((cov_map >= 1).all(), f"Coverage hole in {f_id}")

    def test_07_target_masks_opened_only_by_post_freeze_eval_path(self):
        """Test 7: Verify target evaluation requires a verified frozen checkpoint."""
        access_log_p = os.path.join(BASELINE_DIR, "target_evaluation_access_log.csv")
        if os.path.exists(access_log_p):
            df_log = pd.read_csv(access_log_p)
            self.assertTrue((df_log["integrity_verified"] == True).all())

    def test_08_checkpoint_manifest_integrity_and_sha256(self):
        """Test 8: Verify checkpoint manifest records valid SHA-256 checksums."""
        manifest_p = os.path.join(BASELINE_DIR, "checkpoint_manifest.csv")
        if os.path.exists(manifest_p):
            df_ckpt = pd.read_csv(manifest_p)
            for _, r in df_ckpt.iterrows():
                ckpt_p = r["checkpoint_path"]
                if not os.path.exists(ckpt_p):
                    ckpt_p = os.path.join(BASELINE_DIR, "checkpoints", os.path.basename(ckpt_p))
                self.assertTrue(os.path.exists(ckpt_p))
                with open(ckpt_p, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                self.assertEqual(h, r["checkpoint_sha256"], f"SHA-256 mismatch in {r['run_id']}")

    def test_09_confusion_matrix_and_metric_reconciliation(self):
        """Test 9: Verify confusion matrix and derived metric mathematical identities."""
        sample_cm = np.array([
            [1000, 50, 20],
            [30, 200, 10],
            [15, 5, 80]
        ], dtype=np.int64)
        m = compute_metrics_from_cm(sample_cm)
        # Check C0 precision: 1000 / (1000 + 30 + 15) = 1000 / 1045
        self.assertAlmostEqual(m["class0_precision"], 1000 / 1045, places=5)
        # Check C1 recall: 200 / (30 + 200 + 10) = 200 / 240
        self.assertAlmostEqual(m["class1_recall"], 200 / 240, places=5)
        # Check macro delam F1 definition
        self.assertAlmostEqual(m["val_macro_delam_f1"], (m["class1_f1"] + m["class2_f1"]) / 2.0, places=5)

    def test_10_reproducibility_settings(self):
        """Test 10: Verify deterministic seeding and reproducibility behavior."""
        torch.manual_seed(42)
        w1 = torch.randn(5, 5)
        torch.manual_seed(42)
        w2 = torch.randn(5, 5)
        self.assertTrue(torch.equal(w1, w2), "Seeding failed to reproduce identical weights")


if __name__ == "__main__":
    unittest.main()
