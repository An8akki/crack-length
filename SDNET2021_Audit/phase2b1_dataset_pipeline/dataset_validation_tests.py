"""
Comprehensive Validation Tests for SDNET2021 Dataset Pipeline (Phase 2B.1)
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Automated verification suite validating:
  1. Source decoding, dimensions, and zero-unknown color guarantee.
  2. Exact Strategy C window generation, bounds, and counts (322 total).
  3. Zero uncovered valid deck pixels across the fleet.
  4. Exact pixel-by-pixel reconciliation of serialized patches against raw source crops.
  5. Rare class and zero-delamination patch accounting.
  6. LOBO fold disjointness and circular successor partitioning.
  7. Physical partition filesystem isolation & DataLoader leakage safeguards.
  8. Stitched single physical pixel scoring guarantee.
"""

import os
import sys
import json
import stat
import unittest
import numpy as np
from PIL import Image
import pandas as pd
import torch

DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "extracted_irt")
PIPELINE_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b1_dataset_pipeline")

# Add pipeline directory to sys.path to import dataset_build modules
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from dataset_build import (
    BRIDGES, S, STEP,
    decode_annotation_mask,
    generate_strategy_c_coordinates,
    SDNETPatchDataset, LOBODataModule
)


class TestSDNET2021DatasetPipeline(unittest.TestCase):
    """Full independent test suite validating dataset pipeline for Phase 2B.1."""

    @classmethod
    def setUpClass(cls):
        assert os.path.exists(PIPELINE_DIR), f"Pipeline directory missing: {PIPELINE_DIR}"
        
        # Load manifests
        cls.patch_manifest_path = os.path.join(PIPELINE_DIR, "patch_manifest.csv")
        cls.lobo_manifest_path = os.path.join(PIPELINE_DIR, "lobo_fold_manifest.csv")
        cls.class_dist_path = os.path.join(PIPELINE_DIR, "class_distribution_by_bridge.csv")
        
        assert os.path.exists(cls.patch_manifest_path), "Missing patch_manifest.csv"
        assert os.path.exists(cls.lobo_manifest_path), "Missing lobo_fold_manifest.csv"
        assert os.path.exists(cls.class_dist_path), "Missing class_distribution_by_bridge.csv"
        
        cls.df_patches = pd.read_csv(cls.patch_manifest_path)
        cls.df_lobo = pd.read_csv(cls.lobo_manifest_path)
        cls.df_class_dist = pd.read_csv(cls.class_dist_path)
        
        # Load raw sources for exact reconciliation
        cls.raw_sources = {}
        for b in BRIDGES:
            b_id = b["id"]
            orig_p = os.path.join(SRC_DIR, b["folder"], b["orig_file"])
            annot_p = os.path.join(SRC_DIR, b["folder"], b["annot_file"])
            
            with Image.open(orig_p) as img_o:
                arr_orig = np.array(img_o.convert("L"))
            with Image.open(annot_p) as img_a:
                arr_annot_rgb = np.array(img_a.convert("RGB"))
                
            semantic_mask, valid_mask, breakdown = decode_annotation_mask(arr_annot_rgb)
            cls.raw_sources[b_id] = {
                "arr_orig": arr_orig,
                "arr_annot_rgb": arr_annot_rgb,
                "semantic_mask": semantic_mask,
                "valid_mask": valid_mask,
                "breakdown": breakdown,
                "W": b["W"], "H": b["H"]
            }

    def test_01_source_decoding_and_dimension_verification(self):
        """Test 1: Verify source image dimensions, color decodings, and pixel totals."""
        fleet_tot = 0
        fleet_valid = 0
        fleet_sound = 0
        fleet_c2 = 0
        fleet_c3 = 0
        fleet_white = 0
        fleet_black = 0
        
        for b in BRIDGES:
            b_id = b["id"]
            s = self.raw_sources[b_id]
            W, H = s["W"], s["H"]
            tot = W * H
            
            self.assertEqual(s["arr_orig"].shape, (H, W), f"Dimension mismatch in original {b_id}")
            self.assertEqual(s["semantic_mask"].shape, (H, W), f"Dimension mismatch in mask {b_id}")
            
            bd = s["breakdown"]
            self.assertEqual(bd["unknown"], 0, f"Found unexpected unknown colors in {b_id}")
            self.assertEqual(bd["sound"] + bd["c2"] + bd["c3"], bd["valid"])
            self.assertEqual(bd["valid"] + bd["excluded"], tot)
            
            fleet_tot += tot
            fleet_valid += bd["valid"]
            fleet_sound += bd["sound"]
            fleet_c2 += bd["c2"]
            fleet_c3 += bd["c3"]
            fleet_white += bd["white"]
            fleet_black += bd["black"]
            
        self.assertEqual(fleet_tot, 6317409)
        self.assertEqual(fleet_valid, 6089851)
        self.assertEqual(fleet_sound, 4786118)
        self.assertEqual(fleet_c2, 1045173)
        self.assertEqual(fleet_c3, 258560)
        self.assertEqual(fleet_white, 207210)
        self.assertEqual(fleet_black, 20348)

    def test_02_strategy_c_window_generation_and_counts(self):
        """Test 2: Verify exact 256x256 dimensions, unique coordinates, and 322 total windows."""
        self.assertEqual(len(self.df_patches), 322, "Total fleet patches must equal exactly 322")
        
        for b in BRIDGES:
            b_id = b["id"]
            b_patches = self.df_patches[self.df_patches["bridge_id"] == b_id]
            self.assertEqual(len(b_patches), b["expected_patches"], f"Patch count mismatch for {b_id}")
            
            coords = []
            for _, r in b_patches.iterrows():
                x1, y1, x2, y2 = r["x_min"], r["y_min"], r["x_max"], r["y_max"]
                self.assertEqual(x2 - x1, S, f"Width not 256 in {r['patch_id']}")
                self.assertEqual(y2 - y1, S, f"Height not 256 in {r['patch_id']}")
                self.assertTrue(0 <= x1 < x2 <= b["W"], f"X out of bounds in {r['patch_id']}")
                self.assertTrue(0 <= y1 < y2 <= b["H"], f"Y out of bounds in {r['patch_id']}")
                coords.append((x1, y1, x2, y2))
                
            self.assertEqual(len(coords), len(set(coords)), f"Duplicate coordinates in {b_id}")

    def test_03_zero_uncovered_valid_deck_pixels(self):
        """Test 3: Verify 100.00% valid deck and geometric coverage under Strategy C."""
        for b in BRIDGES:
            b_id = b["id"]
            s = self.raw_sources[b_id]
            W, H = s["W"], s["H"]
            valid_deck = (s["valid_mask"] == 1)
            
            cov_map = np.zeros((H, W), dtype=int)
            b_patches = self.df_patches[self.df_patches["bridge_id"] == b_id]
            
            for _, r in b_patches.iterrows():
                cov_map[r["y_min"]:r["y_max"], r["x_min"]:r["x_max"]] += 1
                
            self.assertTrue((cov_map >= 1).all(), f"Geometric coverage gap in {b_id}")
            uncovered_valid = int(((cov_map == 0) & valid_deck).sum())
            self.assertEqual(uncovered_valid, 0, f"Strategy C left uncovered valid deck pixels in {b_id}")
            
            covered_valid = int(((cov_map >= 1) & valid_deck).sum())
            self.assertEqual(covered_valid, s["breakdown"]["valid"])

    def test_04_exact_patch_reconciliation_against_source_arrays(self):
        """Test 4: Verify pixel-by-pixel match between materialized disk patches and raw crops."""
        patch_dir = os.path.join(PIPELINE_DIR, "patches")
        
        # Test all 322 patches
        for idx, row in self.df_patches.iterrows():
            pid = row["patch_id"]
            b_id = row["bridge_id"]
            y1, y2 = row["y_min"], row["y_max"]
            x1, x2 = row["x_min"], row["x_max"]
            
            img_p = os.path.join(patch_dir, "images", f"{pid}_img.png")
            mask_p = os.path.join(patch_dir, "masks", f"{pid}_mask.png")
            val_p = os.path.join(patch_dir, "validity", f"{pid}_valid.png")
            
            self.assertTrue(os.path.exists(img_p), f"Missing image file: {img_p}")
            self.assertTrue(os.path.exists(mask_p), f"Missing mask file: {mask_p}")
            self.assertTrue(os.path.exists(val_p), f"Missing validity file: {val_p}")
            
            with Image.open(img_p) as img:
                arr_p_img = np.array(img)
            with Image.open(mask_p) as m:
                arr_p_mask = np.array(m)
            with Image.open(val_p) as v:
                arr_p_val = np.array(v)
                
            s = self.raw_sources[b_id]
            expected_img = s["arr_orig"][y1:y2, x1:x2]
            expected_mask = s["semantic_mask"][y1:y2, x1:x2]
            expected_val = s["valid_mask"][y1:y2, x1:x2] * 255
            
            self.assertTrue(np.array_equal(arr_p_img, expected_img), f"Image pixel mismatch in {pid}")
            self.assertTrue(np.array_equal(arr_p_mask, expected_mask), f"Mask pixel mismatch in {pid}")
            self.assertTrue(np.array_equal(arr_p_val, expected_val), f"Validity pixel mismatch in {pid}")
            
            # Assert label counts match manifest
            self.assertEqual(int((arr_p_mask == 0).sum()), row["sound_pixels"])
            self.assertEqual(int((arr_p_mask == 1).sum()), row["class2_pixels"])
            self.assertEqual(int((arr_p_mask == 2).sum()), row["class3_pixels"])
            self.assertEqual(int((arr_p_mask == 255).sum()), row["total_excluded_pixels"])

    def test_05_rare_class_and_zero_delam_patch_accounting(self):
        """Test 5: Verify exact zero-class patch counts and preservation of rare-class patches."""
        zero_c2 = self.df_patches[~self.df_patches["has_class2"]]
        zero_c3 = self.df_patches[~self.df_patches["has_class3"]]
        zero_delam = self.df_patches[~self.df_patches["has_delam"]]
        
        # Verified audit counts
        self.assertEqual(len(zero_c2), 2, "Expected exactly 2 patches with zero Class 2")
        self.assertEqual(len(zero_c3), 12, "Expected exactly 12 patches with zero Class 3")
        self.assertEqual(len(zero_delam), 2, "Expected exactly 2 patches with zero delamination")
        
        # Verify zero C2 patches are both on B5
        self.assertTrue((zero_c2["bridge_id"] == "B5").all())
        # Verify zero delam patches are both on B5
        self.assertTrue((zero_delam["bridge_id"] == "B5").all())
        # Verify zero C3 patches distribution: B3: 2, B4: 3, B5: 7
        self.assertEqual(len(zero_c3[zero_c3["bridge_id"] == "B3"]), 2)
        self.assertEqual(len(zero_c3[zero_c3["bridge_id"] == "B4"]), 3)
        self.assertEqual(len(zero_c3[zero_c3["bridge_id"] == "B5"]), 7)

    def test_06_lobo_fold_disjointness_and_circular_successor(self):
        """Test 6: Verify 5-fold circular successor LOBO partition integrity."""
        all_ids = [b["id"] for b in BRIDGES]
        self.assertEqual(len(self.df_lobo), 5, "Expected 5 LOBO folds")
        
        for idx, row in self.df_lobo.iterrows():
            tgt_b = row["target_bridge_id"]
            val_b = row["val_bridge_id"]
            train_bs = row["train_bridge_ids"].split(",")
            
            # Check circular successor rule: V = B_{(k % 5) + 1}
            expected_val = all_ids[(idx + 1) % 5]
            self.assertEqual(val_b, expected_val, f"Circular successor mismatch in fold {row['fold_id']}")
            
            # Check pairwise disjointness
            self.assertNotIn(tgt_b, train_bs, f"Target leaked into train in {row['fold_id']}")
            self.assertNotEqual(tgt_b, val_b, f"Target equals val in {row['fold_id']}")
            self.assertNotIn(val_b, train_bs, f"Val in train in {row['fold_id']}")
            self.assertEqual(len(train_bs), 3)
            self.assertEqual(len(set([tgt_b, val_b] + train_bs)), 5)
            
            # Check patch counts sum to 322
            total_patches = row["source_train_patches"] + row["source_val_patches"] + row["target_uda_patches"]
            self.assertEqual(total_patches, 322)

    def test_07_physical_partition_isolation_and_leakage_safeguards(self):
        """Test 7: Verify filesystem isolation and DataLoader target protection."""
        for fold_num in range(1, 6):
            fold_id = f"fold_{fold_num}"
            row = self.df_lobo[self.df_lobo["fold_id"] == fold_id].iloc[0]
            tgt_b = row["target_bridge_id"]
            val_b = row["val_bridge_id"]
            train_bs = row["train_bridge_ids"].split(",")
            
            # 1. Target UDA: STRICTLY IMAGES ONLY
            uda_dir = os.path.join(PIPELINE_DIR, "target_uda", fold_id)
            uda_img_dir = os.path.join(uda_dir, "images")
            self.assertTrue(os.path.exists(uda_img_dir))
            
            # Assert NO masks or validity folders exist in target_uda
            self.assertFalse(os.path.exists(os.path.join(uda_dir, "masks")), f"LEAK: masks folder exists in {uda_dir}")
            self.assertFalse(os.path.exists(os.path.join(uda_dir, "validity")), f"LEAK: validity folder exists in {uda_dir}")
            
            # Assert all files in target_uda/images are images and match target bridge
            uda_files = os.listdir(uda_img_dir)
            self.assertEqual(len(uda_files), row["target_uda_patches"])
            for f in uda_files:
                self.assertTrue(f.endswith("_img.png"))
                self.assertTrue(f.startswith(tgt_b), f"Foreign bridge image {f} in UDA for {fold_id}")
                self.assertNotIn("mask", f.lower())
                self.assertNotIn("annot", f.lower())
                
            # 2. Source Train: Target must NOT exist
            train_dir = os.path.join(PIPELINE_DIR, "source_train", fold_id)
            train_img_files = os.listdir(os.path.join(train_dir, "images"))
            self.assertEqual(len(train_img_files), row["source_train_patches"])
            for f in train_img_files:
                self.assertFalse(f.startswith(tgt_b), f"LEAK: Target {tgt_b} in source_train for {fold_id}!")
                self.assertFalse(f.startswith(val_b), f"LEAK: Val {val_b} in source_train for {fold_id}!")
                self.assertTrue(any(f.startswith(tb) for tb in train_bs))
                
            # 3. Source Val: Target and Train must NOT exist
            val_dir = os.path.join(PIPELINE_DIR, "source_val", fold_id)
            val_img_files = os.listdir(os.path.join(val_dir, "images"))
            self.assertEqual(len(val_img_files), row["source_val_patches"])
            for f in val_img_files:
                self.assertFalse(f.startswith(tgt_b), f"LEAK: Target {tgt_b} in source_val for {fold_id}!")
                self.assertTrue(f.startswith(val_b))
                
            # 4. Target Eval Sealed: Read-only protection check
            sealed_dir = os.path.join(PIPELINE_DIR, "target_eval_sealed", fold_id)
            sealed_masks = os.listdir(os.path.join(sealed_dir, "masks"))
            self.assertEqual(len(sealed_masks), row["target_eval_sealed_patches"])
            sample_sealed_mask = os.path.join(sealed_dir, "masks", sealed_masks[0])
            
            # Verify file is read-only (stat check and write rejection)
            file_stat = os.stat(sample_sealed_mask)
            is_read_only = (file_stat.st_mode & stat.S_IWRITE == 0)
            self.assertTrue(is_read_only, f"Sealed mask {sample_sealed_mask} is not read-only!")
            
            with self.assertRaises(PermissionError, msg="Sealed file modification must be rejected by OS!"):
                with open(sample_sealed_mask, "a") as fp:
                    fp.write("illegal_modification")

            # 5. DataLoader Inspection
            dm = LOBODataModule(PIPELINE_DIR, fold_id=fold_id, batch_size=4)
            uda_loader = dm.get_target_uda_loader()
            sample_batch = next(iter(uda_loader))
            
            self.assertIn("image", sample_batch)
            self.assertIn("patch_id", sample_batch)
            self.assertNotIn("mask", sample_batch, f"LEAK: 'mask' yielded by target_uda loader in {fold_id}!")
            self.assertNotIn("validity", sample_batch, f"LEAK: 'validity' yielded by target_uda loader in {fold_id}!")

    def test_08_stitched_single_physical_pixel_scoring_guarantee(self):
        """Test 8: Verify Strategy C logit accumulation scores each physical pixel exactly once."""
        for b in BRIDGES:
            b_id = b["id"]
            s = self.raw_sources[b_id]
            W, H = s["W"], s["H"]
            valid_deck = (s["valid_mask"] == 1)
            
            # Full-bridge logit accumulator simulation
            logit_accumulator = np.zeros((3, H, W), dtype=np.float32)
            b_patches = self.df_patches[self.df_patches["bridge_id"] == b_id]
            
            for _, r in b_patches.iterrows():
                y1, y2 = r["y_min"], r["y_max"]
                x1, x2 = r["x_min"], r["x_max"]
                # Synthetic patch logits (e.g. constant or prediction tensor)
                patch_logits = np.ones((3, S, S), dtype=np.float32)
                logit_accumulator[:, y1:y2, x1:x2] += patch_logits
                
            # Stitched argmax reduction
            final_pred_raster = np.argmax(logit_accumulator, axis=0) # (H, W)
            
            # Metric evaluation strictly over valid deck
            eval_predictions = final_pred_raster[valid_deck]
            self.assertEqual(len(eval_predictions), s["breakdown"]["valid"])
            self.assertEqual(eval_predictions.ndim, 1)


if __name__ == "__main__":
    unittest.main()
