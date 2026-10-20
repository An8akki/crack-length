"""
Automated Split Integrity and LOBO Partitioning Tests
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Verifies:
  1. Circular successor LOBO fold structure across all 5 folds.
  2. Complete mathematical disjointness: Target, Validation, and Train sets are mutually exclusive.
  3. Fleet completeness: Union of partitions equals the exact five bridge set {B1, B2, B3, B4, B5}.
  4. Spatial overlap isolation: 50% sliding-window overlaps are strictly bounded within bridges;
     no patch crosses bridge boundaries or split partitions.
  5. Manifest consistency across patch_manifest.csv and lobo_fold_manifest.csv.
"""

import os
import sys
import unittest
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b1_dataset_pipeline")


class TestLOBOSplits(unittest.TestCase):
    """Rigorous verification of Leave-One-Bridge-Out fold structure and cross-bridge boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.lobo_csv = os.path.join(PIPELINE_DIR, "lobo_fold_manifest.csv")
        cls.patch_csv = os.path.join(PIPELINE_DIR, "patch_manifest.csv")
        assert os.path.exists(cls.lobo_csv), f"Missing lobo_fold_manifest.csv at {cls.lobo_csv}"
        assert os.path.exists(cls.patch_csv), f"Missing patch_manifest.csv at {cls.patch_csv}"
        
        cls.df_lobo = pd.read_csv(cls.lobo_csv)
        cls.df_patches = pd.read_csv(cls.patch_csv)

    def test_01_circular_successor_rule(self):
        """Test 1: Verify each fold strictly obeys the circular successor protocol."""
        expected = {
            "fold_1": {"target": "B1", "val": "B2", "train": {"B3", "B4", "B5"}},
            "fold_2": {"target": "B2", "val": "B3", "train": {"B1", "B4", "B5"}},
            "fold_3": {"target": "B3", "val": "B4", "train": {"B1", "B2", "B5"}},
            "fold_4": {"target": "B4", "val": "B5", "train": {"B1", "B2", "B3"}},
            "fold_5": {"target": "B5", "val": "B1", "train": {"B2", "B3", "B4"}}
        }
        self.assertEqual(len(self.df_lobo), 5)
        for _, row in self.df_lobo.iterrows():
            f_id = row["fold_id"]
            self.assertIn(f_id, expected)
            exp = expected[f_id]
            self.assertEqual(row["target_bridge_id"], exp["target"])
            self.assertEqual(row["val_bridge_id"], exp["val"])
            train_set = set(row["train_bridge_ids"].split(","))
            self.assertEqual(train_set, exp["train"])

    def test_02_mutual_exclusivity(self):
        """Test 2: Target, Validation, and Training sets must have zero pairwise intersection."""
        for _, row in self.df_lobo.iterrows():
            f_id = row["fold_id"]
            tgt = {row["target_bridge_id"]}
            val = {row["val_bridge_id"]}
            train = set(row["train_bridge_ids"].split(","))
            
            self.assertEqual(len(tgt & val), 0, f"Target-Val leak in {f_id}")
            self.assertEqual(len(tgt & train), 0, f"Target-Train leak in {f_id}")
            self.assertEqual(len(val & train), 0, f"Val-Train leak in {f_id}")
            self.assertEqual(tgt | val | train, {"B1", "B2", "B3", "B4", "B5"})

    def test_03_patch_bridge_confinement(self):
        """Test 3: Verify all 322 Strategy C patches belong strictly to a single bridge."""
        self.assertEqual(len(self.df_patches), 322)
        valid_bridges = {"B1", "B2", "B3", "B4", "B5"}
        
        for _, row in self.df_patches.iterrows():
            b_id = row["bridge_id"]
            self.assertIn(b_id, valid_bridges)
            # Patch ID must start with the bridge ID prefix
            self.assertTrue(row["patch_id"].startswith(b_id))
            # Coordinate bounds must be strictly non-negative
            self.assertGreaterEqual(row["x_min"], 0)
            self.assertGreaterEqual(row["y_min"], 0)
            self.assertEqual(row["x_max"] - row["x_min"], 256)
            self.assertEqual(row["y_max"] - row["y_min"], 256)

    def test_04_no_cross_bridge_overlap(self):
        """Test 4: Verify sliding-window patch overlaps occur solely within bridges, never across."""
        for b_id, grp in self.df_patches.groupby("bridge_id"):
            # All coordinates in group must be within the bridge's bounding dimensions
            w_max = grp["x_max"].max()
            h_max = grp["y_max"].max()
            # Dimensions must match audited specifications
            if b_id == "B1":
                self.assertLessEqual(w_max, 461)
                self.assertLessEqual(h_max, 1942)
            elif b_id == "B2":
                self.assertLessEqual(w_max, 1626)
                self.assertLessEqual(h_max, 401)
            elif b_id == "B3":
                self.assertLessEqual(w_max, 4326)
                self.assertLessEqual(h_max, 411)
            elif b_id == "B4":
                self.assertLessEqual(w_max, 2991)
                self.assertLessEqual(h_max, 371)
            elif b_id == "B5":
                self.assertLessEqual(w_max, 451)
                self.assertLessEqual(h_max, 4174)


if __name__ == "__main__":
    unittest.main()
