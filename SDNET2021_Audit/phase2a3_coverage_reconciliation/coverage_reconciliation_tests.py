"""
Phase 2A.3: Coverage Reconciliation and Dataset Readiness Gate Tests
Executable Verification Suite for SDNET2021 Delamination Segmentation Audit
"""

import os
import unittest
import numpy as np
from PIL import Image
import pandas as pd

SRC_DIR = r"C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt"
OUT_DIR = r"C:\Users\Dell\Documents\SDNET2021_Audit\phase2a3_coverage_reconciliation"

BRIDGES = [
    {
        "id": "B1",
        "name": "Forest River NB",
        "dir": os.path.join(SRC_DIR, "1.Forest River NB"),
        "orig_file": "FR_NB_Original.png",
        "annot_file": "FR_NB_Annotated.png",
        "W": 461, "H": 1942, "tot": 895262, "valid": 858114, "sound": 662477, "c2": 152149, "c3": 43488, "white": 37131, "black": 17,
        "strat_a_win": 7, "strat_a_valid": 447560, "strat_c_win": 45, "c_max_depth": 9
    },
    {
        "id": "B2",
        "name": "Forest River SB",
        "dir": os.path.join(SRC_DIR, "2.Forest River SB"),
        "orig_file": "FR_SB_Original.png",
        "annot_file": "FR_SB_Annotated.png",
        "W": 1626, "H": 401, "tot": 652026, "valid": 621161, "sound": 489219, "c2": 107496, "c3": 24446, "white": 30803, "black": 62,
        "strat_a_win": 6, "strat_a_valid": 393216, "strat_c_win": 36, "c_max_depth": 9
    },
    {
        "id": "B3",
        "name": "Park River Median",
        "dir": os.path.join(SRC_DIR, "3.Park River Median"),
        "orig_file": "PR_MD_Original.png",
        "annot_file": "PR_MD_Annotated.png",
        "W": 4326, "H": 411, "tot": 1777986, "valid": 1757327, "sound": 1380331, "c2": 298123, "c3": 78873, "white": 421, "black": 20238,
        "strat_a_win": 16, "strat_a_valid": 1036234, "strat_c_win": 99, "c_max_depth": 9
    },
    {
        "id": "B4",
        "name": "Park River North Bound",
        "dir": os.path.join(SRC_DIR, "4.Park River North Bound"),
        "orig_file": "PR_NB_Original.png",
        "annot_file": "PR_NB_Annotaed.png",
        "W": 2991, "H": 371, "tot": 1109661, "valid": 1109661, "sound": 844908, "c2": 215113, "c3": 49640, "white": 0, "black": 0,
        "strat_a_win": 11, "strat_a_valid": 720896, "strat_c_win": 46, "c_max_depth": 6
    },
    {
        "id": "B5",
        "name": "Park River South Bound",
        "dir": os.path.join(SRC_DIR, "5.Park River South Bound"),
        "orig_file": "PR_SB_Original.png",
        "annot_file": "PR_SB_Annotated.png",
        "W": 451, "H": 4174, "tot": 1882474, "valid": 1743588, "sound": 1409183, "c2": 272292, "c3": 62113, "white": 138855, "black": 31,
        "strat_a_win": 16, "strat_a_valid": 1023331, "strat_c_win": 96, "c_max_depth": 9
    }
]


class TestPhase2A3CoverageReconciliation(unittest.TestCase):
    """Full independent test suite validating dataset reconciliation for Phase 2A.3."""

    @classmethod
    def setUpClass(cls):
        cls.loaded_bridges = {}
        for b in BRIDGES:
            orig_p = os.path.join(b["dir"], b["orig_file"])
            annot_p = os.path.join(b["dir"], b["annot_file"])
            
            assert os.path.exists(orig_p), f"Missing original: {orig_p}"
            assert os.path.exists(annot_p), f"Missing annotated: {annot_p}"
            
            with Image.open(orig_p) as img_orig:
                W, H = img_orig.size
            with Image.open(annot_p) as img_annot:
                arr_annot = np.array(img_annot.convert("RGB"))
            
            R = arr_annot[:, :, 0]
            G = arr_annot[:, :, 1]
            B = arr_annot[:, :, 2]
            
            mask_c2 = (R == 0) & (G == 255) & (B == 0)
            mask_c3 = (R == 255) & (G == 0) & (B == 0)
            mask_white = (R == 255) & (G == 255) & (B == 255)
            mask_black = (R == 0) & (G == 0) & (B == 0)
            mask_sound = (R == G) & (G == B) & (R > 0) & (R < 255)
            mask_unknown = ~(mask_c2 | mask_c3 | mask_white | mask_black | mask_sound)
            valid_deck = mask_sound | mask_c2 | mask_c3
            
            cls.loaded_bridges[b["id"]] = {
                "b": b,
                "W": W, "H": H,
                "mask_c2": mask_c2, "mask_c3": mask_c3,
                "mask_sound": mask_sound, "mask_white": mask_white,
                "mask_black": mask_black, "mask_unknown": mask_unknown,
                "valid_deck": valid_deck
            }

    def test_01_exact_pixel_accounting_per_bridge(self):
        """Test 1: Verify exact pixel totals and mutual exclusivity across all bridges."""
        fleet_tot = 0
        fleet_valid = 0
        fleet_sound = 0
        fleet_c2 = 0
        fleet_c3 = 0
        fleet_white = 0
        fleet_black = 0
        fleet_unknown = 0
        
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            W, H = d["W"], d["H"]
            tot = W * H
            
            self.assertEqual(W, b["W"], f"Width mismatch {b_id}")
            self.assertEqual(H, b["H"], f"Height mismatch {b_id}")
            self.assertEqual(tot, b["tot"], f"Total pixels mismatch {b_id}")
            
            n_sound = int(d["mask_sound"].sum())
            n_c2 = int(d["mask_c2"].sum())
            n_c3 = int(d["mask_c3"].sum())
            n_white = int(d["mask_white"].sum())
            n_black = int(d["mask_black"].sum())
            n_unknown = int(d["mask_unknown"].sum())
            n_valid = int(d["valid_deck"].sum())
            n_excl = n_white + n_black + n_unknown
            
            self.assertEqual(n_unknown, 0, f"Found unexpected unknown colors in {b_id}")
            self.assertEqual(n_sound, b["sound"], f"Sound count mismatch {b_id}")
            self.assertEqual(n_c2, b["c2"], f"Class 2 count mismatch {b_id}")
            self.assertEqual(n_c3, b["c3"], f"Class 3 count mismatch {b_id}")
            self.assertEqual(n_white, b["white"], f"White count mismatch {b_id}")
            self.assertEqual(n_black, b["black"], f"Black count mismatch {b_id}")
            self.assertEqual(n_valid, b["valid"], f"Valid count mismatch {b_id}")
            
            # Exact conservation
            self.assertEqual(n_sound + n_c2 + n_c3, n_valid)
            self.assertEqual(n_valid + n_excl, tot)
            
            fleet_tot += tot
            fleet_valid += n_valid
            fleet_sound += n_sound
            fleet_c2 += n_c2
            fleet_c3 += n_c3
            fleet_white += n_white
            fleet_black += n_black
            fleet_unknown += n_unknown
            
        self.assertEqual(fleet_tot, 6317409)
        self.assertEqual(fleet_valid, 6089851)
        self.assertEqual(fleet_sound, 4786118)
        self.assertEqual(fleet_c2, 1045173)
        self.assertEqual(fleet_c3, 258560)
        self.assertEqual(fleet_white, 207210)
        self.assertEqual(fleet_black, 20348)
        self.assertEqual(fleet_unknown, 0)

    def test_02_exact_strategy_a_counts_and_reconciliation(self):
        """Test 2: Verify exact Strategy A counts and resolve 28,498 px difference."""
        historical_p2a = {"B1": 447211, "B2": 393216, "B3": 1041833, "B4": 720896, "B5": 989583}
        recomputed_valid_A = {}
        total_windows_A = 0
        
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            W, H = d["W"], d["H"]
            valid_deck = d["valid_deck"]
            
            S = 256
            cols = W // S
            rows = H // S
            n_win = cols * rows
            total_windows_A += n_win
            
            self.assertEqual(n_win, b["strat_a_win"], f"Strategy A window count mismatch {b_id}")
            
            cov_mask = np.zeros((H, W), dtype=bool)
            for r in range(rows):
                for c in range(cols):
                    cov_mask[r*S:(r+1)*S, c*S:(c+1)*S] = True
                    
            cov_valid = int((cov_mask & valid_deck).sum())
            self.assertEqual(cov_valid, b["strat_a_valid"], f"Strategy A valid pixels mismatch {b_id}")
            recomputed_valid_A[b_id] = cov_valid
            
        fleet_cov_A = sum(recomputed_valid_A.values())
        fleet_p2a_sum = sum(historical_p2a.values())
        delta = fleet_cov_A - fleet_p2a_sum
        
        self.assertEqual(total_windows_A, 56)
        self.assertEqual(fleet_cov_A, 3621237)
        self.assertEqual(fleet_p2a_sum, 3592739)
        self.assertEqual(delta, 28498, "Difference must exactly equal 28,498 pixels")
        
        # Verify bridge-by-bridge deltas
        self.assertEqual(recomputed_valid_A["B1"] - historical_p2a["B1"], 349)
        self.assertEqual(recomputed_valid_A["B2"] - historical_p2a["B2"], 0)
        self.assertEqual(recomputed_valid_A["B3"] - historical_p2a["B3"], -5599)
        self.assertEqual(recomputed_valid_A["B4"] - historical_p2a["B4"], 0)
        self.assertEqual(recomputed_valid_A["B5"] - historical_p2a["B5"], 33748)
        self.assertEqual(349 + 0 - 5599 + 0 + 33748, 28498)

    def test_03_unique_strategy_c_coordinates_and_bounds(self):
        """Test 3: Verify all 322 Strategy C window coordinates are strictly unique and within bounds."""
        total_windows_C = 0
        S = 256
        step = 128
        
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            W, H = d["W"], d["H"]
            
            xs = list(range(0, W - S + 1, step))
            if xs[-1] != W - S: xs.append(W - S)
            xs = sorted(list(set(xs)))
            
            ys = list(range(0, H - S + 1, step))
            if ys[-1] != H - S: ys.append(H - S)
            ys = sorted(list(set(ys)))
            
            coords = []
            for y in ys:
                for x in xs:
                    coords.append((x, y, x + S, y + S))
                    
            self.assertEqual(len(coords), len(set(coords)), f"Duplicate coordinates generated in {b_id}")
            self.assertEqual(len(coords), b["strat_c_win"], f"Window count mismatch in {b_id}")
            
            for (x1, y1, x2, y2) in coords:
                self.assertTrue(0 <= x1 < x2 <= W, f"X coordinate out of bounds in {b_id}: {x1}, {x2}")
                self.assertTrue(0 <= y1 < y2 <= H, f"Y coordinate out of bounds in {b_id}: {y1}, {y2}")
                self.assertEqual(x2 - x1, S)
                self.assertEqual(y2 - y1, S)
                
            total_windows_C += len(coords)
            
        self.assertEqual(total_windows_C, 322)

    def test_04_zero_uncovered_valid_deck_pixels_under_strategy_c(self):
        """Test 4: Verify 100% valid deck and geometric coverage under Strategy C with exact max depth."""
        S = 256
        step = 128
        
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            W, H = d["W"], d["H"]
            valid_deck = d["valid_deck"]
            
            xs = list(range(0, W - S + 1, step))
            if xs[-1] != W - S: xs.append(W - S)
            xs = sorted(list(set(xs)))
            
            ys = list(range(0, H - S + 1, step))
            if ys[-1] != H - S: ys.append(H - S)
            ys = sorted(list(set(ys)))
            
            cov_map = np.zeros((H, W), dtype=int)
            for y in ys:
                for x in xs:
                    cov_map[y:y+S, x:x+S] += 1
                    
            self.assertEqual(int(cov_map.min()), 1, f"Minimum coverage depth must be >= 1 in {b_id}")
            self.assertEqual(int(cov_map.max()), b["c_max_depth"], f"Max coverage depth mismatch in {b_id}")
            
            # Geometric coverage check
            uncovered_geom = int((cov_map == 0).sum())
            self.assertEqual(uncovered_geom, 0, f"Geometric coverage holes in {b_id}")
            
            # Valid deck coverage check
            uncovered_valid = int(((cov_map == 0) & valid_deck).sum())
            self.assertEqual(uncovered_valid, 0, f"Strategy C left uncovered valid pixels in {b_id}")
            
            covered_valid = int(((cov_map >= 1) & valid_deck).sum())
            self.assertEqual(covered_valid, b["valid"])

    def test_05_exclusion_of_label_index_255(self):
        """Test 5: Verify that label index 255 (excluded pixels) is completely masked from evaluation."""
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            
            H, W = d["H"], d["W"]
            gt_semantic = np.full((H, W), 255, dtype=np.uint8)
            gt_semantic[d["mask_sound"]] = 0
            gt_semantic[d["mask_c2"]] = 1
            gt_semantic[d["mask_c3"]] = 2
            
            pred = np.zeros((H, W), dtype=np.uint8)
            valid_mask = (gt_semantic != 255)
            self.assertEqual(int(valid_mask.sum()), b["valid"])
            
            y_true_valid = gt_semantic[valid_mask]
            y_pred_valid = pred[valid_mask]
            
            self.assertTrue((y_true_valid >= 0).all() and (y_true_valid <= 2).all())
            self.assertFalse((y_true_valid == 255).any(), "Index 255 leaked into evaluation vector!")
            self.assertEqual(len(y_true_valid), b["valid"])

    def test_06_single_physical_pixel_scoring_guarantee(self):
        """Test 6: Verify Strategy C logit accumulation yields exactly one score per physical pixel."""
        for b in BRIDGES:
            b_id = b["id"]
            d = self.loaded_bridges[b_id]
            W, H = d["W"], d["H"]
            valid_deck = d["valid_deck"]
            
            S = 256
            step = 128
            xs = sorted(list(set(list(range(0, W - S + 1, step)) + [W - S])))
            ys = sorted(list(set(list(range(0, H - S + 1, step)) + [H - S])))
            
            logit_accumulator = np.zeros((3, H, W), dtype=np.float32)
            for y in ys:
                for x in xs:
                    patch_logits = np.ones((3, S, S), dtype=np.float32)
                    logit_accumulator[:, y:y+S, x:x+S] += patch_logits
                    
            final_pred_raster = np.argmax(logit_accumulator, axis=0)
            eval_preds = final_pred_raster[valid_deck]
            self.assertEqual(len(eval_preds), b["valid"], "Evaluation vector length must match valid deck pixels exactly")
            self.assertEqual(eval_preds.ndim, 1)

    def test_07_lobo_fold_isolation_and_target_leakage_safeguards(self):
        """Test 7: Verify 5-fold LOBO isolation and data independence."""
        all_ids = [b["id"] for b in BRIDGES]
        self.assertEqual(len(all_ids), 5)
        
        folds = []
        for k in range(5):
            tgt_idx = k
            val_idx = (k + 1) % 5
            train_indices = [i for i in range(5) if i != tgt_idx and i != val_idx]
            
            tgt_b = all_ids[tgt_idx]
            val_b = all_ids[val_idx]
            train_bs = [all_ids[i] for i in train_indices]
            
            self.assertNotIn(tgt_b, train_bs)
            self.assertNotEqual(tgt_b, val_b)
            self.assertNotIn(val_b, train_bs)
            self.assertEqual(len(train_bs), 3)
            self.assertEqual(len(set([tgt_b, val_b] + train_bs)), 5)
            
            folds.append({
                "fold": k + 1,
                "target": tgt_b,
                "val": val_b,
                "train": train_bs
            })
            
        targets = [f["target"] for f in folds]
        vals = [f["val"] for f in folds]
        self.assertEqual(sorted(targets), sorted(all_ids))
        self.assertEqual(sorted(vals), sorted(all_ids))


if __name__ == "__main__":
    unittest.main()
