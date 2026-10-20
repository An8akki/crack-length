# Comprehensive Reproducibility and Scientific Integrity Audit

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Repository:** `https://github.com/An8akki/crack-length` (`depth-estimation-integration`)  
**Audit Date:** 2026-10-20  

---

## 1. Traceability & Cryptographic Seals

| Artifact Category | File Location | Cryptographic Signature / Verification Status |
| :--- | :--- | :--- |
| **Fold 1 Checkpoint (Seed 42)** | `checkpoints/fold_1_seed42_best_checkpoint.pt` | `5bf64696b6f112156a60d2ba50a2967add1ab8b5ed561af791c954717789918c` (100% Bitwise Match) |
| **Fold 1 Checkpoint (Seed 123)** | `checkpoints/fold_1_seed123_best_checkpoint.pt` | `8a45c7daf3687fd0fc9e377d89b06826dcdacafb160c099195dfd47b594d4413` (100% Bitwise Match) |
| **Fold 1 Checkpoint (Seed 999)** | `checkpoints/fold_1_seed999_best_checkpoint.pt` | `025bbe4221adb3325f622b86cd8539442330d392b4becd99b42e79c3041ff8d5` (100% Bitwise Match) |
| **Sealed Target Eval Masks** | `phase2b1_dataset_pipeline/target_eval_sealed/` | All 5 folds verified against `checksums.sha256` |
| **LOBO Fold Manifest** | `phase2b1_dataset_pipeline/lobo_fold_manifest.csv` | Verified circular successor structure |
| **Strategy C Patch Manifest** | `phase2b1_dataset_pipeline/patch_manifest.csv` | 322 patches, 100% valid physical deck coverage |
| **Access Authorization Log** | `phase2b2_source_only_baseline/target_evaluation_access_log.csv` | Pre-evaluation integrity verified for all executed evaluations |

---

## 2. Automated Test Suite Performance

| Test Suite | Purpose | Tests Run | Result | Execution Time |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_deep_coral.py` | Covariance calculation, gradient flow, dimension handling | 7 | **PASS** | 0.180s |
| `tests/test_lobo_splits.py` | Circular successor LOBO fold structure & mutual exclusivity | 4 | **PASS** | 0.016s |
| `tests/test_data_isolation.py` | Target label access denial, ignore_index gradient masking | 5 | **PASS** | 0.006s |
| `phase2b2_source_only_baseline/resume_experiment.py --status` | Checkpoint manifest, loadability, and run verification | 15 runs | **PASS** | 2.100s |
| **Total Test Assertions** | **Comprehensive Automated Verification** | **16+** | **100% PASS** | **< 3s** |

---

## 3. Explicit Documentation of Known Limitations

In accordance with Global Rule 4 and scholarly standards, the following structural limitations are recorded:
1. **Limited Domain Sample Size ($N=5$ Bridges):**
   With only five bridge decks in SDNET2021, the maximum attainable statistical power under exact non-parametric tests is strictly limited ($p_{\text{two-sided}} \ge 0.0625$). Effect size reporting and pre-declared engineering targets take precedence over null-hypothesis rejection.
2. **Ambiguous Infrared Boundary Pixels:**
   Boundary canvas regions (pure white and pure black pixels) represent scanning limits, text labels, and colorbars. These pixels are provisionally masked using label index `255`, and are never scored as concrete.
3. **Unresolved Bridge 4 Spatial Alignment:**
   A possible 2-pixel translation between infrared and annotation on Bridge 4 remains unverified by physical ground truth; the unshifted raw alignment is preserved to prevent artificial distortion.
4. **Radiometric Nature:**
   SDNET2021 images represent 8-bit digital infrared pixel values, not calibrated thermodynamic temperatures.
5. **Class Imbalance:**
   Class 3 delamination (sub-rebar) is exceptionally scarce, representing $<0.5\%$ of bridge deck surface area, demanding class-disaggregated precision, recall, and F1 metrics.
