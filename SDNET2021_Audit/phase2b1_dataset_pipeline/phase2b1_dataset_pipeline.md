# Phase 2B.1 — Build and Validate the SDNET2021 Dataset Pipeline Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Source Dataset:** `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`  
**Verified Audit Reference:** `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a3_coverage_reconciliation\`  
**Output Directory:** `C:\Users\Dell\Documents\SDNET2021_Audit\phase2b1_dataset_pipeline\`  

---

## Executive Summary & Final Decision Gate

### Decision: **PASS**

The dataset construction and partitioning pipeline for five-bridge Leave-One-Bridge-Out (LOBO) cross-domain segmentation has been fully implemented, materialized on disk, and rigorously validated via an automated test suite. All source arrays were decoded directly from disk without altering original files, all 322 canonical Strategy C patches were serialized and reconciled against source crops with zero pixel delta, and five leakage-resistant LOBO partitions were constructed with physical and programmatic isolation.

**Key Execution Highlights:**
1. **Materialized Dataset:** Exactly 322 canonical $256 \times 256$ patches were cropped, encoded, and saved with corresponding 4-class semantic masks (`{0, 1, 2, 255}`) and binary validity masks (`{0, 255}`).
2. **Deterministic Partitions:** Five LOBO folds were generated under the verified **Circular Successor Rule** ($V = B_{(k \pmod 5) + 1}$).
3. **Physical Target-Label Isolation:** The `target_uda/` partitions contain **strictly images only** (zero masks, zero validity masks, zero label metadata). The `target_eval_sealed/` partitions are locked with OS-level read-only permissions and are inaccessible to training DataLoaders.
4. **Zero Double-Counting:** Strategy C sliding-window inference is coupled with full-deck logit accumulation and single physical pixel scoring, ensuring that overlapping patch appearances do not distort test metrics.
5. **Zero Training Restraint:** No models were initialized, trained, or tuned in this phase.
6. **Recommendation:** **Phase 2B.2 (Source-Only Baseline U-Net Model & LOBO Training Benchmark) is CLEARED to begin.**

---

## Task 1 — Source Decoding & Representation Integrity

All five source pairs were inspected and decoded directly from disk using exact color encodings:
- **`0` — Sound Concrete:** Grayscale pixels where $R=G=B$ and $0 < R < 255$.
- **`1` — Class 2 Delamination (Above Top Rebar):** Pure green `[0, 255, 0]`.
- **`2` — Class 3 Delamination (Below Top Rebar):** Pure red `[255, 0, 0]`.
- **`255` — Excluded / Boundary Pixels:** Pure white `[255, 255, 255]` (canvas margins) and pure black `[0, 0, 0]` (scanner zero-padding / CAD margins).
- **Informative Error Guard:** The decoding function asserts `mask_known.all()`. If any unexpected RGB triplet is encountered, it halts immediately with an informative exception specifying the coordinates and RGB value. Across all 6,317,409 physical pixels in the fleet, exactly **0** unknown pixels were detected.

### Scientific Safeguards
- **Relative Radiance Preservation:** Original 8-bit digital numbers ($0 \le I \le 255$) are preserved without claiming they represent calibrated thermodynamic bridge temperatures.
- **Bridge 4 Semantic Alignment:** Bridge 4's 2-pixel background cropping offset remains documented as **UNRESOLVED**. In accordance with scientific integrity, no spatial translation, affine warping, or registration shifts were applied. Source images and masks are ingested strictly in their native published geometry.

---

## Task 2 — Strategy C Patch Generation & Geometry

Strategy C deploys $256 \times 256$ sliding windows with stride $S=128$ and mandatory end-anchoring along both axes:
$$x \in \{0, 128, 256, \dots\} \cup \{W - 256\}, \quad y \in \{0, 128, 256, \dots\} \cup \{H - 256\}$$

All window coordinates were programmatically generated, verified for strict within-bounds containment ($0 \le x < x+256 \le W$, $0 \le y < y+256 \le H$), and deduplicated.

### Fleet Patch Manifest Summary
Documented in [`patch_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/patch_manifest.csv):

| Bridge ID | Bridge Name | Source Dimensions ($W \times H$) | $X$ Steps | $Y$ Steps | Strategy C Patches | Fleet Proportion (%) | Valid Deck Pixels | Excluded Pixels | Mean Valid Fraction |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Forest River NB | 461 × 1942 | 3 | 15 | **45** | 13.98% | 858,114 | 37,148 | 0.9575 |
| **B2** | Forest River SB | 1626 × 401 | 12 | 3 | **36** | 11.18% | 621,161 | 30,865 | 0.9525 |
| **B3** | Park River Median | 4326 × 411 | 33 | 3 | **99** | 30.75% | 1,757,327 | 20,659 | 0.9882 |
| **B4** | Park River North Bound | 2991 × 371 | 23 | 2 | **46** | 14.29% | 1,109,661 | 0 | 1.0000 |
| **B5** | Park River South Bound | 451 × 4174 | 3 | 32 | **96** | 29.81% | 1,743,588 | 138,886 | 0.9255 |
| **Fleet** | **Five Bridge Fleet** | — | — | — | **322** | **100.00%** | **6,089,851** | **227,558** | **0.9634** |

Every patch is stored as:
1. `patches/images/<patch_id>_img.png`: 8-bit grayscale image crop ($256 \times 256$).
2. `patches/masks/<patch_id>_mask.png`: 8-bit 4-class semantic mask crop ($256 \times 256$, values in $\{0, 1, 2, 255\}$).
3. `patches/validity/<patch_id>_valid.png`: 8-bit binary valid-deck mask ($256 \times 256$, value 255 where mask $\in \{0, 1, 2\}$, 0 where 255).

---

## Task 3 — Leakage-Resistant LOBO Partitions & Isolation Verification

### 3.1 Five-Fold Partition Architecture
Under the **Circular Successor Rule** ($V = B_{(k \pmod 5) + 1}$), each bridge serves as target test exactly once, source validation exactly once, and source training exactly 3 times across the five folds.

Documented in [`lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/lobo_fold_manifest.csv):

| Fold ID | Target Bridge (Test) | Source Val Bridge | Source Train Bridges | Source Train Patches | Source Val Patches | Target UDA Patches | Target Eval Sealed Patches | Total Fold Patches | Target UDA Images Only? | Target Labels Sealed? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fold_1** | **B1** (Forest River NB) | **B2** (Forest River SB) | B3, B4, B5 | 241 (74.8%) | 36 (11.2%) | 45 (14.0%) | 45 (14.0%) | 322 | **YES (Images only)** | **YES (Read-only)** |
| **fold_2** | **B2** (Forest River SB) | **B3** (Park River Median) | B1, B4, B5 | 187 (58.1%) | 99 (30.7%) | 36 (11.2%) | 36 (11.2%) | 322 | **YES (Images only)** | **YES (Read-only)** |
| **fold_3** | **B3** (Park River Median) | **B4** (Park River NB) | B1, B2, B5 | 177 (55.0%) | 46 (14.3%) | 99 (30.7%) | 99 (30.7%) | 322 | **YES (Images only)** | **YES (Read-only)** |
| **fold_4** | **B4** (Park River NB) | **B5** (Park River SB) | B1, B2, B3 | 180 (55.9%) | 96 (29.8%) | 46 (14.3%) | 46 (14.3%) | 322 | **YES (Images only)** | **YES (Read-only)** |
| **fold_5** | **B5** (Park River SB) | **B1** (Forest River NB) | B2, B3, B4 | 181 (56.2%) | 45 (14.0%) | 96 (29.8%) | 96 (29.8%) | 322 | **YES (Images only)** | **YES (Read-only)** |

### 3.2 Partition Directory Structure & Physical Isolation
Each partition was created as an independent filesystem directory tree:
```
phase2b1_dataset_pipeline/
├── patches/                               <-- Central canonical patch bank
│   ├── images/                            (322 files)
│   ├── masks/                             (322 files)
│   └── validity/                          (322 files)
├── source_train/
│   ├── fold_1/ [images/, masks/, validity/] (241 patches from B3, B4, B5)
│   ├── fold_2/ ...                          (187 patches from B1, B4, B5)
│   ├── fold_3/ ...                          (177 patches from B1, B2, B5)
│   ├── fold_4/ ...                          (180 patches from B1, B2, B3)
│   └── fold_5/ ...                          (181 patches from B2, B3, B4)
├── source_val/
│   ├── fold_1/ [images/, masks/, validity/] (36 patches from B2)
│   ├── fold_2/ ...                          (99 patches from B3)
│   ├── fold_3/ ...                          (46 patches from B4)
│   ├── fold_4/ ...                          (96 patches from B5)
│   └── fold_5/ ...                          (45 patches from B1)
├── target_uda/
│   ├── fold_1/ [images/]                    (45 B1 images ONLY; NO masks, NO validity)
│   ├── fold_2/ [images/]                    (36 B2 images ONLY; NO masks, NO validity)
│   ├── fold_3/ [images/]                    (99 B3 images ONLY; NO masks, NO validity)
│   ├── fold_4/ [images/]                    (46 B4 images ONLY; NO masks, NO validity)
│   └── fold_5/ [images/]                    (96 B5 images ONLY; NO masks, NO validity)
└── target_eval_sealed/
    ├── fold_1/ [images/, masks/, validity/, eval_stitch_manifest.json] (Read-only locked)
    ├── fold_2/ ...
    ├── fold_3/ ...
    ├── fold_4/ ...
    └── fold_5/ ...
```

### 3.3 Executable Leakage Safeguards Testing
Isolation is not merely asserted; it was verified by test assertions in [`dataset_validation_tests.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/dataset_validation_tests.py):
1. **Target UDA Directory Inspection (`test_07`):** Asserted that `masks/` and `validity/` directories do **not** exist in `target_uda/<fold_id>/`. Asserted that every file in `target_uda/<fold_id>/images/` ends with `_img.png` and begins with the target bridge prefix. Asserted that zero metadata files containing class statistics exist.
2. **Train/Val Bridge Separation (`test_07`):** Asserted that zero target bridge patches appear in `source_train` or `source_val`.
3. **OS Read-Only Permission Enforcement (`test_07`):** All mask files in `target_eval_sealed/` were locked using `stat.S_IREAD`. The test executed an illegal write attempt to the sealed mask files and confirmed that the operating system threw a `PermissionError`.
4. **DataLoader Protection (`test_07`):** Instantiated `SDNETPatchDataset(..., partition='target_uda')`. Asserted that the dictionary returned by `__getitem__` contains only `image` and `patch_id`, and that attempting to access `mask` raises a `KeyError`.

---

## Task 4 — Label Accounting & Rare-Class Preservation

Documented in [`class_distribution_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/class_distribution_by_bridge.csv):

| Bridge ID | Bridge Name | Dimensions | Sound Concrete (Class 0) | Class 2 Above Rebar (Class 1) | Class 3 Below Rebar (Class 2) | Excluded White Canvas | Excluded Black Margin | Strategy C Patches | Patches Zero Class 2 | Patches Zero Class 3 | Patches Zero Delamination |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Forest River NB | 461 × 1942 | 662,477 (74.00%) | 152,149 (17.00%) | 43,488 (4.86%) | 37,131 (4.15%) | 17 (0.00%) | **45** | 0 | 0 | 0 |
| **B2** | Forest River SB | 1626 × 401 | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 30,803 (4.72%) | 62 (0.01%) | **36** | 0 | 0 | 0 |
| **B3** | Park River Median | 4326 × 411 | 1,380,331 (77.63%) | 298,123 (16.77%) | 78,873 (4.44%) | 421 (0.02%) | 20,238 (1.14%) | **99** | 0 | 2 | 0 |
| **B4** | Park River NB | 2991 × 371 | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 0 (0.00%) | 0 (0.00%) | **46** | 0 | 3 | 0 |
| **B5** | Park River SB | 451 × 4174 | 1,409,183 (74.86%) | 272,292 (14.46%) | 62,113 (3.30%) | 138,855 (7.38%) | 31 (0.00%) | **96** | 2 | 7 | 2 |
| **Fleet** | **Five Bridge Fleet** | — | **4,786,118 (75.76%)** | **1,045,173 (16.54%)** | **258,560 (4.09%)** | **207,210 (3.28%)** | **20,348 (0.32%)** | **322** | **2** | **12** | **2** |

### Rare-Class Preservation Policy
- Across the fleet of 322 patches:
  - Exactly **2 patches** contain zero Class 2 pixels (both on B5: `B5_c_r00_c00_y0000_x0000` and `B5_c_r31_c00_y3918_x0000`).
  - Exactly **12 patches** contain zero Class 3 pixels (B3: 2 patches, B4: 3 patches, B5: 7 patches).
  - Exactly **2 patches** contain zero delamination of either class (both on B5).
- **Audit Decision:** **Zero patches were discarded.** All 322 patches are retained in the patch bank and partition manifests. Any class-balancing or hard-mining sampling will be implemented as a separate, documented DataLoader policy in Phase 2B.2.

---

## Task 5 — Automated Test Results & Verification Proof

The test suite [`dataset_validation_tests.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/dataset_validation_tests.py) was executed. All 8 tests passed cleanly without warnings.  
Log output recorded in [`verification_results.txt`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/verification_results.txt):

```text
=== Phase 2B.1 Dataset Pipeline Validation Test Results ===

Command: python dataset_validation_tests.py
Exit Code: 0

STDERR:
........
----------------------------------------------------------------------
Ran 8 tests in 10.040s

OK
```

### Summary of Verification Tests
1. `test_01_source_decoding_and_dimension_verification`: PASSED. Reconciled all 6,317,409 physical pixels across B1–B5 with 0 unknown colors.
2. `test_02_strategy_c_window_generation_and_counts`: PASSED. Confirmed 322 unique, within-bounds $256 	imes 256$ windows.
3. `test_03_zero_uncovered_valid_deck_pixels`: PASSED. Confirmed 100.00% valid-deck and geometric coverage (0 uncovered valid pixels).
4. `test_04_exact_patch_reconciliation_against_source_arrays`: PASSED. Evaluated all 322 materialized disk images, masks, and validity files against raw source crops; confirmed 100% pixel-level equality.
5. `test_05_rare_class_and_zero_delam_patch_accounting`: PASSED. Verified exact zero-class counts (2 zero-C2, 12 zero-C3, 2 zero-delam).
6. `test_06_lobo_fold_disjointness_and_circular_successor`: PASSED. Verified 5-fold circular successor partitioning and mutual disjointness.
7. `test_07_physical_partition_isolation_and_leakage_safeguards`: PASSED. Confirmed filesystem target isolation, read-only permissions on sealed masks, and DataLoader label suppression.
8. `test_08_stitched_single_physical_pixel_scoring_guarantee`: PASSED. Simulated full-deck logit accumulation and confirmed exactly 1 score per physical valid pixel.

---

## Deliverables Generated in Phase 2B.1

All deliverables are saved in `C:\Users\Dell\Documents\SDNET2021_Audit\phase2b1_dataset_pipeline\`:

1. [`phase2b1_dataset_pipeline.md`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/phase2b1_dataset_pipeline.md): Comprehensive pipeline construction and validation report.
2. [`patch_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/patch_manifest.csv): Master manifest of all 322 Strategy C patches with exact coordinates and pixel statistics.
3. [`lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/lobo_fold_manifest.csv): Specification of the 5 Leave-One-Bridge-Out folds under the circular successor rule.
4. [`class_distribution_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/class_distribution_by_bridge.csv): Bridge-by-bridge class accounting across all 6,317,409 physical pixels.
5. [`dataset_build.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/dataset_build.py): Standalone, reproducible Python script containing pipeline generation and PyTorch `SDNETPatchDataset` / `LOBODataModule` classes.
6. [`dataset_validation_tests.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/dataset_validation_tests.py): Automated test suite containing 8 unit tests covering all verification requirements.
7. [`verification_results.txt`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2b1_dataset_pipeline/verification_results.txt): Raw console output and exit code from running the test suite.

---

## Final Recommendation & Next Phase Scope

- **Status:** **PASS**
- **Justification:** All 322 patches are materialized and reconciled with zero delta against source data. Physical and programmatic target-label isolation is implemented and enforced by passing unit tests. Full-bridge stitched scoring is guaranteed.
- **Next Phase:** **Phase 2B.2 (Source-Only Baseline U-Net Model & LOBO Training Benchmark)** is authorized to proceed.
