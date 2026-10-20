# Phase 2A.3 — Coverage Reconciliation and Dataset Readiness Gate Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Input Dataset:** `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`  
**Read-Only Audit Inputs:**
- `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a_dataset_plan\`
- `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a1_protocol_correction\`
- `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a2_independent_verification\`  
**Output Directory:** `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a3_coverage_reconciliation\`  

---

## Executive Summary & Final Decision Gate

### Decision: **PASS**
All numerical, geometric, mask-decoding, and coverage metrics across all five bridges have been independently recomputed directly from the source PNG files, reconciled to the exact single pixel, and validated by executable unit tests. The protocol is fully specified, mathematically watertight, and implementable without ambiguity.

**Authorization Status:**
- **Audit-Only Phase Completed:** No models were trained, tuned, or initialized in this phase.
- **Data Integrity Preserved:** No modifications were made to the source dataset or previous audit outputs.
- **Bridge 4 Position Maintained:** Bridge 4's spatial alignment remains documented as **UNRESOLVED**; no artificial image translations or mask shifts were applied.
- **Physical Pixel Integrity Guaranteed:** Evaluation over overlapping patches strictly scores each physical pixel exactly once via full-raster logit accumulation and argmax reduction.
- **Recommendation:** **Phase 2B (Dataset Materialization and Model Training) is CLEARED to begin.**

---

## Task 1 — Reconcile Strategy A Coverage

### 1.1 Resolution of the 28,498-Pixel Difference
Earlier audit reports presented two seemingly contradictory figures for fleet-level Strategy A ($256 \times 256$ non-overlapping top-left grid) coverage:
- **Phase 2A Preliminary Draft:** 3,592,739 retained valid pixels.
- **Phase 2A.1 / 2A.2 Recomputed Audit:** 3,621,237 covered valid pixels.
- **Discrepancy:** $3,621,237 - 3,592,739 = \mathbf{+28,498\text{ pixels}}$.

By independently recomputing every window and pixel directly from the five source image-mask pairs, we identified the exact bridge-by-bridge physical origin of this difference:

| Bridge ID | Bridge Name | Source Dimensions ($W \times H$) | Valid Deck Pixels | Strategy A Windows ($256 \times 256$) | Covered Valid Pixels (Recomputed) | Uncovered Valid Pixels | Recomputed Coverage Pct (%) | Phase 2A Preliminary Retained Px | Delta (Recomputed − Phase 2A) | Precise Physical Root Cause |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | Forest River NB | 461 × 1942 | 858,114 | 7 ($1 \times 7$) | 447,560 | 410,554 | 52.16% | 447,211 | **+349** | Phase 1C pilot mask border edge handling vs canonical mask decoding inclusion. |
| **B2** | Forest River SB | 1626 × 401 | 621,161 | 6 ($6 \times 1$) | 393,216 | 227,945 | 63.30% | 393,216 | **0** | Exact match across all audits. |
| **B3** | Park River Median | 4326 × 411 | 1,757,327 | 16 ($16 \times 1$) | 1,036,234 | 721,093 | 58.97% | 1,041,833 | **-5,599** | Scanner black zero-padding `[0,0,0]` along the bottom margin was initially lumped into deck concrete; strictly excluded in Phase 2A.1/2A.2. |
| **B4** | Park River North Bound | 2991 × 371 | 1,109,661 | 11 ($11 \times 1$) | 720,896 | 388,765 | 64.97% | 720,896 | **0** | Exact match across all audits. |
| **B5** | Park River South Bound | 451 × 4174 | 1,743,588 | 16 ($1 \times 16$) | 1,023,331 | 720,257 | 58.69% | 989,583 | **+33,748** | Boundary tiles `r1_c0` and `r4_c0` valid deck pixels (52,960 and 63,871 px) reconciled under strict per-pixel mask decoding. |
| **Fleet** | **Five Bridge Fleet** | — | **6,089,851** | **56** | **3,621,237** | **2,468,614** | — | **3,592,739** | **+28,498** | **Net algebraic sum: $+349 + 0 - 5,599 + 0 + 33,748 = +28,498\text{ px}$.** |

The 28,498-pixel difference is **not** a rounding error, floating-point imprecision, or macro-versus-micro averaging artifact. It is the exact net sum of physical mask boundary reconciliations on B1 (+349 px), B3 (-5,599 px), and B5 (+33,748 px).

**Definitive Figure:** The true, verified covered valid-deck count under Strategy A across the 56 canonical non-overlapping tiles is **3,621,237 pixels**.

### 1.2 Micro vs. Macro Fleet Coverage Metrics
The reports previously cited both 59.46% and 59.62% for Strategy A coverage. We recomputed both metrics using consistent definitions and unambiguous denominators:

1. **Fleet Micro Coverage (Pooled Global Ratio):**
   $$\text{Micro Coverage} = \frac{\sum_{i=1}^5 \text{Covered Valid Pixels}_i}{\sum_{i=1}^5 \text{Total Valid Deck Pixels}_i} = \frac{3,621,237}{6,089,851} = \mathbf{59.4635\% \approx 59.46\%}$$
   - **Numerator:** Total covered valid pixels across all 56 tiles = $3,621,237$.
   - **Denominator:** Total valid-deck pixels across all five bridges = $6,089,851$.
   - **Meaning:** Aggregate proportion of physical concrete deck pixels captured by the top-left non-overlapping grid.

2. **Unweighted Macro Coverage (Domain-Averaged Mean):**
   $$\text{Macro Coverage} = \frac{1}{5} \sum_{i=1}^5 \left( \frac{\text{Covered Valid Pixels}_i}{\text{Total Valid Deck Pixels}_i} \right) = \frac{52.16\% + 63.30\% + 58.97\% + 64.97\% + 58.69\%}{5} = \mathbf{59.6165\% \approx 59.62\%}$$
   - **Denominator:** 5 bridge domains (each bridge weighted equally regardless of deck surface area).
   - **Meaning:** Expected per-bridge evaluation coverage.

**Conclusion:** Both metrics are mathematically exact and derive from the identical underlying counts ($3,621,237 / 6,089,851$).

---

## Task 2 — Independently Verify the Denominator

### 2.1 Source Mask Decoding and Color Encodings
All five source annotation PNGs were audited directly from disk using exact RGB color tuples:
- **`0` — Sound Concrete:** Grayscale pixels where $R=G=B$ and $0 < R < 255$.
- **`1` — Class 2 Delamination (Above Top Rebar):** Pure green `[0, 255, 0]`.
- **`2` — Class 3 Delamination (Below Top Rebar):** Pure red `[255, 0, 0]`.
- **`255` — Excluded / Boundary Pixels:**
  - Pure white `[255, 255, 255]` (canvas border surrounding skewed deck mosaics).
  - Pure black `[0, 0, 0]` (scanner zero-padding and CAD raster clamping).
- **Unknown Colors:** Exactly **0** unknown pixels across all 6,317,409 physical pixels in the dataset.

### 2.2 Class Accounting and Conservation Proof
We validated the following exact conservation identities for every bridge:
$$\text{valid\_deck\_pixels} = \text{sound\_pixels} + \text{class2\_pixels} + \text{class3\_pixels}$$
$$\text{excluded\_pixels} = \text{white\_pixels} + \text{black\_pixels} + \text{other\_excluded}$$
$$\text{total\_pixels} = \text{valid\_deck\_pixels} + \text{excluded\_pixels} = W \times H$$

Documented in [`class_accounting_reconciliation.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/class_accounting_reconciliation.csv):

| Bridge ID | Bridge Name | Dimensions ($W \times H$) | Total Pixels | Sound Concrete (Class 0) | Class 2 Above Rebar (Class 1) | Class 3 Below Rebar (Class 2) | Total Delamination | Total Valid Deck Pixels | Excluded White Canvas | Excluded Black Margin | Total Excluded Pixels | Unknown Colors | Accounting Delta |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Forest River NB | 461 × 1942 | 895,262 | 662,477 (74.00%) | 152,149 (17.00%) | 43,488 (4.86%) | 195,637 (21.85%) | 858,114 (95.85%) | 37,131 (4.15%) | 17 (0.00%) | 37,148 (4.15%) | **0** | **0** |
| **B2** | Forest River SB | 1626 × 401 | 652,026 | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 131,942 (20.24%) | 621,161 (95.27%) | 30,803 (4.72%) | 62 (0.01%) | 30,865 (4.73%) | **0** | **0** |
| **B3** | Park River Median | 4326 × 411 | 1,777,986 | 1,380,331 (77.63%) | 298,123 (16.77%) | 78,873 (4.44%) | 376,996 (21.20%) | 1,757,327 (98.84%) | 421 (0.02%) | 20,238 (1.14%) | 20,659 (1.16%) | **0** | **0** |
| **B4** | Park River North Bound | 2991 × 371 | 1,109,661 | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 264,753 (23.86%) | 1,109,661 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | **0** | **0** |
| **B5** | Park River South Bound | 451 × 4174 | 1,882,474 | 1,409,183 (74.86%) | 272,292 (14.46%) | 62,113 (3.30%) | 334,405 (17.76%) | 1,743,588 (92.62%) | 138,855 (7.38%) | 31 (0.00%) | 138,886 (7.38%) | **0** | **0** |
| **Fleet** | **Five Bridge Fleet** | — | **6,317,409** | **4,786,118 (75.76%)** | **1,045,173 (16.54%)** | **258,560 (4.09%)** | **1,303,733 (20.64%)** | **6,089,851 (96.40%)** | **207,210 (3.28%)** | **20,348 (0.32%)** | **227,558 (3.60%)** | **0** | **0** |

**Accounting Audit Result:** Zero discrepancy across all 6,317,409 physical pixels. No black or white pixels are counted inconsistently between mask decoding, tile coverage, and metrics.

---

## Task 3 — Revalidate Strategy C Coverage

### 3.1 Window Generation and Overlap Geometry
Strategy C deploys sliding-window inference with a $256 \times 256$ window, stride $S=128$, and mandatory end-anchoring to eliminate edge truncation:
- $X\text{-coordinates: } x \in \{0, 128, 256, \dots\} \cup \{W - 256\}$ (deduplicated and sorted).
- $Y\text{-coordinates: } y \in \{0, 128, 256, \dots\} \cup \{H - 256\}$ (deduplicated and sorted).

Documented in [`strategy_c_coverage_map_summary.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/strategy_c_coverage_map_summary.csv):

| Bridge ID | Bridge Name | Dimensions ($W \times H$) | $X$ Steps | $Y$ Steps | Strategy C Windows | Total Window-Pixel Appearances | Geometric Image Coverage | Valid Deck Coverage | Excluded Pixel Coverage | Min Overlap Depth | Max Overlap Depth | Mean Overlap Depth | Median Overlap Depth | Single Physical Pixel Scoring Rule |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | Forest River NB | 461 × 1942 | 3 | 15 | **45** | 2,949,120 | 895,262 (100.00%) | 858,114 (100.00%) | 37,148 (100.00%) | 1 | 9 | 3.2941 | 4.0 | Logit accumulation + argmax |
| **B2** | Forest River SB | 1626 × 401 | 12 | 3 | **36** | 2,359,296 | 652,026 (100.00%) | 621,161 (100.00%) | 30,865 (100.00%) | 1 | 9 | 3.6184 | 4.0 | Logit accumulation + argmax |
| **B3** | Park River Median | 4326 × 411 | 33 | 3 | **99** | 6,488,064 | 1,777,986 (100.00%) | 1,757,327 (100.00%) | 20,659 (100.00%) | 1 | 9 | 3.6491 | 4.0 | Logit accumulation + argmax |
| **B4** | Park River North Bound | 2991 × 371 | 23 | 2 | **46** | 3,014,656 | 1,109,661 (100.00%) | 1,109,661 (100.00%) | 0 (100.00%) | 1 | 6 | 2.7167 | 2.0 | Logit accumulation + argmax |
| **B5** | Park River South Bound | 451 × 4174 | 3 | 32 | **96** | 6,291,456 | 1,882,474 (100.00%) | 1,743,588 (100.00%) | 138,886 (100.00%) | 1 | 9 | 3.3421 | 4.0 | Logit accumulation + argmax |
| **Fleet** | **Five Bridge Fleet** | — | — | — | **322** | **21,102,592** | **6,317,409 (100.00%)** | **6,089,851 (100.00%)** | **227,558 (100.00%)** | **1** | **9** | **3.3404** | **4.0** | **1 evaluation per physical valid pixel** |

### 3.2 Four Separate Coverage Dimensions
To prevent scientific ambiguity, four distinct metrics are reported separately:
1. **Geometric Image Coverage:** **100.00%** ($6,317,409 / 6,317,409\text{ pixels}$). Every coordinate in the original raster $(x, y) \in [0, W) \times [0, H)$ falls within at least one window footprint.
2. **Valid-Deck Coverage:** **100.00%** ($6,089,851 / 6,089,851\text{ pixels}$). Exactly **0** valid concrete pixels are missed.
3. **Excluded-Pixel Coverage:** **100.00%** ($227,558 / 227,558\text{ pixels}$). Canvas borders and zero-padding are naturally covered by the sliding grid but masked out during scoring.
4. **Window-Pixel Appearances:** **21,102,592** appearances across **322 windows**.

> [!IMPORTANT]
> **Scientific Precision Warning:**  
> A 100.00% valid-deck coverage figure indicates **geometric completeness of spatial observation** under sliding windows. It **must not** be conflated with 100% ground-truth label certainty or 100% thermal radiance reliability. Unresolved boundary effects, thermal shadow artifacts, and sub-surface diffusion remain intrinsic physical characteristics of infrared thermography.

### 3.3 Single Physical Pixel Scoring Guarantee
Under Strategy C, patch predictions are not evaluated as independent samples (which would introduce artificial sample weighting and spatial double-counting). Instead:
1. A global logit accumulator tensor $\mathbf{L} \in \mathbb{R}^{3 \times H \times W}$ is initialized to zero for the entire bridge deck.
2. Model logit outputs for patch $(x, y, x+256, y+256)$ are accumulated additively into $\mathbf{L}[:, y:y+256, x:x+256]$.
3. The final discrete prediction map is computed once per physical pixel:
   $$\hat{Y}(y, x) = \arg\max_{c \in \{0, 1, 2\}} \mathbf{L}[c, y, x]$$
4. Evaluation metrics (Precision, Recall, F1, IoU, Accuracy) are calculated strictly on physical pixels where $M_{\text{valid}}(y, x) == \text{True}$.
5. Exactly $N_{\text{valid}}$ physical pixels are scored for each bridge. Spatial double-counting is mathematically impossible.

---

## Task 4 — Excluded-Pixel Handling & Spatial Audit

### 4.1 Empirical Nature of Excluded Pixels
- **White Pixels (`[255, 255, 255]`, 207,210 px):**
  - **Distribution:** Concentrated along the exterior margins of B1 (4.15%), B2 (4.72%), and B5 (7.38%).
  - **Physical Cause:** Exterior CAD canvas padding resulting from stitching and homography/perspective rectification of skewed aerial infrared mosaics onto a rectangular image raster.
- **Black Pixels (`[0, 0, 0]`, 20,348 px):**
  - **Distribution:** Predominantly on B3 (20,238 px, 1.14%) forming rectangular zero-padding borders along the bottom scanner raster edge. Minor 1-pixel boundary clamping on B1 (17 px), B2 (62 px), and B5 (31 px).
  - **Physical Cause:** Sensor zero-padding where the thermal camera sensor did not record signal or CAD margin clamping.
- **Documentation Review:**
  - Neither the SDNET2021 README nor the original published papers (Dorafshan et al. 2021) designate white or black pixels as sound deck concrete or delamination.
- **Operational Rule:**
  - White and black pixels are provisionally and strictly retained as **EXCLUDED / NON-DECK BOUNDARY (Label Index 255)**.
  - They are ignored in cross-entropy loss computation (`ignore_index = 255`) and excluded from all confusion matrices and evaluation metrics.

### 4.2 Bridge 4 Spatial Alignment Status: UNRESOLVED
- As demonstrated in Phase 1E.2, Bridge 4's original image (`PR_NB_Original.png`) and annotated mask (`PR_NB_Annotaed.png`) exhibit a 2-pixel vertical background cropping offset.
- Applying a vertical translation $\Delta y = -2$ improves visual grayscale alignment on non-deck borders, but does not prove semantic correctness across the interior concrete deck.
- **Scientific Decision:** The alignment status of Bridge 4 remains marked as **UNRESOLVED**. In accordance with scientific integrity, **no shifting, warping, or spatial interpolation is applied**. Source arrays are consumed strictly as published.

---

## Task 5 — Reproducibility, Test Suite & Leakage Safeguards

### 5.1 Verification Test Results
A dedicated, reproducible test suite was authored and executed: [`coverage_reconciliation_tests.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/coverage_reconciliation_tests.py).

All 7 test cases executed cleanly with zero failures (`OK` in 0.639s). Full execution log is captured in [`verification_results.txt`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/verification_results.txt):

1. `test_01_exact_pixel_accounting_per_bridge`: PASSED (reconciles 6,317,409 pixels across all 5 bridges with zero delta).
2. `test_02_exact_strategy_a_counts_and_reconciliation`: PASSED (verifies 56 windows, 3,621,237 valid covered pixels, and exact 28,498 px delta breakdown).
3. `test_03_unique_strategy_c_coordinates_and_bounds`: PASSED (verifies 322 unique windows within bounds $[0, W] \times [0, H]$).
4. `test_04_zero_uncovered_valid_deck_pixels_under_strategy_c`: PASSED (verifies 100.00% valid-deck and geometric coverage; max depth = 9).
5. `test_05_exclusion_of_label_index_255`: PASSED (verifies label 255 is completely purged from evaluation vectors).
6. `test_06_single_physical_pixel_scoring_guarantee`: PASSED (verifies full-raster logit accumulation produces exactly 1 score per physical pixel).
7. `test_07_lobo_fold_isolation_and_target_leakage_safeguards`: PASSED (verifies disjoint 5-fold circular successor partitioning).

### 5.2 Implementation Status of Leakage Safeguards
To maintain rigorous scientific standards, safeguards that are currently implemented and tested are explicitly distinguished from those planned for Phase 2B:

| Safeguard Mechanism | Status | Implementation Details & Evidence |
| :--- | :---: | :--- |
| **Deterministic Disjoint LOBO Manifest** | **IMPLEMENTED & TESTED** | Circular Successor Rule ($V = B_{(k \pmod 5) + 1}$) verified in `test_07`. Target test, source val, and source train sets are strictly pairwise disjoint in all 5 folds. |
| **Single Physical Pixel Scoring** | **IMPLEMENTED & TESTED** | Full-bridge raster logit accumulation verified in `test_06`. No duplicate scoring or area-weight distortion. |
| **Label Index 255 Masking** | **IMPLEMENTED & TESTED** | Exclusion of white/black pixels from loss and evaluation verified in `test_05`. |
| **Deterministic Grid Coordinates** | **IMPLEMENTED & TESTED** | Fixed stride-128 end-anchoring generated without data-dependent cropping verified in `test_03`. |
| **Physical DataLoader Directory Isolation** | *PLANNED FOR PHASE 2B* | Physical separation of materialized patches into distinct directory trees per fold (`train/`, `val/`, `test/`) will be implemented and audited during patch extraction. |
| **Checkpoint Target Sealing Callback** | *PLANNED FOR PHASE 2B* | Model training loops will receive only source training and source validation DataLoaders. Checkpoint callbacks will monitor only source-val loss/mIoU. Target evaluation will execute in an offline, decoupled post-training pass. |

---

## Deliverables Summary

All required artifacts have been generated, saved, and cross-verified in `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a3_coverage_reconciliation\`:

1. [`phase2a3_coverage_reconciliation.md`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/phase2a3_coverage_reconciliation.md): Comprehensive reconciliation and readiness gate report.
2. [`strategy_a_pixel_reconciliation.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/strategy_a_pixel_reconciliation.csv): Bridge-by-bridge reconciliation of Strategy A counts, resolving the 28,498 px delta.
3. [`strategy_c_coverage_map_summary.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/strategy_c_coverage_map_summary.csv): Full Strategy C coverage map summary across 322 windows, reporting overlap depth and 100% valid coverage.
4. [`class_accounting_reconciliation.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/class_accounting_reconciliation.csv): Recomputed semantic class counts across all 6,317,409 physical pixels.
5. [`coverage_reconciliation_tests.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/coverage_reconciliation_tests.py): Automated test suite containing 7 unit test cases with assertions for all accounting identities.
6. [`verification_results.txt`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a3_coverage_reconciliation/verification_results.txt): Raw command execution output and test passing confirmation log.

---

## Final Decision Gate & Recommendation

- **Verdict:** **PASS**
- **Justification:** Every numerical discrepancy has been resolved to the exact single pixel directly from the source images. The 28,498-pixel discrepancy is fully explained by boundary mask reconciliations on B1, B3, and B5. The evaluation protocol achieves 100.00% valid-deck coverage without spatial double-counting. All unit tests pass.
- **Next Phase:** **Phase 2B (Dataset Construction & Materialization)** is authorized to proceed under the validated LOBO and Strategy C specifications.
