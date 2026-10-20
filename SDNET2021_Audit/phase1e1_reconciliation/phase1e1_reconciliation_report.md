# Phase 1E.1 — Audit Reconciliation and Contradiction Resolution Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** October 20, 2026  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Input Directories:**
- Source Raw Images: `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`
- Phase 1D Audit: `C:\Users\Dell\Documents\SDNET2021_Audit\phase1d_five_bridge_audit\`
- Phase 1E Audit: `C:\Users\Dell\Documents\SDNET2021_Audit\phase1e_alignment_verification\`
**Output Directory:**
- `C:\Users\Dell\Documents\SDNET2021_Audit\phase1e1_reconciliation\`

---

## Executive Summary & Final Verdict

### Final Decision: **GO for Phase 2 Dataset Construction**

All numerical discrepancies and reporting contradictions from previous audit phases have been fully, independently investigated and mathematically resolved using raw source imagery and executable code. All programmatic assertions have passed without exception:
1. **Per-Image Pixel Reconciliation:** All 6,317,409 pixels across the five bridges are accounted for across mutually exclusive semantic categories with **exactly 0 unknown pixels and zero reconciliation delta**.
2. **Tile Manifest Recomputation:** All 56 complete 256 x 256 tiles have been re-extracted directly from raw arrays. Every tile's pixel sum matches 256 x 256 = 65,536 identically.
3. **Contradiction Resolution for Bridge 5:**
   - The Phase 1E table typographical error listing minimum delamination as 567 px has been corrected. `B5_256_tl_r0_c0` contains **0 Class 2 pixels and 0 Class 3 pixels**, giving a total delamination of **exactly 0 pixels**.
   - The minimum total delamination statistic across the 56 retained tiles is **0 pixels**, matching the recomputed row-level data.
   - The conflicting Class 2 counts between Phase 1D and Phase 1E for `r1_c0` and `r4_c0` are definitively resolved: Phase 1D reported metrics for both a **centered grid** (x in [97, 353]) and a **top-left grid** (x in [0, 256]). In the standard top-left grid, the true counts are **1,539 Class 2 px for `r1_c0`** and **6,920 Class 2 px for `r4_c0`**.
4. **Single Primary Retention Rule Established:**
   - Primary standard: **Retain only complete 256 x 256 tiles with valid deck fraction >= 80.0%**.
   - Under this rule, exactly **56 uncorrupted physical tiles** are retained (B1: 7, B2: 6, B3: 16, B4: 11, B5: 16).
   - Zero complete tiles are rejected under either 50% or 80% (all 56 complete tiles have >= 80.81% valid deck).
   - Boundary edge patches (W < 256 or H < 256) are excluded from the primary dataset to avoid synthetic padding artifacts.
5. **No-Wrap Translation Diagnostic Completed:**
   - Replacing `numpy.roll` with explicit overlapping array slices confirms that on Bridge 4, a vertical integer shift of (dy = +2, dx = 0) reduces mean absolute difference from **2.8867 to 0.3702** on the identical N = 838,926 overlapping sound pixels.
   - However, in accordance with research protocol, **no translation is applied to the dataset**. Input X is loaded from `Original.png` and ground-truth targets Y from `Annotated.png`.

---

## 1. Resolution of Phase 1E Contradictions & B5 Tile Verification

### 1.1 Resolution of the Minimum Delamination Discrepancy
- **Contradiction Identified:** The Phase 1E summary table reported `Total Delam (px) Minimum: 567`, while the text noted that `B5_256_tl_r0_c0` had 0 Class 2 and 0 Class 3 pixels.
- **Root Cause:** The value `567` was the minimum Class 3 count on Bridge 3 (`B3_256_tl_r0_c15`), which was inadvertently transcribed into the minimum total delamination column in the Phase 1E markdown table.
- **Measured Source Truth:**
  - `B5_256_tl_r0_c0` contains:
    - Class 1 Sound: 64,953 px
    - Class 2 Above-Mat: **0 px**
    - Class 3 Below-Mat: **0 px**
    - White Canvas Padding: 583 px
    - Total Delamination (C2 + C3): **0 px**
    - Total Pixels: 64,953 + 0 + 0 + 583 = 65,536 px
  - Therefore, the true minimum total delamination across all 56 retained tiles is **0 pixels**.

### 1.2 Reconciliation of Bridge 5 Tiles `r0_c0`, `r1_c0`, and `r4_c0`
Direct recomputation from `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\5.Park River South Bound\PR_SB_Annotated.png`:

| Tile ID | Grid Row | Y-Range | X-Range | Sound Pixels | Class 2 Pixels | Class 3 Pixels | Total Delam Pixels | Canvas Pixels | Total Pixels | Valid Fraction | Phase 1D Centered Class 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B5_256_tl_r0_c0`** | 0 | 0–256 | 0–256 | 64,953 | **0** | **0** | **0** | 583 | 65,536 | 99.11% | 6,598 |
| **`B5_256_tl_r1_c0`** | 1 | 256–512 | 0–256 | 51,421 | **1,539** | **0** | **1,539** | 12,576 | 65,536 | 80.81% | 17,615 |
| **`B5_256_tl_r4_c0`** | 4 | 1024–1280 | 0–256 | 56,951 | **6,920** | **0** | **6,920** | 1,665 | 65,536 | 97.46% | 23,710 |

#### Explanation of Conflicting Numbers:
In Phase 1D, two distinct tiling grids were generated:
1. **Centered Grid:** Horizontal coordinates centered across the bridge deck width (x in [97, 353] for B5). This grid captured interior delaminations, yielding 17,615 Class 2 px for `r1` and 23,710 Class 2 px for `r4`.
2. **Top-Left Grid:** Horizontal coordinates anchored at the left edge (x in [0, 256]). Because B5 has white canvas padding along its western edge, the top-left grid captures the deck boundary, resulting in 1,539 Class 2 px for `r1` and 6,920 Class 2 px for `r4`.
Both Phase 1E and Phase 1E.1 standardize strictly on the **Top-Left non-overlapping grid**. The exact measured counts are **1,539 px for `r1_c0`** and **6,920 px for `r4_c0`**.

---

## 2. Retention Rule Audit: 50% vs 80% Valid Deck Threshold

### 2.1 Complete Grid Tiles vs. Partial Boundary Edge Patches
To eliminate ambiguity, image regions are partitioned into:
1. **Complete 256 x 256 Tiles:** Fully bounded grid cells where x2 - x1 = 256 and y2 - y1 = 256.
2. **Partial Edge Patches:** Boundary strips along the right or bottom edges where raw image dimensions are not integer multiples of 256 (patch width < 256 or patch height < 256).

### 2.2 Recomputed Per-Bridge Tile Counts
| Bridge | Raw Dimensions | Complete Grid Cells | Retained Complete (>= 50%) | Retained Complete (>= 80%) | Rejected Complete | Partial Edge Patches | Partial Retained If Padded (>= 50%) | Partial Retained If Padded (>= 80%) | Recommended Primary Retained |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** (Forest River NB) | 461 x 1942 | 7 | 7 | 7 | 0 | 9 | 7 | 3 | **7** |
| **B2** (Forest River SB) | 1626 x 401 | 6 | 6 | 6 | 0 | 8 | 2 | 0 | **6** |
| **B3** (Park River Median) | 4326 x 411 | 16 | 16 | 16 | 0 | 18 | 18 | 1 | **16** |
| **B4** (Park River NB) | 2991 x 371 | 11 | 11 | 11 | 0 | 13 | 1 | 0 | **11** |
| **B5** (Park River SB) | 451 x 4174 | 16 | 16 | 16 | 0 | 18 | 14 | 0 | **16** |
| **Fleet Total** | — | **56** | **56** | **56** | **0** | **66** | **42** | **4** | **56** |

### 2.3 Primary Retention Rule Recommendation & Justification
- **Recommended Standard:** **Retain only complete 256 x 256 tiles with valid deck fraction >= 80.0%**.
- **Justification:**
  1. **Zero Complete Tile Loss:** Across all five bridges, every single one of the 56 complete grid cells has a valid deck fraction of at least **80.81%** (`B5_256_tl_r1_c0` is the lowest at 80.81%; all others are >= 88.95%, with 46 tiles at 100.0%). Therefore, the 80% threshold preserves 100% of the complete grid cells while enforcing high signal density.
  2. **Prevention of Boundary Artifacts:** The 66 partial edge patches have physical dimensions as small as 90 px or 78 px. Padding them to 256 x 256 with synthetic zero or white canvas introduces severe boundary step discontinuities that distort convolutional filters and bias domain-adaptation batch norm statistics.
  3. **Preservation of Authentic Receptive Fields:** Retaining only complete 256 x 256 tiles ensures that every pixel processed by the U-Net architecture represents an authentic physical sensor measurement.

---

## 3. Corrected Translation Diagnostic (Explicit Array Slices, No Wraparound)

### 3.1 Implementation Details
The Phase 1E translation search used `numpy.roll`, which circularly wraps boundary pixels. In Phase 1E.1, this was replaced with **explicit overlapping sub-array slices**.
- For translation (dy, dx) of `Original` relative to `Annotated`:
  - Overlap height: H - |dy|; Overlap width: W - |dx|
  - Both baseline (unshifted) and translated differences are evaluated **strictly on the identical boolean mask of sound concrete pixels within the exact same spatial window**.

### 3.2 Quantitative Results Across Bridges
| Bridge | Shift (dy, dx) | Overlap Sound Pixels | Baseline Mean Abs Diff | Translated Mean Abs Diff | Mean Diff Delta | Baseline Exact % | Translated Exact % | Exact % Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | (0, 0) | 662,477 | 0.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% | 0.00% |
| **B1** | (+2, 0) | 661,555 | 0.0000 | 1.9891 | +1.9891 | 100.00% | 29.98% | -70.02% |
| **B2** | (0, 0) | 489,219 | 0.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% | 0.00% |
| **B2** | (+2, 0) | 485,967 | 0.0000 | 4.9255 | +4.9255 | 100.00% | 22.06% | -77.94% |
| **B3** | (0, 0) | 1,380,331 | 0.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% | 0.00% |
| **B3** | (+2, 0) | 1,372,423 | 0.0000 | 5.0452 | +5.0452 | 100.00% | 18.86% | -81.14% |
| **B4** | (0, 0) | 844,908 | 2.9793 | 2.9793 | 0.0000 | 23.25% | 23.25% | 0.00% |
| **B4** | (+1, 0) | 841,917 | 2.9329 | 1.5034 | **-1.4295** | 23.24% | 53.64% | +30.40% |
| **B4** | (+2, 0) | 838,926 | 2.8867 | 0.3702 | **-2.5164** | 23.23% | **84.33%** | **+61.10%** |
| **B4** | (+3, 0) | 835,935 | 2.8407 | 1.5126 | -1.3281 | 23.23% | 53.61% | +30.38% |
| **B5** | (0, 0) | 1,409,183 | 7.8237 | 7.8237 | 0.0000 | 8.04% | 8.04% | 0.00% |
| **B5** | (+2, 0) | 1,408,312 | 7.8209 | 7.4803 | -0.3405 | 8.04% | 8.42% | +0.38% |

### 3.3 Critical Methodological Conclusion on Translation
1. On **Bridge 4**, the integer translation dy = +2 pixels achieves an **87.2% reduction in mean absolute error** (dropping from 2.8867 to 0.3702) and boosts exact sound pixel matches from 23.2% to 84.3%. This confirms that the background in `PR_NB_Annotaed.png` was shifted vertically by 2 pixels during annotation authoring.
2. **Registration Independence Policy:** Even though grayscale agreement is maximized at (dy = +2), **no translation is applied to the dataset**.
   - Input X is extracted solely from `PR_NB_Original.png`.
   - Labels Y are extracted from `PR_NB_Annotaed.png`.
   - The original physical deck features remain untouched, avoiding synthetic sub-pixel interpolation blur.

---

## 4. Recomputed Summary Statistics Across Retained Tiles

Across all 56 retained complete 256 x 256 tiles (under the >= 80% valid deck rule):

| Class / Feature | Minimum | Median | Mean | Maximum | Total Pixels Across Fleet |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Class 1 (Sound Concrete)** | 26,699 | 49,241.0 | 48,195.5 | 64,953 | 2,698,947 |
| **Class 2 (Above-Mat Delamination)** | **0** | 12,673.5 | 13,229.1 | 33,380 | 740,831 |
| **Class 3 (Below-Mat Delamination)** | **0** | 2,297.0 | 3,240.3 | 10,365 | 181,459 |
| **Total Delamination (C2 + C3)** | **0** | 15,728.0 | 16,469.5 | 38,837 | 922,290 |
| **Valid Deck Pixels** | 52,960 | 65,536.0 | 64,664.9 | 65,536 | 3,621,237 |
| **Valid Deck Fraction** | **80.81%** | 100.00% | 98.67% | 100.00% | — |

---

## 5. Verification Deliverables Inventory

The following authoritative files have been generated in `C:\Users\Dell\Documents\SDNET2021_Audit\phase1e1_reconciliation\`:
1. `phase1e1_reconciliation_report.md`: This comprehensive audit report.
2. `tile_counts_50_vs_80_percent.csv`: Complete comparison of 50% vs 80% thresholds for complete vs partial edge patches.
3. `tile_manifest_recomputed.csv`: Full manifest of all 56 complete tiles with 28 verified geometric and class attributes.
4. `b5_tile_discrepancy_check.csv`: Itemized reconciliation of Bridge 5 tiles `r0_c0`, `r1_c0`, and `r4_c0`.
5. `translation_metrics_no_wrap.csv`: Complete slice-based translation metrics for all bridges without array wraparound.
6. `audit_phase1e1.py`: The executable audit script with all programmatic assertions verified.

---

## 6. Phase 2 Dataset Construction Recommendations

1. **Standardize on Complete 256 x 256 Grid:** Use the 56 recomputed tiles as the benchmark non-overlapping evaluation set.
2. **Handle Zero-Class Tiles in Evaluation:** Metrics must support tiles with zero delamination (`B5_256_tl_r0_c0`) and zero Class 3 delaminations (`B5_256_tl_r0_c0`, `r1_c0`, `r4_c0`) without zero-division runtime errors.
3. **Training Stride Consideration:** For model training, a sliding window with 50% overlap (128-pixel stride) on training bridges can expand training volume from ~40 to ~120 tiles, while keeping test bridges strictly evaluated on the standard 56 non-overlapping tiles.