# Phase 1D: Five-Bridge Dataset Consistency Audit Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Target Dataset:** SDNET2021 Infrared Thermography (IRT) — Bridges 1 through 5  
**Audit Directory:** C:\Users\Dell\Documents\SDNET2021_Audit\phase1d_five_bridge_audit\  
**Date:** October 2026  
**Status:** Audit Completed. No models trained, no packages installed, no original files modified.

---

## 1. Executive Summary & Verdict

- **Decision:** **CONDITIONAL GO — MANAGEABLE DISCREPANCIES REQUIRE DOCUMENTED HANDLING**
- **Core Feasibility Finding:** All five bridges (B1–B5) decode without error and exhibit 100% dimensional match between original thermal and annotated image pairs. The class-to-color mapping defined in the official README is perfectly consistent across all 5 bridges with **0.0000% ambiguous fringe pixels**.
- **Delamination Prevalence:** Subsurface delamination is remarkably consistent across all five decks, averaging **20.64%** of total pixels (Class 2: 16.54%, Class 3: 4.09%, Sound: 75.76%).
- **Crucial Architectural Discovery (Tiling Constraint):**
  - **256 x 256 tiles:** Universally compatible across all 5 bridges. Produces **56 complete non-overlapping tiles**, 100% of which meet the retention threshold (>= 50% valid deck) and 100% of which contain both Class 2 and Class 3 delaminations.
  - **384 x 384 tiles:** Physically incompatible with Bridge 4 without resizing or padding (Bridge 4 height is 371 px < 384 px, yielding 0 complete tiles).

---

## 2. Five-Bridge Image Metrology & Inspection Matrix

| Bridge ID | Bridge Name | Stored Filename (Original) | Stored Filename (Annotated) | Dimensions (W x H) | Orientation | Total Pixels | Mode (Orig / Anno) | Decoding Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B1** | Forest River NB | FR_NB_Original.png | FR_NB_Annotated.png | 461 x 1,942 | Vertical | 895,262 | L (8-bit) / RGB | VERIFIED (100% Match) |
| **B2** | Forest River SB | FR_SB_Original.png | FR_SB_Annotated.png | 1,626 x 401 | Horizontal | 652,026 | L (8-bit) / RGB | VERIFIED (100% Match) |
| **B3** | Park River Median | PR_MD_Original.png | PR_MD_Annotated.png | 4,326 x 411 | Horizontal | 1,777,986 | L (8-bit) / RGB | VERIFIED (100% Match) |
| **B4** | Park River NB | PR_NB_Original.png | PR_NB_Annotaed.png *(typo)*| 2,991 x 371 | Horizontal | 1,109,661 | L (8-bit) / RGB | VERIFIED (100% Match) |
| **B5** | Park River SB | PR_SB_Original.png | PR_SB_Annotated.png | 451 x 4,174 | Vertical | 1,882,474 | L (8-bit) / RGB | VERIFIED (100% Match) |
| **Total**| **Full IRT Fleet** | **5 Image Pairs** | **5 Image Pairs** | **Varied** | **Mixed** | **6,317,409** | **Uniform dtypes** | **All 5 Bridges Decoded** |

---

## 3. Ground-Truth Semantic Reconciliation & Validity Masking

### 3.1 Certified Semantic Partitioning Rules
Each pixel is assigned to one of five mutually exclusive categories:
1. **Class 1 (Sound Concrete):** Grayscale deck pixels where R = G = B (excluding white canvas and black boundary).
2. **Class 2 (Above-Mat Delamination):** Pure Green pixels where R = 0, G = 255, B = 0.
3. **Class 3 (Below-Mat Delamination):** Pure Red pixels where R = 255, G = 0, B = 0.
4. **Excluded (Canvas Padding):** Pure White pixels where R = 255, G = 255, B = 255.
5. **Excluded (Boundary Artifacts):** Pure Black pixels where R = 0, G = 0, B = 0.
6. **Ambiguous / Unassigned:** Any non-standard color combination.

### 3.2 Five-Bridge Class Breakdown Table

| Bridge | Class 1 Sound (%) | Class 2 Above (%) | Class 3 Below (%) | Delam Total (%) | Excluded White (%) | Excluded Black (%) | Ambiguous (%) | Reconciled Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 662,477 (74.00%) | 152,149 (16.99%) | 43,488 (4.86%) | 195,637 (21.85%) | 37,131 (4.15%) | 17 (0.00%) | 0 (0.00%) | 895,262 (Delta=0) |
| **B2** | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 131,942 (20.24%) | 30,803 (4.72%) | 62 (0.01%) | 0 (0.00%) | 652,026 (Delta=0) |
| **B3** | 1,380,331 (77.63%) | 298,123 (16.77%) | 78,873 (4.44%) | 376,996 (21.20%) | 421 (0.02%) | 20,238 (1.14%)| 0 (0.00%) | 1,777,986 (Delta=0) |
| **B4** | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 264,753 (23.86%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 1,109,661 (Delta=0) |
| **B5** | 1,409,183 (74.86%) | 272,292 (14.46%) | 62,113 (3.30%) | 334,405 (17.76%) | 138,855 (7.38%)| 31 (0.00%) | 0 (0.00%) | 1,882,474 (Delta=0) |
| **Fleet**| **4,786,118 (75.76%)**| **1,045,173 (16.54%)**| **258,560 (4.09%)**| **1,303,733 (20.64%)**| **207,210 (3.28%)**| **20,348 (0.32%)**| **0 (0.00%)**| **6,317,409 (Delta=0)** |

---

## 4. Annotation-to-Image Correspondence Audit

Grayscale intensities of unannotated sound concrete were compared between the Original.png and Annotated.png files to determine whether annotations were drawn directly on the original raster:

| Bridge | Sound Pixels Audited | Exact Match Pixels | Exact Match % | Mean Abs Diff | Max Abs Diff | Root Cause Analysis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | 662,477 | 662,477 | **100.00%** | 0.00 | 0 | Perfect bit-for-bit identical background. |
| **B2** | 489,219 | 489,219 | **100.00%** | 0.00 | 0 | Perfect bit-for-bit identical background. |
| **B3** | 1,380,331 | 1,380,331 | **100.00%** | 0.00 | 0 | Perfect bit-for-bit identical background. |
| **B4** | 844,908 | 196,451 | **23.25%** | 2.98 | 107 | Sub-pixel 1-px vertical shift during export (corr = 0.9442, median diff = 0.0). |
| **B5** | 1,409,183 | 113,260 | **8.04%** | 7.82 | 248 | Edge anti-aliasing against canvas padding (corr = 0.9439, median diff = 4.0; 95% diff <= 24). |

### Metrological Takeaway:
To avoid contrast or export bias, the machine learning pipeline must feed Original.png as input tensor X and use Annotated.png strictly to derive target labels Y. Under no circumstances should input features be extracted from the annotated image.

---

## 5. Tiling Feasibility & Fleet Yield Evaluation

### 5.1 Non-Overlapping 256 x 256 Grid (Top-Left Anchored)

| Bridge | Width x Height | Grid Layout (cols x rows) | Complete Tiles | Retained Tiles (>=50% deck) | Rejected Tiles | Tiles with Class 2 | Tiles with Class 3 | Uncovered Deck Area % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 461 x 1,942 | 1 x 7 | 7 | 7 (100%) | 0 | 7 (100%) | 7 (100%) | 48.76% |
| **B2** | 1,626 x 401 | 6 x 1 | 6 | 6 (100%) | 0 | 6 (100%) | 6 (100%) | 39.69% |
| **B3** | 4,326 x 411 | 16 x 1 | 16 | 16 (100%) | 0 | 16 (100%) | 16 (100%) | 41.02% |
| **B4** | 2,991 x 371 | 11 x 1 | 11 | 11 (100%) | 0 | 11 (100%) | 11 (100%) | 35.03% |
| **B5** | 451 x 4,174 | 1 x 16 | 16 | 16 (100%) | 0 | 16 (100%) | 16 (100%) | 44.30% |
| **Total**| — | — | **56** | **56 (100%)** | **0** | **56 (100%)** | **56 (100%)** | **Mean = 41.76%** |

### 5.2 Non-Overlapping 384 x 384 Grid (Top-Left Anchored)

| Bridge | Width x Height | Grid Layout (cols x rows) | Complete Tiles | Retained Tiles (>=50% deck) | Rejected Tiles | Tiles with Class 2 | Tiles with Class 3 | Uncovered Deck Area % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 461 x 1,942 | 1 x 5 | 5 | 5 (100%) | 0 | 5 (100%) | 5 (100%) | 17.65% |
| **B2** | 1,626 x 401 | 4 x 1 | 4 | 4 (100%) | 0 | 4 (100%) | 4 (100%) | 9.54% |
| **B3** | 4,326 x 411 | 11 x 1 | 11 | 11 (100%) | 0 | 11 (100%) | 11 (100%) | 8.77% |
| **B4** | 2,991 x 371 | 7 x 0 | **0** | **0 (BLOCKED)** | 0 | 0 | 0 | **100.00% (FAILED)**|
| **B5** | 451 x 4,174 | 1 x 10 | 10 | 10 (100%) | 0 | 10 (100%) | 10 (100%) | 21.67% |
| **Total**| — | — | **30** | **30** | **0** | **30** | **30** | **Incompatible with B4**|

---

## 6. Documented Handling for Conditional Discrepancies

To transition from CONDITIONAL GO to complete dataset pipeline execution, three documented handling rules are established:

1. **Mandatory Adoption of 256 x 256 Resolution:** Because Bridge 4 height is 371 px, standard unpadded 384 x 384 tiles cannot be extracted. The dataset pipeline must standardize on **256 x 256 patches** to guarantee uniform, unpadded extraction across all five bridges (yielding 56 high-quality tiles).
2. **Decoupled Input/Label Provenance:** Model input X must always be read from Original.png (8-bit grayscale), while target mask Y must be extracted from Annotated.png (color palette thresholding).
3. **Bridge Deck Orientation Normalization:** Bridges B1 and B5 are vertically oriented ( \gg W$), while B2, B3, and B4 are horizontally oriented ( \gg H$). During dataset construction, horizontal bridges must be rotated 90 degrees or extracted along their longitudinal axis so convolutional kernels observe a consistent transverse/longitudinal aspect ratio.

---

## 7. Artifact Manifest in phase1d_five_bridge_audit/

- ive_bridge_summary.csv: Machine-readable master metrics table for all 5 bridges.
- ive_bridge_all_tiles.csv: Comprehensive manifest detailing all 172 evaluated tile candidates.
- 1_tiles_manifest.csv through 5_tiles_manifest.csv: Per-bridge CSV manifests across all 4 tiling configurations.
- 1_diagnostic.png through 5_diagnostic.png: 4-panel visual verification images per bridge.
- phase1d_five_bridge_audit.md: This comprehensive audit report.
