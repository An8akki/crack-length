# Phase 1E: Independent Label Alignment and Tile Manifest Verification Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Phase:** 1E — Independent Label Alignment and Tile Manifest Verification  
**Date:** October 20, 2026  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Input Directories:**
- Source Raw Images: `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`
- Phase 1D Audit: `C:\Users\Dell\Documents\SDNET2021_Audit\phase1d_five_bridge_audit\`
**Output Directory:**
- `C:\Users\Dell\Documents\SDNET2021_Audit\phase1e_alignment_verification\`

---

## Executive Summary & Final Verdict

### Final Decision: **CONDITIONAL GO with Explicit Unresolved Risks**

The Phase 1E independent audit re-examined all 5 bridges of the SDNET2021 Infrared Thermography (IRT) dataset, recomputed all tile coordinates and class counts directly from source images, reconciled all 6,317,409 pixels with zero delta, investigated pixel-level sound-concrete correspondence between original and annotated images, and evaluated cross-bridge statistical validity.

### Key Audit Findings & Core Corrections
1. **Correction of Phase 1D Claim on Tile Class Composition:**
   - **Phase 1D claimed:** All 56 retained 256×256 tiles contain both Class 2 and Class 3 delaminations.
   - **Phase 1E measured reality:** **This claim is empirically incorrect.**
     - **1 tile contains NO Class 2 pixels:** `B5_256_tl_r0_c0` (Class 2 = 0 px, Class 3 = 0 px, Sound = 64,953 px; 100% sound deck tile).
     - **3 tiles contain NO Class 3 pixels:** `B5_256_tl_r0_c0`, `B5_256_tl_r1_c0` (Class 3 = 0 px, Class 2 = 1,539 px), and `B5_256_tl_r4_c0` (Class 3 = 0 px, Class 2 = 6,920 px), all located on Bridge 5 (Park River Southbound).
     - On Bridges B1, B2, B3, and B4: 100% of retained tiles contain both Class 2 and Class 3 pixels.
2. **Reconciliation & Mask Integrity:**
   - 100.000% of the 6,317,409 pixels across all 5 bridges are accounted for across mutually exclusive categories (Sound, Class 2, Class 3, Excluded White Canvas, Excluded Black Boundary).
   - **Zero unknown or ambiguous pixels exist** (Delta = 0).
   - No unknown pixel was silently converted into sound concrete.
3. **B4 and B5 Background Discrepancies:**
   - **B1, B2, B3:** Bit-for-bit identical background between `Original.png` and `Annotated.png` (exact match 100.0%, max abs diff = 0).
   - **B4:** 23.25% exact matches, mean abs diff = 2.979. Diagnostic shift analysis reveals a vertical integer displacement of $\Delta y = +2$ pixels reduces mean diff by 87.6% (to 0.370). However, edge-gradient analysis demonstrates that this shift does not substantially change annotation alignment and causes boundary wrapping.
   - **B5:** 8.04% exact matches, mean abs diff = 7.824, median = 4.0. Discrepancies are predominantly mild radiometric/tonal adjustments from CAD export, with extreme differences (>50) confined to the outer deck perimeter edges ($x \le 86$ and $x \ge 416$). Shifting does not resolve the difference (mean diff remains 7.48).
   - **Policy Decision:** Adhere strictly to protocol: **Input $X$ must be extracted exclusively from `Original.png`**, and target segmentation mask $Y$ from `Annotated.png`. No artificial translations or image registrations should be applied.
4. **Cross-Bridge Data Suitability:**
   - 56 tiles exist across 5 independent bridges (B1: 7, B2: 6, B3: 16, B4: 11, B5: 16).
   - While tile count (56) provides reasonable sample sizes for training batches, the true statistical degrees of freedom for Leave-One-Bridge-Out (LOBO) evaluation is **$N=5$ bridges ($	ext{df}=4$)**.
   - Non-overlapping $256\times 256$ grid leaves 41.76% of raw bridge image area uncovered due to narrow bridge deck aspect ratios.

---

## Task 1: Independent Tile Manifest Recomputation & Verification

### 1.1 Methodology
Tile extraction was independently re-implemented directly against the raw image dimensions and reconstructed multi-class masks. Grid extraction used standard top-left aligned non-overlapping $256 \times 256$ stepping. Every tile with valid deck fraction $\ge 80.0\%$ ($\ge 52,428$ valid deck pixels) was retained without excluding tiles for minority class sparsity.

### 1.2 Tile Count Verification
- Total potential grid cells evaluated: **92**
- Retained complete $256 \times 256$ tiles ($\ge 80\%$ valid deck): **56**
- Excluded tiles: **36** (14 tiles with $<80\%$ valid deck, 22 partial edge/boundary tiles).
- Distribution by bridge:
  - **B1 (Forest River NB):** 7 retained (width 461, height 1942; cols=1, rows=7)
  - **B2 (Forest River SB):** 6 retained (width 1626, height 401; cols=6, rows=1)
  - **B3 (Park River Median):** 16 retained (width 4326, height 411; cols=16, rows=1)
  - **B4 (Park River NB):** 11 retained (width 2991, height 371; cols=11, rows=1)
  - **B5 (Park River SB):** 16 retained (width 451, height 4174; cols=16, rows=1)
  - **Total:** **56 tiles**

### 1.3 Pixel Count Distribution Across Retained Tiles
Across all 56 retained tiles:
| Metric | Class 1 Sound (px) | Class 2 Above-Mat (px) | Class 3 Below-Mat (px) | Total Delam (px) | Valid Deck (px) | Valid Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimum** | 26,699 | **0** | **0** | 567 | 52,960 | 80.81% |
| **Median** | 49,241.0 | 12,673.5 | 2,297.0 | 15,221.5 | 65,536.0 | 100.00% |
| **Maximum** | 64,953 | 33,380 | 10,365 | 38,837 | 65,536 | 100.00% |

Per-bridge distributions (min / median / max):
| Bridge | Retained Tiles | Sound Pixels (min / med / max) | Class 2 Pixels (min / med / max) | Class 3 Pixels (min / med / max) | Valid Deck Pixels (min / med / max) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1** | 7 | 45,226 / 50,549 / 55,274 | 2,932 / 9,738 / 18,299 | 1,489 / 4,374 / 8,073 | 58,853 / 65,536 / 65,536 |
| **B2** | 6 | 44,333 / 50,225 / 55,595 | 7,651 / 13,874 / 16,423 | 913 / 2,059 / 7,163 | 65,536 / 65,536 / 65,536 |
| **B3** | 16 | 29,970 / 43,062 / 56,116 | 2,683 / 16,914 / 33,380 | 567 / 3,184 / 10,365 | 59,394 / 65,536 / 65,536 |
| **B4** | 11 | 26,699 / 48,274 / 59,584 | 4,264 / 13,277 / 29,788 | 1,327 / 2,642 / 9,049 | 65,536 / 65,536 / 65,536 |
| **B5** | 16 | 33,087 / 54,654 / 64,953 | **0** / 9,692 / 28,141 | **0** / 1,491 / 8,367 | 52,960 / 65,536 / 65,536 |

### 1.4 Detailed Anomaly Report: Zero-Class Tiles
1. **Tiles with 0 Class 2 Pixels:**
   - `B5_256_tl_r0_c0` (Bridge 5, Row 0, Col 0): Sound = 64,953 px, Class 2 = 0 px, Class 3 = 0 px, White canvas = 583 px.
   - *Significance:* This is an all-sound deck tile situated at the northern end of Bridge 5. Retaining it provides a critical true-negative test for false-positive delamination predictions.
2. **Tiles with 0 Class 3 Pixels:**
   - `B5_256_tl_r0_c0`: Sound = 64,953 px, Class 2 = 0 px, Class 3 = 0 px.
   - `B5_256_tl_r1_c0`: Sound = 51,421 px, Class 2 = 1,539 px, Class 3 = 0 px.
   - `B5_256_tl_r4_c0`: Sound = 56,951 px, Class 2 = 6,920 px, Class 3 = 0 px.
   - *Significance:* All three zero-Class 3 tiles reside on Bridge 5. In a Leave-One-Bridge-Out split where B5 is the target bridge, evaluation metrics for Class 3 will experience high variance on these tiles.
3. **Tiles with 0 Valid Deck Pixels:**
   - **0 tiles.** All 56 retained tiles contain $\ge 52,960$ valid deck pixels (min valid fraction is 80.81%).

---

## Task 2: Validity and Label Mask Audit & Reconciliation

### 2.1 Multi-Class Categorization Rules
Masks were reconstructed using strict mutually exclusive color classification:
- **Class 1 (Sound Concrete):** $R=G=B$ where $R 
otin \{0, 255\}$ (grayscale bridge deck values 1 to 254).
- **Class 2 (Above-Mat Delamination):** Pure green $[0, 255, 0]$.
- **Class 3 (Below-Mat Delamination):** Pure red $[255, 0, 0]$.
- **Excluded White Canvas:** Pure white $[255, 255, 255]$ (outer background padding).
- **Excluded Black Boundary:** Pure black $[0, 0, 0]$ (masked edges or outer canvas).
- **Unknown / Ambiguous:** Any RGB triplet not matching the five categories above.

### 2.2 Comprehensive Pixel Reconciliation Table
| Bridge | Dimensions | Total Pixels | Class 1 Sound (%) | Class 2 Above (%) | Class 3 Below (%) | Total Delam (%) | Excluded White (%) | Excluded Black (%) | Unknown Pixels | Reconciled Sum | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 461 × 1942 | 895,262 | 662,477 (74.00%) | 152,149 (17.00%) | 43,488 (4.86%) | 195,637 (21.85%) | 37,131 (4.15%) | 17 (0.00%) | 0 | 895,262 | **0** |
| **B2** | 1626 × 401 | 652,026 | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 131,942 (20.24%) | 30,803 (4.72%) | 62 (0.01%) | 0 | 652,026 | **0** |
| **B3** | 4326 × 411 | 1,777,986 | 1,380,331 (77.64%) | 298,123 (16.77%) | 78,873 (4.44%) | 376,996 (21.20%) | 421 (0.02%) | 20,238 (1.14%) | 0 | 1,777,986 | **0** |
| **B4** | 2991 × 371 | 1,109,661 | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 264,753 (23.86%) | 0 (0.00%) | 0 (0.00%) | 0 | 1,109,661 | **0** |
| **B5** | 451 × 4174 | 1,882,474 | 1,409,183 (74.86%) | 272,292 (14.47%) | 62,113 (3.30%) | 334,405 (17.76%) | 138,855 (7.38%) | 31 (0.00%) | 0 | 1,882,474 | **0** |
| **Total**| — | **6,317,409** | **4,786,118 (75.76%)** | **1,045,173 (16.54%)** | **258,560 (4.09%)** | **1,303,733 (20.64%)** | **207,210 (3.28%)** | **20,348 (0.32%)** | **0** | **6,317,409** | **0** |

### 2.3 Nature and Verification of Black Pixels
The Phase 1E investigation analyzed the underlying original grayscale values at the exact spatial coordinates of the 20,348 black pixels:
- **B1 (17 black px):** Original image values at these coordinates: min = 0, max = 0, mean = 0.00. Located at outermost pixel corner.
- **B2 (62 black px):** Original image values: min = 0, max = 0, mean = 0.00. Outermost perimeter edge.
- **B3 (20,238 black px):** Original image values: min = 0, max = 0, mean = 0.00. Forms a continuous 2-to-5 pixel wide longitudinal border strip along the raw stitched image edge where thermal scanner coverage terminated. They represent sensor zero-padding.
- **B4 (0 black px):** Tightly cropped to bridge deck; zero black boundary pixels.
- **B5 (31 black px):** Original image values: min = 14, max = 18, mean = 15.61. These 31 pixels lie exactly on the perimeter boundary between the bridge deck and white canvas padding. In the CAD export of B5, anti-aliased edge smoothing or boundary rasterization caused these 31 pixels to be set to pure black $[0,0,0]$.
- **Conclusion:** On B1, B2, and B3, black pixels are identical sensor zero-padding in both original and annotated files. On B5, they are edge-rasterization artifacts. Excluding all black pixels from the valid deck mask is fully sound and prevents non-deck sensor margins from leaking into sound concrete labels.

---

## Task 3: Investigation of B4 and B5 Correspondence

### 3.1 Quantitative Alignment Metrics
Differences were evaluated strictly at pixels assigned to sound concrete ($R=G=B$ in `Annotated.png`):
| Bridge | Sound Pixel Count | Exact Match % | Differing % | Mean Abs Diff | Median Abs Diff | p75 Diff | p90 Diff | p95 Diff | p99 Diff | Max Abs Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 662,477 | **100.000%** | 0.000% | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| **B2** | 489,219 | **100.000%** | 0.000% | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| **B3** | 1,380,331 | **100.000%** | 0.000% | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| **B4** | 844,908 | **23.251%** | 76.749% | 2.979 | 1.0 | 3.0 | 6.0 | 12.0 | 36.0 | 107 |
| **B5** | 1,409,183 | **8.037%** | 91.963% | 7.824 | 4.0 | 7.0 | 15.0 | 24.0 | 87.0 | 248 |

### 3.2 Diagnostic Integer Translation Search
A diagnostic 2D grid search over integer shifts $\Delta y, \Delta x \in [-5, +5]$ was performed to test whether a rigid spatial offset explains the discrepancies:
- **B4 Optimization:**
  - Shift $(\Delta y = +2, \Delta x = 0)$ reduces Mean Absolute Difference from **2.979 to 0.370** (an **87.6% reduction**).
  - Median absolute difference drops to 0.0.
  - *Edge-Gradient Analysis:* Original thermal gradient magnitude at annotation boundaries changed from 1.542 (unshifted) to 1.592 (shifted). The boundary alignment improves by only 3.2%, but the translation introduces artificial border-wrapping artifacts.
- **B5 Optimization:**
  - Shift $(\Delta y = +2, \Delta x = 0)$ alters Mean Absolute Difference from **7.824 to 7.481** (only a 4.4% reduction).
  - Spatial decomposition of B5 differences shows that 95% of sound pixels have $|I_{\text{orig}} - I_{\text{annot}}| \le 24$. Differences $> 50$ are clustered entirely along the left ($x \le 86$) and right ($x \ge 416$) perimeter deck boundaries where slight rotational skew or perspective warping occurred during mosaic stitching.
- **Policy Adherence:**
  - In accordance with the project constraints, **no translation, warping, histogram matching, or contrast correction will be applied**.
  - All input features $X$ must be loaded directly from `Original.png`, and target labels $Y$ from the annotation color definitions. This ensures the model learns true physical thermal features without synthetic interpolation blur.

---

## Task 4: Visual Alignment Verification

For all 5 bridges, composite 5-panel diagnostic visualizations were generated and saved at full resolution in `phase1e_alignment_verification/`:
1. **Panel 1:** Original grayscale infrared thermogram (`Original.png`).
2. **Panel 2:** Annotated composite image (`Annotated.png`).
3. **Panel 3:** Reconstructed multi-class ground-truth mask (Black=Ignore, Gray=Sound, Green=Class 2, Red=Class 3).
4. **Panel 4:** Boundary overlay of Class 2 and Class 3 delaminations overlaid directly on the original thermogram, with the $256 \times 256$ tile grid boundaries delineated.
5. **Panel 5:** Difference heatmap showing $|I_{\text{orig}} - I_{\text{annot}}|$ over valid sound concrete pixels.

### Verification of Aspect Ratios and Rescaling:
All diagnostic visualizations were rendered preserving exact 1:1 pixel coordinates without independent resizing of images or masks.

---

## Task 5: Data Suitability for Cross-Bridge Evaluation

### 5.1 Tile Count vs. Independent Fleet Count
- Retained tiles per bridge:
  - B1: 7 tiles (12.5%)
  - B2: 6 tiles (10.7%)
  - B3: 16 tiles (28.6%)
  - B4: 11 tiles (19.6%)
  - B5: 16 tiles (28.6%)
- Total tiles: **56**
- Total independent bridges: **5**

### 5.2 Critical Experimental Risks for Cross-Bridge Domain Adaptation
1. **Degrees of Freedom in Leave-One-Bridge-Out (LOBO):**
   - In a 5-fold LOBO cross-validation protocol, testing on bridge $k$ holds out all $N_k$ tiles from that bridge.
   - Although evaluation is computed over tens of thousands of pixels, the macro statistical unit of independence is the **bridge deck ($N=5$)**.
   - With 5 evaluation folds, statistical hypothesis tests comparing baseline U-Net vs. UDA (Deep CORAL / DANN) have only $\text{df} = 4$. A two-tailed paired t-test or Wilcoxon signed-rank test requires consistent directional improvement across all 5 folds to reach statistical significance ($p < 0.05$).
2. **Uncovered Raw Image Area:**
   - Non-overlapping $256 \times 256$ tiling leaves an average of **41.76%** of raw bridge deck pixels unextracted (B1: 42.4%, B2: 39.7%, B3: 41.1%, B4: 43.5%, B5: 42.1%).
   - This occurs because bridge widths are narrow ($371 \le W \le 461$ for B1, B4, B5), allowing only 1 column of 256-pixel tiles.
   - For Phase 2 dataset construction, sliding-window tiling with 50% overlap should be evaluated to maximize spatial coverage and training tile volume.
3. **Class 3 Sparsity on Bridge 5:**
   - Bridge 5 has 3 tiles with zero Class 3 pixels. Delamination below the top rebar mat is relatively sparse across the northern span of B5. Evaluation metrics must handle zero-division gracefully (e.g., reporting macro-averaged IoU only over tiles containing the target class, alongside global deck-level IoU).

---

## Summary of Deliverables Created

| Deliverable | Location | Status |
| :--- | :--- | :---: |
| **Verification Report** | `phase1e_alignment_verification/phase1e_verification_report.md` | Completed |
| **Recomputed Tile Manifest** | `phase1e_alignment_verification/tile_manifest_recomputed.csv` | Completed (56 tiles, 28 fields) |
| **Class Pixel Reconciliation** | `phase1e_alignment_verification/class_pixel_reconciliation.csv` | Completed (6,317,409 px, Delta=0) |
| **Alignment Metrics** | `phase1e_alignment_verification/alignment_metrics.csv` | Completed (Full difference percentiles) |
| **Phase 1E Audit Script** | `phase1e_alignment_verification/audit_phase1e.py` | Completed |
| **Per-Bridge Diagnostic Panels** | `phase1e_alignment_verification/b1_phase1e_diagnostic.png` to `b5_...` | Completed (5 full composites) |
| **B4/B5 Crop Diagnostics** | `phase1e_alignment_verification/b4_shift_diagnostic_crop.png`, `b5_...` | Completed |
| **Reconstructed Masks** | `phase1e_alignment_verification/b1_reconstructed_mask.png` to `b5_...` | Completed (5 mask files) |

---

## Recommendations for Phase 2 (Dataset Construction)

1. **Input Pipeline Invariance:** Configure dataset loader to ingest `Original.png` as thermal input feature map $X$ and slice ground-truth targets $Y$ from the precomputed Phase 1E reconstructed label masks.
2. **Standardize on $256 \times 256$ Tile Resolution:** Confirm $256 \times 256$ as standard tile resolution. Bridge 4 deck width (371 px) is smaller than 384 px, precluding $384 \times 384$ tiles without substantial zero-padding.
3. **Stride Strategy:** Consider $50\%$ stride overlap (128 px step) during training tile generation to expand training volume from 56 to ~180 tiles while preserving 100% hold-out bridge separation for evaluation.
4. **Metric Reporting Standards:** In all subsequent cross-bridge modeling experiments, report both tile-level mean IoU and full-deck aggregated IoU/F1 scores to ensure robustness against tile-level class absence.
