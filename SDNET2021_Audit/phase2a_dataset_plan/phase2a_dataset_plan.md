# Phase 2A — SDNET2021 Dataset Construction and Evaluation Protocol

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** October 20, 2026  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Source Dataset:** `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`  
**Output Directory:** `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a_dataset_plan\`  

---

## Executive Summary & Final Verdict

### Final Decision: **GO for Dataset Construction Under Standardized Protocol**

**Key Decisions & Verified Facts:**
1. **Dataset Construction Feasibility:** Fully verified. All 6,317,409 pixels across five in-service bridge decks decode with 100% dimensional match and zero unknown pixels. Mutually exclusive class masks have been established.
2. **Patch Strategy Selection:**
   - **Strategy A ($256 \times 256$ Non-Overlapping, 56 patches):** Codified as the **Primary Evaluation Benchmark** and baseline model training set. Every test pixel is evaluated exactly once without double-counting.
   - **Strategy B ($384 \times 384$ Non-Overlapping, 30 patches):** **REJECTED.** Bridge 4 physical deck height is 371 pixels ($371 < 384$), yielding exactly **0 complete patches**. Adopting 384-pixel patches would preclude 5-bridge cross-domain evaluation.
   - **Strategy C ($256 \times 256$, Stride 128, 197 retained patches):** **RECOMMENDED for Source-Bridge Training Expansion Only.** Increases source training volume by 3.5× (expanding valid deck training coverage from ~60% to ~90%) while holding out the target bridge strictly on non-overlapping Strategy A benchmark tiles.
3. **Leave-One-Bridge-Out (LOBO) Protocol Codified:** Five independent evaluation folds are established. In each fold, the target bridge is 100% held out. A complete independent source bridge is held out for source validation, eliminating spatial adjacency leakage.
4. **Strict Treatment of Bridge 4 Alignment Status:** Bridge 4's spatial alignment remains documented as **UNRESOLVED**. In accordance with research integrity, no image translation or registration warping is applied. Input $X$ is loaded from `PR_NB_Original.png` and ground-truth $Y$ from `PR_NB_Annotaed.png`.

---

## 1. Re-Audit of Source Image Pairs and Pixel Accounting

### 1.1 Source Array Verification
All five original and annotated PNG pairs were read directly from source storage. All dimensions, pillow channel modes, and pixel counts reconciled identically:

| Bridge ID | Bridge Name | Orientation | Dimensions ($W \times H$) | Pillow Modes | Total Pixels | Class 1 Sound (%) | Class 2 Above (%) | Class 3 Below (%) | Delam Total (%) | Excluded White (%) | Excluded Black (%) | Unknown Pixels |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Forest River Northbound | Vertical | 461 × 1942 | L / RGB | 895,262 | 662,477 (74.00%) | 152,149 (17.00%) | 43,488 (4.86%) | 195,637 (21.85%) | 37,131 (4.15%) | 17 (0.00%) | **0** |
| **B2** | Forest River Southbound | Horizontal | 1626 × 401 | L / RGB | 652,026 | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 131,942 (20.24%) | 30,803 (4.72%) | 62 (0.01%) | **0** |
| **B3** | Park River Median | Horizontal | 4326 × 411 | L / RGB | 1,777,986 | 1,380,331 (77.64%) | 298,123 (16.77%) | 78,873 (4.44%) | 376,996 (21.20%) | 421 (0.02%) | 20,238 (1.14%) | **0** |
| **B4** | Park River Northbound | Horizontal | 2991 × 371 | L / RGB | 1,109,661 | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 264,753 (23.86%) | 0 (0.00%) | 0 (0.00%) | **0** |
| **B5** | Park River Southbound | Vertical | 451 × 4174 | L / RGB | 1,882,474 | 1,409,183 (74.86%) | 272,292 (14.47%) | 62,113 (3.30%) | 334,405 (17.76%) | 138,855 (7.38%) | 31 (0.00%) | **0** |
| **Fleet** | **Five Bridge Fleet** | — | — | — | **6,317,409** | **4,786,118 (75.76%)** | **1,045,173 (16.54%)** | **258,560 (4.09%)** | **1,303,733 (20.64%)** | **207,210 (3.28%)** | **20,348 (0.32%)** | **0** |

### 1.2 Nature of Original Grayscale Intensities
- **Critical Scientific Note:** The original thermograms (`Original.png`) are 8-bit single-channel grayscale raster arrays ($0 \le I \le 255$). They represent relative radiant infrared energy mapped via camera color palettes (e.g., FLIR ironbow or thermal grayscale lookup).
- Without raw camera radiometric calibration metadata, atmospheric compensation curves, or surface emissivity logs, pixel intensity values **must not be interpreted as absolute thermodynamic temperature measurements ($^\circ\text{C}$)**.

---

## 2. Segmentation Target Representation & Encoding

### 2.1 Integer Class Mapping
The ground-truth segmentation masks are formalized as single-channel `uint8` tensors:
- **`0` — Sound Concrete:** Grayscale pixels where $R=G=B$ and $1 \le R \le 254$. Represents intact bridge deck concrete.
- **`1` — Class 2 Delamination:** Pure green pixels `[0, 255, 0]`. Subsurface delamination above or at the top reinforcing mat.
- **`2` — Class 3 Delamination:** Pure red pixels `[255, 0, 0]`. Deep subsurface delamination below the top reinforcing mat.
- **`255` — Ignored / Excluded Pixels:**
  - Pure white pixels `[255, 255, 255]` (canvas padding outside physical bridge deck).
  - Pure black pixels `[0, 0, 0]` (scanner zero-padding along mosaic borders and CAD boundary clamping).
  - Any unassigned / ambiguous pixel (currently 0 across fleet).

### 2.2 Loss Masking Rule
All training objective functions (Cross-Entropy, Focal Loss, Dice Loss) and all evaluation metrics must mask out index `255` using standard PyTorch `ignore_index = 255`. Excluded pixels will never contribute to gradient updates or evaluation denominators.

---

## 3. Patch Extraction Strategy Comparison

Documented in [`patch_strategy_comparison.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/patch_strategy_comparison.csv) and [`coverage_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/coverage_by_bridge.csv):

| Evaluation Feature | Strategy A (256×256 Non-Overlap) | Strategy B (384×384 Non-Overlap) | Strategy C (256×256 Stride 128) |
| :--- | :---: | :---: | :---: |
| **Tile Resolution** | 256 × 256 | 384 × 384 | 256 × 256 |
| **Stride** | 256 px (0% overlap) | 384 px (0% overlap) | 128 px (50% overlap) |
| **Fleet Total Retained ($\ge 80\%$)** | **56 patches** | **30 patches** | **197 patches** |
| **B1 Patches** | 7 | 5 | 27 |
| **B2 Patches** | 6 | 4 | 22 |
| **B3 Patches** | 16 | 11 | 64 |
| **B4 Patches** | **11** | **0 (Height 371 < 384)** | **22** |
| **B5 Patches** | 16 | 10 | 62 |
| **Patches with 0 Delamination** | 1 (`B5_r0_c0`) | 0 | 2 |
| **Patches with 0 Class 3** | 3 (all on B5) | 0 | 5 (1 on B4, 4 on B5) |
| **Valid Deck Fleet Coverage** | 59.27% | 40.85% (B4: 0.0%) | **84.82%** (B1: 82.7%, B2: 94.2%, B3: 91.6%, B4: 67.9%, B5: 87.6%) |
| **Suitability for Training** | Baseline only (small sample) | **Infeasible** | **Recommended for Source Training** |
| **Suitability for Evaluation** | **Recommended Benchmark** | Unsuitable | Prohibited (Double-counts test pixels) |

### 3.1 Failure Analysis of Strategy B (384×384)
Bridge 4 has dimensions $W = 2991, H = 371$. Because $H = 371 < 384$, zero complete $384 \times 384$ patches can be extracted. Applying Strategy B would force zero-padding 13 pixels across the entire bridge deck or resizing/distorting physical aspect ratios. More critically, holding out Bridge 4 in LOBO would leave zero test tiles. Strategy B is therefore **definitively rejected**.

### 3.2 Protocol Formulation: Dual-Grid Paradigm
- **Evaluation Standard:** Strictly standardized on **Strategy A (56 non-overlapping tiles)**. This ensures every test pixel is evaluated once, preserving 1:1 ground-truth correspondence.
- **Training Expansion:** When training models on source bridges, **Strategy C (197 tiles)** may be utilized to provide 3.5× more training samples and expand physical spatial coverage from ~59% to ~85%.

---

## 4. Five-Fold Leave-One-Bridge-Out (LOBO) Protocol

Documented in [`lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/lobo_fold_manifest.csv):

| Fold ID | Target Bridge (Test) | Source Bridge Pool | Source Train Bridges | Source Val Bridge | Target Test Tiles (Strat A) | Source Train Tiles (Strat A) | Source Train Tiles (Strat C) | Source Val Tiles (Strat A / C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **B1** (Forest River NB) | B2, B3, B4, B5 | B3, B4, B5 | **B2** | **7** | 43 | 148 | 6 / 22 |
| **Fold 2** | **B2** (Forest River SB) | B1, B3, B4, B5 | B3, B4, B5 | **B1** | **6** | 43 | 148 | 7 / 27 |
| **Fold 3** | **B3** (Park River Median) | B1, B2, B4, B5 | B1, B2, B5 | **B4** | **16** | 29 | 111 | 11 / 22 |
| **Fold 4** | **B4** (Park River NB) | B1, B2, B3, B5 | B1, B2, B5 | **B3** | **11** | 29 | 111 | 16 / 64 |
| **Fold 5** | **B5** (Park River SB) | B1, B2, B3, B4 | B1, B2, B3 | **B4** | **16** | 33 | 113 | 11 / 22 |

### 4.1 Strict Spatial Data Leakage Safeguards
1. **Bridge-Level Isolation:** In every fold, validation data is formed by holding out an entire **independent source bridge**. This ensures validation loss evaluates true cross-bridge domain transfer rather than memorization of adjacent spatial patches.
2. **Complete Target Sealing:** All target bridge pixels and labels are strictly isolated from training, normalization calculation, hyperparameter tuning, and early-stopping decisions.
3. **Unsupervised Domain Adaptation (UDA) Rules:**
   - Allowed target inputs during training: Unlabeled patches from `Original.png` of the target bridge (used solely for domain-invariance objectives, e.g., CORAL covariance loss or DANN discriminator).
   - Target labels are **sealed** in cold storage and loaded solely by the offline evaluation script after model weights are frozen.

---

## 5. Evaluation Policy & Zero-Division Standards

### 5.1 Primary Evaluation Metric: Full-Deck Confusion Matrix Aggregation
To prevent small-tile variance and zero-division distortion, principal evaluation metrics must be computed by **globally aggregating all predictions across the target bridge evaluation set** before calculating ratios:

$$\text{IoU}_c = \frac{C_{cc}}{\sum_j C_{cj} + \sum_i C_{ic} - C_{cc}}, \quad \text{Precision}_c = \frac{C_{cc}}{\sum_i C_{ic}}, \quad \text{Recall}_c = \frac{C_{cc}}{\sum_j C_{cj}}, \quad \text{F1}_c = \frac{2 P_c R_c}{P_c + R_c}$$

Because every target bridge contains thousands of Class 2 and Class 3 pixels (B1: 195k, B2: 132k, B3: 377k, B4: 265k, B5: 334k), **deck-level denominators are never zero**.

### 5.2 Zero-Division Policy for Tile-Level Secondary Analysis
When reporting secondary tile-level distributions:
- If $N_{\text{true}} = 0$ and $N_{\text{pred}} = 0$: IoU is mathematically undefined ($0/0$). **It must NOT be reported as 1.0 (100%)**, as awarding 1.0 rewards an empty prediction and heavily inflates macro averages. It must be logged as `NaN` and excluded from the tile-averaged mean IoU.
- If $N_{\text{true}} = 0$ and $N_{\text{pred}} > 0$: $\text{Precision} = 0.0$, $\text{Recall} = \text{NaN}$, $\text{IoU} = 0.0$.

### 5.3 Test Prediction Stitching
To evaluate full-bridge segmentation without double-counting:
- Standard benchmark scoring is performed on the 56 non-overlapping Strategy A tiles.
- For full-deck raster visualization, adjacent patches are stitched using non-overlapping tiling. Each physical pixel $(y, x)$ receives exactly one prediction.

---

## 6. Audit Deliverables Inventory

All required planning deliverables have been generated and validated in `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a_dataset_plan\`:
1. [`phase2a_dataset_plan.md`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/phase2a_dataset_plan.md): Comprehensive dataset construction and evaluation protocol.
2. [`patch_strategy_comparison.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/patch_strategy_comparison.csv): Metric comparison across Strategy A (256), Strategy B (384), and Strategy C (stride 128).
3. [`lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/lobo_fold_manifest.csv): 5-fold Leave-One-Bridge-Out manifest with explicit source train/val splits.
4. [`class_distribution_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/class_distribution_by_bridge.csv): Full pixel accounting and semantic distributions per bridge.
5. [`coverage_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/coverage_by_bridge.csv): Per-bridge physical coverage, patch counts, and uncovered margins.
6. [`build_dataset_audit.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a_dataset_plan/build_dataset_audit.py): Executable verification script containing all data integrity assertions.

---

## 7. Limitations and Recommendations for Phase 2B Implementation

1. **Bridge 4 Label Noise Limitation:** Bridge 4's 2-pixel background cropping offset must be reported transparently in any future publication. Training must strictly ingest `PR_NB_Original.png` to avoid contrast artifacts.
2. **Small Macro Fleet Size ($N=5$ Bridges):** Although 197 training patches can be extracted, the statistical degrees of freedom for cross-bridge generalization remains $\text{df} = 4$. Reporting standard deviations across the 5 LOBO folds is mandatory.
3. **Zero Model Training in Phase 2A:** In accordance with instructions, zero model training or tensor caching was executed in this planning phase. Phase 2B may now proceed to materialize the patch manifests and tensors on disk.