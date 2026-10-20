# Phase 2A.1 — Evaluation Coverage and LOBO Protocol Correction Report

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** October 20, 2026  
**Auditor:** Antigravity Autonomous Research Agent (DeepMind)  
**Source Dataset:** `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\`  
**Output Directory:** `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a1_protocol_correction\`  

---

## Executive Summary & Final Verdict

### Final Decision: **GO for Dataset Construction Under Corrected Protocol**

**Key Audit Corrections & Architectural Decisions:**
1. **Correction of Evaluation Coverage Deficit (59.62% $\to$ 100.00%):** The Phase 2A non-overlapping patch benchmark (Strategy A, 56 patches) left **40.38% (1,497,112 pixels)** of valid deck concrete unextracted due to fixed grid alignment against narrow bridge dimensions. Phase 2A.1 establishes **Strategy C (Full-Bridge Sliding-Window Inference with End-Anchoring and Logit Accumulation)** as the primary evaluation benchmark. Strategy C achieves **100.00% valid deck coverage across all five bridges (5,089,851 / 5,089,851 valid pixels)**, while ensuring that every physical pixel receives exactly one final prediction.
2. **Single Physical Pixel Scoring Guarantee:** In Strategy C, overlapping window predictions accumulate class logits across the bridge raster. The final class is assigned via $\arg\max$ over the accumulated logit tensor. Every physical pixel is scored exactly once in the global confusion matrix, strictly eliminating spatial double-counting.
3. **Deterministic, Disjoint LOBO Manifest:** The 5-fold Leave-One-Bridge-Out split is formalized using the **Deterministic Circular Successor Rule** ($V = B_{(k \pmod 5) + 1}$). Target test, source validation, and source training sets are strictly pairwise disjoint in every fold. Every bridge serves as target test exactly once, source validation exactly once, and source train exactly 3 times.
4. **Strict Target Data Isolation Policy:** Target labels remain sealed in cold storage and are accessible only by the post-training offline evaluation script after model weights and hyperparameters are frozen.
5. **Unresolved Status of Bridge 4 Documented:** Bridge 4's 2-pixel vertical background cropping offset remains documented as **UNRESOLVED**. In accordance with scientific integrity, no image translation or registration warping is applied. Input $X$ is loaded from `PR_NB_Original.png` and ground-truth $Y$ from `PR_NB_Annotaed.png`.

---

## 1. Revalidation of the Five Source Pairs

### 1.1 Direct Source Array Inspection
All five original and annotated PNG pairs were re-audited directly from disk. Dimensions, channel modes, and pixel counts reconciled with zero delta across all 6,317,409 pixels:

| Bridge ID | Bridge Name | Dimensions ($W \times H$) | Modes | Total Pixels | Sound Pixels (%) | Class 2 Above (%) | Class 3 Below (%) | Total Delam (%) | Excluded White (%) | Excluded Black (%) | Unknown Pixels | Alignment Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | Forest River NB | 461 × 1942 | L / RGB | 895,262 | 662,477 (74.00%) | 152,149 (17.00%) | 43,488 (4.86%) | 195,637 (21.85%) | 37,131 (4.15%) | 17 (0.00%) | **0** | Identical (0 diff) |
| **B2** | Forest River SB | 1626 × 401 | L / RGB | 652,026 | 489,219 (75.03%) | 107,496 (16.49%) | 24,446 (3.75%) | 131,942 (20.24%) | 30,803 (4.72%) | 62 (0.01%) | **0** | Identical (0 diff) |
| **B3** | Park River Median | 4326 × 411 | L / RGB | 1,777,986 | 1,380,331 (77.64%) | 298,123 (16.77%) | 78,873 (4.44%) | 376,996 (21.20%) | 421 (0.02%) | 20,238 (1.14%) | **0** | Identical (0 diff) |
| **B4** | Park River NB | 2991 × 371 | L / RGB | 1,109,661 | 844,908 (76.14%) | 215,113 (19.39%) | 49,640 (4.47%) | 264,753 (23.86%) | 0 (0.00%) | 0 (0.00%) | **0** | **UNRESOLVED** (2-px background offset; zero shift applied) |
| **B5** | Park River SB | 451 × 4174 | L / RGB | 1,882,474 | 1,409,183 (74.86%) | 272,292 (14.47%) | 62,113 (3.30%) | 334,405 (17.76%) | 138,855 (7.38%) | 31 (0.00%) | **0** | Mild tonal shift; edge clamping |
| **Fleet**| **5 Bridges** | — | — | **6,317,409** | **4,786,118 (75.76%)** | **1,045,173 (16.54%)** | **258,560 (4.09%)** | **1,303,733 (20.64%)** | **207,210 (3.28%)** | **20,348 (0.32%)** | **0** | — |

### 1.2 Nature of Mask Encodings and Excluded Pixels
- **`0` — Sound Concrete:** Grayscale pixels where $R=G=B$ and $1 \le R \le 254$.
- **`1` — Class 2 Delamination:** Pure green `[0, 255, 0]`.
- **`2` — Class 3 Delamination:** Pure red `[255, 0, 0]`.
- **`255` — Excluded Pixels:**
  - Pure white `[255, 255, 255]` (canvas border surrounding skewed deck mosaics).
  - Pure black `[0, 0, 0]` (scanner zero-padding and CAD raster clamping).
  - These pixels are excluded solely on empirical verification that they lie outside the physical concrete bridge deck. They are assigned label index `255` and masked out of all loss functions and evaluation metrics.
- **Scientific Caution:** Original grayscale intensities are 8-bit digital numbers ($0 \le I \le 255$) reflecting relative radiance, not calibrated absolute temperatures.

---

## 2. Comparison of Evaluation Coverage Strategies

Documented in [`evaluation_coverage_comparison.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/evaluation_coverage_comparison.csv) and [`pixel_coverage_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/pixel_coverage_by_bridge.csv):

| Strategy | Windows Fleet | Fleet Valid Deck Pixels | Covered Valid Pixels | Covered Valid Pct | Uncovered Valid Pixels | Class 2 Coverage | Class 3 Coverage | Single Physical Pixel Scoring | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** (Non-Overlap 256×256) | 56 | 5,089,851 | 3,592,739 | **59.62%** | 1,497,112 (40.38%) | 71.70% | 63.49% | Yes | Secondary Diagnostic Benchmark |
| **Strategy B** (Sliding Stride 128, No Anchor) | 198 | 5,089,851 | 4,317,358 | **84.82%** | 772,493 (15.18%) | 90.77% | 94.66% | No (Overlaps double-counted) | Rejected (Arbitrary edge loss) |
| **Strategy C** (Full-Bridge Sliding + End-Anchor) | **322** | **5,089,851** | **5,089,851** | **100.00%** | **0 (0.00%)** | **100.00%** | **100.00%** | **Yes (Logit accumulation)** | **PRIMARY EVALUATION BENCHMARK** |

### 2.1 Per-Bridge Coverage Breakdown Under Strategy C (Primary Benchmark)
- **Bridge 1 (Forest River NB):** 45 windows ($3 \times 15$ grid) $\to$ **858,114 / 858,114 valid pixels covered (100.00%)**.
- **Bridge 2 (Forest River SB):** 36 windows ($12 \times 3$ grid) $\to$ **621,161 / 621,161 valid pixels covered (100.00%)**.
- **Bridge 3 (Park River Median):** 99 windows ($33 \times 3$ grid) $\to$ **1,757,327 / 1,757,327 valid pixels covered (100.00%)**.
- **Bridge 4 (Park River NB):** 46 windows ($23 \times 2$ grid) $\to$ **1,109,661 / 1,109,661 valid pixels covered (100.00%)**.
- **Bridge 5 (Park River SB):** 96 windows ($3 \times 32$ grid) $\to$ **1,743,588 / 1,743,588 valid pixels covered (100.00%)**.
- **Fleet Total:** **322 inference windows $\to$ 5,089,851 / 5,089,851 valid deck pixels (100.00%)**.

### 2.2 Mechanism for Single Physical Pixel Scoring
In Strategy C:
1. For each $256 \times 256$ window $k$, the neural network outputs unnormalized class logits $Z_k \in \mathbb{R}^{3 \times 256 \times 256}$.
2. Logits are accumulated across full-bridge rasters:
   $$\text{LogitAccum}[c, y, x] = \sum_{k \in \Omega(y, x)} Z_k[c, y - y_k, x - x_k], \quad \text{CountAccum}[y, x] = |\Omega(y, x)|$$
3. The mean logit is computed: $\bar{Z}[c, y, x] = \text{LogitAccum}[c, y, x] / \text{CountAccum}[y, x]$.
4. The final class prediction is assigned: $\hat{Y}(y, x) = \arg\max_{c \in \{0, 1, 2\}} \bar{Z}[c, y, x]$.
5. Evaluation is computed strictly between $\hat{Y}(y, x)$ and ground truth $Y(y, x)$ at pixels where $Y(y, x) \neq 255$.
**Result:** Every valid physical pixel on the entire bridge is scored **exactly once**, providing a true full-bridge benchmark with zero double-counting.

---

## 3. Corrected Five-Fold Leave-One-Bridge-Out (LOBO) Protocol

Documented in [`corrected_lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/corrected_lobo_fold_manifest.csv):

| Fold ID | Target Test Bridge | Source Train Bridges | Source Validation Bridge | Validation Selection Rule | Target Test Windows (Strat C) | Target Valid Pixels Evaluated | Source Train Windows (Strat C) | Source Val Windows (Strat C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **B1** (Forest River NB) | B3 + B4 + B5 | **B2** | Circular Successor: $B_{(1 \pmod 5) + 1}$ | **45** | **858,114 (100%)** | 241 | 36 |
| **Fold 2** | **B2** (Forest River SB) | B1 + B4 + B5 | **B3** | Circular Successor: $B_{(2 \pmod 5) + 1}$ | **36** | **621,161 (100%)** | 187 | 99 |
| **Fold 3** | **B3** (Park River Median) | B1 + B2 + B5 | **B4** | Circular Successor: $B_{(3 \pmod 5) + 1}$ | **99** | **1,757,327 (100%)** | 177 | 46 |
| **Fold 4** | **B4** (Park River NB) | B1 + B2 + B3 | **B5** | Circular Successor: $B_{(4 \pmod 5) + 1}$ | **46** | **1,109,661 (100%)** | 180 | 96 |
| **Fold 5** | **B5** (Park River SB) | B2 + B3 + B4 | **B1** | Circular Successor: $B_{(5 \pmod 5) + 1}$ | **96** | **1,743,588 (100%)** | 181 | 45 |

### 3.1 Properties of the Circular Successor Selection Rule
1. **Strict Disjointness:** In every fold $k$, the target test bridge, source validation bridge, and three source training bridges are **strictly pairwise disjoint** ($\{T_k\} \cap \{V_k\} = \emptyset$, $\{T_k\} \cap \mathcal{S}_{\text{train}, k} = \emptyset$, $\{V_k\} \cap \mathcal{S}_{\text{train}, k} = \emptyset$).
2. **Equitable Fleet Utilization:** Across the 5 folds:
   - Every bridge serves as Target Test exactly once.
   - Every bridge serves as Source Validation exactly once.
   - Every bridge serves in Source Training exactly 3 times.
3. **Zero Boundary Crossing:** Patches are sliced strictly within each bridge deck array ($W_i \times H_i$). No patch can cross bridge boundaries, eliminating cross-image spatial autocorrelation.

---

## 4. Target-Data Isolation Policy

To guarantee scientific integrity under domain shift:
1. **Source Training Labels:** Used exclusively for supervised training loss (Cross-Entropy, Focal, Dice loss).
2. **Source Validation Labels:** Used strictly for monitoring validation loss, evaluating early-stopping criteria, and selecting optimal checkpoints.
3. **Target Images (Unlabeled):**
   - In Source-Only Baseline experiments: Target images are **never accessed** during training.
   - In Unsupervised Domain Adaptation (UDA) experiments (e.g., Deep CORAL, DANN, MMD): Unlabeled target patches extracted from `PR_XX_Original.png` may be used as inputs to align latent feature distributions.
4. **Target Ground-Truth Labels (Strictly Sealed):** Target labels from `PR_XX_Annotaed.png` are stored in an isolated cold-storage directory. They are **never loaded** during data preprocessing, normalization fitting, feature extraction, hyperparameter tuning, validation monitoring, early stopping, or domain adaptation loss calculation.
5. **Offline Scoring:** Target labels are accessed only once by the offline evaluation script after model weights are permanently frozen.

---

## 5. Final Evaluation Metrics & Zero-Division Standards

### 5.1 Primary Evaluation Metric: Full-Deck Confusion Matrix Aggregation
For each target bridge $B_k$, the full-bridge prediction raster $\hat{Y}_k(y, x)$ is compared against ground truth $Y_k(y, x)$ strictly over valid deck pixels ($Y_k \neq 255$):

$$C_{ij} = \sum_{(y, x) \in \text{ValidDeck}} \mathbb{I}[Y_k(y, x) = i \;\wedge\; \hat{Y}_k(y, x) = j]$$

From this global $3 \times 3$ confusion matrix, metrics are computed:
- **Class 2 (Above-Mat Delamination):** Precision, Recall, F1, and IoU.
- **Class 3 (Below-Mat Delamination):** Precision, Recall, F1, and IoU.
- **Combined Delamination (Class 2 + Class 3):** Precision, Recall, F1, and IoU.
- **Sound Concrete:** Precision, Recall, F1, and IoU.

Because every target bridge contains thousands of Class 2 and Class 3 pixels (B1: 195k, B2: 132k, B3: 377k, B4: 265k, B5: 334k), **deck-level denominators are never zero**.

### 5.2 Macro-Averaging Standard
Fleet-level cross-bridge performance is reported as the macro-average across the five independent target bridge domains:

$$\text{Macro-IoU}_c = \frac{1}{5} \sum_{k=1}^5 \text{IoU}_c(B_k), \quad \text{Macro-F1}_c = \frac{1}{5} \sum_{k=1}^5 \text{F1}_c(B_k)$$

$$\text{mIoU} = \frac{1}{3} \left( \text{Macro-IoU}_{\text{Sound}} + \text{Macro-IoU}_{\text{Class 2}} + \text{Macro-IoU}_{\text{Class 3}} \right)$$

### 5.3 Zero-Division Policy
- If a model predicts zero pixels for a class ($\sum_i C_{ic} = 0$), Precision is defined as $0.0$, and IoU is $0.0$.
- Under no circumstances is an absent prediction assigned a score of 1.0 (100%).

---

## 6. Audit Deliverables Inventory

All corrected planning artifacts and the validation script have been created and verified in `C:\Users\Dell\Documents\SDNET2021_Audit\phase2a1_protocol_correction\`:
1. [`phase2a1_protocol_correction.md`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/phase2a1_protocol_correction.md): Comprehensive architectural plan, protocol specification, and evaluation standard.
2. [`evaluation_coverage_comparison.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/evaluation_coverage_comparison.csv): High-level comparison across Strategy A (56 patches), Strategy B (198 patches), and Strategy C (322 windows).
3. [`corrected_lobo_fold_manifest.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/corrected_lobo_fold_manifest.csv): 5-fold Leave-One-Bridge-Out manifest with circular successor validation splits.
4. [`pixel_coverage_by_bridge.csv`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/pixel_coverage_by_bridge.csv): Detailed per-bridge pixel coverage, patch counts, and edge analysis.
5. [`protocol_validation.py`](file:///C:/Users/Dell/Documents/SDNET2021_Audit/phase2a1_protocol_correction/protocol_validation.py): Self-contained executable verification script with all integrity assertions active.

---

## 7. Protocol Readiness for Phase 2B Implementation

1. **Evaluation Coverage Blocker Resolved:** The 40.38% coverage deficit of Strategy A is resolved by codifying Strategy C as the primary evaluation standard, ensuring 100% full-bridge valid deck evaluation.
2. **Zero Model Training Executed:** In accordance with constraints, zero model training was conducted.
3. **Ready for Phase 2B:** The protocol is mathematically sound, fully reproducible, and ready for physical patch slicing and tensor serialization in Phase 2B.