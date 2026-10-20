# Dataset Validation Report: SDNET2021 Infrared Thermography

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Dataset Reference:** SDNET2021 (DOI: `10.31356/data019`)  
**Modalities:** Infrared Thermography (IRT) exclusively; GPR and impact-echo are strictly excluded.  
**Tiling Strategy:** Strategy C ($256 \times 256$, 128px stride, 50% overlap, end-anchored)  
**Date:** 2026-10-20  

---

## 1. Bridge Domains and Geometric Specifications

The benchmark evaluates five real-world concrete highway bridge decks under a Leave-One-Bridge-Out (LOBO) domain shift protocol:

| Bridge ID | Bridge Name | Orientation / Location | Original Dimensions ($W \times H$) | Strategy C Patch Count |
| :---: | :--- | :--- | :---: | :---: |
| **B1** | Forest River North Bound | Highway bridge, NB lane | $461 \times 1942$ px | 45 patches |
| **B2** | Forest River South Bound | Highway bridge, SB lane | $1626 \times 401$ px | 36 patches |
| **B3** | Park River Median | Highway bridge, median strip | $4326 \times 411$ px | 99 patches |
| **B4** | Park River North Bound | Highway bridge, NB lane | $2991 \times 371$ px | 46 patches |
| **B5** | Park River South Bound | Highway bridge, SB lane | $451 \times 4174$ px | 96 patches |
| **Total** | | | | **322 patches** |

*Note on Bridge 4 annotation filename:* In the official SDNET2021 dataset distribution, Bridge 4's annotation is spelled `PR_NB_Annotaed.png` (source typographical variant preserved in all pipeline loaders).

---

## 2. Semantic Color Decoding & Label Protocol

SDNET2021 annotated images encode ground truth in RGB space, decoded deterministically as follows:

| Class Value | Semantic Interpretation | Physical Definition | Ground-Truth RGB Signature |
| :---: | :--- | :--- | :---: |
| **`0`** | Sound Concrete | Concrete deck exhibiting no thermal anomaly | Grayscale background: $R=G=B$, with $0 < R < 255$ |
| **`1`** | Class 2 Delamination | Delamination located above the top rebar mat | Pure Green: $[0, 255, 0]$ |
| **`2`** | Class 3 Delamination | Delamination located below the top rebar mat | Pure Red: $[255, 0, 0]$ |
| **`255`** | Excluded / Ambiguous | Boundary canvas, text annotations, colorbar legend | Pure White $[255, 255, 255]$ or Pure Black $[0, 0, 0]$ |

### Methodological Safeguards:
1. **Uncalibrated Radiant Intensity:** Grayscale values represent 8-bit digital infrared thermogram intensity, not thermodynamic temperature measurements. No radiometric calibration is assumed.
2. **Ambiguous Boundary Pixels:** Pure white and pure black pixels are excluded (`255`) and masked from both loss computation and accuracy metrics, ensuring no canvas border artifacts distort model evaluation.
3. **Class Imbalance:** Class 3 delamination (red) is extremely rare across highway bridges, representing $<0.5\%$ of valid deck area. Models must be evaluated using class-specific precision, recall, and F1 to expose defect drop-off.

---

## 3. Leave-One-Bridge-Out (LOBO) Circular Successor Partitioning

To ensure zero target leakage and prevent over-optimistic validation, a circular successor protocol assigns each bridge to target, validation, and training roles:

| Fold ID | Target Bridge (Held-out Evaluation) | Source-Validation Bridge (Model Selection) | Source-Training Bridges | Source-Train Patches | Source-Val Patches |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **B1** (45 patches) | **B2** (36 patches) | **B3, B4, B5** | 241 patches | 36 patches |
| **Fold 2** | **B2** (36 patches) | **B3** (99 patches) | **B1, B4, B5** | 187 patches | 99 patches |
| **Fold 3** | **B3** (99 patches) | **B4** (46 patches) | **B1, B2, B5** | 177 patches | 46 patches |
| **Fold 4** | **B4** (46 patches) | **B5** (96 patches) | **B1, B2, B3** | 180 patches | 96 patches |
| **Fold 5** | **B5** (96 patches) | **B1** (45 patches) | **B2, B3, B4** | 181 patches | 45 patches |

---

## 4. Bridge 4 Spatial Alignment Audit

In Phase 1E.2, an independent spatial audit identified a potential 2-pixel vertical translation between Bridge 4's infrared thermogram and annotation raster. 

- **Scientific Finding:** While shifting the annotation raster by 2 pixels alters cross-entropy loss, there is no physical ground-truth justification for translating the annotation.
- **Protocol Decision:** In accordance with Global Rule 8, **no spatial shift has been applied to Bridge 4**. The original, unshifted annotation raster is preserved as the frozen baseline.

---

## 5. Current File Integrity & Blocker Assessment

1. **Manifest Integrity:**
   - `lobo_fold_manifest.csv`: 100% Verified bitwise.
   - `patch_manifest.csv`: 100% Verified bitwise.
   - `class_distribution_by_bridge.csv`: 100% Verified bitwise.
2. **Target Evaluation Seals:**
   - `target_eval_sealed/{fold_id}/`: Ground truth masks, validity rasters, and `eval_stitch_manifest.json` for all 5 folds are present and verified against `checksums.sha256`.
3. **Missing Raw IR Dataset:**
   - The 10 original image pairs in `extracted_irt/` were not tracked in Git.
   - Materialization of the 322 patches (`source_train/`, `source_val/`) requires placing the 10 raw PNG pairs into `SDNET2021_Audit/extracted_irt/` and executing `dataset_build.py`.
