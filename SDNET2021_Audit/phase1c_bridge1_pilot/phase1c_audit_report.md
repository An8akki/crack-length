# Phase 1C: Bridge 1 Mask Generation and Tiling Feasibility Pilot Report

**Bridge Under Audit:** Bridge 1 (Forest River Northbound — FR_NB)
**Input Files:**
- FR_NB_Original.png
- FR_NB_Annotated.png
**Output Directory:** C:\Users\Dell\Documents\SDNET2021_Audit\phase1c_bridge1_pilot\
**Date:** October 2026

---

## 1. Input Verification & Dimensional Conformance

| File | Format | Mode | Dimensions (W x H) | Bit Depth | Total Pixels | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FR_NB_Original.png | PNG | L (Grayscale) | 461 x 1,942 | 8-bit uint8 | 895,262 | VERIFIED |
| FR_NB_Annotated.png | PNG | RGB (Color) | 461 x 1,942 | 8-bit uint8 | 895,262 | VERIFIED |

**Dimensional Conformance:** Both images decode with zero errors and possess identical dimensions (461 x 1,942 pixels).

---

## 2. Mask Generation & Exact Pixel Reconciliation

### 2.1 Class Definitions & Semantic Mapping Rules
- **Class 1 (Sound Deck):** Unannotated deck pixels where R = G = B (excluding white canvas padding and black boundary points).
- **Class 2 (Above-Mat Delamination):** Pure Green pixels where R = 0, G = 255, B = 0.
- **Class 3 (Below-Mat Delamination):** Pure Red pixels where R = 255, G = 0, B = 0.
- **Excluded (Canvas Padding):** Pure White pixels where R = 255, G = 255, B = 255.
- **Excluded (Boundary Artifacts):** Pure Black pixels where R = 0, G = 0, B = 0.
- **Uncertain / Fringe Pixels:** Any transitional color with partial green/red blending (Count = 0).

### 2.2 Complete Pixel Reconciliation Table

| Semantic Category | RGB Rule in Annotated Image | Mask Pixel Value | Pixel Count | Area Fraction | Metrological Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Class 1: Sound Concrete** | R = G = B (not 0, not 255) | 0 | 662,477 | 73.998% | Usable Deck |
| **Class 2: Above-Mat Delam** | [0, 255, 0] | 1 | 152,149 | 16.995% | Usable Deck |
| **Class 3: Below-Mat Delam** | [255, 0, 0] | 2 | 43,488 | 4.858% | Usable Deck |
| **Excluded: White Padding** | [255, 255, 255] | 255 | 37,131 | 4.148% | Non-Deck Canvas |
| **Excluded: Black Boundary** | [0, 0, 0] | 255 | 17 | 0.002% | Boundary Artifact |
| **Total Usable Deck Pixels** | Classes 1, 2, and 3 | 0, 1, 2 | **858,114** | **95.851%** | Ground Truth |
| **Total Excluded Pixels** | White + Black | 255 | **37,148** | **4.149%** | Masked Out |
| **Reconciled Image Total** | All Pixels | Any | **895,262** | **100.000%** | Exact Match (Delta = 0) |

---

## 3. Non-Overlapping Tiling Feasibility Evaluation

Because the Bridge 1 orthomosaic is tall and narrow (W = 461, H = 1,942), non-overlapping tile grids fit exactly 1 tile horizontally. Both Top-Left (x0=0, y0=0) and Centered grids were evaluated.

### 3.1 256 x 256 Tile Grid Summary

- **Total Complete Tiles:** 7 tiles (1 x 7 grid).
- **Retention Threshold:** Tiles with >= 50% valid deck pixels are retained.
- **Top-Left Grid:** 7 of 7 retained (100%), 0 rejected. Valid deck ratios: 89.8% to 100.0%.
- **Centered Grid:** 7 of 7 retained (100%), 0 rejected. Valid deck ratios: 100.0% across all 7 tiles.
- **Class Representation:**
  - Class 2 (Green) is present in **7 of 7 tiles** (100%).
  - Class 3 (Red) is present in **7 of 7 tiles** (100%).
  - Sound-only tiles: **0 tiles**.
  - Dead / empty tiles: **0 tiles**.

### 3.2 384 x 384 Tile Grid Summary

- **Total Complete Tiles:** 5 tiles (1 x 5 grid).
- **Top-Left Grid:** 5 of 5 retained (100%), 0 rejected. Valid deck ratios: 90.1% to 100.0%.
- **Centered Grid:** 5 of 5 retained (100%), 0 rejected. Valid deck ratios: 97.4% to 100.0%.
- **Deck Coverage:** Captures 384 / 461 = 83.3% of transverse deck width per tile.
- **Class Representation:**
  - Class 2 (Green) is present in **5 of 5 tiles** (100%).
  - Class 3 (Red) is present in **5 of 5 tiles** (100%).
  - Sound-only tiles: **0 tiles**.
  - Dead / empty tiles: **0 tiles**.

---

## 4. Generated Artifacts in phase1c_bridge1_pilot/

1. b1_multiclass_mask_raw.png: Single-channel uint8 ground-truth raster (0=Sound, 1=Class 2, 2=Class 3, 255=Excluded).
2. b1_multiclass_mask_colorized.png: RGB palette visualization ([128,128,128]=Sound, [0,255,0]=Class 2, [255,0,0]=Class 3, [255,255,255]=Excluded).
3. b1_validity_mask.png: Binary uint8 mask (255=Usable Deck, 0=Canvas Background).
4. b1_tiles_256_topleft.csv & b1_tiles_256_centered.csv: Detailed tile manifests for 256 x 256.
5. b1_tiles_384_topleft.csv & b1_tiles_384_centered.csv: Detailed tile manifests for 384 x 384.
6. b1_pilot_5panel_diagnostic.png: 5-panel composite visual verification.
7. b1_grid_overlay_256.png & b1_grid_overlay_384.png: Thermal images overlaid with grid boundary coordinates.

---

## 5. Decision & Next Step

**Verdict:** **GO FOR DATASET PIPELINE DEVELOPMENT**

**Justification:**
1. Bridge 1 labels are completely unambiguous, non-anti-aliased, and reconcile to the exact pixel.
2. 100% of complete non-overlapping tiles at both 256 x 256 and 384 x 384 pass the >= 50% deck validity test.
3. Every retained tile contains both Class 2 and Class 3 delamination defects alongside sound concrete, providing rich supervision per patch.
4. 384 x 384 centered tiling is identified as the superior spatial configuration, capturing 83.3% of deck width with > 97.4% valid pixels per tile.

*Constraint Reminder:* This decision validates the labeling and tiling logic for Bridge 1. Bridges 2-5 must be processed and verified with equivalent rigor before initiating cross-bridge model training.
