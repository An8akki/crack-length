# Baseline Failure Modes & Empirical Error Analysis

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Component:** Source-Only Baseline Failure Diagnostics (Target Bridge B1)  
**Date:** 2026-10-20  

---

## 1. Executive Summary of Failure Modes

Analysis of the three verified source-only baseline runs on Bridge 1 (`fold_1_seed42`, `fold_1_seed123`, `fold_1_seed999`) reveals two fundamental structural failure modes that motivate domain adaptation:

1. **Complete Blindness to Deep Delamination (Class 3):**
   - Out of 43,488 ground-truth Class 3 delamination pixels on Bridge 1, the baseline achieved **exactly 0 true positive pixels** across all three seeds ($F1 = 0.00\%$).
   - 96.4% of Class 3 pixels were misclassified as sound concrete, and 3.6% were misclassified as shallow Class 2 delamination.
2. **Extreme Sensitivity to Initialization Bias (Class 2):**
   - Depending on stochastic initialization seed, the model swings between severe under-detection (Seed 42: 3.2% recall) and aggressive over-detection (Seed 999: 44.1% recall, accompanied by 222,349 false positive sound pixels).

---

## 2. In-Depth Class-Specific Error Breakdown

### A. Class 3 Delamination (Below Rebar Mat)
- **Physical Defect Nature:** Class 3 delaminations reside beneath the top reinforcing steel grid ($\approx 50\text{--}75\text{ mm}$ below the deck surface). Due to thermal diffusion resistance of the overlying concrete, radiant thermal signatures at the surface are diffuse, exhibiting low peak temperature contrast relative to surrounding sound deck.
- **Cross-Bridge Domain Shift Vulnerability:**
  In source training bridges (B3, B4, B5), radiant intensity distributions reflect different ambient environmental heating cycles. The U-Net encoder maps subtle thermal gradients to the dominant background representation (Sound Concrete, Class 0).
- **Confusion Matrix Evidence (Bridge 1):**
  - Seed 42: 41,916 pixels $\to$ Sound, 1,572 $\to$ Class 2, **0 $\to$ Class 3**
  - Seed 123: 31,626 pixels $\to$ Sound, 11,862 $\to$ Class 2, **0 $\to$ Class 3**
  - Seed 999: 20,915 pixels $\to$ Sound, 22,573 $\to$ Class 2, **0 $\to$ Class 3**

### B. Class 2 Delamination (Above Rebar Mat)
- **Physical Defect Nature:** Shallow delaminations ($\approx 10\text{--}35\text{ mm}$ depth) create pronounced localized thermal traps that warm quickly under solar irradiance.
- **Error Dynamic:**
  While shallow delaminations are detectable, the unadapted decision threshold suffers high variance:
  - **Seed 42:** Over-regularized. Fails to activate on 96.8% of defects (Macro Delam F1: **2.84%**).
  - **Seed 123:** Balanced operating point. Captures core defect areas with moderate boundaries (Macro Delam F1: **11.14%**).
  - **Seed 999:** Under-conservative. Recalls 44.1% of Class 2 pixels but falsely classifies 25.9% of sound concrete pixels as delaminated (Macro Delam F1: **14.45%**).

---

## 3. Visual Artifacts & Sliding-Window Stitching Dynamics

Colorized full-bridge prediction rasters were generated in `results/visualizations/`:
- [`results/visualizations/fold_1_seed42_target_prediction_colorized.png`](file:///C:/Project/ml_pipeline/crack-length/results/visualizations/fold_1_seed42_target_prediction_colorized.png)
- [`results/visualizations/fold_1_seed123_target_prediction_colorized.png`](file:///C:/Project/ml_pipeline/crack-length/results/visualizations/fold_1_seed123_target_prediction_colorized.png)
- [`results/visualizations/fold_1_seed999_target_prediction_colorized.png`](file:///C:/Project/ml_pipeline/crack-length/results/visualizations/fold_1_seed999_target_prediction_colorized.png)

### Key Spatial Observations:
1. **Sliding-Window Blurring:** Accumulating window logits with 50% overlap effectively suppresses checkerboard boundary artifacts, producing spatially continuous anomaly maps.
2. **Defect Edge Attenuation:** True delamination patches exhibit sharp boundary transitions that are softened by sliding-window averaging, reducing pixel-level precision at defect perimeters.
3. **Bridge Deck Boundary Integrity:** Excluded boundary regions (`255`) remain cleanly masked (black pixels), with zero spurious edge classifications leaking outside the physical bridge deck outline.

---

## 4. Implications for Deep CORAL Adaptation

The complete collapse on Class 3 delaminations provides an empirical justification for Deep CORAL:
- **Hypothesis:** By forcing the bottleneck feature covariance of target thermograms to match source feature covariance, the network will map subtle target temperature gradients into the active delamination feature manifold rather than collapsing them into the background sound concrete cluster.
- **Critical Milestone:** If Deep CORAL achieves nonzero recall on Class 3 delaminations or stabilizes Class 2 detection across seeds, it will represent a substantial engineering advance over the source-only baseline.
