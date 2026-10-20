# Research Gap, Methodological Demarcation, and Contributions

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** 2026-10-20  

---

## 1. Demarcation of Novelty and Prior Work

To preserve scientific integrity and avoid overstated claims, this study explicitly delineates prior established work from our experimental contributions:

```
+------------------------------------------------------------------------------------------------+
| PRIOR ESTABLISHED WORK (DO NOT CLAIM AS NOVEL)                                                 |
| - SDNET2021 Dataset creation and initial manual annotation (Dorafshan et al., 2021).           |
| - Standard U-Net convolutional semantic segmentation architecture (Ronneberger et al., 2015).   |
| - Deep CORAL second-order covariance alignment loss (Sun & Saenko, 2016).                      |
| - Demonstration that naive CNNs degrade when tested across SDNET2021 bridges (Ichi et al., 2022)|
+------------------------------------------------------------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------------------+
| IDENTIFIED SCIENTIFIC RESEARCH GAPS                                                            |
| 1. Lack of Unsupervised Domain Adaptation (UDA) on real infrastructure thermography benchmarks.|
| 2. Pervasive cross-split data leakage in prior literature (overlapping patches evaluated       |
|    as independent physical pixels, mixing tiles from the same structure across splits).        |
| 3. Unaddressed depth-dependent failure modes: prior benchmarks pooled delaminations into a    |
|    single binary class, masking the total collapse on deep Class 3 (sub-rebar) delaminations.  |
| 4. Methodological inflation of statistical power (treating N=322 patches as independent       |
|    domains rather than the N=5 physical bridges).                                              |
+------------------------------------------------------------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------------------+
| OUR METHODOLOGICAL CONTRIBUTIONS & STUDY DESIGN                                                |
| 1. Leak-Free Circular Successor LOBO Protocol: Exact Leave-One-Bridge-Out partitioning with    |
|    pre-evaluation cryptographic checkpoint freezing (SHA-256).                                 |
| 2. Deep CORAL Bottleneck Adaptation for Infrared Thermography: Differentiable covariance       |
|    alignment applied to latent representations of uncalibrated infrared radiation signals.      |
| 3. Single Physical Pixel Scoring Guarantee: Full-bridge sliding-window logit accumulation      |
|    divided by coverage count, scoring each physical deck pixel once.                           |
| 4. Depth-Resolved Delamination Diagnosis: Disaggregated evaluation of shallow (Class 2) vs.   |
|    deep (Class 3) delaminations, exposing the zero-recall failure mode on deep defects.        |
| 5. Statistically Defensible Evaluation: Transparent power analysis acknowledging the minimum    |
|    attainable two-sided p-value of 0.0625 for N=5 domains.                                     |
+------------------------------------------------------------------------------------------------+
```

---

## 2. Research Questions & Formal Hypotheses

### Primary Question:
Does aligning second-order feature statistics between labeled source-bridge thermograms and unlabeled target-bridge thermograms using Deep CORAL improve cross-bridge concrete delamination segmentation compared with source-only training?

### Hypotheses:
- **$H_1$ (Overall Defect Generalization):** Deep CORAL improves mean Macro Delamination F1 across the five held-out bridge decks compared with source-only training ($\Delta \text{Macro F1} \ge +0.05$).
- **$H_2$ (Depth-Dependent Recovery):** Deep CORAL mitigates the complete detection collapse on deep Class 3 delaminations by transferring second-order latent representations from source bridges with active Class 3 signatures.
- **$H_0$ (Null Hypothesis):** Feature covariance alignment provides no measurable gain on held-out target decks ($\bar{M}_{\text{CORAL}} \le \bar{M}_{\text{baseline}}$), indicating that uncalibrated thermal contrast differences cannot be resolved by second-order statistical alignment alone.

---

## 3. Engineering and Practical Implications

If Deep CORAL demonstrates positive transfer on held-out bridge decks, transportation departments (DOTs) can deploy trained thermal defect models to newly scanned bridge decks without expensive, hazardous, and labor-intensive manual bridge-specific ground-truth labeling. If transfer is limited, the study provides definitive, leak-free empirical proof that physical radiometric calibration or multi-modal sensing (e.g. impact-echo / GPR coupling) is required to cross domain barriers in bridge deck NDT.
