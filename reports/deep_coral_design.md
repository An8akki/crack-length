# Deep CORAL Domain Adaptation: Architecture & Experimental Design

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Method:** Deep CORAL (Correlation Alignment)  
**Reference:** Sun, B., & Saenko, K. (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. European Conference on Computer Vision (ECCV).  
**Date:** 2026-10-20  

---

## 1. Research Question & Scientific Hypothesis

**Primary Research Question:**
Does aligning second-order feature statistics between labeled source-bridge thermograms and unlabeled target-bridge thermograms using Deep CORAL improve cross-bridge concrete delamination segmentation compared with source-only training?

**Scientific Hypothesis:**
In infrared thermography, cross-bridge domain shift arises from differing concrete mix designs, thermal diffusivity, environmental irradiance, ambient temperature cycles, and bridge deck geometry. While raw thermal intensity values vary substantially across highway structures, higher-level convolutional representations of delaminations (subsurface thermal anomalies with localized gradient boundaries) share structural correlations. 

By penalizing the Frobenius norm distance between the feature covariance matrices of source-bridge and target-bridge thermograms, Deep CORAL aligns the principal axes of feature dispersion in the network's latent space, mitigating feature distribution collapse on previously unseen bridges.

---

## 2. Mathematical Formulation

Let $\mathcal{D}_S = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$ denote labeled source-bridge training patches, and $\mathcal{D}_T = \{x_j^t\}_{j=1}^{n_t}$ denote unlabeled target-bridge patches.

Let $\Phi(x) \in \mathbb{R}^{d}$ represent the extracted feature vector from an intermediate network layer for input $x$.

For a batch of source features $F_S \in \mathbb{R}^{N_S \times d}$ and target features $F_T \in \mathbb{R}^{N_T \times d}$, the unbiased sample covariance matrices $C_S, C_T \in \mathbb{R}^{d \times d}$ are computed as:

$$C_S = \frac{1}{N_S - 1} (F_S - \mathbf{1}\mu_S^T)^T (F_S - \mathbf{1}\mu_S^T)$$
$$C_T = \frac{1}{N_T - 1} (F_T - \mathbf{1}\mu_T^T)^T (F_T - \mathbf{1}\mu_T^T)$$

where $\mu_S = \frac{1}{N_S} \mathbf{1}^T F_S$ and $\mu_T = \frac{1}{N_T} \mathbf{1}^T F_T$.

The Deep CORAL covariance loss is defined as:

$$\mathcal{L}_{\mathrm{CORAL}} = \frac{1}{4d^2} \|C_S - C_T\|_F^2 = \frac{1}{4d^2} \sum_{i=1}^d \sum_{j=1}^d \left(C_S(i, j) - C_T(i, j)\right)^2$$

where $\|\cdot\|_F$ denotes the matrix Frobenius norm, and $d$ is the feature dimension.

### Total Objective Function:
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{seg}}(y_S, \hat{y}_S) + \lambda \cdot \mathcal{L}_{\mathrm{CORAL}}(F_S, F_T)$$

where:
- $\mathcal{L}_{\mathrm{seg}}$ is the supervised Cross-Entropy Loss with `ignore_index=255`, evaluated strictly on labeled source patches.
- $\lambda \ge 0$ is the adaptation hyperparameter weight (default $\lambda = 0.5$).
- Target patches $x_T$ contribute **only** to $\mathcal{L}_{\mathrm{CORAL}}$; zero target labels enter the training graph.

---

## 3. Architectural Selection: Why the U-Net Bottleneck?

The baseline model is a 4-stage convolutional U-Net with channel features $[8, 16, 32, 64]$. 

Three candidate layers for covariance alignment were evaluated:
1. **Input Space (Pixel Level):** Aligning raw 1-channel pixels would merely match raw grayscale contrast without accounting for non-linear thermal spatial patterns.
2. **Decoder / Penultimate Layer ($C=8, 256 \times 256$):** The final decoder layers are heavily committed to pixel-level reconstruction and spatial localization. Forcing covariance alignment here can impair boundary delineation.
3. **Encoder Bottleneck ($C=128, 16 \times 16$):** The bottleneck captures the most abstract, high-level semantic representations of the deck surface (anomaly presence vs. sound concrete) while abstracting away local pixel coordinate artifacts. With batch size $B=8$, spatial flattening yields $N = 8 \times 16 \times 16 = 2,048$ samples for estimating a $128 \times 128$ covariance matrix, providing high statistical stability and smooth gradient flow.

**Design Decision:** The bottleneck representation ($d=128$) is selected as the primary feature layer for Deep CORAL domain adaptation.

---

## 4. Leakage-Proof Training Architecture

1. **Unlabeled Target Branch:** The target DataLoader reads strictly from `target_uda/{fold_id}/images/`. It possesses no `masks/` directory and cannot access ground truth annotations.
2. **Model Selection Isolation:** Early stopping and best checkpoint selection are computed strictly on the designated source-validation bridge (`source_val/{fold_id}/`) using Macro Delamination F1. No target data is involved in selecting the best epoch.
3. **Cryptographic Checkpoint Freezing:** The selected checkpoint `.pt` is permanently frozen and its SHA-256 signature recorded before the offline target evaluation pipeline is authorized to run.
4. **Target Evaluation Verification:** Full-bridge sliding-window prediction maps are accumulated and evaluated on the held-out target bridge only after training has terminated.
