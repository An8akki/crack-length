# Comprehensive Literature Review: Cross-Bridge Infrared Thermography & Domain Adaptation

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** 2026-10-20  

---

## 1. Introduction and Foundations

Non-destructive testing (NDT) of reinforced concrete bridge decks is essential for maintaining transportation infrastructure safety. Subsurface concrete delaminations—originating primarily from rebar corrosion, freeze-thaw cracking, and cyclical mechanical loading—compromise structural integrity and lead to spalling if undetected. Infrared thermography (IRT) has emerged as an efficient, wide-area NDT methodology that senses radiant heat emissions caused by disrupted heat conduction across delamination voids.

However, translating machine learning segmentation models across differing physical bridges remains an acute open scientific challenge due to severe cross-structure domain shifts.

---

## 2. SDNET2021 Dataset & Prior Concrete Benchmark Studies

### SDNET2021 Benchmark
- **Citation:** Dorafshan, S., Thomas, R. J., & Maguire, M. (2021). *SDNET2021: Annotated data set of non-destructive evaluation of reinforced concrete bridge decks*. Utah State University. DOI: `10.31356/data019`.
- **Significance:** SDNET2021 is the first large-scale, multi-modal, public benchmark dataset providing aligned NDT modalities across five operational concrete highway bridges in Utah and Wyoming. The infrared thermography component contains complete 2D stitched thermograms alongside manual expert annotations.

### Benchmark Evaluation by Ichi, Jafari, and Dorafshan (2022)
- **Citation:** Ichi, E., Jafari, M., & Dorafshan, S. (2022). "Deep Learning for Delamination Detection in Bridge Decks from Infrared Thermography: A Benchmark Study." *Infrastructures*, 7(9), 107. DOI: `10.3390/infrastructures7090107`.
- **Findings:** The authors benchmarked standard convolutional neural networks (FCN, U-Net, DeepLabv3) on SDNET2021. They established that intra-bridge evaluation (splitting tiles from the same bridge into train/test) yields seemingly high performance, but performance degrades sharply when models are evaluated on unseen bridge structures.

### Evaluation of IRT Datasets by Ichi and Dorafshan (2024)
- **Citation:** Ichi, E., & Dorafshan, S. (2024). "Evaluation of Infrared Thermography Dataset for Delamination Detection in Reinforced Concrete Bridge Decks." *Applied Sciences*, 14(6), 2455. DOI: `10.3390/app14062455`.
- **Findings:** This study systematically investigated the physical environmental factors causing domain divergence across bridge decks: variable ambient temperature, solar solar irradiance angles, concrete deck age, aggregate composition, and surface texture. They demonstrated that uncalibrated 8-bit digital infrared images capture radiant emission contrast that is highly bridge-dependent, confirming that naive supervised models suffer severe cross-bridge transfer degradation.

---

## 3. Unsupervised Domain Adaptation & Deep CORAL

### The Domain Shift Problem in Feature Space
When training on source domain $\mathcal{D}_S \sim P(X_S, Y_S)$ and testing on target domain $\mathcal{D}_T \sim Q(X_T, Y_T)$ where $P(X) \ne Q(X)$, deep networks learn representations biased toward source-specific feature distributions.

### Deep CORAL Formulation
- **Citation:** Sun, B., & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." In *European Conference on Computer Vision (ECCV)*, Workshops, pp. 443–450. Springer, Cham. DOI: `10.1007/978-3-319-49409-8_35`.
- **Citation:** Sun, B., Feng, J., & Saenko, K. (2016). "Return of Frustratingly Easy Domain Adaptation." In *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), pp. 2058–2065.
- **Mathematical Principle:** Deep CORAL aligns the second-order statistics (covariance) of source and target feature representations without requiring target labels. Unlike Maximum Mean Discrepancy (MMD) which matches only first-order means or adversarial methods (DANN, ADDA) that require training unstable discriminator networks, CORAL minimizes:
  $$\mathcal{L}_{\mathrm{CORAL}} = \frac{1}{4d^2} \|C_S - C_T\|_F^2$$
  providing an exact, stable, and differentiable loss function that directly aligns the shape and orientation of feature clusters in latent space.

### Domain Adaptation in Semantic Segmentation
- **Citation:** Tsai, Y. H., Hung, W. C., Schulter, S., Sohn, K., Yang, M. H., & Chandraker, M. (2018). "Learning to Adapt Structured Output Spaces for Semantic Segmentation." In *CVPR*, pp. 7472–7481.
- **Application to Dense Prediction:** While originally formulated for image classification, feature alignment techniques have been extended to dense prediction architectures. In encoder-decoder models (like U-Net), aligning bottleneck feature covariance enables the network to learn bridge-invariant defect representations prior to spatial decoding.

---

## 4. Reference List (Standard Scholarly Format)

1. **Dorafshan, S., Thomas, R. J., & Maguire, M. (2021).** SDNET2021: Annotated data set of non-destructive evaluation of reinforced concrete bridge decks. Utah State University. DOI: `10.31356/data019`.
2. **Ichi, E., Jafari, M., & Dorafshan, S. (2022).** Deep learning for delamination detection in bridge decks from infrared thermography: A benchmark study. *Infrastructures*, 7(9), 107. DOI: `10.3390/infrastructures7090107`.
3. **Ichi, E., & Dorafshan, S. (2024).** Evaluation of infrared thermography dataset for delamination detection in reinforced concrete bridge decks. *Applied Sciences*, 14(6), 2455. DOI: `10.3390/app14062455`.
4. **Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015).** Learning transferable features with deep adaptation networks. In *International Conference on Machine Learning (ICML)*, pp. 97–105.
5. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** U-Net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, pp. 234–241. Springer, Cham. DOI: `10.1007/978-3-319-24574-4_28`.
6. **Sun, B., & Saenko, K. (2016).** Deep CORAL: Correlation alignment for deep domain adaptation. In *European Conference on Computer Vision (ECCV)*, Workshops, pp. 443–450. Springer, Cham. DOI: `10.1007/978-3-319-49409-8_35`.
7. **Sun, B., Feng, J., & Saenko, K. (2016).** Return of frustratingly easy domain adaptation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), pp. 2058–2065.
8. **Tsai, Y. H., Hung, W. C., Schulter, S., Sohn, K., Yang, M. H., & Chandraker, M. (2018).** Learning to adapt structured output spaces for semantic segmentation. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7472–7481.
