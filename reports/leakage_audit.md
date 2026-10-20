# Data Isolation & Leakage Prevention Audit

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Component:** Anti-Leakage Protocol & Adversarial Verification Suite  
**Date:** 2026-10-20  

---

## 1. Threat Model & Potential Leakage Vectors

In cross-domain semantic segmentation research, data leakage commonly corrupts findings through six insidious failure modes:

1. **Target-Label Exposure During Training:** Loading target annotations into memory or passing them into loss computation during unsupervised adaptation.
2. **Target-Driven Model Selection:** Using target-bridge performance to tune hyperparameters, adjust early-stopping patience, or choose the "best" checkpoint.
3. **Target-Contaminated Preprocessing:** Computing pixel intensity normalization statistics ($\mu, \sigma$) across the entire dataset (including target bridges) rather than source-training partitions only.
4. **Partition Overlap:** Patch tiling schemes that sample overlapping windows spanning both training and validation splits.
5. **Canvas Boundary Bias:** Unmasked canvas borders (pure white or pure black pixels) propagating spurious gradients that artificially inflate background accuracy.
6. **Double-Counting Overlapping Target Pixels:** Evaluating overlapping patch predictions independently as if they were independent physical measurements.

---

## 2. Enforced Experimental Safeguards

| Leakage Vector | Enforced Experimental Safeguard | Verification Status |
| :--- | :--- | :---: |
| **Target Labels** | Physical separation into sealed directory (`target_eval_sealed/`). The adaptation target DataLoader (`target_uda/`) contains zero mask files and yields only single-channel images. | **VERIFIED (PASS)** |
| **Model Selection** | Early stopping and checkpoint selection monitor strictly source-validation Macro Delamination F1 ($[F1_{C1} + F1_{C2}]/2$) on the designated source-validation bridge deck. | **VERIFIED (PASS)** |
| **Normalization** | Intensity mean ($\mu$) and standard deviation ($\sigma$) are computed exclusively from valid concrete pixels of designated source-training patches (`arr_mask != 255`). | **VERIFIED (PASS)** |
| **Patch Overlap** | Sliding-window overlap is 50% *within* each individual bridge deck. Because Leave-One-Bridge-Out assigns entire bridges to splits, no patch ever spans cross-bridge boundaries. | **VERIFIED (PASS)** |
| **Canvas Boundary** | Multiclass Cross-Entropy Loss strictly enforces `ignore_index=255`, ensuring mathematically zero gradient is backpropagated through canvas boundary pixels. | **VERIFIED (PASS)** |
| **Physical Pixel Scoring** | Target bridge sliding-window logits ($3 \times H \times W$) are accumulated and divided by the pixel coverage count, scoring each physical valid deck pixel ($M_{\text{valid}}$) exactly once. | **VERIFIED (PASS)** |

---

## 3. Adversarial Test Results

Two automated test suites were constructed to deliberately stress and test data isolation:

1. **`tests/test_lobo_splits.py` (4 of 4 Tests Passed):**
   - Verified circular successor fold structure across Folds 1–5.
   - Proved mutual exclusivity: $\text{Target} \cap \text{Val} = \emptyset$, $\text{Target} \cap \text{Train} = \emptyset$, $\text{Val} \cap \text{Train} = \emptyset$.
   - Confirmed complete fleet union: $\text{Target} \cup \text{Val} \cup \text{Train} = \{B1, B2, B3, B4, B5\}$.
   - Confirmed zero cross-bridge spatial coordinate overlap.
2. **`tests/test_data_isolation.py` (5 of 5 Tests Passed):**
   - Confirmed attempting to access target masks from adaptation DataLoader raises `KeyError`.
   - Verified label index `255` produces exactly $0.0$ gradient on network logits.
   - Proved checkpoint selection is strictly invariant to target evaluation scores.
   - Verified post-freeze target authorization guards reject unverified checkpoints.
   - Confirmed deterministic seeding behavior.

---

## 4. Conclusion

The data isolation architecture completely eliminates both direct and indirect target label contamination. All 9 automated isolation and partition tests pass with 100% compliance.
