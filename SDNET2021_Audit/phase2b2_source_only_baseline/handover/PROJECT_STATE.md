# Research Project State and Roadmap

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Repository:** `https://github.com/An8akki/crack-length`  
**Branch:** `depth-estimation-integration`  
**Last Updated:** 2026-10-20  

---

## 1. Research Overview & Primary Scientific Objective

This research investigates the performance, domain vulnerability, and cross-bridge generalizability of deep learning semantic segmentation models applied to concrete bridge deck infrared thermography (IRT) from the benchmark SDNET2021 dataset.

The core objective is to evaluate unsupervised domain adaptation (UDA) methods against a rigorous, reproducible, and leak-free source-only baseline across five real-world highway bridges evaluated under a Leave-One-Bridge-Out (LOBO) protocol.

### Fundamental Scientific Ground Rules:
1. **Source-Only Baseline Isolation:** Phase 2B.2 is strictly a source-only benchmark. No target-domain adaptation methods (such as Deep CORAL, maximum classifier discrepancy, entropy minimization, pseudo-labeling, or target fine-tuning) are permitted in this baseline phase.
2. **Strict Evaluation Isolation:** Target bridge ground-truth masks and target class distributions are sealed during training and checkpoint selection. Checkpoints are frozen with cryptographic SHA-256 signatures before target evaluation occurs.
3. **No Unverified Data Tweaks:** The documented Bridge 4 spatial alignment uncertainty (a potential 2-pixel vertical shift between infrared and annotation) is explicitly recorded as an unresolved open question; masks are **never shifted**, preventing artificial or unverified distortion.
4. **Physical Integrity:** 8-bit grayscale intensity represents uncalibrated radiant infrared thermal signal, not true thermodynamic temperature.

---

## 2. Completed Research Phases & Verified Milestones

### Phase 1C — Bridge 1 Pilot Audit
- Established the ground-truth semantic color decoding:
  * Label `0`: Sound concrete (grayscale background).
  * Label `1`: Class 2 delamination (delamination above the top rebar mat; green `[0,255,0]`).
  * Label `2`: Class 3 delamination (delamination below the top rebar mat; red `[255,0,0]`).
  * Label `255`: Excluded / ambiguous (pure white `[255,255,255]`, pure black `[0,0,0]`, boundary text, and colorbar artifacts).

### Phases 1D, 1E, 1E.1, & 1E.2 — Multi-Bridge Geometry & Alignment Audits
- Audited all 5 bridge infrared image pairs: B1 (Forest River NB), B2 (Forest River SB), B3 (Park River median), B4 (Park River NB), and B5 (Park River SB).
- Reconciled tile manifests and pixel counts across independent audit passes.
- Thoroughly audited Bridge 4 spatial alignment, documenting that while a 2px shift alters cross-entropy, there is no physical ground-truth justification for translating the annotation raster. The raw original alignment was frozen.

### Phases 2A, 2A.1, 2A.2, & 2A.3 — LOBO Evaluation Protocol Reconciliation
- Reconciled coverage across candidate tiling strategies:
  * Strategy A (non-overlapping): Covered only ~59.62% of valid bridge deck pixels.
  * Strategy C (50% overlapping sliding window, $256 \times 256$ with 128px stride): Achieved **100% full-bridge valid deck coverage** across all five bridges.
- Materialized the fleet-wide total of **322 Strategy C patches**:
  * Bridge 1: 42 patches
  * Bridge 2: 81 patches
  * Bridge 3: 158 patches
  * Bridge 4: 28 patches
  * Bridge 5: 13 patches
- Formalized the Leave-One-Bridge-Out (LOBO) circular successor fold structure:
  * **Fold 1:** Target B1 | Validation B2 | Training B3, B4, B5 (199 train patches)
  * **Fold 2:** Target B2 | Validation B3 | Training B1, B4, B5 (83 train patches)
  * **Fold 3:** Target B3 | Validation B4 | Training B1, B2, B5 (136 train patches)
  * **Fold 4:** Target B4 | Validation B5 | Training B1, B2, B3 (281 train patches)
  * **Fold 5:** Target B5 | Validation B1 | Training B2, B3, B4 (267 train patches)

### Phase 2B.1 — Dataset Pipeline Construction & Materialization
- Implemented `dataset_build.py` which materialized the 322 patches into strictly separated directories:
  * `source_train/{fold_id}/` (images and labels)
  * `source_val/{fold_id}/` (images and labels)
  * `target_uda/{fold_id}/` (images only — zero masks or labels)
  * `target_eval_sealed/{fold_id}/` (sealed evaluation masks and stitching manifests)
- Validated pipeline integrity using `dataset_validation_tests.py` (100% test pass).

### Phase 2B.2 (Current Phase) — Source-Only U-Net Baseline & LOBO Benchmark
- **Model Architecture:** Conventional 4-stage PyTorch U-Net (`features=[8, 16, 32, 64]`, 486,427 parameters) designed for stable CPU convergence without severe overfitting on small fold datasets (~83–281 patches).
- **Optimization:** AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), batch size 8, unweighted CrossEntropyLoss with `ignore_index=255`.
- **Early Stopping & Model Selection:** Monitored on source-validation Macro Delamination F1 ($[F1_{C1} + F1_{C2}] / 2$) with patience = 12 epochs. Zero division defined as 0.0.
- **Normalization:** Computed strictly on valid deck pixels of source-training patches ($\mu, \sigma$); zero target-derived statistics used.
- **Evaluation Mechanism:** Full-bridge sliding-window logit accumulation ($3 \times H \times W$), divided by prediction counts per pixel, argmax applied once to produce the final raster. Evaluated strictly on physical valid deck pixels ($M_{\text{valid}}$), scoring each physical pixel exactly once.
- **Completed Runs on Current Machine:**
  * `fold_1_seed42`: Target B1 Delam Macro F1 = 2.84%, Defect Recall = 11.35%
  * `fold_1_seed123`: Target B1 Delam Macro F1 = 11.14%, Defect Recall = 34.75%
  * `fold_1_seed999`: Target B1 Delam Macro F1 = 14.45%, Defect Recall = 53.20%

---

## 3. Outstanding Work & Next Immediate Actions

1. **Complete Remaining 12 Baseline Runs:**
   - Execute `python resume_experiment.py` on the target machine.
   - The runner will verify Fold 1 runs (skipping them without retraining) and execute Folds 2 through 5 across seeds `42`, `123`, and `999`.
2. **Run Comprehensive Validation Suite:**
   - Execute `python baseline_validation_tests.py`.
   - Verify all 10 integrity tests pass (circular fold matching, zero target leakage, ignore_index=255 masking, source-only normalization, full-coverage stitching, SHA-256 pre-evaluation freezing, confusion matrix mathematical consistency, and determinism).
3. **Compile Fleet Summary Tables:**
   - Inspect `baseline_metrics_by_bridge.csv` (mean $\pm$ standard deviation across seeds for each bridge).
   - Inspect `baseline_macro_summary.csv` (macro fleet-wide baseline average).
4. **Author Final Baseline Report:**
   - Finalize `phase2b2_baseline_report.md` evaluating the baseline findings, Class 3 drop-off under cross-bridge domain shift, and decision gate readiness.
5. **Phase 2C Transition (Future Phase):**
   - Once Phase 2B.2 is completely signed off, proceed to the separately specified UDA experiments (e.g., Deep CORAL / Adversarial Domain Adaptation). Do NOT start UDA automatically.
