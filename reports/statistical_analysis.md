# Statistical Analysis & Power Protocol: Cross-Bridge Delamination

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Domain Unit of Analysis:** Bridge Deck ($N=5$)  
**Stochastic Repeats:** 3 Independent Seeds ($42, 123, 999$)  
**Date:** 2026-10-20  

---

## 1. Unit of Comparison & Aggregation Protocol

A frequent methodological error in deep learning applied to civil infrastructure is treating overlapping patches as independent statistical observations ($N=322$). Because overlapping patches share physical pixels and originate from the same structure, treating them as independent inflates sample size, induces autocorrelation, and produces artificially deflated $p$-values.

In this research:
- The **bridge structure ($N=5$) is the sole independent domain-level unit of analysis**.
- Seeds ($42, 123, 999$) represent repeated stochastic draws within a bridge domain.
- For each bridge domain $B_i$, the domain-level baseline performance is the mean across seeds:
  $$\bar{M}_{\text{baseline}}(B_i) = \frac{1}{3} \sum_{s \in \{42, 123, 999\}} M_{\text{baseline}}(B_i, s)$$
  $$\bar{M}_{\text{CORAL}}(B_i) = \frac{1}{3} \sum_{s \in \{42, 123, 999\}} M_{\text{CORAL}}(B_i, s)$$

---

## 2. Mathematical Limits of Statistical Power with $N=5$ Bridges

Because the benchmark dataset contains five highway bridge decks (B1–B5), statistical inference under non-parametric hypothesis testing is bounded by strict combinatorial constraints:

### Paired Wilcoxon Signed-Rank Test:
For $N=5$ paired differences $D_i = \bar{M}_{\text{CORAL}}(B_i) - \bar{M}_{\text{baseline}}(B_i)$:
- Number of possible sign permutations: $2^5 = 32$.
- The maximum test statistic $W^+ = \sum_{i=1}^5 i = 15$ occurs when all five differences are strictly positive ($D_i > 0$).
- Under the null hypothesis $H_0$, the exact probability of observing $W^+ = 15$ is:
  $$p_{\text{one-sided}} = \frac{1}{32} = 0.03125$$
  $$p_{\text{two-sided}} = \frac{2}{32} = 0.0625$$

### Critical Scientific Deduction:
Under a conventional two-sided significance threshold of $\alpha = 0.05$, **it is mathematically impossible to reject the null hypothesis with $N=5$ bridge domains**, even if Deep CORAL outperforms the baseline on 100% of the bridges.

Therefore:
1. **No Claims of Significance:** The study will **not** claim "statistically significant improvement at $p < 0.05$ (two-sided)". Any paper making such a claim on five bridges is mathematically fallacious.
2. **Effect Sizes:** Evaluation must report non-parametric effect sizes (e.g., Wilcoxon $r = W / \sqrt{N}$), absolute delamination F1 gain $\Delta$, and relative percentage gain.
3. **Predeclared Practical Threshold:** We evaluate against the project's predeclared engineering threshold of:
   $$\Delta \text{Macro Delamination F1} \ge +0.05 \quad (+5.0\%)$$
   labeled explicitly as a project-specific engineering objective rather than a universal statistical standard.

---

## 3. Empirical Baseline Findings Across All 5 Bridges (N=5 Domains)

| Target Bridge Domain | Seed 42 | Seed 123 | Seed 999 | Domain Mean $\pm$ SD | Defect Recall | Sound F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bridge 1 (Forest River NB)** | 2.84% | 11.14% | 14.45% | **$9.48 \pm 5.98\%$** | $33.10 \pm 20.97\%$ | $80.84 \pm 7.16\%$ |
| **Bridge 2 (Forest River SB)** | 6.66% | 17.00% | 18.35% | **$14.00 \pm 6.40\%$** | $76.77 \pm 40.24\%$ | $57.00 \pm 20.26\%$ |
| **Bridge 3 (Park River Median)** | 9.10% | 0.00% | 19.24% | **$9.45 \pm 9.62\%$** | $48.94 \pm 43.97\%$ | $80.18 \pm 7.87\%$ |
| **Bridge 4 (Park River NB)** | 12.44% | 12.88% | 16.93% | **$14.08 \pm 2.48\%$** | $57.52 \pm 35.14\%$ | $61.15 \pm 39.88\%$ |
| **Bridge 5 (Park River SB)** | 12.97% | 14.62% | 14.84% | **$14.14 \pm 1.02\%$** | $65.86 \pm 20.07\%$ | $68.63 \pm 10.44\%$ |
| **Fleet Macro Average (15 Runs)** | — | — | — | **$12.23 \pm 5.54\%$** | **$56.44 \pm 32.26\%$** | **$69.56 \pm 20.45\%$** |

### Key Observations Across the 5-Bridge Fleet:
1. **Severe Sub-Rebar (Class 3) Delamination Collapse:**
   - Fleet-wide Class 3 F1 averaged only **$0.45 \pm 1.72\%$**.
   - In 11 out of 15 runs, Class 3 F1 was exactly **0.00%** (zero true positive detections).
2. **Above-Rebar (Class 2) Delamination Sensitivity:**
   - Fleet-wide Class 2 F1 averaged **$24.02 \pm 11.16\%$**, indicating models capture superficial near-surface delaminations with modest capability but struggle with generalization across differing solar/structural conditions.
3. **Stochastic Optimization Sensitivity:**
   - Seed variation accounts for up to a $19.24\%$ performance spread on the same bridge (e.g., Bridge 3 from 0.00% in seed 123 to 19.24% in seed 999), proving that multi-seed reporting is mathematically mandatory for cross-bridge domain adaptation evaluation.
4. **Baseline Benchmark Ceiling:**
   - The established baseline benchmark target is **$12.23\%$ Macro Delamination F1**. Unsupervised Domain Adaptation (Deep CORAL) must target $\Delta \ge +5.0\%$ Macro F1 to achieve practical significance.
