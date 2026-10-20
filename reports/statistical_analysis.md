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

## 3. Empirical Baseline Findings on Bridge 1 (Target B1)

| Metric | Seed 42 | Seed 123 | Seed 999 | Bridge 1 Mean $\pm$ SD |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Delam F1** | 2.84% | 11.14% | 14.45% | **$9.48 \pm 5.99\%$** |
| **Class 2 Delam F1 (Above Rebar)** | 5.67% | 22.28% | 28.91% | **$18.95 \pm 11.97\%$** |
| **Class 3 Delam F1 (Below Rebar)** | 0.00% | 0.00% | 0.00% | **$0.00 \pm 0.00\%$** |
| **Defect Recall** | 11.35% | 34.75% | 53.20% | **$33.10 \pm 20.97\%$** |
| **Sound Concrete F1** | 86.53% | 83.18% | 72.80% | **$80.84 \pm 7.16\%$** |

### Key Observations:
- **Zero Detection of Class 3:** In 100% of baseline runs on Bridge 1, Class 3 delaminations yielded zero true positive pixels.
- **Stochastic Variance:** Seed variation accounts for a $11.61\%$ spread in Macro Delam F1 (from 2.84% to 14.45%), demonstrating that single-seed reporting is invalid for domain adaptation benchmarks on this dataset.
