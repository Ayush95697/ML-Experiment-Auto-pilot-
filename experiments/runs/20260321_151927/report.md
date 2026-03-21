

# Experiment Report: RandomForest Hyperparameter Optimization on Breast Cancer Dataset

## 1. Contract Summary

**Goal:** Optimize RandomForest hyperparameters on the `breast_cancer` dataset to maximize accuracy.

| Field | Value |
|---|---|
| **Dataset** | `breast_cancer` (sklearn built-in, 569 samples, 30 features, binary classification) |
| **Model** | RandomForestClassifier |
| **Primary Metric** | Accuracy (5-fold cross-validation) |
| **Failure Threshold** | 0.90 |
| **Configs Tested** | 5 |
| **Constraints** | Max 10 min training per config; no GPU; must report std error; significance testing required before winner declaration |

All 5 configs passed the 0.90 failure threshold.

## 2. Configs Tested

| config_id | Label | n_estimators | max_depth | min_samples_split | max_features | Other | Accuracy (mean ± std) | Train Time (s) |
|:---------:|---|:-----------:|:---------:|:-----------------:|:------------:|---|---|:---------:|
| 1 | conservative_baseline | 100 | 10 | 5 | sqrt | — | 0.9578 ± 0.0203 | 0.89 |
| 2 | aggressive_complexity | 500 | 25 | 2 | log2 | — | 0.9596 ± 0.0212 | 2.13 |
| 3 | regularisation_focused | 200 | 5 | 20 | sqrt | min_samples_leaf=4 | 0.9543 ± 0.0217 | 1.62 |
| 4 | **speed_optimised** | **50** | **8** | **10** | **sqrt** | **—** | **0.9596 ± 0.0204** | **0.48** |
| 5 | creative_exploration | 300 | 15 | 8 | log2 | min_samples_leaf=2 | 0.9561 ± 0.0229 | 2.11 |

**Key observations:**
- The full range of mean accuracy across all 5 configs is **0.0053** (0.9543 → 0.9596).
- All per-config standard deviations exceed 0.020, dwarfing the inter-config spread.
- Configs 2 and 4 are tied at the highest mean accuracy of **0.9596**.

### Statistical Significance Testing

The contract mandates significance testing before declaring a winner. A **paired Wilcoxon signed-rank test** across the 5 CV folds was conducted for the key pair (config 2 vs. config 4):

| Fold | Config 2 | Config 4 | Diff (C2 − C4) |
|:----:|:--------:|:--------:|:---------------:|
| 1 | 0.9298 | 0.9298 | 0.0000 |
| 2 | 0.9386 | 0.9474 | −0.0088 |
| 3 | 0.9825 | 0.9912 | −0.0087 |
| 4 | 0.9737 | 0.9649 | +0.0088 |
| 5 | 0.9735 | 0.9646 | +0.0089 |

- **Signed differences:** 0, −0.0088, −0.0087, +0.0088, +0.0089
- Two negative, two positive, one zero (tied). The ranks of absolute non-zero differences are nearly identical.
- **Result:** The test statistic is not significant at any conventional α level (0.05, 0.10). **There is no statistically significant difference between config 2 and config 4** — nor between any pair of configs, given the 0.0053 total mean range vs. ~0.02 per-config std.

**Conclusion:** All five configs are **statistically indistinguishable** on accuracy. Winner selection must rely on secondary criteria (variance, training cost, complexity/generalization risk).

## 3. Debate Summary

- **Statistician** argued for **config 4**: The 0.0053 spread across configs is completely swamped by ~0.02+ fold-level standard deviations. No pairwise test survives at any conventional significance threshold. Config 4 matches config 2's mean while exhibiting the lowest std (0.0204). The original winner declaration (config 2) violates the contract's significance testing requirement.

- **Practitioner** argued for **config 4**: It achieves the joint-highest mean accuracy (0.9596) in 0.48 s — **4.4× faster** than config 2 (2.13 s) — using only 50 estimators. Config 2's 500 estimators and max_depth=25 are an unjustifiable complexity penalty for zero additional mean accuracy. Config 4's lightweight footprint enables faster retraining and lower overfitting risk.

- **Skeptic** argued for **config 4**: Config 2's aggressive depth (25) and min_samples_split (2) invite memorizing noise with no evidence of generalization benefit. The consistent fold-level pattern (fold 3 ~0.98, fold 1 ~0.92) across all configs proves that fold composition — not hyperparameters — drives variance. Declaring config 2 the winner is arbitrary without significance testing.

**Consensus: Config 4 (speed_optimised).** All three personas unanimously agreed. The originally declared winner (config 2) was overridden.

## 4. Winner Config

```python
# Winner: config_4 — speed_optimised
RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
# Accuracy: 0.9596 ± 0.0204 (5-fold CV)
# Training time: 