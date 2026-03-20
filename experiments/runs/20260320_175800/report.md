

# Experiment Report: RandomForest Hyperparameter Optimization on Breast Cancer Dataset

## 1. Contract Summary

| Field | Detail |
|---|---|
| **Goal** | Optimize RandomForest hyperparameters on the `breast_cancer` dataset to maximize accuracy |
| **Dataset** | `breast_cancer` (sklearn built-in, ~569 samples, 30 features, binary classification) |
| **Model** | `RandomForestClassifier` (scikit-learn) |
| **Primary Metric** | Accuracy (stratified 5-fold cross-validation) |
| **Failure Threshold** | 0.92 (any config below this is rejected) |
| **Configs Tested** | 5 |
| **Constraints** | Max 10 min training time per config; no GPU required; stratified 5-fold CV |
| **Output Format** | Markdown report with metric table + winner config |

## 2. Configs Tested

| config_id | Label | n_estimators | max_depth | min_samples_split | max_features | Other | Accuracy (mean ± std) | Train Time (s) | Passed |
|:-:|---|:-:|:-:|:-:|---|---|:-:|:-:|:-:|
| 1 | conservative_baseline | 100 | 10 | 5 | sqrt | — | 0.9578 ± 0.0203 | 0.67 | ✅ |
| 2 | aggressive_complexity | 500 | 30 | 2 | log2 | — | 0.9596 ± 0.0212 | 3.01 | ✅ |
| 3 | regularisation_focused | 200 | 5 | 20 | sqrt | min_samples_leaf=4 | 0.9543 ± 0.0217 | 1.41 | ✅ |
| **4** | **speed_optimised** | **50** | **8** | **10** | **sqrt** | **—** | **0.9596 ± 0.0204** | **0.47** | **✅** |
| 5 | creative_exploration | 300 | 15 | 8 | 0.8 | min_samples_leaf=2 | 0.9579 ± 0.0262 | 1.69 | ✅ |

**Per-fold CV scores:**

| config_id | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.9298 | 0.9386 | 0.9825 | 0.9649 | 0.9735 |
| 2 | 0.9298 | 0.9386 | 0.9825 | 0.9737 | 0.9735 |
| 3 | 0.9211 | 0.9386 | 0.9825 | 0.9649 | 0.9646 |
| **4** | **0.9298** | **0.9474** | **0.9912** | **0.9649** | **0.9646** |
| 5 | 0.9123 | 0.9474 | 0.9825 | 0.9649 | 0.9823 |

All 5 configs passed the 0.92 failure threshold. The spread between the worst mean (config 3, 0.9543) and the best mean (configs 2 & 4, 0.9596) is only **0.0053**, which is smaller than every config's standard deviation.

## 3. Debate Summary

- **Statistician** argued that configs 2 and 4 are tied at 0.9596 mean accuracy, and with overlapping confidence intervals across all five configs (within-config variance spans up to 6.1 pp), no config has a statistically defensible advantage over any other. Recommended **config 4** as the tie-break is arbitrary on statistical grounds alone.
- **Practitioner** argued that config 4 delivers the same 0.9596 accuracy as config 2 in **0.47 s vs. 3.01 s** (6.4× speedup), with a simpler parameter set (50 trees, max_depth=8) that is far easier to deploy, debug, and maintain in production. Flagged config 2's aggressive depth (30) and min_samples_split (2) as overfitting risks. Recommended **config 4**.
- **Skeptic** argued that config 2's aggressive hyperparameters are classic overfitting signatures on a ~569-sample dataset, that the 0.92 threshold is too lenient to discriminate, and that identical CV scores across folds 1–3 for configs 1, 2, and 4 raise structural concerns about the CV setup itself. Recommended **config 4**, contingent on auditing the fold splits and the 0.9912 outlier.

**All three personas reached consensus: config 4 is the winner.** They disagreed only on the primary *reason* to reject config 2 (statistical indistinguishability vs. production overhead vs. overfitting risk).

## 4. Winner Config

> **Note:** The automated pipeline originally declared config 2 as the winner (first config encountered at the max mean score). Post-debate and post-verification, the winner is revised to **config 4** based on unanimous expert consensus: identical accuracy, 6.4× faster training, lower model complexity, and reduced overfitting risk.

```python
# Config 4 — speed_optimised (REVISED WINNER)
{
    "n_estimators": 50,
    "max_depth": 8,
    "min_samples_split": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}
# Mean accuracy: 0.9596 ± 0.0204 (stratified 5-fold CV)
# Training time: 0.47 seconds
```

**Why it won:**

1