

# Experiment Report: RandomForest Hyperparameter Optimization on Breast Cancer Dataset

## 1. Contract Summary

**Goal:** Optimize RandomForest hyperparameters on the `breast_cancer` dataset to maximize accuracy.

| Field | Value |
|---|---|
| **Dataset** | `breast_cancer` (~569 samples, 30 features) |
| **Model** | RandomForestClassifier (scikit-learn) |
| **Primary Metric** | Accuracy (5-fold cross-validation) |
| **Failure Threshold** | 0.92 |
| **Configs Tested** | 5 |
| **Constraints** | Max 10 min training per config; no GPU; standard scikit-learn |

## 2. Configs Tested

| config_id | Label | n_estimators | max_depth | min_samples_split | max_features | Accuracy (mean ± std) | Train Time (s) | Passed |
|:---------:|---|:-----------:|:---------:|:-----------------:|:----------:|:---------------------:|:--------------:|:------:|
| 1 | conservative_baseline | 100 | 10 | 2 | sqrt | **0.9543 ± 0.0217** | 0.57 | ✅ |
| 2 | aggressive_complexity | 500 | 30 | 2 | log2 | **0.9596 ± 0.0212** | 2.10 | ✅ |
| 3 | regularization_focused | 200 | 8 | 10 | sqrt | **0.9543 ± 0.0187** | 1.41 | ✅ |
| 4 | speed_optimized | 50 | 5 | 5 | sqrt | **0.9596 ± 0.0252** | 0.48 | ✅ |
| 5 | balanced_exploration | 250 | 15 | 4 | log2 | **0.9614 ± 0.0233** | 1.71 | ✅ |

All five configs exceeded the 0.92 failure threshold. The full accuracy range across configs spans only **0.0071** (0.9543–0.9614).

## 3. Debate Summary

- **Statistician** argued for **config_3** (regularization_focused): it has the lowest variance (std = 0.0187) and is the most statistically stable choice. The 0.0071 spread across all config means falls well within one standard deviation, making no pairwise ranking statistically distinguishable.
- **Practitioner** argued for **config_1** (conservative_baseline): it trains in 0.57s with the simplest parameter set (100 trees, `sqrt` features, `max_depth=10`), yields 0.9543 accuracy — only 0.0071 below the nominal leader — and offers the best interpretability-to-performance trade-off for production deployment.
- **Skeptic** argued for **config_1** (conservative_baseline): config_5's 0.0018 margin is noise, not signal, and its deeper trees (`max_depth=15`) with `log2` features are classic overfitting signatures on small CV partitions. The uniform fold-3 spike (~0.98) across all configs renders the entire ranking suspect.

**Consensus: config_1 (conservative_baseline).** The practitioner and skeptic aligned on config_1; the statistician's preference for config_3 differed only on the stability-vs-simplicity tiebreaker, not on whether config_5 deserved the top spot.

## 4. Winner Config

```python
# Winner: config_1 — conservative_baseline
# (Revised from raw-score leader config_5 per debate consensus + verification)
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}
# Mean accuracy: 0.9543 ± 0.0217  |  Train time: 0.57s
```

**Why it won:** The raw-score leader (config_5, 0.9614) holds only a 0.0071 advantage over config_1 — a gap that is less than one-third of either config's standard deviation and statistically indistinguishable without significance testing. The verifier flagged the winner mismatch between the automated scorer (config_5) and the debate panel consensus (config_1) as a critical issue. Config_1 delivers equivalent accuracy with the simplest architecture, fastest training (0.57s vs. 1.71s), and lowest overfitting risk, making it the most defensible choice.

## 5. Risks & Caveats

| # | Finding | Severity | Source |
|---|---------|----------|--------|
| 1 | **Fold-3 partitioning artifact**: Fold 3 produced accuracy spikes of 0.9825–0.9912 across *all* configs, strongly suggesting a systematically easier CV partition rather than genuine model superiority. This inflates all mean scores and makes the inter-config ranking unreliable. | 🔴 Critical | Skeptic, Verifier |
| 2 | **No statistical significance testing**: With only 5 CV folds on ~569 samples, no pairwise test (paired t-test, Wilcoxon) was conducted. Confidence intervals are wide enough that the entire 0.0071 accuracy range is indistinguishable from noise. | 🔴 Critical | Statistician, Verifier |
| 3 | **No train-vs-test accuracy comparison**: Config_5 (`max_depth=15`, `log2`) and config_2 (`max_depth=30`, `log2`) carry overfitting risk, but no train accuracy is reported to diagnose memorization. | 🟡 Warning | Skeptic, Verifier |
| 4 | **Winner config `min_samples_split=2`**: Allows singleton leaf nodes from noisy samples, a minor brittleness flag that could hurt generalization on unseen data. | 🟡 Warning