

# Experiment Report: Optimise RandomForest on Titanic for Accuracy

## 1. Contract Summary

**Goal:** Optimise a RandomForest classifier on the Titanic dataset for accuracy.

| Field | Value |
|---|---|
| **Dataset** | Titanic (`Survived` target column) |
| **Model** | RandomForest |
| **Primary Metric** | Accuracy (5-fold CV) |
| **Configs Tested** | 5 |
| **Failure Threshold** | 0.85 |
| **Outcome** | **FAILED** — no configuration met the 0.85 threshold. Best mean accuracy was 0.652, a shortfall of ~0.20. |

> ⚠️ **This experiment did not produce a deployable model.** All results below describe relative performance among failing configurations only.

---

## 2. Configs Tested

| config_id | Label | n_estimators | max_depth | min_samples_split | max_features | Accuracy (mean ± std) | Train Time (s) | Passed? |
|:---------:|---|:-----------:|:---------:|:-----------------:|:----------:|:---------------------:|:-------------:|:------:|
| 1 | conservative_baseline | 100 | 10 | 10 | sqrt | **0.652 ± 0.078** | 0.55 | ❌ |
| 2 | aggressive_complexity | 500 | 20 | 2 | log2 | 0.604 ± 0.091 | 2.53 | ❌ |
| 3 | regularisation_focused | 150 | 5 | 20 | sqrt | 0.644 ± 0.093 | 0.79 | ❌ |
| 4 | speed_optimised | 50 | 8 | 8 | log2 | 0.620 ± 0.085 | 0.28 | ❌ |
| 5 | creative_exploration | 200 | 12 | 5 | sqrt | 0.644 ± 0.076 | 1.07 | ❌ |

**Fold-level detail (all configs):**

| Fold | Config 1 | Config 2 | Config 3 | Config 4 | Config 5 |
|:----:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 0.500 | 0.462 | 0.462 | 0.462 | 0.500 |
| 2 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 |
| 3 | 0.720 | 0.720 | 0.720 | 0.600 | 0.720 |
| 4 | 0.680 | 0.600 | 0.680 | 0.680 | 0.680 |
| 5 | 0.680 | 0.560 | 0.680 | 0.680 | 0.640 |

Fold 1 collapses to 0.46–0.50 across **every** configuration — a structural anomaly independent of hyperparameters.

---

## 3. Debate Summary

- **Statistician** selected Config 1 but emphasized the 0.008 margin over Configs 3 and 5 is statistically indistinguishable from noise (std ≈ 0.077–0.093, n=5 folds). Called the fold-1 collapse a likely stratification failure inflating variance across the board.

- **Practitioner** selected Config 1 for its textbook-reasonable defaults (sqrt features, max_depth=10), fastest competitive train time (0.55s), and noted Config 2's 5× slower runtime yields *worse* accuracy (0.604). Argued the bottleneck is clearly not hyperparameters but the data pipeline.

- **Skeptic** nominally selected Config 1 but challenged the validity of the entire experiment: the fold-1 collapse (0.46–0.50 vs. 0.68–0.72 in folds 2–5) screams data contamination or stratification failure, and the 0.20 gap below the 0.85 threshold means "we are debating which configuration failed least badly."

**Consensus:** Config 1 (unanimous, but qualified). All three personas agree the fold-1 anomaly is the most important finding and that no configuration should be considered production-ready.

---

## 4. Winner Config

```yaml
config_id: 1
label: conservative_baseline
params:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 10
  max_features: sqrt
mean_accuracy: 0.652
std_accuracy: 0.078
train_time_sec: 0.55
passed_threshold: false
```

**Why it "won" (with heavy caveats):**

1. Highest mean accuracy at 0.652, marginally above Configs 3 (0.6443) and 5 (0.644).
2. Second-fastest train time (0.55s vs. 0.28s for Config 4), with meaningful accuracy advantage over Config 4 (+0.032).
3. Most stable fold scores among top contenders (std = 0.078, lowest after Config 5's 0.076).

**However:** The 0.008 margin over Configs 3 and 5 falls within one standard error. A paired t-test or Wilcoxon signed-rank test across folds would not reject the null hypothesis of equal performance. This "winner" is the least-bad configuration in a failed experiment, not a statistically validated champion.

---

## 5. Risks & Caveats

### Critical Issues

1. **Experiment failure — threshold not met.** The best score (0.652) is 0.198 points below the 0.85 failure threshold. No configuration is deployable. Continued hyperparameter tuning within this RandomForest search space is unlikely to close a 20-point gap