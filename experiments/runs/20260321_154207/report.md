

# Experiment Report: RandomForest Hyperparameter Optimization on Breast Cancer Dataset

## 1. Contract Summary

**Goal:** Optimize RandomForest hyperparameters on the `breast_cancer` dataset (569 samples, 30 features) to maximize accuracy.

| Field | Value |
|---|---|
| **Dataset** | `breast_cancer` (sklearn) |
| **Model** | RandomForestClassifier |
| **Primary Metric** | Accuracy (5-fold CV, `random_state=42`) |
| **Failure Threshold** | 0.92 |
| **Configs Tested** | 5 |
| **Constraints** | ≤10 min train/config · no GPU · k=5 CV · `random_state=42` |
| **Required Output** | Markdown report with metric table, winner config, and statistical significance test |

All 5 configs exceeded the 0.92 failure threshold. **No statistical significance test was conducted**, which is a contract violation acknowledged below.

---

## 2. Configs Tested

| config_id | Label | n_estimators | max_depth | min_samples_split | max_features | Other | Accuracy (mean ± std) | Train Time (s) |
|:-:|---|:-:|:-:|:-:|:-:|---|:-:|:-:|
| 1 | conservative_baseline | 100 | 10 | 10 | sqrt | — | 0.9596 ± 0.0263 | 0.88 |
| 2 | aggressive_complexity | 500 | 30 | 2 | log2 | — | 0.9596 ± 0.0212 | 2.60 |
| 3 | regularisation_focused | 200 | 8 | 20 | sqrt | min_samples_leaf=5 | 0.9561 ± 0.0192 | 1.47 |
| 4 | **speed_optimised** | 50 | 12 | 8 | log2 | — | **0.9649 ± 0.0229** | **0.53** |
| 5 | balanced_exploration | 250 | 15 | 5 | sqrt | min_samples_leaf=2 | 0.9596 ± 0.0226 | 2.09 |

**Fold-level scores (for transparency):**

| Fold | Config 1 | Config 2 | Config 3 | Config 4 | Config 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.9211 | 0.9298 | 0.9298 | 0.9298 | 0.9211 |
| 2 | 0.9386 | 0.9386 | 0.9386 | 0.9474 | 0.9474 |
| 3 | **0.9912** | **0.9825** | **0.9825** | **0.9912** | **0.9825** |
| 4 | 0.9649 | 0.9737 | 0.9649 | 0.9825 | 0.9737 |
| 5 | 0.9823 | 0.9735 | 0.9646 | 0.9735 | 0.9735 |

**Key observation:** Fold 3 produces the highest score for every config (range 0.9825–0.9912), inflating all means. This anomaly is discussed in §5.

---

## 3. Debate Summary

- **Statistician** endorsed config 4 as the numerical leader (0.9649) but warned the 0.0053 margin over three tied configs is smaller than every config's within-fold standard deviation, making it **statistically indistinguishable from noise** on only 5 folds.
- **Practitioner** endorsed config 4 for its 3–5× speed advantage (0.53 s vs. 1.47–2.60 s) and lean design (50 estimators), arguing speed compounds into real operational value — while acknowledging the accuracy edge alone is marginal.
- **Skeptic** dissented, favoring config 3 (regularisation_focused) because its explicit regularisation (max_depth=8, min_samples_split=20, min_samples_leaf=5) is the most defensible design for unseen data, and argued the fold-3 spike may be artificially boosting less-regularised configs.

**Consensus: None.** The core disagreement is whether a 0.0053 noisy margin constitutes a real win, and whether speed should factor into an accuracy-focused evaluation.

---

## 4. Winner Config

**Provisional winner: Config 4 (`speed_optimised`)** — highest mean accuracy, fastest training time. **This declaration is provisional** because the margin is not statistically significant and the required significance test has not been performed.

```yaml
config_id: 4
label: speed_optimised
n_estimators: 50
max_depth: 12
min_samples_split: 8
max_features: log2
random_state: 42
n_jobs: -1

mean_accuracy: 0.9649
std_accuracy: 0.0229
train_time_sec: 0.53
```

**Why it is the provisional pick:**

1. **Numerically highest accuracy:** 0.9649 vs. 0.9596 for configs 1, 2, 5 and 0.9561 for config 3.
2. **Fastest training:** 0.53 s — 1.7× faster than the next fastest (config 1 at 0.88 s) and 4.9× faster than config 2 (2.60 s).
3. **Lowest complexity per unit accuracy:** 50 estimators achieve more than 500 (config 2), suggesting diminishing returns beyond ~50 trees on this dataset.

**Why this is NOT a confirmed winner:**

- The 0.0053 margin is **0.23× the winner's own std (0.0229)** — well below any reasonable significance threshold.
- A