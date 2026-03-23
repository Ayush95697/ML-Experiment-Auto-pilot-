

# Experiment Report: Optimise LogisticRegression on Titanic for Accuracy

## 1. Contract Summary

| Field | Value |
|---|---|
| **Goal** | Optimise LogisticRegression on the Titanic dataset for accuracy |
| **Dataset** | Titanic (`Survived` as target column) |
| **Model** | `LogisticRegression` (scikit-learn) |
| **Primary Metric** | Accuracy (5-fold cross-validation) |
| **Configs Tested** | 5 |
| **Failure Threshold** | 0.85 |
| **Outcome** | **FAILED** — No configuration met the 0.85 threshold. Best mean accuracy: **0.7065** (config 3), 14.35 pp below threshold. |

---

## 2. Configs Tested

| config_id | Label | C | Solver | max_iter | Penalty | Mean Accuracy | Std | Passed? |
|---|---|---|---|---|---|---|---|---|
| 1 | conservative_baseline | 1.0 | lbfgs | 100 | l2 | 0.6846 | 0.1341 | ❌ |
| 2 | aggressive_learning | 10.0 | lbfgs | 500 | l2 | 0.6846 | 0.1341 | ❌ |
| 3 | **regularisation_focused** | **0.1** | **lbfgs** | **200** | **l2** | **0.7065** | **0.0305** | ❌ |
| 4 | speed_optimised | 1.0 | liblinear | 50 | l2 | 0.6846 | 0.1341 | ❌ |
| 5 | elastic_exploration | 0.5 | saga | 300 | l1 | 0.6757 | 0.0748 | ❌ |

**Critical anomaly:** Configs 1, 2, and 4 produced **identical** cv_scores `[0.4231, 0.8, 0.72, 0.76, 0.72]` despite C ranging from 1.0 to 10.0 and different solvers (`lbfgs` vs `liblinear`). This is statistically implausible and suggests a pipeline bug, caching artefact, or data leakage that invalidates those three results.

---

## 3. Debate Summary

- **Statistician** — Backed config 3 on the basis of its 4× lower variance (std 0.0305 vs 0.1341) and highest mean (0.7065). Flagged that the identical scores across configs 1/2/4 render their means "nearly meaningless." Stressed the 0.7065 result is still 14 pp below the 0.85 threshold, making the win purely relative.

- **Practitioner** — Agreed on config 3 for production reliability: a tight fold range of 0.68–0.76 allows reasoning about real-world behaviour, whereas configs 1/2/4's instability (0.42–0.80) is unusable. Argued the identical scores across varying C values prove feature saturation, meaning stronger regularisation (C=0.1) is doing genuine useful work. Concluded that hyperparameter tuning alone is "rearranging deck chairs" and the bottleneck is feature quality.

- **Skeptic** — Concurred config 3 wins "by default in a field of failures." Questioned whether config 3's low std reflects true stability or accidental fold homogeneity. Warned that accuracy on a class-imbalanced dataset like Titanic may mask poor recall/precision and that the experiment needs upstream fixes, not more hyperparameter search.

**Consensus:** All three personas unanimously selected **config 3**. Key disagreement centred on *interpretation* of config 3's low variance (genuine signal vs. artefact) and whether a winner should be declared at all given all configs fail the threshold.

---

## 4. Winner Config

```python
# Config 3 — regularisation_focused (PROVISIONAL)
LogisticRegression(
    C=0.1,
    solver="lbfgs",
    max_iter=200,
    penalty="l2"
)
```

**Why it won (provisionally):**

1. **Highest mean accuracy:** 0.7065 — the only config above 0.70.
2. **Lowest variance:** std = 0.0305, a 4.4× reduction versus configs 1/2/4 (std = 0.1341). Fold scores ranged 0.68–0.76 compared to 0.42–0.80.
3. **No pathological fold:** Configs 1/2/4 all collapsed to 0.4231 on fold 1; config 3's worst fold was 0.68.
4. **Unanimous debate consensus** across all three personas.

⚠️ **This is a provisional winner.** Mean accuracy of 0.7065 **does not pass** the 0.85 failure threshold. The experiment cannot be marked complete.

---

## 5. Risks & Caveats

| # | Severity | Finding | Source |
|---|---|---|---|
| 1 | 🔴 Critical | **All 5 configs fail the 0.85 threshold.** Best score is 14.35 pp below. No config is production-viable. | Verifier |
| 2 | 🔴 Critical | **Configs 1, 2, and 4 share identical cv_scores** despite C ∈ {1.0, 10.0} and different solvers. This is statistically implausible and indicates a potential pipeline bug, caching artefact, or data leakage corrupting 60% of the comparison set. | Verifier, Statistician |
| 3 | 🟡 Warning | **Config 3's low std (0.0305) is unverified.** It may reflect accidental fold stratification or class-balance alignment rather than genuine model stability. No stratification or class-balance audit was performed. |