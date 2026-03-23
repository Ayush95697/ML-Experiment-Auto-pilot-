

# Experiment Report: Optimise RandomForest on Titanic for Accuracy

## 1. Contract Summary

| Field | Value |
|---|---|
| **Goal** | Optimise a RandomForest classifier on the Titanic dataset for accuracy |
| **Dataset** | Titanic (CSV at `C:\Users\Ayush\AppData\Local\Temp\tmpxg9he20v.csv`) |
| **Model** | RandomForest |
| **Primary Metric** | Accuracy |
| **Configs to Test** | 5 |
| **Failure Threshold** | 0.85 (minimum acceptable accuracy) |
| **Target Column (as specified)** | `PassengerId\tSurvived\tPclass\tName\tSex\tAge\tSibSp\tParch\tTicket\tFare\tCabin\tEmbarked` |

**Critical contract defect:** The `target_column` field contains a tab-delimited string of all CSV headers instead of a single column name (likely `Survived`). This malformation is the root cause of total experimental failure.

## 2. Configs Tested

| config_id | label | key params | accuracy score |
|---|---|---|---|
| 1 | Config 1 | — | ❌ FAILED |
| 2 | Config 2 | — | ❌ FAILED |
| 3 | Config 3 | — | ❌ FAILED |
| 4 | Config 4 | — | ❌ FAILED |
| 5 | Config 5 | — | ❌ FAILED |

All 5 configurations failed with the identical error:

```
Found array with 0 feature(s) (shape=(156, 0)) while a minimum of 1 is required by StandardScaler.
```

**Zero accuracy scores were produced.** No configuration can be compared against the 0.85 failure threshold. No mean, standard deviation, or variance can be computed.

## 3. Debate Summary

- **Statistician** argued that with zero valid accuracy scores across all five configs, no statistical analysis is possible — zero means, zero variances, zero signal. The uniform failure pattern is diagnostic of a systemic upstream data-parsing issue, not a hyperparameter issue.
- **Practitioner** argued that evaluation on simplicity, speed, or generalization is impossible when no config produced a single prediction. The shared preprocessing failure reveals a pipeline design flaw: the absence of a basic input-validation layer (e.g., `assert X.shape[1] > 0`) before model training.
- **Skeptic** argued this is a complete experimental collapse originating upstream of hyperparameter tuning. The malformed `target_column` field caused the preprocessing pipeline to strip all features, and the experiment provides exactly zero evidence about model performance.

**Consensus: No winner (`config_#NONE`).** All three personas agreed unanimously. There was no disagreement on outcome — only minor differences in framing (statistical signal vs. pipeline design vs. upstream root cause).

## 4. Winner Config

```
# NO WINNER
# All 5 configurations failed before producing any predictions.
# Root cause: malformed target_column field in the experiment contract.
# The feature matrix collapsed to shape=(156, 0) during preprocessing.
```

**No configuration won** because no configuration produced a valid accuracy score. The failure is not attributable to any hyperparameter choice; it is a data-ingestion defect that prevented all configs from executing.

## 5. Risks & Caveats

### Critical Issues (from Verifier — 4 critical, 2 warnings)

1. **Total experimental failure:** All 5 configs produced runtime errors. Zero accuracy scores exist. The experiment yields no valid results and cannot support any model comparison or winner selection.
2. **Malformed contract:** The `target_column` field is a tab-separated string of all column headers (`PassengerId\tSurvived\tPclass\t...`) rather than a single column name. This caused the preprocessing pipeline to select zero features, producing `shape=(156, 0)` for every configuration.
3. **Threshold unassessable:** The primary metric (accuracy) was never evaluated. The 0.85 failure threshold cannot be assessed. There is zero evidence that RandomForest on this dataset meets or fails the threshold.
4. **No defensive validation:** The pipeline lacks a post-preprocessing feature-shape assertion. All five configs reached `StandardScaler` with an empty matrix rather than failing fast at the point of ingestion. This silent propagation of the error wasted all five configuration runs.
5. **Unverified root cause:** The diagnosis (malformed `target_column`) is the most plausible explanation but has not been confirmed by direct inspection of the CSV file or pipeline source code. This should be verified explicitly before rerunning.

## 6. Recommended Next Steps

Based on the outlier insight that all five configs failed identically and silently — revealing a critical absence of defensive validation in the pipeline — the following steps are required:

1. **Fix the `target_column` field** in the experiment contract to contain only the single target column name (e.g., `"Survived"`). Verify the CSV at the specified path is correctly formatted, tab-vs-comma delimited as expected, and readable with the intended parser.

2. **Add a post-preprocessing assertion** immediately after feature selection / encoding:
   ```python
   assert X.shape[1] > 0, f"Feature matrix has 0 columns after preprocessing. Check target_column and CSV parsing."
   ```
   This single check would have surfaced the CSV parsing bug before any of the 5 configs reached `StandardScaler`, saving the entire cost of this failed experiment run.

3. **Rerun all 5 configurations from scratch** after fixes are applied. Report per-fold accuracy scores, mean ± std across CV folds, and train-vs-test accuracy to check for overfitting. Only after obtaining valid results can a winner be declared and compared against the 0.85 failure threshold. Per learned rules, apply statistical significance testing before declaring any winner — marginal score differences within standard error are not sufficient.