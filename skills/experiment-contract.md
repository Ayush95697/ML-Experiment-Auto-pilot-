---
name: experiment-contract
description: Before running any experiment, generate a structured contract with goal, constraints, metrics, and failure conditions. Trigger on any new experiment request.
---

## Steps
1. Ask the user these 5 questions (all at once, numbered):
   1. What dataset are you using? (name or path)
   2. What model type? (RandomForest, SVM, LogisticRegression, XGBoost, etc.)
   3. What metric matters most? (accuracy, f1, roc_auc, rmse, etc.)
   4. How many config variants to test? (3–10 recommended)
   5. What is the minimum acceptable metric value? (failure threshold)

2. Once answered, output ONLY valid JSON:
{
  "goal": "one sentence describing what we're optimising",
  "dataset": "...",
  "model": "...",
  "primary_metric": "...",
  "n_configs": N,
  "failure_threshold": 0.XX,
  "constraints": ["max 10 min training time", "no GPU required"],
  "output_format": "markdown report with metric table + winner config"
}

3. Ask: "Does this contract look correct? Reply yes to proceed."