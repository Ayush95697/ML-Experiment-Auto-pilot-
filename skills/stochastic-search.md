---
name: stochastic-search
description: Generate N hyperparameter configs with slight framing variations to explore the config space broadly. Use when spawning parallel experiment configs.
---

## Steps
Given a model type and dataset info, generate {n_configs} distinct hyperparameter
configurations. Each config must explore a DIFFERENT region of the search space.

Rules:
- Config 1: conservative defaults (safe baseline)
- Config 2: aggressive learning rate / high complexity
- Config 3: regularisation-focused (prevent overfitting)
- Config 4: speed-optimised (fast convergence)
- Config 5+: random creative exploration

Output ONLY valid JSON — a list of config objects:
[
  {
    "config_id": 1,
    "label": "conservative_baseline",
    "framing": "Safe defaults, low risk",
    "params": { ... model-specific params ... }
  },
  ...
]

For RandomForest params include: n_estimators, max_depth, min_samples_split, max_features
For SVM params include: C, kernel, gamma
For LogisticRegression params include: C, solver, max_iter, penalty