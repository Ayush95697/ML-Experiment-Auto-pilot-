import json, time
from datetime import datetime
from pathlib import Path
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np


DATASETS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
}
MODELS = {
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "LogisticRegression": LogisticRegression,
}

def load_dataset(name: str):
    loader = DATASETS.get(name.lower())
    if not loader:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    data = loader()
    return data.data, data.target

def build_model(model_name: str, params: dict):
    cls = MODELS.get(model_name)
    if not cls:
        raise ValueError(f"Unknown model: {model_name}")
    # Filter params to only valid ones for the model
    import inspect
    valid = inspect.signature(cls.__init__).parameters
    clean = {k: v for k, v in params.items() if k in valid}
    return cls(**clean)

def run_single_config(config: dict, contract: dict) -> dict:
    X, y = load_dataset(contract["dataset"])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = build_model(contract["model"], config["params"])
    metric = contract.get("primary_metric", "accuracy")

    start = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring=metric)
    elapsed = round(time.time() - start, 2)

    return {
        "config_id": config["config_id"],
        "label": config["label"],
        "params": config["params"],
        "metric": metric,
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score":  round(float(np.std(scores)), 4),
        "cv_scores":  [round(float(s), 4) for s in scores],
        "train_time_sec": elapsed,
        "passed_threshold": float(np.mean(scores)) >= contract.get("failure_threshold", 0.0)
    }

def run_all_experiments(contract: dict, configs: list[dict]) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = Path(__file__).resolve().parent.parent
    run_dir = BASE_DIR / "experiments" / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    (run_dir / "contract.json").write_text(json.dumps(contract, indent=2))
    (run_dir / "configs.json").write_text(json.dumps(configs, indent=2))

    print(f"\n[Runner] Starting {len(configs)} experiments...")
    results = []
    for cfg in configs:
        print(f"  Running config #{cfg['config_id']}: {cfg['label']}...", end=" ")
        try:
            r = run_single_config(cfg, contract)
            results.append(r)
            print(f"{r['metric']}={r['mean_score']} ± {r['std_score']}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"config_id": cfg["config_id"], "error": str(e)})

    # THIS PART WAS MISSING — add everything below
    valid = [r for r in results if "mean_score" in r]
    winner = max(valid, key=lambda r: r["mean_score"]) if valid else None

    output = {
        "run_id": timestamp,
        "results": results,
        "winner_config_id": winner["config_id"] if winner else None
    }

    (run_dir / "results.json").write_text(json.dumps(output, indent=2))
    print(f"\n[Runner] Done. Winner: config #{winner['config_id'] if winner else 'none'}")
    print(f"[Runner] Saved to: {run_dir}")
    return output, run_dir

if __name__ == "__main__":
    if __name__ == "__main__":
        from agents.contract_agent import run_contract
        from agents.config_agent import generate_configs

        contract = run_contract("Test RandomForest on iris, optimize for accuracy")
        configs = generate_configs(contract)
        run_output, run_dir = run_all_experiments(contract, configs)
        print(json.dumps(run_output, indent=2))