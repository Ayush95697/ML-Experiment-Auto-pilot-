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
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASETS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
}
MODELS = {
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "LogisticRegression": LogisticRegression,
}

def load_dataset(name: str, csv_path: str = None, target_column: str = None):
    # Built-in sklearn dataset
    if name.lower() in DATASETS:
        loader = DATASETS[name.lower()]
        data = loader()
        return data.data, data.target

    # CSV file — either from path or uploaded temp file
    if csv_path:
        df = pd.read_csv(csv_path, sep=None, engine="python")

        if not target_column:
            # Default: last column is the target
            target_column = df.columns[-1]

        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found. Available: {list(df.columns)}")

        X = df.drop(columns=[target_column])

        # Drop non-numeric columns automatically
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"[Runner] Dropping non-numeric columns: {non_numeric}")
            X = X.select_dtypes(include=[np.number])

        # Encode target if it's a string
        y = df[target_column]
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"[Runner] Encoded target classes: {list(le.classes_)}")

        # Drop rows with missing values
        mask = ~(X.isna().any(axis=1) | pd.isna(y))
        X, y = X[mask].values, np.array(y)[mask]

        print(f"[Runner] Loaded CSV: {X.shape[0]} rows, {X.shape[1]} features")
        return X, y

    raise ValueError(f"Unknown dataset: '{name}'. Use 'iris', 'breast_cancer', or provide a csv_path.")

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
        # Get csv_path from contract if provided
    csv_path = contract.get("csv_path", None)
    target_column = contract.get("target_column", None)

    X, y = load_dataset(contract["dataset"], csv_path=csv_path, target_column=target_column)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = build_model(contract["model"], config["params"])
    metric = contract.get("primary_metric", "accuracy")

    start = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring=metric)
    elapsed = round(time.time() - start, 2)
    mean_score = round(float(np.mean(scores)), 4)

    # Fix MLflow path — always save to project root
    BASE_DIR = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"file:///{BASE_DIR}/mlruns")   # ← add this line

    mlflow.set_experiment(f"autopilot_{contract['dataset']}_{contract['model']}")
    with mlflow.start_run(run_name=config["label"]):
        mlflow.log_params(config["params"])
        mlflow.log_param("config_id", config["config_id"])
        mlflow.log_param("framing", config.get("framing", ""))
        mlflow.log_metric(metric, mean_score)
        mlflow.log_metric(f"{metric}_std", round(float(np.std(scores)), 4))
        mlflow.log_metric("train_time_sec", elapsed)

    return {
        "config_id": config["config_id"],
        "label": config["label"],
        "params": config["params"],
        "metric": metric,
        "mean_score": mean_score,
        "std_score": round(float(np.std(scores)), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
        "train_time_sec": elapsed,
        "passed_threshold": mean_score >= contract.get("failure_threshold", 0.0)
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

    from agents.contract_agent import run_contract
    from agents.config_agent import generate_configs

    contract = run_contract("Test RandomForest on iris, optimize for accuracy")
    configs = generate_configs(contract)
    run_output, run_dir = run_all_experiments(contract, configs)
    print(json.dumps(run_output, indent=2))