import streamlit as st
import json
import tempfile
import subprocess
import sys
import time
import os
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="ML Experiment AutoPilot", layout="wide")
st.title("ML Experiment AutoPilot")
st.caption("Multi-agent hyperparameter search with debate + verification")

BASE_DIR = Path(__file__).resolve().parent
is_local = os.environ.get("STREAMLIT_SERVER_HEADLESS") != "true"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment setup")

    dataset_mode = st.radio("Dataset source",
                            ["Built-in", "Upload CSV", "CSV file path"],
                            horizontal=True)

    csv_path = None
    target_column = None
    dataset = None

    if dataset_mode == "Built-in":
        dataset = st.selectbox("Dataset", ["iris", "breast_cancer"])

    elif dataset_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])

        if uploaded:
            if "tmp_csv_path" not in st.session_state or \
               st.session_state.get("tmp_csv_name") != uploaded.name:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded.read())
                tmp.close()
                st.session_state["tmp_csv_path"] = tmp.name
                st.session_state["tmp_csv_name"] = uploaded.name

            csv_path = st.session_state["tmp_csv_path"]
            dataset = uploaded.name.replace(".csv", "")

            df_preview = pd.read_csv(csv_path, sep=None, engine="python")
            st.caption(f"{df_preview.shape[0]} rows x {df_preview.shape[1]} cols")

            cols = df_preview.columns.tolist()
            target_column = st.selectbox(
                "Target column",
                cols,
                index=len(cols) - 1,
                key="target_col_upload"
            )
            st.caption(f"Selected target: **{target_column}**")
            st.caption(f"Features: {len(cols) - 1} columns")
        else:
            st.session_state.pop("tmp_csv_path", None)
            st.session_state.pop("tmp_csv_name", None)
            dataset = "uploaded_csv"
            st.caption("No file uploaded yet")

    elif dataset_mode == "CSV file path":
        csv_path = st.text_input(
            "Full path to CSV file",
            placeholder=r"C:\Users\Ayush\data\mydata.csv",
            key="csv_path_input"
        )
        if csv_path and Path(csv_path).exists():
            df_preview = pd.read_csv(csv_path, sep=None, engine="python")
            st.caption(f"{df_preview.shape[0]} rows x {df_preview.shape[1]} cols")

            cols = df_preview.columns.tolist()
            target_column = st.selectbox(
                "Target column",
                cols,
                index=len(cols) - 1,
                key="target_col_path"
            )
            st.caption(f"Selected target: **{target_column}**")
            dataset = Path(csv_path).stem
        elif csv_path:
            st.error("File not found at that path")
            dataset = "csv_path"
        else:
            dataset = "csv_path"

    st.divider()

    model     = st.selectbox("Model", ["RandomForest", "LogisticRegression", "SVM"])
    metric    = st.selectbox("Primary metric", ["accuracy", "f1_weighted", "roc_auc"])
    n_configs = st.slider("Configs to test", 3, 8, 5)
    threshold = st.slider("Minimum acceptable score", 0.5, 1.0, 0.85, step=0.01)

    ready = True
    if dataset_mode == "Upload CSV" and not csv_path:
        st.warning("Please upload a CSV file first")
        ready = False
    if dataset_mode == "CSV file path" and (not csv_path or not Path(csv_path).exists()):
        st.warning("Please enter a valid CSV file path")
        ready = False

    run_btn = st.button("Run AutoPilot", type="primary",
                        use_container_width=True, disabled=not ready)

    st.divider()

    # ── Run history ───────────────────────────────────────
    st.subheader("Run history")
    runs_dir = BASE_DIR / "experiments" / "runs"

    if runs_dir.exists():
        run_folders = sorted(
            [f for f in runs_dir.iterdir() if f.is_dir()],
            reverse=True
        )
        history = []
        for folder in run_folders:
            results_file  = folder / "results.json"
            contract_file = folder / "contract.json"
            if results_file.exists() and contract_file.exists():
                r = json.loads(results_file.read_text())
                c = json.loads(contract_file.read_text())
                valid = [x for x in r["results"] if "mean_score" in x]
                if valid:
                    best = max(valid, key=lambda x: x["mean_score"])
                    history.append({
                        "run_id":        folder.name,
                        "dataset":       c.get("dataset", "—"),
                        "model":         c.get("model", "—"),
                        "metric":        c.get("primary_metric", "—"),
                        "best_score":    best["mean_score"],
                        "best_config":   best["label"],
                        "configs_tested": len(valid),
                    })

        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.caption("No runs yet")
    else:
        st.caption("No runs yet")

    # ── MLflow (local only) ───────────────────────────────
    if is_local:
        st.divider()
        st.caption("MLflow experiment tracker")
        mlflow_port = st.number_input("MLflow port", value=5000, step=1)

        if st.button("Open MLflow UI", use_container_width=True):
            subprocess.Popen(
                [sys.executable, "-m", "mlflow", "ui",
                 "--backend-store-uri", str(BASE_DIR / "mlruns"),
                 "--port", str(int(mlflow_port))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
            st.success("MLflow running at:")
            st.markdown(
                f"[http://localhost:{int(mlflow_port)}]"
                f"(http://localhost:{int(mlflow_port)})"
            )

# ── Main area — status metrics ────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Status",      st.session_state.get("status", "Ready"))
col2.metric("Configs run", st.session_state.get("n_run",   "—"))
col3.metric("Best score",  st.session_state.get("best",    "—"))

# ── Run pipeline ──────────────────────────────────────────
if run_btn:
    contract = {
        "goal":              f"Optimise {model} on {dataset} for {metric}",
        "dataset":           dataset,
        "model":             model,
        "primary_metric":    metric,
        "n_configs":         n_configs,
        "failure_threshold": threshold,
        "csv_path":          csv_path,
        "target_column":     target_column,
    }
    with st.expander("Experiment contract", expanded=True):
        st.json(contract)

    st.session_state["contract"] = contract
    st.session_state["status"]   = "Running..."
    st.session_state.pop("results", None)
    st.session_state.pop("report",  None)
    st.rerun()

if st.session_state.get("contract") and st.session_state.get("status") == "Running...":
    contract = st.session_state["contract"]

    from agents.config_agent      import generate_configs
    from agents.experiment_runner import run_all_experiments
    from agents.debate_agent      import run_debate
    from agents.verifier_agent    import run_verification
    from agents.synthesizer_agent import synthesize_report

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Generating configs...")
        configs = generate_configs(contract)

        st.write(f"Training {len(configs)} models...")
        results, run_dir = run_all_experiments(contract, configs)

        st.write("Debating results...")
        debate = run_debate(results, contract, run_dir)

        st.write("Verifying with fresh agent...")
        review = run_verification(results, debate, contract, run_dir)

        st.write("Writing final report (Opus)...")
        report = synthesize_report(contract, results, debate, review, run_dir)
        status.update(label="Done!", state="complete")

    winner_id = results.get("winner_config_id")
    winner = next(
        (r for r in results["results"] if r.get("config_id") == winner_id), {}
    )
    st.session_state["best"]    = str(winner.get("mean_score", "—"))
    st.session_state["n_run"]   = len(results["results"])
    st.session_state["status"]  = "Complete"
    st.session_state["report"]  = report
    st.session_state["results"] = results
    st.rerun()

# ── Show results ──────────────────────────────────────────
if st.session_state.get("results"):
    results = st.session_state["results"]

    st.subheader("Config comparison")
    rows = [r for r in results["results"] if "mean_score" in r]
    if rows:
        df = pd.DataFrame(rows)[
            ["config_id", "label", "mean_score", "std_score",
             "train_time_sec", "passed_threshold"]
        ]
        st.dataframe(df.sort_values("mean_score", ascending=False),
                     use_container_width=True)

    st.subheader("Final report")
    st.markdown(st.session_state.get("report", ""))
    st.download_button(
        "Download report.md",
        st.session_state.get("report", ""),
        file_name="experiment_report.md"
    )