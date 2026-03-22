import streamlit as st, json
from pathlib import Path

st.set_page_config(page_title="ML Experiment AutoPilot", layout="wide")
st.title("ML Experiment AutoPilot")
st.caption("Multi-agent hyperparameter search with debate + verification")

# Sidebar — experiment config
with st.sidebar:
    st.header("Experiment setup")
    with st.sidebar:
        st.header("Experiment setup")

        # Dataset selection
        dataset_mode = st.radio("Dataset source",
                                ["Built-in", "Upload CSV", "CSV file path"],
                                horizontal=True)

        csv_path = None
        target_column = None

        if dataset_mode == "Built-in":
            dataset = st.selectbox("Dataset", ["iris", "breast_cancer"])

        elif dataset_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload your CSV", type=["csv"])
            if uploaded:
                # Save to temp file so sklearn can read it
                import tempfile, os

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded.read())
                tmp.close()
                csv_path = tmp.name
                dataset = uploaded.name.replace(".csv", "")

                # Preview
                import pandas as pd

                df_preview = pd.read_csv(csv_path)
                st.caption(f"{df_preview.shape[0]} rows × {df_preview.shape[1]} cols")
                target_column = st.selectbox("Target column",
                                             df_preview.columns.tolist(),
                                             index=len(df_preview.columns) - 1)
            else:
                dataset = "uploaded_csv"
                st.caption("No file uploaded yet")

        elif dataset_mode == "CSV file path":
            csv_path = st.text_input("Full path to CSV file",
                                     placeholder=r"C:\Users\Ayush\data\mydata.csv")
            if csv_path and Path(csv_path).exists():
                import pandas as pd

                df_preview = pd.read_csv(csv_path)
                st.caption(f"{df_preview.shape[0]} rows × {df_preview.shape[1]} cols")
                target_column = st.selectbox("Target column",
                                             df_preview.columns.tolist(),
                                             index=len(df_preview.columns) - 1)
                dataset = Path(csv_path).stem
            elif csv_path:
                st.error("File not found at that path")
                dataset = "csv_path"
            else:
                dataset = "csv_path"

        model = st.selectbox("Model", ["RandomForest", "LogisticRegression", "SVM"])
        metric = st.selectbox("Primary metric", ["accuracy", "f1_weighted", "roc_auc"])
        n_configs = st.slider("Configs to test", 3, 8, 5)
        threshold = st.slider("Minimum acceptable score", 0.5, 1.0, 0.85, step=0.01)
        run_btn = st.button("Run AutoPilot", type="primary", use_container_width=True)

# Main area — results
col1, col2, col3 = st.columns(3)
col1.metric("Status", st.session_state.get("status", "Ready"))
col2.metric("Configs run", st.session_state.get("n_run", "—"))
col3.metric("Best score", st.session_state.get("best", "—"))

if run_btn:
    contract = {
        "goal": f"Optimise {model} on {dataset} for {metric}",
        "dataset": dataset,
        "model": model,
        "primary_metric": metric,
        "n_configs": n_configs,
        "failure_threshold": threshold,
        "csv_path": csv_path,        # None for built-in, path string for CSV
        "target_column": target_column,  # None = auto-detect last column
    }
    st.session_state["contract"] = contract
    st.session_state["status"] = "Running..."
    st.rerun()