import streamlit as st, json
from pathlib import Path

st.set_page_config(page_title="ML Experiment AutoPilot", layout="wide")
st.title("ML Experiment AutoPilot")
st.caption("Multi-agent hyperparameter search with debate + verification")

# Sidebar — experiment config
with st.sidebar:
    st.header("Experiment setup")
    dataset   = st.selectbox("Dataset", ["iris", "breast_cancer"])
    model     = st.selectbox("Model", ["RandomForest", "LogisticRegression", "SVM"])
    metric    = st.selectbox("Primary metric", ["accuracy", "f1", "roc_auc"])
    n_configs = st.slider("Configs to test", 3, 8, 5)
    threshold = st.slider("Minimum acceptable score", 0.5, 1.0, 0.85, step=0.01)
    run_btn   = st.button("Run AutoPilot", type="primary", use_container_width=True)

# Main area — results
col1, col2, col3 = st.columns(3)
col1.metric("Status", st.session_state.get("status", "Ready"))
col2.metric("Configs run", st.session_state.get("n_run", "—"))
col3.metric("Best score", st.session_state.get("best", "—"))

if run_btn:
    contract = {
        "goal": f"Optimise {model} on {dataset} for {metric}",
        "dataset": dataset, "model": model,
        "primary_metric": metric,
        "n_configs": n_configs,
        "failure_threshold": threshold
    }
    st.session_state["contract"] = contract
    st.session_state["status"] = "Running..."
    st.rerun()