# ML Experiment AutoPilot

> An autonomous multi-agent system that searches hyperparameters, debates results across analyst personas, verifies findings with a zero-context reviewer, and produces structured experiment reports — all from a single click.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)](https://streamlit.io)
[![Claude API](https://img.shields.io/badge/Claude-API-orange.svg)](https://anthropic.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.18-blue.svg)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-green.svg)](https://scikit-learn.org)

---

## What it does

Most ML engineers spend hours manually trying hyperparameter configs, comparing results in spreadsheets, and writing experiment summaries. ML Experiment AutoPilot automates the entire loop — from config generation to final report — using a pipeline of specialized AI agents that each handle one part of the process.

You describe what you want to optimise. The system does the rest.

---

## Live demo

[**Try it on Streamlit Cloud**](https://dybshuuuakt8v8gqwvappza.streamlit.app/)

---

## Pipeline overview

```
User input
    │
    ▼
┌─────────────────────────────┐
│  1. Prompt Contract         │  Defines goal, constraints, metrics,
│     + Reverse Prompting     │  failure conditions before any work starts
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. Stochastic Config       │  Spawns N agents with different framings
│     Generation              │  (conservative / aggressive / regularised
│                             │  / speed-optimised / creative)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. Parallel Training       │  Trains each config with 5-fold CV
│     + MLflow Logging        │  Logs params + metrics to MLflow
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Agent Chat Room         │  3 personas debate the results:
│     Debate                  │  Statistician · Practitioner · Skeptic
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. Sub-Agent Verification  │  Fresh-context agent reviews the report
│                             │  with zero bias — catches wrong winners,
│                             │  contradictions, unsupported claims
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. Final Report Synthesis  │  Claude Opus writes the structured
│     + Memory Update         │  markdown report. agents.md updated
│                             │  with new learned rules for next run
└──────────────┬──────────────┘
               │
               ▼
        report.md + MLflow run
```

---

## Agent concepts implemented

| Concept | Where it's used |
|---|---|
| Core agent loop (Observe → Think → Act) | Every agent call cycles through context, reasoning, and tool use |
| Stochastic multi-agent consensus | 5 configs generated with intentionally varied framings to explore different regions of the search space |
| Agent chat room | Statistician, Practitioner, and Skeptic debate results with different priorities — sharpens conclusions through disagreement |
| Sub-agent verification loop | A completely fresh agent (no `agents.md`, no prior context) reviews the output — eliminates sunk-cost bias from the builder agent |
| Self-modifying memory | `agents.md` accumulates learned rules across sessions automatically — the system gets better at your preferences over time |
| Prompt contracts | Experiment definition (goal, constraints, metric, failure threshold) locked in before any training starts |
| Token cost strategy | Haiku for bulk config generation, Sonnet for debate and verification, Opus only for final synthesis — ~60% cost reduction vs. using Opus throughout |

---

## Features

**Dataset support**
- Built-in sklearn datasets (iris, breast_cancer)
- Upload any CSV file directly through the UI
- Paste a local file path for large datasets
- Auto-detects tab-separated vs comma-separated files
- Target column selector with preview of row/column counts
- Non-numeric columns dropped automatically, string targets label-encoded

**Model support**
- Random Forest
- Logistic Regression
- SVM (Support Vector Machine)
- Invalid hyperparameters filtered automatically per model

**Experiment configuration**
- Choose number of configs to test (3–8)
- Set minimum acceptable score threshold
- Choose primary metric: accuracy, f1_weighted, roc_auc
- All settings validated before run starts

**Results**
- Config comparison table sorted by best score
- Structured markdown report with 6 sections
- Downloadable `report.md`
- Run history table showing all past experiments
- MLflow integration for detailed param/metric tracking (local)

---

## Project structure

```
ml_autopilot/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env                        # API keys (never committed)
│
├── memory/
│   └── agents.md               # Self-modifying rules file
│
├── skills/
│   ├── experiment-contract.md  # Prompt contract skill
│   ├── stochastic-search.md    # Parallel config generation
│   ├── debate.md               # Chat room debate skill
│   ├── verifier.md             # Zero-context reviewer
│   └── synthesizer.md          # Final report writer
│
├── agents/
│   ├── base.py                 # Shared Claude API wrapper
│   ├── contract_agent.py       # Prompt contract + reverse prompting
│   ├── config_agent.py         # Stochastic config generation
│   ├── experiment_runner.py    # Model training + MLflow logging
│   ├── debate_agent.py         # 3-persona chat room debate
│   ├── verifier_agent.py       # Fresh-context verification
│   ├── synthesizer_agent.py    # Opus final report synthesis
│   └── orchestrator.py         # Master pipeline controller
│
└── experiments/
    └── runs/
        └── {timestamp}/
            ├── contract.json
            ├── configs.json
            ├── results.json
            ├── chat.json       # Full debate transcript
            ├── review.json     # Verifier findings
            └── report.md       # Final output
```

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/Ayush95697/ML-Experiment-Auto-pilot-/tree/main
cd ml-experiment-autopilot
```

**2. Create virtual environment and install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

**3. Add your Anthropic API key**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get your key from [console.anthropic.com](https://console.anthropic.com)

**4. Run the app**
```bash
streamlit run app.py
```

**5. (Optional) Open MLflow UI in a separate terminal**
```bash
mlflow ui --backend-store-uri mlruns
```
Then visit `http://localhost:5000`

---

## Running from terminal (without UI)

```bash
python agents/orchestrator.py
```

This runs the full pipeline interactively — the agent asks 5 clarifying questions, generates a prompt contract, and proceeds through all 6 steps with terminal output.

---

## Example output

After running RandomForest on breast_cancer dataset with 5 configs:

```
Config comparison:
┌──────────┬─────────────────────────┬────────────┬───────────┬──────────────────┐
│ config_id│ label                   │ mean_score │ std_score │ train_time_sec   │
├──────────┼─────────────────────────┼────────────┼───────────┼──────────────────┤
│ 3        │ regularisation_focused  │ 0.9649     │ 0.0143    │ 1.37             │
│ 2        │ aggressive_complexity   │ 0.9613     │ 0.0312    │ 3.24             │
│ 5        │ creative_exploration    │ 0.9578     │ 0.0229    │ 2.11             │
│ 1        │ conservative_baseline   │ 0.9542     │ 0.0187    │ 0.89             │
│ 4        │ speed_optimised         │ 0.9508     │ 0.0204    │ 0.47             │
└──────────┴─────────────────────────┴────────────┴───────────┴──────────────────┘

Debate outcome:
- Statistician: config_3 wins — smallest std (0.014) with top score
- Practitioner: config_3 — regularised configs generalise better in production
- Skeptic: margin between config_2 and config_3 is within noise on this dataset size

Verifier verdict: approved (0 critical issues)

Winner: config_3 — regularisation_focused
```

---

## How the self-modifying memory works

Every run, the system checks whether the verifier found critical issues or the debate surfaced an outlier insight. If it did, it automatically appends a new rule to `memory/agents.md`:

```
## Learned rules
[RandomForest] Always flag when aggressive configs have std > 2x conservative baseline — high variance signals overfitting risk.
[Verification] Check train_time ratio — configs taking 6x longer for <1% gain are not production-viable.
[Dataset] On small datasets (<500 rows), prefer regularised configs — high n_estimators overfits without meaningful gain.
```

By session 5, the agent has learned your preferences and avoids the mistakes it made in sessions 1–4.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Anthropic Claude API | All agent reasoning (Haiku / Sonnet / Opus) |
| scikit-learn | Model training and cross-validation |
| MLflow | Experiment tracking and run comparison |
| Streamlit | Web UI and deployment |
| pandas + numpy | Data loading and processing |
| python-dotenv | Environment variable management |

---

## Token cost per run

Using the tiered model strategy (60% Haiku / 30% Sonnet / 10% Opus):

| Step | Model | Approx. cost |
|---|---|---|
| Config generation | Haiku | ~$0.01 |
| Debate (3 personas) | Sonnet + Haiku | ~$0.08 |
| Verification | Sonnet | ~$0.04 |
| Final synthesis | Opus | ~$0.15 |
| **Total per run** | | **~$0.28–$0.40** |

Running 100% Opus throughout would cost ~$1.20 per run — the tiered approach saves ~70%.

---

## Deploying to Streamlit Cloud

1. Push repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set main file to `app.py`
4. Add secrets: Settings → Secrets:
```
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
IS_STREAMLIT_CLOUD = "true"
```
5. Deploy

The `IS_STREAMLIT_CLOUD = "true"` secret hides the local-only MLflow button on the deployed version.

---

## Roadmap

- [ ] Add XGBoost and LightGBM model support
- [ ] Gemini API integration for video-to-action pipeline (watch tutorial videos and replicate configs)
- [ ] DagsHub integration for cloud MLflow tracking
- [ ] Export results to PDF
- [ ] Regression task support (currently classification only)
- [ ] Batch mode — queue multiple experiments and run overnight

---

## About

Built by **Ayush Mishra** — 3rd year B.Tech CSE student at Raj Kumar Goel Institute of Technology (RKGIT), India.

This project implements advanced AI agent patterns including stochastic multi-agent consensus, debate-based analysis, sub-agent verification loops, and self-modifying memory — patterns used in production AI systems at scale.

Connect with me on [LinkedIn](www.linkedin.com/in/ayush-mishra-183b61275) · [GitHub](https://github.com/Ayush95697)

---

## License

MIT License — free to use, modify, and distribute.
