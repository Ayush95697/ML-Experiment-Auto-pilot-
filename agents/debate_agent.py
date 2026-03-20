import json
from agents.base import call_claude, load_file
from pathlib import Path

PERSONAS = [
    ("statistician", "claude-sonnet-4-6",
     "You are a STATISTICIAN. Focus on: Is the margin between configs statistically meaningful? "
     "Look at std scores. A small mean difference with high variance means nothing."),
    ("practitioner", "claude-sonnet-4-6",
     "You are a PRACTITIONER. Focus on: Which config is simplest and most reproducible? "
     "Penalise overly complex configs. Prefer configs that generalise."),
    ("skeptic", "claude-haiku-4-5-20251001",
     "You are a SKEPTIC. Challenge the apparent winner. What could inflate these scores? "
     "Any signs of overfitting? What does the worst config reveal about the problem?"),
]

def run_debate(results: dict, contract: dict, run_dir: Path) -> dict:
    skill = load_file("skills/debate.md")
    results_str = json.dumps(results, indent=2)
    contract_str = json.dumps(contract, indent=2)

    debate_turns = []
    print("\n[Debate Agent] Starting 3-persona debate...")

    for role, model, persona_context in PERSONAS:
        prompt = f"""
{skill}

You are playing the role of: {role.upper()}
Your perspective: {persona_context}

Experiment contract:
{contract_str}

Results:
{results_str}

Previous debate so far:
{json.dumps(debate_turns, indent=2) if debate_turns else "You are speaking first."}

State your analysis (3–5 sentences). Be specific — reference config IDs and numbers.
End with: WINNER: config_#N, CONCERN: one sentence.
"""
        response = call_claude(prompt, model=model,
                               system="You are a focused ML analyst. Be concise and evidence-based.",
                               max_tokens=400)
        turn = {"role": role, "model": model, "analysis": response}
        debate_turns.append(turn)
        print(f"  [{role.upper()}]: {response[:120]}...")

    # Synthesise debate into structured JSON
    synthesis_prompt = f"""
{skill}

Here is the full debate transcript:
{json.dumps(debate_turns, indent=2)}

Now produce ONLY the final summary JSON as defined in the skill. No markdown, no extra text.
"""
    raw = call_claude(synthesis_prompt, model="claude-sonnet-4-6",
                      system="Output only valid JSON.", max_tokens=1200)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


    # print(f"DEBUG synthesis raw:\n{raw}\n")

    summary = json.loads(raw)

    debate_output = {"transcript": debate_turns, "summary": summary}
    (run_dir / "chat.json").write_text(json.dumps(debate_output, indent=2))
    print(f"[Debate Agent] Consensus: {summary.get('consensus', 'none')}")
    print(f"[Debate Agent] Key disagreement: {summary.get('key_disagreement', '')}")
    return debate_output

if __name__ == "__main__":
    dummy_results = {
        "results": [
            {"config_id":1,"label":"conservative","mean_score":0.934,"std_score":0.018,"passed_threshold":True},
            {"config_id":2,"label":"aggressive","mean_score":0.947,"std_score":0.041,"passed_threshold":True},
            {"config_id":3,"label":"regularised","mean_score":0.940,"std_score":0.012,"passed_threshold":True},
        ],
        "winner_config_id": 2
    }
    dummy_contract = {"model":"RandomForest","dataset":"iris","primary_metric":"accuracy","failure_threshold":0.85}
    from pathlib import Path
    Path("experiments/runs/test").mkdir(parents=True, exist_ok=True)
    run_debate(dummy_results, dummy_contract, Path("experiments/runs/test"))