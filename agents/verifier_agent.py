import json
from agents.base import call_claude, load_file, client
from pathlib import Path

def run_verification(results: dict, debate: dict, contract: dict, run_dir: Path) -> dict:
    skill = load_file("skills/verifier.md")

    fresh_system = (
        "You are a strict ML peer reviewer with no prior context. "
        "You only see what is given to you. Output only valid JSON."
    )

    prompt = f"""
{skill}

IMPORTANT: If winner_config_id does not match the config with the highest
mean_score in the results list, this is a CRITICAL issue and verdict MUST
be "needs_revision". State exactly which config is wrong and what the
correct winner should be.

Experiment contract:
{json.dumps(contract, indent=2)}

Results to review:
{json.dumps(results, indent=2)}

Debate summary:
{json.dumps(debate.get('summary', {}), indent=2)}

Perform your review. Output ONLY the JSON defined in the skill. No extra text.
"""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=fresh_system,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    print(f"DEBUG raw:\n{raw}\n")   # remove after testing

    review = json.loads(raw)

    BASE_DIR = Path(__file__).resolve().parent.parent
    save_dir = BASE_DIR / "experiments" / "runs" / run_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "review.json").write_text(json.dumps(review, indent=2))

    print(f"\n[Verifier] Issues found: {review.get('issues_found', 0)}")
    print(f"[Verifier] Verdict: {review.get('verdict', 'unknown')}")
    if review.get("critical"):
        print("[Verifier] CRITICAL issues:")
        for issue in review["critical"]:
            print(f"  - {issue}")
    return review


if __name__ == "__main__":
    dummy_results = {
        "results": [
            {"config_id": 1, "mean_score": 0.951, "std_score": 0.018, "passed_threshold": True},
            {"config_id": 2, "mean_score": 0.963, "std_score": 0.041, "passed_threshold": True},
            {"config_id": 3, "mean_score": 0.958, "std_score": 0.012, "passed_threshold": True},
            {"config_id": 4, "mean_score": 0.947, "std_score": 0.022, "passed_threshold": True},
        ],
        "winner_config_id": 4   # WRONG — config 4 is the lowest scorer
    }
    dummy_debate = {
        "summary": {
            "consensus": "2",
            "key_disagreement": "config 3 vs config 2 on stability",
            "outlier_insight": "config 4 trains fastest but scores lowest"
        }
    }
    dummy_contract = {
        "model": "RandomForest",
        "primary_metric": "accuracy",
        "failure_threshold": 0.85
    }

    BASE_DIR = Path(__file__).resolve().parent.parent
    test_dir = BASE_DIR / "experiments" / "runs" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    run_verification(dummy_results, dummy_debate, dummy_contract, test_dir)
