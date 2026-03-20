import json
from agents.base import call_claude, load_file
from pathlib import Path

def synthesize_report(contract: dict, results: dict,
                      debate: dict, review: dict, run_dir: Path) -> str:
    skill = load_file("skills/synthesizer.md")
    winner_id = results.get("winner_config_id")
    winner_cfg = next((r for r in results["results"] if r.get("config_id") == winner_id), {})

    prompt = f"""
{skill}

Contract: {json.dumps(contract, indent=2)}
All results: {json.dumps(results["results"], indent=2)}
Winner config: {json.dumps(winner_cfg, indent=2)}
Debate summary: {json.dumps(debate.get("summary", {}), indent=2)}
Verification: {json.dumps(review, indent=2)}

Write the full experiment report now. Follow the structure in the skill exactly.
"""
    # Opus — most expensive, used ONCE per run for final synthesis only
    report = call_claude(prompt, model="claude-opus-4-6",
                         system=load_file("memory/agents.md"), max_tokens=1500)

    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[Synthesizer] Report saved to: {report_path}")
    return report