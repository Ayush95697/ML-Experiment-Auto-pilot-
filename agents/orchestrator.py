from agents.contract_agent    import run_contract
from agents.config_agent      import generate_configs
from agents.experiment_runner import run_all_experiments
from agents.debate_agent      import run_debate
from agents.verifier_agent    import run_verification
from agents.synthesizer_agent import synthesize_report
from agents.base import call_claude
from pathlib import Path


def update_agents_md(new_rule: str):
    path = Path(__file__).resolve().parent.parent / "memory" / "agents.md"
    current = path.read_text(encoding="utf-8")
    if "## Learned rules" in current:
        updated = current.replace(
            "## Learned rules",
            f"## Learned rules\n{new_rule}"
        )
        path.write_text(updated, encoding="utf-8")


def generate_learned_rule(review: dict, debate: dict, contract: dict) -> str | None:
    """Ask Claude to write a rule in the standard format based on this run's findings."""
    if review.get("verdict") != "needs_revision" and not debate.get("summary",{}).get("outlier_insight"):
        return None  # Nothing to learn this session

    from agents.base import call_claude
    prompt = f"""
Based on these findings from an ML experiment run, write ONE new rule for the agent's
memory file. Format: [Category] Never/Always do X because Y.
Keep it under 20 words. Be specific to ML experiments.

Verifier critical issues: {review.get('critical', [])}
Debate outlier insight: {debate.get('summary', {}).get('outlier_insight', '')}
Model used: {contract.get('model')}
"""
    rule = call_claude(prompt, model="claude-haiku-4-5-20251001",
                       system="Output ONLY the rule. One line. No explanation.")
    return rule.strip()

# In run_autopilot(), replace the update block with:
rule = generate_learned_rule(review, debate, contract)
if rule:
    update_agents_md(rule)
    print(f"\n[Memory] New rule learned: {rule}")


def run_autopilot(user_request: str = None):
    print("=" * 55)
    print("  ML EXPERIMENT AUTOPILOT")
    print("=" * 55)

    if not user_request:
        user_request = input("\nDescribe your experiment: ")

    print("\n[1/5] Running experiment contract...")
    contract = run_contract(user_request)

    print("\n[2/5] Generating hyperparameter configs...")
    configs = generate_configs(contract)

    print("\n[3/5] Running experiments...")
    results, run_dir = run_all_experiments(contract, configs)

    print("\n[4/5] Running debate analysis...")
    debate = run_debate(results, contract, run_dir)

    print("\n[5/5] Verifying results...")
    review = run_verification(results, debate, contract, run_dir)

    print("\n[6/6] Synthesizing final report (Opus)...")
    report = synthesize_report(contract, results, debate, review, run_dir)

    rule = generate_learned_rule(review, debate, contract)
    if rule:
        update_agents_md(rule)
        print(f"\n[Memory] New rule learned: {rule}")

    print(f"\n{'='*55}")
    print("  DONE. Report preview:")
    print(f"{'='*55}")
    print(report[:800] + "..." if len(report) > 800 else report)
    return run_dir


if __name__ == "__main__":
    run_autopilot()