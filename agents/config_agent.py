import json
from agents.base import call_claude, load_file

def generate_configs(contract: dict) -> list[dict]:
    skill = load_file("skills/stochastic-search.md")
    n = contract.get("n_configs", 5)

    prompt = f"""
{skill}

Contract:
{json.dumps(contract, indent=2)}

Generate exactly {n} hyperparameter configs for a {contract['model']} model.
Each config explores a different region as described in the skill.
Output ONLY a valid JSON list. No markdown, no explanation.
"""
    raw = call_claude(prompt, model="claude-haiku-4-5-20251001", max_tokens=1200)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    configs = json.loads(raw)
    print(f"[Config Agent] Generated {len(configs)} configs")
    for c in configs:
        print(f"  #{c['config_id']} {c['label']}: {c['params']}")
    return configs

if __name__ == "__main__":
    dummy_contract = {
        "model": "RandomForest",
        "dataset": "iris",
        "primary_metric": "accuracy",
        "n_configs": 4,
        "failure_threshold": 0.85
    }
    configs = generate_configs(dummy_contract)
    print(json.dumps(configs, indent=2))