import json
from agents.base import call_claude, load_file

def run_contract(user_request: str) -> dict:
    skill = load_file("skills/experiment-contract.md")
 # Step 1: reverse prompting — agent asks 5 questions
    questions_prompt = f"""
{skill}

The user wants to run an experiment. Their request:
"{user_request}"

Ask your 5 clarifying questions now. Output ONLY the 5 numbered questions,
nothing else.
"""
    questions = call_claude(questions_prompt, model="claude-haiku-4-5-20251001")
    print("\n[Contract Agent] Questions:\n", questions)

    # Step 2: collect answers (terminal for now, Streamlit later)
    print("\nAnswer each question (press Enter after each):")
    answers = []
    for i in range(1, 6):
        ans = input(f"  Q{i}: ")
        answers.append(ans)

    # Step 3: generate contract JSON
    contract_prompt = f"""
{skill}

User request: "{user_request}"
Answers:
{chr(10).join(f"Q{i+1}: {a}" for i, a in enumerate(answers))}

Now output ONLY the JSON contract. No explanation, no markdown fences.
"""
    raw = call_claude(contract_prompt, model="claude-haiku-4-5-20251001", max_tokens=500)

    # Strip markdown fences if present
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    contract = json.loads(raw)

    print("\n[Contract Agent] Contract:\n", json.dumps(contract, indent=2))
    confirm = input("\nDoes this look correct? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Restarting contract...")
        return run_contract(user_request)

    return contract

if __name__ == "__main__":
    c = run_contract("I want to test different Random Forest configs on the iris dataset")
    print("Final contract:", c)