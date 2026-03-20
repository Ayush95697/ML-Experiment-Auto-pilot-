import json
import re
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
{chr(10).join(f"Q{i + 1}: {a}" for i, a in enumerate(answers))}

Now output ONLY the JSON contract. No explanation, no markdown fences.
"""
    raw = call_claude(contract_prompt, model="claude-haiku-4-5-20251001", max_tokens=1000)

    # Robust JSON extraction using regex
    # This finds the first '{' and the last '}' to handle potential extra data
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        clean_json = match.group(1)
    else:
        # Fallback: if no braces found, try stripping in case it's raw but messy
        clean_json = raw.strip()

    try:
        contract = json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"\n[Error] Failed to parse JSON contract.")
        print(f"Raw Output from model:\n{raw}")
        # Optional: Print the exact position of the error for debugging
        raise e

    print("\n[Contract Agent] Contract:\n", json.dumps(contract, indent=2))
    confirm = input("\nDoes this look correct? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Restarting contract...")
        return run_contract(user_request)

    return contract


if __name__ == "__main__":
    c = run_contract("I want to test different Random Forest configs on the iris dataset")
    print("Final contract:", c)