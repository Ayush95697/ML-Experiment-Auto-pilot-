import anthropic, os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def load_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001",
                system: str = "", max_tokens: int = 1000) -> str:
    if not system:
        system_path = BASE_DIR / "memory" / "agents.md"
        system = load_file(str(system_path))
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text