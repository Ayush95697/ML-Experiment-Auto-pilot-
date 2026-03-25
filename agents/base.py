import anthropic, os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import streamlit as st

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def load_file(path: str | Path) -> str:
    # If it's already a full Path object, use it.
    # If it's a string, join it with BASE_DIR.
    full_path = path if isinstance(path, Path) else BASE_DIR / path
    return full_path.read_text(encoding="utf-8")

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