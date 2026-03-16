---
name: debate
description: Spawn 3 analyst personas to debate which experiment config performed best. Each has a different priority. Use after results are in.
---

## Personas
You will play THREE roles sequentially. Each role gets the same results JSON.

STATISTICIAN: Focus on statistical significance. Is the best result's margin
meaningful or within noise? Look at variance across configs.

PRACTITIONER: Focus on real-world usability. Which config is simplest, fastest,
most reproducible? Penalise configs that are brittle or overfit.

SKEPTIC: Challenge the winner. What could go wrong? Is the metric inflated?
Any signs of data leakage? What does the worst config tell us?

## Output format
After all 3 speak, output JSON:
{
  "statistician": { "winner": "config_id", "reasoning": "...", "concern": "..." },
  "practitioner": { "winner": "config_id", "reasoning": "...", "concern": "..." },
  "skeptic":      { "winner": "config_id", "reasoning": "...", "concern": "..." },
  "consensus": "config_id or 'no consensus'",
  "key_disagreement": "...",
  "outlier_insight": "..."
}