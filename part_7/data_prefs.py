from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

@dataclass
class PrefExample:
    prompt: str
    chosen: str
    rejected: str


def load_preferences(split: str = "train[:200]") -> List[PrefExample]:
    """Load a tiny preference set. Tries Anthropic HH; falls back to a toy set.
    HH fields: 'chosen', 'rejected' (full conversations). We use an empty prompt.
    """
    items: List[PrefExample] = []
    if load_dataset is not None:
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split=split)
            for row in ds:
                ch = str(row.get("chosen", "")).strip()
                rj = str(row.get("rejected", "")).strip()
                if ch and rj:
                    items.append(PrefExample(prompt="", chosen=ch, rejected=rj))
        except Exception:
            print("Failed to load Anthropic/hh-rlhf dataset. Using fallback toy pairs.")
            pass
    if not items:
        # fallback toy pairs
        items = [
            PrefExample("Summarize: Scaling laws for neural language models.",
                        "Scaling laws describe how performance improves predictably as model size, data, and compute increase.",
                        "Scaling laws are when you scale pictures to look bigger."),
            PrefExample("Give two uses of attention in transformers.",
                        "It lets the model focus on relevant tokens and enables parallel context integration across positions.",
                        "It remembers all past words exactly without any computation."),
        ]
    return items