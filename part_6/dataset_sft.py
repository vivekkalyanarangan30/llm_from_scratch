from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import traceback

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from formatters import Example

@dataclass
class SFTItem:
    prompt: str
    response: str


def load_tiny_hf(split: str = "train[:200]") -> List[SFTItem]:
    """Try to load a tiny instruction dataset from HF; fall back to a baked-in list.
    We use `tatsu-lab/alpaca` as a familiar schema (instruction, input, output) and keep only a slice.
    """
    items: List[SFTItem] = []
    if load_dataset is not None:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                out = row.get("output", "").strip()
                if inp:
                    instr = instr + "\n" + inp
                if instr and out:
                    items.append(SFTItem(prompt=instr, response=out))
        except Exception:
            pass
    if not items:
        # fallback tiny list
        seeds = [
            ("Explain what a transformer is in one sentence.", "A transformer is a neural network that uses attention to weigh relationships between tokens and model sequences effectively."),
            ("List three prime numbers.", "2, 3, 5."),
            ("Write a Python function that returns the square of a number.", "def square(x):\n    return x*x"),
            ("Translate to French: 'good morning'", "bonjour"),
        ]
        items = [SFTItem(prompt=p, response=r) for p,r in seeds]
    return items