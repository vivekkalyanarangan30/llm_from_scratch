from __future__ import annotations
import re
from typing import List, Tuple

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))

def token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    gp = g.copy()
    for t in p:
        if t in gp:
            gp.remove(t); common += 1
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec  = common / len(g)
    return 2*prec*rec/(prec+rec)