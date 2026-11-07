"""Content-safety flags by CODE (Step 8c fix #1).

Why: in the full run the AI never set "graphic" or "sexual": 0 of 74 promos, even a bedroom scene ranked #5.
So code scans the script lines INSIDE each clip (dialogue, action, sound, camera) with weighted words:
strong words count 2, weak words count 1; a clip is flagged when its total reaches FLAG_AT.
Single words are too broad in this show ("bedchamber" is mostly a sickroom, "intimate" is used for hugs),
hence weights + a threshold. The editor sees the matched words as evidence.
"""
from __future__ import annotations

import re

from src.parser import Block

FLAG_AT = 3
WORDS = {
    "graphic": {
        2: r"blood\w*|bleed\w*|gash\w*|lacerat\w*|flesh|red-hot|branding|searing|scald\w*|tortur\w*|corpse",
        1: r"wound\w*|whip\w*|riding crop|throat|noose|gallows|hang(s|ing|ed)?|punch\w*|chok\w*|slap\w*",
    },
    "sexual": {
        2: r"naked|shirtless|carnal|bare (?:chest|skin|shoulders?)|entangle\w*|seduct\w*|seductiv\w*|undress\w*",
        # "in bed" is weak: in this show it is mostly the sick Duchess (it caused a false flag on Ep 25-26)
        1: r"kiss\w*|caress\w*|intimate\w*|passion\w*|nestl\w*|in bed|on the bed",
    },
}
_RX = {flag: [(w, re.compile(rf"\b(?:{pat})\b", re.I)) for w, pat in levels.items()] for flag, levels in WORDS.items()}


def content_flags(p: dict, blocks: dict[str, Block]) -> dict[str, list[str]]:
    """Return {flag: [matched words]} for flags whose weighted total reaches FLAG_AT inside the clip."""
    hits: dict[str, list[tuple[str, int]]] = {f: [] for f in _RX}
    for s in p.get("segments", []):
        for b in blocks.values():
            if b.episode == s["episode"] and b.end > s["start"] and b.start < s["end"]:
                for flag, rxs in _RX.items():
                    for weight, rx in rxs:
                        hits[flag] += [(m.group(0).lower(), weight) for m in rx.finditer(b.text)]
    return {f: sorted({w for w, _ in h}) for f, h in hits.items() if sum(wt for _, wt in h) >= FLAG_AT}


def apply(promos: list[dict], blocks: dict[str, Block]) -> list[dict]:
    """Add code-found flags to each promo (keeps any flags the AI already set) + the evidence words."""
    out = []
    for p in promos:
        found = content_flags(p, blocks)
        flags = list(dict.fromkeys((p.get("flags") or []) + sorted(found)))
        out.append({**p, "flags": flags, "safety_evidence": found} if found else {**p, "flags": flags})
    return out
