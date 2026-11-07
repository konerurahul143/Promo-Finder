"""Fix the "one line on why it works" (a field the brief requires) when it talks about EDITING instead of the VIEWER.

Found in the full run: after the repair loop, 20 of 74 "why" lines said things like "...trimmed to fit within the
60-second limit by skipping transitional actions". That explains the cut, not why a stranger would stop scrolling.
CODE detects these lines; the AI rewrites only those (batched), from the clip's lines + hook + cliffhanger.
The original is kept as `why_original`.

Usage (on a finished run; then re-rank):
    python -m src.why --run outputs/FINAL_full_run
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.llm import USAGE, LLMError, complete
from src.parser import Block, parse_folder
from src.rules import timeline
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
# Words that describe the EDIT, not the viewer. First version missed "under 60 seconds" and "condenses".
EDITING_WORDS = re.compile(r"\b(trim\w*|condens\w*|\d+[- ]?seconds?|seconds? limit|time limit|skipp\w*|"
                           r"fit (?:within|under|in)\w*|segment\w*|shorten\w*|cut(?:s|ting)?\b|stitch\w*|splic\w*|"
                           r"combin\w*|transitional|length|duration|pacing|tight(?:ly)? edit\w*)", re.I)
BATCH = 10

# v2: v1 lines were plot summaries ("she crawls through X to save Y"): WHAT happens,
# not WHY it works. v2 asks for the reason first and forbids retelling the plot.
SYSTEM = """You write the one-line "why it works" for short promo clips of a vertical microdrama (TikTok/Reels).
For each clip you get its lines, the hook and the cliffhanger. Write ONE line (max 25 words) giving the REASON a
scrolling stranger stops and wants the next episode. Start with the reason (the emotion it triggers, the stakes,
the open question), e.g. "Instant outrage at a public humiliation, and viewers need to see her get even." or
"A life-or-death ultimatum with no answer yet: will she really crawl through the thorns?"
Do NOT just retell the plot. NEVER mention editing, cuts, segments, length, seconds, trimming or limits.
Reply with JSON only: {"items": [{"id": "P1", "why": "..."}]}"""


def needs_rewrite(why: str | None) -> bool:
    return not why or bool(EDITING_WORDS.search(why))


def rewrite(promos: list[dict], blocks: dict[str, Block], profile: str | None = None) -> tuple[list[dict], int]:
    """Return (promos with fixed "why", number rewritten). Only promos whose "why" needs it are sent to the AI."""
    out = [dict(p) for p in promos]
    todo = [i for i, p in enumerate(out) if needs_rewrite(p.get("why"))]
    done = 0
    for k in range(0, len(todo), BATCH):
        idx = todo[k:k + BATCH]
        text = "\n\n".join(f"=== P{n + 1} ===\nHOOK: {out[i].get('hook', '')}\nCLIFFHANGER: {out[i].get('cliffhanger', '')}\n"
                           + timeline(out[i], blocks) for n, i in enumerate(idx))
        try:
            reply = complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                             json_mode=True, profile=profile, tag=f"why-v2:batch{k // BATCH + 1}")
        except LLMError as err:
            print(f"  why rewrite batch failed, originals kept: {str(err)[:80]}")
            continue
        got = {it.get("id"): (it.get("why") or "").strip() for it in (reply.get("items", []) if isinstance(reply, dict) else [])}
        for n, i in enumerate(idx):
            new = got.get(f"P{n + 1}", "")
            if new and not needs_rewrite(new):                     # code double-checks the AI's line
                out[i] = {**out[i], "why_original": out[i].get("why"), "why": new}
                done += 1
    return out, done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(paths.FINAL_RUN))
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"))
    a = ap.parse_args(argv)
    f = Path(a.run) / "ranked" / "judged.json"
    judged = json.loads(f.read_text(encoding="utf-8"))
    alias = None if a.aliases.lower() == "none" else Path(a.aliases)
    blocks = {b.id: b for e in parse_folder(Path(a.input), alias) for b in e.blocks}
    before = sum(needs_rewrite(p.get("why")) for p in judged)
    fixed, n = rewrite(judged, blocks)
    f.write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding="utf-8")
    after = sum(needs_rewrite(p.get("why")) for p in fixed)
    print(f"'why' lines about editing: {before} -> {after}  (rewrote {n}; originals kept as why_original)")
    print(f"AI usage: {USAGE}")
    print(f"Now re-rank to refresh the report:  python -m src.judge --rerank {a.run}")


if __name__ == "__main__":
    main()
