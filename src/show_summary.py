"""One-time SHOW SUMMARY ("show bible"): 1 AI call per show, from the 50 episode loglines.

Why: each finder/pair call sees only one episode (or two snippets), so it can't know the show's big
secrets and can't flag spoilers (41->42 was labelled new_viewers though it reveals the Duke twist).
The summary lists the major twists + secret identities and is passed to every finder/pair call.

Usage:
    python -m src.show_summary            # builds outputs/show_summaries/<show_id>.json (once per show)
    Finder and pairs call get_for() automatically, so this command is optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.llm import complete
from src.parser import Episode, parse_folder
from src import paths as _paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _paths.DEFAULT_INPUT
PROMPT_VERSION = "summary-v1"

SYSTEM = """You summarise a serialized microdrama for promo editors, so they know what would SPOIL the show
for a brand-new viewer. You get every episode's one-line logline, in order.

Return JSON only:
{"premise": "2 sentences: the setup a new viewer sees in episode 1-3, no twists",
 "characters": [{"name": "...", "role": "...", "secret": "hidden identity or secret, or empty"}],
 "major_twists": [{"twist": "short statement", "first_revealed_in_episode": 0,
                   "spoiler_signs": ["phrases or events in a clip that would give it away"]}],
 "spoiler_rule": "one sentence: when a promo counts as a spoiler for this show"}

Only list twists that would genuinely ruin a surprise (hidden identities, secret plots revealed, who is really
behind something, deaths, big reversals). 3-8 twists. Keep it short."""


def build_messages(episodes: list[Episode]) -> list[dict]:
    lines = [f"Ep {e.episode}: {e.logline}" for e in episodes]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Episode loglines:\n" + "\n".join(lines)}]


def build(episodes: list[Episode], profile: str | None = None) -> dict:
    return complete(build_messages(episodes), json_mode=True, profile=profile, tag=f"{PROMPT_VERSION}:show")


def as_prompt_text(summary: dict | None) -> str:
    """Compact text block added to finder/pair prompts."""
    if not summary:
        return ""
    tw = "\n".join(f"- {t['twist']} (revealed ep {t.get('first_revealed_in_episode')}; signs: "
                   f"{', '.join(t.get('spoiler_signs', [])[:4])})" for t in summary.get("major_twists", []))
    secrets = "; ".join(f"{c['name']}: {c['secret']}" for c in summary.get("characters", []) if c.get("secret"))
    return ("SHOW CONTEXT (use ONLY to judge spoilers and audience; pick moments from the episode text):\n"
            f"Premise new viewers know: {summary.get('premise', '')}\n"
            f"Secrets: {secrets}\nMajor twists:\n{tw}\n"
            f"Spoiler rule: {summary.get('spoiler_rule', '')}\n"
            "If a promo shows or states the ANSWER of any of these twists, flag it \"spoiler\" and set audience "
            "\"retargeting\". A mystery shown without its answer (e.g. 'someone is poisoning her' without who) is a "
            "HOOK, not a spoiler.\n")


SUMMARY_DIR = _paths.SHOW_SUMMARIES


def show_id(episodes: list[Episode]) -> str:
    """Stable ID for a show = hash of its episode loglines, so each show gets ITS OWN summary
    (bug fix: a single global file would reuse this show's twists for a new show)."""
    blob = "\n".join(f"{e.episode}:{e.logline}" for e in sorted(episodes, key=lambda e: e.episode))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def get_for(episodes: list[Episode], profile: str | None = None) -> dict:
    """Load this show's summary, or build it (1 AI call) and save it the first time."""
    path = SUMMARY_DIR / f"{show_id(episodes)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    s = build(episodes, profile)
    s["_show_id"] = show_id(episodes)
    s["_review_status"] = "pending_editor_review"   # human in the loop: skim the twist list once per show
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  built show summary {path.name} (1 AI call); an editor should skim its twist list once")
    return s


def review_text(s: dict, path: Path) -> str:
    """The editor's 1-minute check of the AI-written twist list (Step 8c fix #2).
    In the full run, a wrong "twist" (the visible main plot of several episodes, not a secret)
    flagged the trial promo as a spoiler and sank it to rank 72."""
    lines = [f"SHOW SUMMARY REVIEW  ({s.get('_review_status', 'pending_editor_review')})", f"file: {path}", "",
             f"Premise new viewers know: {s.get('premise', '')}", "", "Twists the tool treats as SPOILERS:"]
    for i, t in enumerate(s.get("major_twists", []), 1):
        lines.append(f"  {i}. [ep {t.get('first_revealed_in_episode')}] {t.get('twist')}")
        lines.append(f"     signs: {', '.join(t.get('spoiler_signs', [])[:4])}")
    lines += ["", "Ask for each: would a NEW viewer feel cheated if a promo showed this? If not (it's the visible plot,",
              "e.g. 'she is framed and put on trial'), delete it from major_twists in the file above.",
              "Add any secret that is missing. Then set \"_review_status\": \"reviewed_by_editor\" and re-run.",
              "(Re-running after an edit makes new AI calls, since the prompts change.)"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--review", action="store_true", help="print the twist list for the editor's 1-minute check")
    a = ap.parse_args(argv)
    eps = parse_folder(Path(a.input), None)
    s = get_for(eps, a.profile)
    path = SUMMARY_DIR / (show_id(eps) + ".json")
    if a.review:
        print(review_text(s, path))
        return
    print(as_prompt_text(s))
    print(f"Show summary: {path}")


if __name__ == "__main__":
    main()
