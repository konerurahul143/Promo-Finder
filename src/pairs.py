"""Step 4b: CROSS-EPISODE promos.

Data-backed rule: in this format each episode ends on a cliffhanger and the next one opens on its answer.
All 3 cross-episode reference promos are "last ~5-35s of Ep N + first ~0-36s of Ep N+1".
So CODE builds the pair (tail of N + head of N+1) and the AI only judges it and picks the exact cuts.

Usage:
    python -m src.pairs --pair 41          # Ep 41 -> Ep 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import rules, show_summary
from src.llm import USAGE, complete
from src.parser import Block, Episode, fmt_ts, parse_folder
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
PROMPT_VERSION = "pair-v2"   # v2 = v1 + one-time show summary (major twists) for spoiler judgment
TAIL_S = 40.0   # how much of the end of Ep N the AI sees
HEAD_S = 45.0   # how much of the start of Ep N+1 the AI sees

SYSTEM = """You are a senior promo editor for vertical microdrama apps (like ReelShort and DramaBox).
Each episode ends on a cliffhanger and the next episode opens on what happens next. You are shown the END of one
episode and the START of the next. Decide if joining them makes a strong 20-60 second promo for TikTok/Reels/Shorts.

A GOOD CROSS-EPISODE PROMO:
- Segment 1 (from episode A's end) STARTS ON the hook: a shock, threat, reveal or action, within the first 3 seconds.
- Segment 2 (from episode B's start) carries the moment forward and ENDS ON a NEW unresolved beat
  (a threat, an ultimatum, an unanswered question, a reveal cut mid-beat). Never end after tension resolves.
- It makes sense to a stranger, and total length is 20-60 seconds (use the times shown).
- Never start or end on a line that is half of a split sentence ("..."). Avoid V.O. as the hook.
- If the join is weak (A's end is calm, or B's start is a scene change / recap / slow), set "use": false.

Cite ONLY block IDs shown in [brackets]. A segment runs from the START of start_id to the END of end_id.
audience: "new_viewers" (works with zero context, no major spoiler) or "retargeting".
flags: "spoiler", "graphic", "sexual" as applicable.

Reply with JSON only:
{"use": true,
 "segments": [{"start_id": "E41-B040", "end_id": "E41-B048"}, {"start_id": "E42-B001", "end_id": "E42-B017"}],
 "hook_id": "E41-B040", "hook": "...", "cliffhanger_id": "E42-B017", "cliffhanger": "...",
 "why": "one line", "standalone": "yes|partly|no", "audience": "new_viewers|retargeting", "flags": []}
or {"use": false, "why": "one line on why this join is weak"}"""


def _lines(blocks: list[Block]) -> list[str]:
    out = []
    for b in blocks:
        if b.kind == "camera":
            continue
        t = f"[{b.id} {fmt_ts(b.start)}-{fmt_ts(b.end)}]"
        if b.kind == "dialogue":
            who = b.character + (f" {b.voice}" if b.voice else "")
            out.append(f"{t} {who} ({b.emotion}): {b.text}")
        else:
            out.append(f"{t} {b.kind.upper()}: {b.text}")
    return out


def build_messages(a: Episode, b: Episode, summary: dict | None = None) -> list[dict]:
    tail = [x for x in a.blocks if x.start >= a.duration - TAIL_S]
    head = [x for x in b.blocks if x.start <= HEAD_S]
    text = "\n".join([
        f"EPISODE A = {a.episode} (its LAST {TAIL_S:.0f}s; episode length {fmt_ts(a.duration)})",
        f"LOGLINE: {a.logline}", *_lines(tail), "",
        f"EPISODE B = {b.episode} (its FIRST {HEAD_S:.0f}s)",
        f"LOGLINE: {b.logline}", f"CONTINUITY: {b.continuity}", *_lines(head)])
    if summary is None:
        print("  WARNING: no show summary given: spoiler judgment is off")
    context = show_summary.as_prompt_text(summary)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": (context + "\n" if context else "") + "Judge this cross-episode join.\n\n" + text}]


def resolve_pair(a: Episode, b: Episode, raw: dict) -> dict:
    by_id = {x.id: x for x in a.blocks + b.blocks}
    out = {**raw, "pair": [a.episode, b.episode], "problems": []}
    if not raw.get("use"):
        out["segments"] = []
        return out
    segs = []
    for s in raw.get("segments", []):
        x, y = by_id.get(s.get("start_id")), by_id.get(s.get("end_id"))
        if not x or not y or x.episode != y.episode:
            out["problems"].append(f"bad segment {s}")
            continue
        segs.append({"episode": x.episode, "start_id": x.id, "end_id": y.id, "start": x.start, "end": y.end,
                     "start_ts": fmt_ts(x.start), "end_ts": fmt_ts(y.end), "seconds": round(y.end - x.start, 2)})
    out["segments"] = segs
    out["duration"] = round(sum(s["seconds"] for s in segs), 2)
    for key in ("hook_id", "cliffhanger_id"):
        blk = by_id.get(raw.get(key))
        if not blk:
            out["problems"].append(f"unknown {key}: {raw.get(key)}")
        else:
            out[key.replace("_id", "_at")] = f"Ep{blk.episode} {fmt_ts(blk.start)}"
            out[key.replace("_id", "_text")] = blk.text
            if key == "hook_id" and segs and blk.episode == segs[0]["episode"]:
                out["hook_delay_s"] = round(blk.start - segs[0]["start"], 2)
    return out


def judge_pair(a: Episode, b: Episode, profile: str | None = None, repair: bool = True,
               summary: dict | None = None) -> dict:
    msgs = build_messages(a, b, summary)
    tag = f"{PROMPT_VERSION}:ep{a.episode:02d}-{b.episode:02d}"
    reply = complete(msgs, json_mode=True, profile=profile, tag=tag)
    reply = reply if isinstance(reply, dict) else {"use": False, "why": "bad reply"}
    res = resolve_pair(a, b, reply)
    if not res.get("use") or not repair:
        return res
    by_id = {x.id: x for x in a.blocks + b.blocks}
    res = rules.repair_loop(msgs, reply, res, lambda raw: resolve_pair(a, b, {"use": True, **raw}),
                            by_id, profile, tag)
    if res["status"] in ("ok", "fixed"):
        res = rules.snap_to_cuts(res, {a.episode: a.shot_cuts(), b.episode: b.shot_cuts()})
        res = rules.merge_touching(res)
        final = rules.check(res, by_id)
        if final:
            res = {**res, "status": "failed", "rule_errors": final}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"))
    ap.add_argument("--pair", type=int, required=True, help="first episode N of the pair N -> N+1")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=str(paths.ARCHIVE / "dev_single_pair"))
    a = ap.parse_args(argv)

    alias = None if a.aliases.lower() == "none" else Path(a.aliases)
    all_eps = parse_folder(Path(a.input), alias)
    eps = {e.episode: e for e in all_eps}
    ea, eb = eps[a.pair], eps[a.pair + 1]
    res = judge_pair(ea, eb, a.profile, summary=show_summary.get_for(all_eps))   # THIS show's summary

    out = Path(a.out) / PROMPT_VERSION
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"ep{ea.episode:02d}-{eb.episode:02d}.json"
    f.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pair Ep {ea.episode} -> {eb.episode}: use={res.get('use')}  ->  {f}")
    if res.get("use"):
        segs = " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in res["segments"])
        print(f"   {segs}  ({res.get('duration')}s)  audience={res.get('audience')}  flags={res.get('flags')}")
        print(f"   hook @{res.get('hook_at')} (+{res.get('hook_delay_s')}s): {res.get('hook')}")
        print(f"   cliff @{res.get('cliffhanger_at')}: {res.get('cliffhanger')}")
    print(f"   why: {res.get('why')}")
    if "status" in res:
        print(f"   rules: {res['status'].upper()}" + (f"  remaining: {res['rule_errors']}" if res["rule_errors"] else ""))
    if "snapped" in res:
        print(f"   cuts: {'SNAPPED' if res['snapped'] else 'NOT snapped'} - {res['snap_note']}")
    print(f"Usage: {USAGE}")


if __name__ == "__main__":
    main()
