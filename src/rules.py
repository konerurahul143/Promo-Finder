"""Hard promo rules (CODE, not AI) + the repair message sent back to the AI when a promo breaks them.

Why: the AI picks good moments but can't add up durations (Ep 12: a good hook + cliffhanger 68.7s apart,
though the prompt said 20-60s). So code measures exactly, and if a rule fails it sends the AI the exact
numbers plus a timeline with t+seconds for every line, so the AI never has to do arithmetic.
"""
from __future__ import annotations

import json

from src.parser import Block, fmt_ts

MIN_S, MAX_S, HOOK_MAX_S = 20.0, 60.0, 3.0
MAX_SEGMENTS = 3
# Max jump between two pieces of the SAME episode (Step 8c fix: "collage" clips stitched distant scenes,
# up to 57s apart). Calibrated on the refs: ref #3 deliberately skips 39.6s of talk, so 30s would reject a
# good promo; 45s keeps it and catches the 3 worst tool clips (57.3s, 52.3s, 48.0s).
MAX_GAP_S = 45.0


def recheck(promos: list[dict]) -> list[dict]:
    """Re-apply the CURRENT structural rules to already-made promos (e.g. after a rule was added),
    marking newly-failing ones as failed with the reason, so they are listed as not ranked."""
    out = []
    for p in promos:
        if p.get("status") in ("ok", "fixed"):
            errs = check({**p, "problems": []})
            if errs:
                p = {**p, "status": "failed", "rule_errors": errs, "failed_by": "rule added after the run"}
        out.append(p)
    return out
# Longest allowed stretch with no dialogue / action / sound inside a promo. Calibrated on the 13 refs:
# 4s wrongly flagged 2 good clips (4.7s slow-mo run, 5.2s door-suspense beat, both camera-only).
# Real dead air in the data is ~21s (Ep 18). 8s = clear margin both ways.
DEAD_AIR_S = 8.0
ELLIPSIS = ("...", "…")


def _inside(s: dict, blocks: dict[str, Block]) -> list[Block]:
    return sorted((b for b in blocks.values() if b.episode == s["episode"] and b.kind != "camera"
                   and b.end > s["start"] and b.start < s["end"]), key=lambda b: (b.start, b.end))


def content_errors(p: dict, blocks: dict[str, Block]) -> list[str]:
    """Checks that need the script text: dead air (item 3) and mid-sentence cuts (item 4)."""
    errs = []
    for s in p.get("segments", []):
        inside = _inside(s, blocks)
        # item 3: longest stretch with nothing happening (only camera shots)
        t, worst, at = s["start"], 0.0, s["start"]
        for b in inside:
            if b.start - t > worst:
                worst, at = b.start - t, t
            t = max(t, b.end)
        if s["end"] - t > worst:
            worst, at = s["end"] - t, t
        if worst > DEAD_AIR_S:
            errs.append(f"dead air: {worst:.1f}s with no dialogue/action/sound from {fmt_ts(at)} in Ep{s['episode']}")
        # item 4: never start on the second half of a sentence, never end on the first half of one
        talk = [b for b in inside if b.kind == "dialogue"]
        if talk and talk[0].start >= s["start"] - 0.05 and talk[0].text.startswith(ELLIPSIS):
            errs.append(f"starts mid-sentence on [{talk[0].id}] \"{talk[0].text[:40]}\": start one line earlier or later")
        if talk and talk[-1].text.rstrip('"').endswith(ELLIPSIS):
            nxt = sorted((b for b in blocks.values() if b.episode == s["episode"] and b.kind == "dialogue"
                          and b.start >= talk[-1].end), key=lambda b: b.start)
            if nxt and nxt[0].text.startswith(ELLIPSIS):
                errs.append(f"ends mid-sentence on [{talk[-1].id}] \"{talk[-1].text[-40:]}\": the sentence continues "
                            f"in [{nxt[0].id}]; end after it or one line earlier")
    return errs


def check(p: dict, blocks: dict[str, Block] | None = None) -> list[str]:
    """Return the list of broken rules for a resolved promo (empty list = passes).
    With `blocks`, also checks dead air and mid-sentence cuts."""
    errs = list(p.get("problems", []))
    segs = p.get("segments", [])
    if not segs:
        return errs or ["no valid segments"]
    if len(segs) > MAX_SEGMENTS:
        errs.append(f"{len(segs)} segments (max {MAX_SEGMENTS})")
    d = p.get("duration", 0)
    if d > MAX_S:
        errs.append(f"too long: {d:.1f}s (must be <= {MAX_S:.0f}s), cut {d - MAX_S:.1f}s or more")
    if d < MIN_S:
        errs.append(f"too short: {d:.1f}s (must be >= {MIN_S:.0f}s), add {MIN_S - d:.1f}s or more")
    hd = p.get("hook_delay_s")
    if hd is None:
        errs.append("hook block is not inside the first segment")
    elif hd < 0:
        errs.append(f"hook block starts {-hd:.1f}s BEFORE the promo starts")
    elif hd > HOOK_MAX_S:
        errs.append(f"hook lands {hd:.1f}s in (must be <= {HOOK_MAX_S:.0f}s): start the promo at or just before the hook")
    for a, b in zip(segs, segs[1:]):
        if a["episode"] == b["episode"] and b["start"] < a["end"]:
            errs.append(f"segments overlap or are out of order ({a['end_ts']} then {b['start_ts']})")
        if a["episode"] == b["episode"] and b["start"] - a["end"] > MAX_GAP_S:
            errs.append(f"pieces {a['end_ts']} and {b['start_ts']} are {b['start'] - a['end']:.0f}s apart "
                        f"(max {MAX_GAP_S:.0f}s): a promo must stay inside one moment, not stitch distant scenes")
        if a["episode"] > b["episode"]:
            errs.append("segments are not in story order")
    for s in segs:
        if s["seconds"] <= 0:
            errs.append(f"segment {s['start_id']}..{s['end_id']} has no length")
    if blocks is not None:
        errs += content_errors(p, blocks)
    return errs


def timeline(p: dict, blocks: dict[str, Block]) -> str:
    """Every non-camera line inside the promo with t+seconds from the promo start (no arithmetic needed)."""
    lines, offset = [], 0.0
    for s in p["segments"]:
        inside = sorted((b for b in blocks.values()
                         if b.episode == s["episode"] and b.kind != "camera"
                         and b.start >= s["start"] and b.end <= s["end"]), key=lambda b: b.start)
        lines.append(f"-- segment Ep{s['episode']} {s['start_ts']}-{s['end_ts']} ({s['seconds']:.1f}s) --")
        for b in inside:
            who = f"{b.character}: " if b.kind == "dialogue" else f"{b.kind.upper()}: "
            lines.append(f"t+{offset + b.start - s['start']:5.1f}s  [{b.id}] {who}{b.text[:80]}")
        offset += s["seconds"]
    lines.append(f"TOTAL = {offset:.1f}s")
    return "\n".join(lines)


SNAP_WINDOW_S = 2.0


def snap_to_cuts(p: dict, cuts: dict[int, list[float]], window: float = SNAP_WINDOW_S) -> dict:
    """Move each segment's start BACK and end FORWARD to the nearest camera cut (within `window` s),
    so no clip begins or ends mid-shot and no line is clipped. If snapping would break a rule,
    the original cut is kept. Adds: snapped (bool), snap_note."""
    if not p.get("segments"):
        return p
    new_segs, moved = [], 0.0
    for s in p["segments"]:
        c = cuts.get(s["episode"], [])
        before = [x for x in c if s["start"] - window <= x <= s["start"]]
        after = [x for x in c if s["end"] <= x <= s["end"] + window]
        a = max(before) if before else s["start"]
        b = min(after) if after else s["end"]
        moved += (s["start"] - a) + (b - s["end"])
        new_segs.append({**s, "start": round(a, 2), "end": round(b, 2), "start_ts": fmt_ts(a), "end_ts": fmt_ts(b),
                         "seconds": round(b - a, 2)})
    shift = p["segments"][0]["start"] - new_segs[0]["start"]          # first start moved earlier by this much
    q = {**p, "segments": new_segs, "duration": round(sum(s["seconds"] for s in new_segs), 2)}
    if p.get("hook_delay_s") is not None:
        q["hook_delay_s"] = round(p["hook_delay_s"] + shift, 2)
    errs = check({**q, "problems": []})
    if errs:
        return {**p, "snapped": False, "snap_note": f"kept original cuts: snapping would break {errs}"}
    return {**q, "snapped": True, "snap_note": f"moved {moved:.2f}s total onto camera cuts"}


MERGE_GAP_S = 1.0


def merge_touching(p: dict, max_gap: float = MERGE_GAP_S) -> dict:
    """Join consecutive segments from the same episode whose gap is <= max_gap seconds
    (the AI sometimes splits one stretch into pieces 0.0-0.6s apart). Re-checks rules; keeps the
    original if merging would break one. Adds: merged (int, how many joins)."""
    segs = p.get("segments", [])
    if len(segs) < 2:
        return {**p, "merged": 0}
    out = [dict(segs[0])]
    for s in segs[1:]:
        last = out[-1]
        if s["episode"] == last["episode"] and 0 <= s["start"] - last["end"] <= max_gap:
            last.update(end=s["end"], end_ts=s["end_ts"], end_id=s["end_id"],
                        seconds=round(s["end"] - last["start"], 2))
        else:
            out.append(dict(s))
    joins = len(segs) - len(out)
    if not joins:
        return {**p, "merged": 0}
    q = {**p, "segments": out, "duration": round(sum(s["seconds"] for s in out), 2), "merged": joins}
    if check({**q, "problems": []}):
        return {**p, "merged": 0}
    return q


def repair_loop(messages: list[dict], first_reply: dict, promo: dict, resolve_fn, blocks: dict[str, Block],
                profile: str | None, tag: str, max_rounds: int = 2) -> dict:
    """Check a resolved promo; if it breaks rules, send exact numbers back to the AI (up to max_rounds).
    Adds: status (ok | fixed | failed | dropped), rule_errors (final), repair_history."""
    from src.llm import complete  # local import keeps rules.py usable without a model

    errs = check(promo, blocks)
    history = [errs]
    if not errs:
        return {**promo, "status": "ok", "rule_errors": [], "repair_history": history}
    convo = messages + [{"role": "assistant", "content": json.dumps(first_reply, ensure_ascii=False)}]
    for r in range(1, max_rounds + 1):
        convo = convo + [{"role": "user", "content": repair_message(promo, errs, blocks)}]
        reply = complete(convo, json_mode=True, profile=profile, tag=f"{tag}:repair{r}")
        if isinstance(reply, dict) and isinstance(reply.get("promos"), list) and reply["promos"]:
            reply = reply["promos"][0]
        if not isinstance(reply, dict) or reply.get("drop"):
            return {**promo, "status": "dropped", "rule_errors": errs, "repair_history": history,
                    "drop_reason": (reply or {}).get("why") if isinstance(reply, dict) else "bad reply"}
        convo = convo + [{"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)}]
        promo = resolve_fn(reply)
        errs = check(promo, blocks)
        history.append(errs)
        if not errs:
            return {**promo, "status": "fixed", "rule_errors": [], "repair_history": history}
    return {**promo, "status": "failed", "rule_errors": errs, "repair_history": history}


def repair_message(p: dict, errs: list[str], blocks: dict[str, Block]) -> str:
    keep = {k: p.get(k) for k in ("hook_id", "cliffhanger_id", "hook", "cliffhanger")}
    return ("Your promo breaks these rules (measured exactly by code):\n- " + "\n- ".join(errs)
            + "\n\nHere is your promo with the exact time of every line from the promo start:\n"
            + timeline(p, blocks)
            + f"\n\nYou chose: {json.dumps(keep, ensure_ascii=False)}\n"
            "Fix it: keep the same hook and cliffhanger if possible. To shorten, start later or split into up to "
            "3 segments that skip slow lines (use the t+ times to make the total 20-60s). Use only block IDs "
            "from the episode text. Return ONE corrected promo as JSON in the same shape as before "
            '(with "segments", "hook_id", "cliffhanger_id", ...). If it cannot be fixed, return {"drop": true, "why": "..."}.')
