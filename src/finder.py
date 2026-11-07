"""Step 4a: the FINDER. One AI call per episode -> candidate promos as block IDs.

The AI does judgment only (which moment hooks, where it's unresolved, does it stand alone).
It never writes timestamps: it cites block IDs, and code converts them to exact times.

Usage:
    python -m src.finder --episode 1                 # one episode, prints + saves JSON
    python -m src.finder --episode 1 --camera        # include camera lines in the prompt (A/B test)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import rules, show_summary
from src.llm import USAGE, complete
from src.parser import Episode, episode_to_prompt_text, fmt_ts, parse_folder
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
PROMPT_VERSION = "finder-v5"   # default; v1-v4 kept for comparison (--prompt finder-v1)

SYSTEM_V1 = """You are a senior promo editor for vertical microdrama apps (like ReelShort and DramaBox).
You pick moments from a show to post as 20-60 second promos on TikTok, Instagram Reels and YouTube Shorts,
to make scrolling strangers stop and start watching.

A GOOD PROMO:
1. HOOK: the first 3 seconds make a scrolling viewer stop: a slap, a shove, a reveal, a shocking line, an explosion.
2. CLIFFHANGER: it ends on an unresolved moment (a threat, an ultimatum, a door opening, a question, a reveal cut mid-beat),
   so the viewer needs to know what happens next. It must NOT end after the tension is resolved.
3. STANDALONE: it makes sense to someone who has never seen the show (who is wronging whom, and what is at stake).
4. LENGTH: 20-60 seconds in total. It can be ONE continuous stretch, or up to 3 segments from this episode, in time order,
   to cut out slow talk.

RULES:
- Cite moments ONLY by the block IDs shown in [brackets] (e.g. E01-B023). Never invent IDs or timestamps.
- A segment runs from the START of start_id to the END of end_id.
- hook_id must be the block inside the first 3 seconds of the promo; cliffhanger_id must be the last meaningful block.
- Return 0 to 3 promos. Quality over quantity: if the episode has no strong moment, return fewer (or none).
- Do not pick near-duplicates of the same moment.
- Flag "spoiler" if it reveals a major twist, "graphic" for gore/torture, "sexual" for sexual content.

Reply with JSON only, in exactly this shape:
{"promos": [
  {"segments": [{"start_id": "E01-B001", "end_id": "E01-B010"}],
   "hook_id": "E01-B001", "hook": "short description of the hook",
   "cliffhanger_id": "E01-B010", "cliffhanger": "short description of the unresolved ending",
   "why": "one line on why it works as a promo",
   "standalone": "yes|partly|no",
   "flags": []}
]}"""

# v2 changes (from the Ep 1 v1 run): the hook was 5.3s in because the clip opened on a lead-in line.
# v2 makes the clip START ON the hook, bans mid-sentence cuts, and asks for the target audience.
SYSTEM_V2 = """You are a senior promo editor for vertical microdrama apps (like ReelShort and DramaBox).
You pick moments from a show to post as 20-60 second promos on TikTok, Instagram Reels and YouTube Shorts,
to make scrolling strangers stop and start watching.

HOW TO BUILD EACH PROMO (in this order):
1. Pick the HOOK block first: the single most stopping moment, a slap, a shove, a throw, a reveal, a shocking
   or cruel line, an explosion. The promo STARTS ON the hook: the first segment's start_id is the hook block itself,
   or at most one short block before it, so the hook lands within the first 3 seconds. No lead-in, no setup lines.
2. Pick the CLIFFHANGER block: an unresolved moment (a threat, an ultimatum, a door opening, an unanswered question,
   a reveal cut mid-beat, a defiant "No"). The promo ENDS ON it. Never continue past the point where tension resolves.
3. Fill the middle so a stranger understands who is wronging whom and what is at stake. Use up to 3 segments from
   this episode, in time order, to skip slow talk. Total length 20-60 seconds (use the times shown).

CUT RULES:
- Never start on a line that begins mid-sentence ("...and it isn't you") and never end on a line that ends
  mid-sentence ("I may leave this chapel..."). Keep both halves of a split sentence together.
- Avoid V.O. (inner thoughts) as the hook; they read as confusing out of context.
- Cite moments ONLY by the block IDs shown in [brackets]. Never invent IDs or timestamps.
- A segment runs from the START of start_id to the END of end_id.

SELECTION:
- Return 0 to 3 promos. Quality over quantity: a slow episode can return none. No near-duplicates.
- audience: "new_viewers" if it works with zero context and no major spoiler; "retargeting" if it needs the story
  or reveals a twist (e.g. a hidden identity).
- flags: "spoiler" (reveals a major twist), "graphic" (gore/torture), "sexual" (sexual content).

Reply with JSON only, in exactly this shape:
{"promos": [
  {"segments": [{"start_id": "E01-B001", "end_id": "E01-B010"}],
   "hook_id": "E01-B001", "hook": "short description of the hook",
   "cliffhanger_id": "E01-B010", "cliffhanger": "short description of the unresolved ending",
   "why": "one line on why it works as a promo",
   "standalone": "yes|partly|no",
   "audience": "new_viewers|retargeting",
   "flags": []}
]}"""

# v4 change (from the Step 6 judge): "start ON the hook" over-corrected. Ep 1's cut opened on the punchline
# the punchline and lost the setup line before it, so a stranger didn't know who the line was about.
# v4 allows 1-2 short setup lines before the hook, as long as the hook still starts within 3 seconds.
SYSTEM_V4 = SYSTEM_V2.replace(
    """   or at most one short block before it, so the hook lands within the first 3 seconds. No lead-in, no setup lines.""",
    """   or 1-2 SHORT setup lines before it when a stranger needs them to understand the hook (e.g. a question
   the hook answers, or a line that names who is involved). The hook block must still START within the first
   3 seconds of the promo. No long lead-in.
   The opening must make clear WHO is involved: a punchline with no setup ("Because he is marrying me." with
   no hint who "he" is) is a weak hook for a stranger.""")
assert SYSTEM_V4 != SYSTEM_V2, "v4 prompt edit did not apply"

# v5 change (from Step 8a evaluation of the full run): the finder returned ONE promo in 49/50 episodes (losing
# the second strong moment) and built "collage" clips (73/87 had 2-4 pieces jumping around the episode), while
# good promos are mostly one continuous stretch. v5 asks for up to 3 DIFFERENT moments and continuous cuts.
SYSTEM_V5 = SYSTEM_V4.replace(
    """3. Fill the middle so a stranger understands who is wronging whom and what is at stake. Use up to 3 segments from
   this episode, in time order, to skip slow talk. Total length 20-60 seconds (use the times shown).""",
    """3. Fill the middle so a stranger understands who is wronging whom and what is at stake. Total length 20-60
   seconds (use the times shown).
   PREFER ONE CONTINUOUS STRETCH. Use a 2nd/3rd segment ONLY to skip a few slow lines inside the SAME moment
   (pieces close together in time). NEVER stitch unrelated moments from different parts of the episode into
   one promo: if the episode has two strong moments, make them two separate promos.""").replace(
    """- Return 0 to 3 promos. Quality over quantity: a slow episode can return none. No near-duplicates.""",
    """- Return 1 to 3 promos. Most episodes have 2-3 strong, DIFFERENT moments (e.g. a humiliation, a threat,
  a reveal): return each one as its own promo. Return only 1 if the episode truly has one, 0 if none.
  No near-duplicates: different promos must cover different moments.""")
assert SYSTEM_V5.count("PREFER ONE CONTINUOUS STRETCH") == 1 and "Return 1 to 3 promos" in SYSTEM_V5, "v5 edit failed"

# v3 = v2 + the one-time SHOW SUMMARY (major twists), so the AI can flag spoilers it can't see in one episode.
PROMPTS = {"finder-v1": SYSTEM_V1, "finder-v2": SYSTEM_V2, "finder-v3": SYSTEM_V2, "finder-v4": SYSTEM_V4,
           "finder-v5": SYSTEM_V5}
USES_SUMMARY = {"finder-v3", "finder-v4", "finder-v5"}


def build_messages(ep: Episode, include_camera: bool = False, version: str = PROMPT_VERSION,
                   summary: dict | None = None) -> list[dict]:
    if version in USES_SUMMARY and summary is None:
        print(f"  WARNING: {version} expects a show summary but none was given: spoiler judgment is off")
    context = show_summary.as_prompt_text(summary) if version in USES_SUMMARY else ""
    return [{"role": "system", "content": PROMPTS[version]},
            {"role": "user", "content": (context + "\n" if context else "") + "Find the best promos in this episode.\n\n"
                                        + episode_to_prompt_text(ep, include_camera=include_camera)}]


def resolve(ep: Episode, raw: dict) -> dict:
    """Turn the AI's block IDs into exact times. Records problems instead of crashing."""
    by_id = {b.id: b for b in ep.blocks}
    out = {**raw, "episode": ep.episode, "problems": []}
    segs = []
    for s in raw.get("segments", []):
        a, b = by_id.get(s.get("start_id")), by_id.get(s.get("end_id"))
        if not a or not b:
            out["problems"].append(f"unknown block id in segment {s}")
            continue
        segs.append({"episode": ep.episode, "start_id": a.id, "end_id": b.id,
                     "start": a.start, "end": b.end,
                     "start_ts": fmt_ts(a.start), "end_ts": fmt_ts(b.end),
                     "seconds": round(b.end - a.start, 2)})
    out["segments"] = segs
    out["duration"] = round(sum(s["seconds"] for s in segs), 2)
    for key in ("hook_id", "cliffhanger_id"):
        blk = by_id.get(raw.get(key))
        if not blk:
            out["problems"].append(f"unknown {key}: {raw.get(key)}")
        else:
            out[key.replace("_id", "_at")] = fmt_ts(blk.start)
            out[key.replace("_id", "_text")] = blk.text
            if key == "hook_id" and segs:
                out["hook_delay_s"] = round(blk.start - segs[0]["start"], 2)   # rule: <= 3s
    return out


def find_in_episode(ep: Episode, include_camera: bool = False, profile: str | None = None,
                    version: str = PROMPT_VERSION, repair: bool = True, summary: dict | None = None) -> list[dict]:
    msgs = build_messages(ep, include_camera, version, summary)
    tag = f"{version}:ep{ep.episode:02d}{':cam' if include_camera else ''}"
    reply = complete(msgs, json_mode=True, profile=profile, tag=tag)
    promos = reply.get("promos", []) if isinstance(reply, dict) else []
    resolved = [resolve(ep, p) for p in promos]
    if not repair:
        return resolved
    by_id = {b.id: b for b in ep.blocks}
    cuts = {ep.episode: ep.shot_cuts()}
    out = []
    for i, p in enumerate(resolved, 1):
        # code checks the hard rules; any failing promo goes back to the AI with exact numbers (max 2 rounds)
        p = rules.repair_loop(msgs, reply, p, lambda raw: resolve(ep, raw), by_id, profile, f"{tag}:p{i}")
        if p["status"] in ("ok", "fixed"):
            p = rules.snap_to_cuts(p, cuts)          # Step 5: clean cuts on camera shot boundaries
            p = rules.merge_touching(p)              # Step 5: join pieces that touch (gap <= 1s)
            final = rules.check(p, by_id)            # final check after cleanup (dead air, mid-sentence, ...)
            if final:
                p = {**p, "status": "failed", "rule_errors": final}
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"))
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--camera", action="store_true", help="include camera lines in the prompt")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--prompt", default=PROMPT_VERSION, choices=sorted(PROMPTS))
    ap.add_argument("--out", default=str(paths.ARCHIVE / "dev_single_episode_finder"))
    a = ap.parse_args(argv)

    alias = None if a.aliases.lower() == "none" else Path(a.aliases)
    eps = parse_folder(Path(a.input), alias)
    ep = next(e for e in eps if e.episode == a.episode)
    summary = show_summary.get_for(eps) if a.prompt in USES_SUMMARY else None   # THIS show's summary
    promos = find_in_episode(ep, a.camera, a.profile, a.prompt, summary=summary)

    out = Path(a.out) / a.prompt
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"ep{ep.episode:02d}{'_cam' if a.camera else ''}.json"
    f.write_text(json.dumps({"prompt_version": a.prompt, "episode": ep.episode,
                             "include_camera": a.camera, "promos": promos}, indent=2, ensure_ascii=False),
                 encoding="utf-8")

    print(f"Episode {ep.episode}: {len(promos)} promo(s)  ->  {f}")
    for i, p in enumerate(promos, 1):
        segs = " + ".join(f"{s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        print(f"\n#{i}  {segs}  ({p['duration']}s)  standalone={p.get('standalone')}  "
              f"audience={p.get('audience')}  flags={p.get('flags')}")
        print(f"   hook @{p.get('hook_at')} (+{p.get('hook_delay_s')}s): {p.get('hook')}")
        print(f"   cliff @{p.get('cliffhanger_at')}: {p.get('cliffhanger')}")
        print(f"   why: {p.get('why')}")
        if "status" in p:
            print(f"   rules: {p['status'].upper()}  (checks per round: {[len(e) for e in p['repair_history']]} errors)"
                  + (f"  remaining: {p['rule_errors']}" if p["rule_errors"] else ""))
        if "snapped" in p:
            print(f"   cuts: {'SNAPPED' if p['snapped'] else 'NOT snapped'} - {p['snap_note']}")
    print(f"\nUsage: {USAGE}")


if __name__ == "__main__":
    main()
