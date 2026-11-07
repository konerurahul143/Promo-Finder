"""Step 8b (part 3): "Why not just paste the script into Gemini?"

The SAME finder model gets the RAW episode file and a plain request, with no block IDs, no code checks
and no repair. Code then checks what came back:
  - real timestamps: does each start/end match an actual line's start/end in the script (within 0.5s)?
  - inside the episode, segments in order
  - 20-60s long
  - hook within the first 3s (using the hook time the model itself reports)
  - reference moments covered
and compares with the tool's output for the same episodes.

Usage:
    python -m src.plain_baseline --episodes 1 3 5 12 13 18 23 24 34 45
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.evaluate import REF, SAME_MOMENT, covered, ref_segments
from src.llm import USAGE, LLMError, complete
from src.parser import parse_folder
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
TOL_S = 0.5

PLAIN_PROMPT = """Here is the screenplay of one episode of a vertical microdrama. Find the best promos in it:
20-60 second clips to post on TikTok/Reels/Shorts. A good promo opens on a hook in the first 3 seconds, ends on a
cliffhanger, and makes sense on its own. A promo can be one stretch or a few joined moments.

Reply with JSON only:
{"promos": [{"segments": [{"start": "MM:SS.cc", "end": "MM:SS.cc"}], "hook_time": "MM:SS.cc", "hook": "...",
 "cliffhanger": "...", "why": "..."}]}

SCREENPLAY:
"""
TS = re.compile(r"^(\d{1,2}):(\d{2}(?:\.\d+)?)$")


def secs(v) -> float | None:
    m = TS.match(str(v).strip()) if v is not None else None
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def check(p: dict, ep, starts: list[float], ends: list[float]) -> dict:
    segs, errs, fake = [], [], 0
    for s in p.get("segments", []):
        a, b = secs(s.get("start")), secs(s.get("end"))
        if a is None or b is None:
            errs.append("unreadable timestamp"); continue
        if not any(abs(a - x) <= TOL_S for x in starts) or not any(abs(b - x) <= TOL_S for x in ends):
            fake += 1
        if b > ep.duration + TOL_S:
            errs.append("timestamp beyond episode end")
        if b <= a:
            errs.append("segment ends before it starts")
        segs.append({"episode": ep.episode, "start": a, "end": b})
    if fake:
        errs.append(f"{fake} segment(s) with a start/end that matches no real line")
    dur = sum(max(0.0, s["end"] - s["start"]) for s in segs)
    if not 20 <= dur <= 60:
        errs.append(f"length {dur:.1f}s outside 20-60s")
    if any(b["start"] < a["end"] for a, b in zip(segs, segs[1:])):
        errs.append("segments overlap or out of order")
    h = secs(p.get("hook_time"))
    if segs and h is not None and h - segs[0]["start"] > 3:
        errs.append(f"hook {h - segs[0]['start']:.1f}s in (> 3s)")
    return {"segments": segs, "duration": round(dur, 1), "errors": errs, "fake_timestamps": fake}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--episodes", type=int, nargs="+", default=[1, 3, 5, 12, 13, 18, 23, 24, 34, 45])
    ap.add_argument("--run", default=str(paths.FINAL_RUN), help="tool run to compare with")
    a = ap.parse_args(argv)

    eps = {e.episode: e for e in parse_folder(Path(a.input), None)}
    results = []
    for n in a.episodes:
        ep = eps[n]
        raw = (Path(a.input) / ep.source_file).read_text(encoding="utf-8")
        try:
            reply = complete([{"role": "user", "content": PLAIN_PROMPT + raw}], json_mode=True,
                             tag=f"plain-baseline:ep{n:02d}")
        except LLMError as err:
            print(f"ep{n:02d} ERROR {str(err)[:80]}"); continue
        starts = [b.start for b in ep.blocks]
        ends = [b.end for b in ep.blocks]
        for p in (reply.get("promos", []) if isinstance(reply, dict) else []):
            results.append({"episode": n, **check(p, ep, starts, ends), "raw": p})
        print(f"ep{n:02d}: {sum(r['episode'] == n for r in results)} promo(s)")

    total = len(results)
    clean = sum(not r["errors"] for r in results)
    fake = sum(r["fake_timestamps"] > 0 for r in results)
    length_bad = sum(any("length" in e for e in r["errors"]) for r in results)
    hook_bad = sum(any("hook" in e for e in r["errors"]) for r in results)

    ref = [rp for rp in json.loads(REF.read_text(encoding="utf-8"))["promos"]
           if {s["episode"] for s in rp["segments"]} <= set(a.episodes)]
    def hit(clips):
        return sum(any(covered(ref_segments(rp), c["segments"]) / sum(s["end"] - s["start"] for s in ref_segments(rp))
                       >= SAME_MOMENT for c in clips) for rp in ref)
    tool = [p for p in json.loads((Path(a.run) / "ranked" / "judged.json").read_text(encoding="utf-8"))
            if all(s["episode"] in a.episodes for s in p["segments"])]
    tool_first = sum(1 for p in tool if p.get("status") == "ok")

    summary = {"episodes": a.episodes, "plain_promos": total,
               "plain_pass_all_checks": clean, "plain_with_fake_timestamps": fake,
               "plain_length_out_of_range": length_bad, "plain_hook_late": hook_bad,
               "plain_ref_moments_covered": hit(results), "tool_promos_same_episodes": len(tool),
               "tool_first_try_pass": tool_first, "tool_final_pass": len(tool),
               "tool_ref_moments_covered": hit(tool), "refs_in_these_episodes": len(ref), "llm_usage": USAGE}
    out = Path(a.run) / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plain_gemini.json").write_text(json.dumps({"summary": summary, "promos": results}, indent=2,
                                                      ensure_ascii=False), encoding="utf-8")
    print(f"\nPLAIN GEMINI (same model, raw script, no system) on {len(a.episodes)} episodes: {total} promos")
    print(f"  pass ALL checks          : {clean}/{total}")
    print(f"  made-up timestamps       : {fake}/{total} promos have a start/end that matches no real line")
    print(f"  length outside 20-60s    : {length_bad}/{total}")
    print(f"  hook later than 3s       : {hook_bad}/{total}")
    print(f"  reference moments covered: {summary['plain_ref_moments_covered']}/{len(ref)}")
    print(f"TOOL on the same episodes: {len(tool)} promos, first try {tool_first}/{len(tool)}, final 100% pass, "
          f"timestamps always real (from code), ref moments {summary['tool_ref_moments_covered']}/{len(ref)}")
    print(f"Saved {out / 'plain_gemini.json'}")


if __name__ == "__main__":
    main()
