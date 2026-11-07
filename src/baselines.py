"""Step 8b: does the tool beat simple methods anyone could use without AI?

Baselines (built by code, snapped to camera cuts like the tool's clips):
  - random : a random 30s clip from each episode (fixed seed, repeatable)
  - ending : the last 30s of each episode (microdramas end on cliffhangers, so this is a strong baseline)
Blind test: for N sampled episodes, the tool's best single-episode clip + both baselines are SHUFFLED and
judged together without labels by the same judge model. Also: how many reference moments each method covers.

Usage:
    python -m src.baselines --run outputs/FINAL_full_run --sample 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from src import judge
from src.evaluate import REF, SAME_MOMENT, covered, ref_segments
from src.parser import Episode, fmt_ts, parse_folder
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
CLIP_S = 30.0
SEED = 7


def _clip(ep: Episode, start: float, length: float = CLIP_S) -> dict:
    cuts = ep.shot_cuts()
    s = max([c for c in cuts if c <= start] or [0.0])
    e = min([c for c in cuts if c >= s + length] or [ep.duration])
    seg = {"episode": ep.episode, "start": s, "end": e, "start_ts": fmt_ts(s), "end_ts": fmt_ts(e),
           "seconds": round(e - s, 2), "start_id": "", "end_id": ""}
    return {"segments": [seg], "duration": round(e - s, 2)}


def random_clip(ep: Episode, rng: random.Random) -> dict:
    return _clip(ep, rng.uniform(0, max(0.0, ep.duration - CLIP_S)))


def ending_clip(ep: Episode) -> dict:
    return _clip(ep, max(0.0, ep.duration - CLIP_S))


def tool_best(judged: list[dict], episode: int) -> dict | None:
    """The tool's best-scoring single-episode clip for this episode (what an editor would post first)."""
    own = [p for p in judged if all(s["episode"] == episode for s in p["segments"]) and "score" in p.get("judge", {})]
    return max(own, key=lambda p: p["judge"]["score"]) if own else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"))
    ap.add_argument("--run", default=str(paths.FINAL_RUN))
    ap.add_argument("--sample", type=int, default=20, help="episodes in the blind judge test")
    ap.add_argument("--judge-profile", default=None, help="judge model profile (default: `judge` in model.yaml)")
    a = ap.parse_args(argv)

    alias = None if a.aliases.lower() == "none" else Path(a.aliases)
    eps = {e.episode: e for e in parse_folder(Path(a.input), alias)}
    blocks = {b.id: b for e in eps.values() for b in e.blocks}
    judged = json.loads((Path(a.run) / "ranked" / "judged.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    rand = {n: random_clip(e, rng) for n, e in sorted(eps.items())}
    end = {n: ending_clip(e) for n, e in sorted(eps.items())}

    # --- (1) reference coverage over ALL 50 episodes (code only) ---
    ref = json.loads(REF.read_text(encoding="utf-8"))["promos"]
    def hits(clips: list[dict]) -> int:
        n = 0
        for rp in ref:
            rs = ref_segments(rp)
            d = sum(s["end"] - s["start"] for s in rs)
            n += any(covered(rs, c["segments"]) / d >= SAME_MOMENT for c in clips)
        return n
    best_per_ep = [b for b in (tool_best(judged, n) for n in eps) if b]
    coverage = {"random (1 clip/ep)": hits(list(rand.values())), "ending (1 clip/ep)": hits(list(end.values())),
                "tool best clip/ep": hits(best_per_ep), "tool all clips": hits(judged)}

    # --- (2) blind judge test on a sample of episodes ---
    sample = sorted(random.Random(SEED + 1).sample(sorted(n for n in eps if tool_best(judged, n)), a.sample))
    items = []
    for n in sample:
        tb = tool_best(judged, n)
        for method, clip in (("tool", {"segments": tb["segments"], "duration": tb["duration"]}),
                             ("random", rand[n]), ("ending", end[n])):
            items.append({**clip, "_method": method, "_episode": n})
    random.Random(SEED + 2).shuffle(items)                      # blind: methods mixed in every batch
    scored = judge.judge([{k: v for k, v in it.items()} for it in items], blocks, a.judge_profile)

    by = {m: [p["judge"] for p in scored if p["_method"] == m and "score" in p["judge"]] for m in ("tool", "random", "ending")}
    table = {m: {k: round(mean(j[k] for j in v), 2) for k in ("score", "hook", "cliffhanger", "standalone")} | {"n": len(v)}
             for m, v in by.items() if v}
    per_ep = {}
    for p in scored:
        if "score" in p["judge"]:
            per_ep.setdefault(p["_episode"], {})[p["_method"]] = p["judge"]["score"]
    wins = {b: {"tool_wins": sum(1 for d in per_ep.values() if d.get("tool", 0) > d.get(b, 0)),
                "ties": sum(1 for d in per_ep.values() if d.get("tool", 0) == d.get(b, 0)),
                "tool_loses": sum(1 for d in per_ep.values() if d.get("tool", 0) < d.get(b, 0))}
            for b in ("random", "ending")}

    out = Path(a.run) / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    res = {"judge_profile": a.judge_profile or "default judge", "sample_episodes": sample, "clip_length_s": CLIP_S, "seed": SEED, "blind_judge": table,
           "tool_vs_baseline_per_episode": wins, "reference_moments_covered_of_13": coverage, "per_episode": per_ep}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"Blind judge test on {len(sample)} episodes (clips shuffled, no labels):")
    print(f"  {'method':8} {'score':>6} {'hook':>5} {'cliff':>6} {'alone':>6}")
    for m, t in table.items():
        print(f"  {m:8} {t['score']:6.2f} {t['hook']:5.2f} {t['cliffhanger']:6.2f} {t['standalone']:6.2f}  (n={t['n']})")
    for b, w in wins.items():
        print(f"  tool vs {b:7}: tool better in {w['tool_wins']}, tie {w['ties']}, worse in {w['tool_loses']} episodes")
    print("Reference moments covered (of 13):", coverage)
    print(f"Saved {out / 'results.json'}")


if __name__ == "__main__":
    main()
