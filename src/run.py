"""Step 7: ONE command runs the whole pipeline on a show.

    parse -> show summary -> finder (every episode) -> pairs (every N->N+1) -> judge -> rank -> report

Usage:
    python -m src.run                                   # this show, all episodes
    python -m src.run --input "<new show folder>" --aliases none   # new show -> outputs/<show>_promos
    python -m src.run --episodes 1 2                    # quick test on a few episodes

A failing episode/pair is logged and skipped; the run continues. Every AI call is in outputs/raw_ai_calls/llm_calls.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src import finder, judge, pairs, safety, show_summary, why
from src.llm import USAGE, LLMError
from src.parser import parse_folder, write_outputs
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find, check and rank promos for a whole show.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="folder of episode .txt files")
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"), help="'none' for a new show")
    ap.add_argument("--out", default=None,
                    help="output folder (default: outputs/<show folder name>_promos)")
    ap.add_argument("--episodes", type=int, nargs="*", help="only these episodes (default: all)")
    ap.add_argument("--no-pairs", action="store_true")
    ap.add_argument("--finder-prompt", default=finder.PROMPT_VERSION, choices=sorted(finder.PROMPTS),
                    help="finder prompt version (default: the current one)")
    a = ap.parse_args(argv)

    t0 = time.time()
    out = Path(a.out) if a.out else paths.run_folder_for(a.input)
    (out / "finder").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    alias = None if a.aliases.lower() == "none" else Path(a.aliases)

    eps = parse_folder(Path(a.input), alias)
    write_outputs(eps, out / "parsed")
    by_num = {e.episode: e for e in eps}
    chosen = [by_num[n] for n in (a.episodes or sorted(by_num)) if n in by_num]
    log(f"parsed {len(eps)} episodes ({sum(len(e.blocks) for e in eps)} blocks); running on {len(chosen)}")

    summary = show_summary.get_for(eps)
    log(f"show summary ready: {len(summary.get('major_twists', []))} major twists")

    errors, promos = [], []
    for e in chosen:
        try:
            found = finder.find_in_episode(e, summary=summary, version=a.finder_prompt)
        except LLMError as err:
            errors.append(f"finder ep{e.episode:02d}: {err}")
            log(f"  ep{e.episode:02d}  ERROR {str(err)[:80]}")
            continue
        (out / "finder" / f"ep{e.episode:02d}.json").write_text(
            json.dumps({"prompt_version": a.finder_prompt, "episode": e.episode, "promos": found},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        promos += [{**p, "source_file": f"finder/ep{e.episode:02d}.json"} for p in found]
        log(f"  ep{e.episode:02d}  {len(found)} promo(s): " + ", ".join(p.get("status", "?") for p in found))

    if not a.no_pairs:
        nums = [e.episode for e in chosen]
        for n in nums:
            if n + 1 not in by_num:
                continue
            try:
                res = pairs.judge_pair(by_num[n], by_num[n + 1], summary=summary)
            except LLMError as err:
                errors.append(f"pair {n}->{n + 1}: {err}")
                log(f"  pair {n:02d}->{n + 1:02d}  ERROR {str(err)[:80]}")
                continue
            (out / "pairs" / f"ep{n:02d}-{n + 1:02d}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            if res.get("use"):
                promos.append({**res, "source_file": f"pairs/ep{n:02d}-{n + 1:02d}.json"})
            log(f"  pair {n:02d}->{n + 1:02d}  " + (f"use: {res.get('status')}" if res.get("use") else "skip"))

    usable = [p for p in promos if p.get("status") in ("ok", "fixed")]
    log(f"judging {len(usable)} rule-passing promos ...")
    blocks = {b.id: b for e in eps for b in e.blocks}
    try:
        judged = judge.judge(usable, blocks)
    except LLMError as err:
        errors.append(f"judge: {err}")
        judged = usable
    judged = safety.apply(judged, blocks)          # code-found graphic/sexual flags (Step 8c fix)
    judged, n_why = why.rewrite(judged, blocks)    # 'why it works' about the viewer, not the edit
    log(f"rewrote {n_why} 'why it works' lines that described the editing")
    ranked = judge.rank(judged)
    dropped = [p for p in promos if p.get("status") not in ("ok", "fixed")] + [p for p in judged if "score" not in p.get("judge", {})]
    judge.write_report(ranked, dropped, out / "ranked", judged)

    status = Counter(p.get("status") for p in promos)
    secs = time.time() - t0
    run = {
        "input": a.input, "episodes_run": len(chosen), "pairs_run": not a.no_pairs,
        "versions": {"finder": a.finder_prompt, "pairs": pairs.PROMPT_VERSION, "judge": judge.PROMPT_VERSION},
        "candidates": len(promos), "status": dict(status),
        "rule_pass_first_try": status.get("ok", 0), "rule_pass_after_repair": status.get("fixed", 0),
        "ranked": len(ranked), "new_viewer_promos": sum(p.get("audience") == "new_viewers" for p in ranked),
        "errors": errors, "wall_time_s": round(secs, 1), "llm_usage": USAGE,
    }
    (out / "run_summary.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"DONE in {secs / 60:.1f} min: {len(promos)} candidates, status {dict(status)}, {len(ranked)} ranked, "
        f"{len(errors)} errors")
    log(f"LLM usage: {USAGE}")
    log(f"report: {out / 'ranked' / 'ranked.md'}")


if __name__ == "__main__":
    main()
