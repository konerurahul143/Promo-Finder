"""Step 6: JUDGE + RANK.

Judge = a DIFFERENT model family from the finder (gpt-oss-120b on Groq vs Gemini), so no model grades its own picks.
It sees ONLY what a viewer would get: the lines inside each clip with t+ times. It does NOT see the finder's
"why" (no sales pitch) and never sees the reference set. It scores hook / cliffhanger / standalone 1-5.
CODE then ranks: new-viewer promos first, then by a weighted score, with a reason for every position.

Usage:
    python -m src.judge          # judges + ranks single-episode dev outputs in outputs/archive (normally use src.run)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.finder import PROMPT_VERSION as FINDER_VERSION
from src.llm import USAGE, complete
from src.parser import Block, parse_folder
from src.rules import timeline
from src import paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = paths.DEFAULT_INPUT
# v2: hook scored ONLY on a separate "first 3 seconds" block cut out by code.
# v3: that block also includes CAMERA shots (as SHOT:), because visual hooks (e.g. Ep 1's close-up of the
#     groom's fingers locked with the sister's hand at +2.8s) exist only in camera lines.
PROMPT_VERSION = "judge-v3"
HOOK_WINDOW_S = 3.0
BATCH = 5                                    # promos per call (stays within Groq free tokens-per-minute)
WEIGHTS = {"hook": 0.40, "cliffhanger": 0.35, "standalone": 0.25}

SYSTEM = """You are a tough, experienced performance-marketing editor for vertical microdrama apps.
You judge short promo clips (20-60s) that will be posted on TikTok / Reels / Shorts to win NEW viewers.
You see each clip as a script: every line with its time from the clip start (t+seconds). Judge only what is
in the clip, as a stranger scrolling past would experience it.

Score each clip 1-5 (5 = best) on:
- hook: score ONLY the "FIRST 3 SECONDS" block given for each clip. Anything after t+3s does NOT count for the
  hook, however dramatic. Would those 3 seconds stop a scrolling stranger? (5 = slap/explosion/shocking line;
  1 = calm talk or setup). A punchline whose meaning is unclear without context scores lower.
  The block includes SHOT lines (what the camera shows). Count strong VISUALS too: a shocking image can hook
  a viewer as much as a shocking line.
- cliffhanger: does it END on an unresolved moment that makes you need the next episode? (1 = resolved/calm)
- standalone: does a stranger understand who is wronging whom and what is at stake? (1 = confusing)
Be strict and use the full scale; don't give everything 4s.

Reply with JSON only:
{"scores": [{"id": "P1", "hook": 1, "cliffhanger": 1, "standalone": 1,
             "first_3s": "what happens in the first 3 seconds",
             "verdict": "one line: would you post this, and why / why not"}]}"""


def first_seconds(p: dict, blocks: dict[str, Block], window: float = HOOK_WINDOW_S) -> str:
    """Lines that START inside the first `window` seconds of the clip, cut out by code (judge v2)."""
    s = p["segments"][0]
    lines = []
    for b in sorted(blocks.values(), key=lambda b: (b.start, b.kind != "camera")):
        if b.episode == s["episode"] and s["start"] <= b.start < s["start"] + window:
            if b.kind == "camera":
                who = "SHOT (what the viewer sees): "
            else:
                who = f"{b.character}: " if b.kind == "dialogue" else f"{b.kind.upper()}: "
            lines.append(f"t+{b.start - s['start']:.1f}s {who}{b.text[:120]}")
    return "\n".join(lines) or "(nothing starts in the first 3 seconds)"


def promo_text(pid: str, p: dict, blocks: dict[str, Block]) -> str:
    return (f"=== {pid} | {p['duration']:.1f}s | {len(p['segments'])} segment(s) ===\n"
            f"FIRST 3 SECONDS (score the hook ONLY on this):\n{first_seconds(p, blocks)}\n"
            f"FULL CLIP:\n" + timeline(p, blocks))


def judge(promos: list[dict], blocks: dict[str, Block], profile: str | None = None) -> list[dict]:
    """Adds p['judge'] = {hook, cliffhanger, standalone, first_3s, verdict, score} to each promo."""
    out = [dict(p) for p in promos]
    for i in range(0, len(out), BATCH):
        batch = out[i:i + BATCH]
        ids = [f"P{j + 1}" for j in range(len(batch))]
        text = "\n\n".join(promo_text(pid, p, blocks) for pid, p in zip(ids, batch))
        reply = complete([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": "Judge these clips.\n\n" + text}],
                         json_mode=True, profile=profile, role="judge", tag=f"{PROMPT_VERSION}:batch{i // BATCH + 1}")
        got = {s.get("id"): s for s in (reply.get("scores", []) if isinstance(reply, dict) else [])}
        for pid, p in zip(ids, batch):
            s = got.get(pid)
            if not s:
                p["judge"] = {"error": "judge returned no score for this clip"}
                continue
            sc = {k: max(1, min(5, int(s.get(k, 1)))) for k in WEIGHTS}
            p["judge"] = {**sc, "first_3s": s.get("first_3s", ""), "verdict": s.get("verdict", ""),
                          "score": round(sum(WEIGHTS[k] * sc[k] for k in WEIGHTS), 2)}
    return out


DUP_OVERLAP = 0.30   # two promos sharing >= 30% of the shorter one's time = two cuts of the same moment


def overlap_s(a: dict, b: dict) -> float:
    """Seconds of footage two promos share (same episode, overlapping times)."""
    return sum(max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
               for x in a["segments"] for y in b["segments"] if x["episode"] == y["episode"])


def rank(promos: list[dict]) -> list[dict]:
    """Code ranks: only promos that passed every rule.
    Fix 1 (full run): a promo flagged "spoiler" is ALWAYS re-targeting, whatever audience the AI wrote
             (3 spoiler promos were labelled new_viewers and ranked among new-viewer promos).
    Fix 2 (full run): overlapping promos (>= 30% of the shorter clip shared) are cuts of the same moment:
             keep the higher-scoring one, attach the others to it as `alternatives` (not ranked separately).
    Order: new-viewer promos first, then weighted judge score (hook 40 / cliffhanger 35 / standalone 25),
    then hook as tie-break."""
    ok = [dict(p) for p in promos if p.get("status") in ("ok", "fixed") and "score" in p.get("judge", {})]
    for p in ok:
        if "spoiler" in (p.get("flags") or []) and p.get("audience") != "retargeting":
            p["audience_note"] = f"AI said '{p.get('audience')}', moved to retargeting because it is flagged spoiler"
            p["audience"] = "retargeting"

    # dedupe: best score first, so the strongest cut of each moment survives
    kept: list[dict] = []
    for p in sorted(ok, key=lambda p: (-p["judge"]["score"], -p["judge"]["hook"])):
        twin = next((k for k in kept if overlap_s(p, k) >= DUP_OVERLAP * min(p["duration"], k["duration"])), None)
        if twin is not None:
            twin.setdefault("alternatives", []).append(
                {"segments": p["segments"], "duration": p["duration"], "score": p["judge"]["score"],
                 "shared_s": round(overlap_s(p, twin), 1), "source_file": p.get("source_file")})
        else:
            kept.append(p)

    ok = kept
    ok.sort(key=lambda p: (p.get("audience") != "new_viewers", -p["judge"]["score"], -p["judge"]["hook"]))
    for i, p in enumerate(ok, 1):
        j = p["judge"]
        aud = "new viewers" if p.get("audience") == "new_viewers" else "re-targeting (existing viewers)"
        flags = f"; flags: {', '.join(p['flags'])}" if p.get("flags") else ""
        alts = f" ({len(p['alternatives'])} alternative cut(s) of this moment listed below it)" if p.get("alternatives") else ""
        note = f" [{p['audience_note']}]" if p.get("audience_note") else ""
        p["rank"] = i
        p["rank_reason"] = (f"#{i} for {aud}: score {j['score']:.2f}/5 (hook {j['hook']}, cliffhanger "
                            f"{j['cliffhanger']}, standalone {j['standalone']}){flags}{note}{alts}. "
                            f"First 3s: {j['first_3s']}. Judge: {j['verdict']}")
    return ok


def load_promos(dirs: list[Path]) -> list[dict]:
    promos = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            items = data["promos"] if "promos" in data else ([data] if data.get("use") else [])
            for p in items:
                promos.append({**p, "source_file": f"{d.name}/{f.name}"})
    return promos


def write_report(ranked: list[dict], dropped: list[dict], out: Path, judged: list[dict] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if judged is not None:   # full records of every judged promo: what --rerank must read (bug fix)
        (out / "judged.json").write_text(json.dumps(judged, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(ranked, dropped, out / "ranked.md")
    _write_csv(ranked, out / "promos.csv")


def _episodes(p: dict) -> str:
    eps = sorted({s["episode"] for s in p["segments"]})
    return "Ep " + " → ".join(str(e) for e in eps)


def _segments(p: dict, sep: str = " → ") -> str:
    return sep.join(f"Ep{s['episode']} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])


def _at(value) -> str:
    """hook_at is 'MM:SS.cc' for single-episode promos and 'EpN MM:SS.cc' for pairs."""
    return str(value) if value else "?"


def _write_md(ranked: list[dict], dropped: list[dict], path: Path) -> None:
    """The editor's report, in the PDF's format: for each promo the episode, every segment's start/end in order,
    the hook, the cliffhanger and one line on why it works, then the rank and why."""
    n_new = sum(p.get("audience") == "new_viewers" for p in ranked)
    lines = ["# Promos, ranked (post #1 first)", "",
             f"{len(ranked)} distinct promos: {n_new} for **new viewers** (ranked first), {len(ranked) - n_new} for "
             "**re-targeting** existing viewers (spoilers). Every promo passed the code checks: 20–60s, hook within 3s, "
             "clean camera cuts, no dead air, no mid-sentence cuts.", "",
             "> **Before posting, an editor must:** (1) check every ⚠ graphic / sexual flag; (2) review the AI-written "
             "twist list once per show (`python -m src.show_summary --review`), because spoiler labels come from it "
             "and it can be wrong both ways.", "",
             "## Summary", "", "| Rank | Episode | Segments (in order) | Length | Audience | Score | Flags |",
             "|---|---|---|---|---|---|---|"]
    for p in ranked:
        lines.append(f"| {p['rank']} | {_episodes(p)} | {_segments(p, ' + ')} | {p['duration']:.1f}s | "
                     f"{p.get('audience')} | {p['judge']['score']:.2f} | {', '.join(p.get('flags') or []) or '–'} |")
    lines += ["", "## Each promo", ""]
    for p in ranked:
        j = p["judge"]
        aud = "new viewers" if p.get("audience") == "new_viewers" else "re-targeting (existing viewers)"
        flags = f" · ⚠ {', '.join(p['flags'])}" if p.get("flags") else ""
        lines += [f"### #{p['rank']} · {_episodes(p)} · {p['duration']:.1f}s · {aud}{flags}", "",
                  f"- **Segments:** {_segments(p)}",
                  f"- **Hook** ({_at(p.get('hook_at'))}): {p.get('hook', '')}  ",
                  f"  line: *\"{(p.get('hook_text') or '')[:140]}\"*",
                  f"- **Cliffhanger** ({_at(p.get('cliffhanger_at'))}): {p.get('cliffhanger', '')}  ",
                  f"  line: *\"{(p.get('cliffhanger_text') or '')[:140]}\"*",
                  f"- **Why it works:** {p.get('why', '')}",
                  f"- **Why rank #{p['rank']}:** score {j['score']:.2f}/5 (hook {j['hook']}, cliffhanger "
                  f"{j['cliffhanger']}, makes sense alone {j['standalone']}). Judge: {j.get('verdict', '')}"]
        if p.get("audience_note"):
            lines.append(f"- **Audience:** {p['audience_note']}")
        for flag, words in (p.get("safety_evidence") or {}).items():
            lines.append(f"- **⚠ {flag} (found by code, editor must check):** {', '.join(words[:8])}")
        for a in p.get("alternatives", []):
            lines.append(f"- *Alternative cut of the same moment:* {_segments(a)} ({a['duration']:.1f}s, "
                         f"score {a['score']:.2f}, shares {a['shared_s']}s)")
        lines.append("")
    if dropped:
        lines += ["## Not ranked (failed a code check or the judge)", ""]
        lines += [f"- {p.get('source_file')}: {p.get('status')}: {p.get('rule_errors') or p.get('judge', {}).get('error', '')}"
                  for p in dropped]
    path.write_text("\n".join(lines), encoding="utf-8")


DECISIONS_FILE = "editor_decisions.json"


def clip_id(p: dict) -> str:
    """Stable ID for a clip from its segments, so an editor's decision survives re-ranking."""
    return "|".join(f"E{s['episode']}:{s['start']:.2f}-{s['end']:.2f}" for s in p["segments"])


def load_decisions(folder: Path) -> dict:
    f = folder / DECISIONS_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _write_csv(ranked: list[dict], path: Path) -> None:
    """One row per promo for Excel / Google Sheets. Segment columns are in order (up to 3 per episode, 2 episodes)."""
    import csv
    # HUMAN IN THE LOOP: the editor fills editor_decision (approve / reject) + editor_note, then runs
    # `python -m src.review`. Saved decisions are re-filled here whenever the report is regenerated.
    decisions = load_decisions(path.parent)
    cols = ["clip_id", "editor_decision", "editor_note", "rank", "audience", "flags", "episodes", "length_s",
            "segment_1", "segment_2", "segment_3", "segment_4",
            "hook_at", "hook", "hook_line", "cliffhanger_at", "cliffhanger", "cliffhanger_line", "why_it_works",
            "score", "hook_score", "cliffhanger_score", "standalone_score", "judge_verdict",
            "alternative_cuts", "source_file"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:    # utf-8-sig: Excel shows quotes/dashes right
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in ranked:
            j = p["judge"]
            segs = [f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]] + [""] * 4
            cid = clip_id(p)
            d = decisions.get(cid, {})
            w.writerow({"clip_id": cid, "editor_decision": d.get("decision", ""), "editor_note": d.get("note", ""),
                        "rank": p["rank"], "audience": p.get("audience"), "episodes": _episodes(p),
                        "length_s": p["duration"], "segment_1": segs[0], "segment_2": segs[1],
                        "segment_3": segs[2], "segment_4": segs[3],
                        "hook_at": p.get("hook_at"), "hook": p.get("hook"), "hook_line": p.get("hook_text"),
                        "cliffhanger_at": p.get("cliffhanger_at"), "cliffhanger": p.get("cliffhanger"),
                        "cliffhanger_line": p.get("cliffhanger_text"), "why_it_works": p.get("why"),
                        "score": j["score"], "hook_score": j["hook"], "cliffhanger_score": j["cliffhanger"],
                        "standalone_score": j["standalone"], "judge_verdict": j.get("verdict"),
                        "flags": ", ".join(p.get("flags") or []),
                        "alternative_cuts": " | ".join(_segments(a, " + ") for a in p.get("alternatives", [])),
                        "source_file": p.get("source_file")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--aliases", default=str(ROOT / "config" / "character_aliases.json"))
    ap.add_argument("--promos", nargs="+", default=[str(paths.ARCHIVE / "dev_single_episode_finder" / FINDER_VERSION),
                                                    str(paths.ARCHIVE / "dev_single_pair" / "pair-v2")])
    ap.add_argument("--profile", default=None, help="judge profile (default: `judge` in model.yaml)")
    ap.add_argument("--out", default=str(paths.ARCHIVE / "dev_judge_ranking"))
    ap.add_argument("--rerank", default=None, help="a finished run folder (e.g. outputs/FINAL_full_run): re-rank its saved "
                                                   "judge scores with the current ranking rules, no AI calls")
    a = ap.parse_args(argv)

    if a.rerank:
        run = Path(a.rerank)
        src_file = run / "ranked" / "judged.json"
        if not src_file.exists():
            raise SystemExit("no ranked/judged.json in this run (made before the fix): re-run the judge step")
        judged = json.loads(src_file.read_text(encoding="utf-8"))
        for p in judged:                                    # start clean: previous rank output fields
            for k in ("rank", "rank_reason", "alternatives", "audience_note"):
                p.pop(k, None)
        all_promos = load_promos([run / "finder", run / "pairs"])
        dropped = [p for p in all_promos if p.get("status") not in ("ok", "fixed")]
        from src.parser import parse_folder as _pf
        from src import safety
        _alias = None if a.aliases.lower() == "none" else Path(a.aliases)
        _blocks = {b.id: b for e in _pf(Path(a.input), _alias) for b in e.blocks}
        judged = safety.apply(judged, _blocks)            # code-found graphic/sexual flags (Step 8c fix)
        from src import rules as _rules
        judged = _rules.recheck(judged)                    # apply rules added after the run (e.g. max gap)
        dropped += [p for p in judged if p.get("status") == "failed"]
        ranked = rank(judged)
        write_report(ranked, dropped, run / "ranked", judged)
        alts = sum(len(p.get("alternatives", [])) for p in ranked)
        moved = sum(1 for p in ranked if p.get("audience_note"))
        summ_file = run / "run_summary.json"          # keep run_summary.json consistent with the new report
        if summ_file.exists():
            import datetime as _dt
            summ = json.loads(summ_file.read_text(encoding="utf-8"))
            summ.update(ranked=len(ranked), grouped_as_alternatives=alts, not_ranked=len(dropped),
                        new_viewer_promos=sum(p.get("audience") == "new_viewers" for p in ranked),
                        last_reranked=_dt.datetime.now().isoformat(timespec="seconds"),
                        rerank_note="ranking re-applied with later code rules (safety flags, max 45s gap, "
                                    "why-it-works rewrite); AI scores unchanged")
            summ_file.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"re-ranked {len(judged)} judged promos -> {len(ranked)} ranked ({alts} grouped as alternative cuts, "
              f"{moved} spoiler(s) moved to retargeting)  ->  {run / 'ranked' / 'ranked.md'}")
        return

    alias = None if a.aliases.lower() == "none" else Path(a.aliases)
    blocks = {b.id: b for e in parse_folder(Path(a.input), alias) for b in e.blocks}
    promos = load_promos([Path(d) for d in a.promos])
    usable = [p for p in promos if p.get("status") in ("ok", "fixed")]
    from src import safety
    judged = safety.apply(judge(usable, blocks, a.profile), blocks)
    ranked = rank(judged)
    dropped = [p for p in promos if p.get("status") not in ("ok", "fixed")] + [p for p in judged if "score" not in p.get("judge", {})]
    write_report(ranked, dropped, Path(a.out), judged)

    print(f"{len(promos)} promos loaded, {len(usable)} passed the rules, {len(ranked)} ranked  ->  {a.out}/ranked.md\n")
    for p in ranked:
        print(p["rank_reason"], "\n")
    print(f"Usage: {USAGE}")


if __name__ == "__main__":
    main()
