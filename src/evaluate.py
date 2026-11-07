"""Step 8a: score a finished run against the 13 reference promos (test_data/). Code only, no AI.

For each reference promo:
  - SAME MOMENT: some tool clip covers >= 50% of the reference clip's footage
  - EXACT:       that clip's every start/end is within +/- tolerance (5s) of the reference's, segment by segment
  - where that clip ranked (ranked clips AND their alternative cuts count; unranked candidates are reported too)

The tool NEVER sees test_data/; this runs only after a run has finished.

Usage:
    python -m src.evaluate --run outputs/FINAL_full_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.judge import load_promos
from src import paths

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "test_data" / "reference_promos.json"
SAME_MOMENT = 0.50


def mmss(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def ref_segments(p: dict) -> list[dict]:
    return [{"episode": s["episode"], "start": mmss(s["start_exact"]), "end": mmss(s["end_exact"])}
            for s in p["segments"]]


def covered(ref: list[dict], tool: list[dict]) -> float:
    """Seconds of the reference footage that the tool clip also contains."""
    return sum(max(0.0, min(r["end"], t["end"]) - max(r["start"], t["start"]))
               for r in ref for t in tool if r["episode"] == t["episode"])


def exact(ref: list[dict], tool: list[dict], tol: float) -> tuple[bool, list[float]]:
    """Each reference segment must have a tool segment in the same episode with start and end within tol."""
    diffs, ok = [], True
    for r in ref:
        best = min((max(abs(r["start"] - t["start"]), abs(r["end"] - t["end"])) for t in tool
                    if t["episode"] == r["episode"]), default=None)
        diffs.append(best if best is not None else float("inf"))
        ok &= best is not None and best <= tol
    return ok, diffs


def candidates(run: Path) -> list[dict]:
    """Every clip the tool produced, labelled with where it ended up (rank N / alternative of N / unranked)."""
    ranked = json.loads((run / "ranked" / "ranked.json").read_text(encoding="utf-8"))
    out = []
    for p in ranked:
        out.append({"segments": p["segments"], "where": f"rank {p['rank']}", "rank": p["rank"],
                    "audience": p.get("audience"), "score": p["judge"]["score"]})
        for a in p.get("alternatives", []):
            out.append({"segments": a["segments"], "where": f"alternative cut of rank {p['rank']}",
                        "rank": p["rank"], "audience": p.get("audience"), "score": a["score"]})
    seen = {json.dumps(c["segments"], sort_keys=True) for c in out}
    for p in load_promos([run / "finder", run / "pairs"]):            # failed / dropped candidates too
        key = json.dumps(p.get("segments", []), sort_keys=True)
        if p.get("segments") and key not in seen:
            out.append({"segments": p["segments"], "where": f"not ranked ({p.get('status')})", "rank": None,
                        "audience": p.get("audience"), "score": None})
    return out


def evaluate(run: Path, ref_path: Path = REF) -> dict:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    tol = ref.get("tolerance_seconds", 5)
    cands = candidates(run)
    rows = []
    for rp in ref["promos"]:
        rs = ref_segments(rp)
        dur = sum(s["end"] - s["start"] for s in rs)
        best = max(cands, key=lambda c: covered(rs, c["segments"]), default=None)
        cov = covered(rs, best["segments"]) / dur if best else 0.0
        same = cov >= SAME_MOMENT
        ex, diffs = exact(rs, best["segments"], tol) if best and same else (False, [])
        rows.append({"ref_rank": rp["rank"], "episodes": sorted({s["episode"] for s in rs}),
                     "hook": rp["hook"][:50], "audience": rp["audience"],
                     "same_moment": same, "exact": ex, "coverage": round(cov, 2),
                     "max_cut_diff_s": round(max(diffs), 1) if diffs and max(diffs) != float("inf") else None,
                     "tool_where": best["where"] if same else "-", "tool_rank": best["rank"] if same else None,
                     "tool_clip": " + ".join(f"Ep{s['episode']} {s['start']:.1f}-{s['end']:.1f}"
                                             for s in best["segments"]) if same else "-"})
    n = len(rows)
    summary = {
        "references": n,
        "same_moment": sum(r["same_moment"] for r in rows),
        "exact_within_tol": sum(r["exact"] for r in rows),
        "found_in_editor_list": sum(1 for r in rows if r["same_moment"] and not r["tool_where"].startswith("not ranked")),
        "found_in_top10": sum(1 for r in rows if r["tool_rank"] and r["tool_rank"] <= 10),
        "found_in_top20": sum(1 for r in rows if r["tool_rank"] and r["tool_rank"] <= 20),
        "new_viewer_refs_found": sum(r["same_moment"] for r in rows if r["audience"] == "new_viewers"),
        "new_viewer_refs": sum(1 for r in rows if r["audience"] == "new_viewers"),
        "tolerance_s": tol, "same_moment_threshold": SAME_MOMENT,
    }
    return {"summary": summary, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(paths.FINAL_RUN))
    a = ap.parse_args(argv)
    run = Path(a.run)
    res = evaluate(run)
    (run / "evaluation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    s = res["summary"]
    print(f"In the editor's list (ranked + alternatives): {s['found_in_editor_list']}/{s['references']}")
    print(f"Same moment found: {s['same_moment']}/{s['references']}   exact (±{s['tolerance_s']}s): "
          f"{s['exact_within_tol']}/{s['references']}   in top 10: {s['found_in_top10']}   in top 20: {s['found_in_top20']}")
    print(f"New-viewer refs found: {s['new_viewer_refs_found']}/{s['new_viewer_refs']}\n")
    print(f"{'ref':>3} {'eps':8} {'found':5} {'exact':5} {'cover':>5} {'maxdiff':>7}  where in tool output")
    for r in res["rows"]:
        print(f"#{r['ref_rank']:<2} {str(r['episodes']):8} {'yes' if r['same_moment'] else 'NO':5} "
              f"{'yes' if r['exact'] else '-':5} {r['coverage']:>5.0%} {str(r['max_cut_diff_s'] or '-'):>7}  "
              f"{r['tool_where']}  {r['hook']}")
    print(f"\nSaved {run / 'evaluation.json'}")


if __name__ == "__main__":
    main()
