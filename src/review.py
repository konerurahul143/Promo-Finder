"""HUMAN IN THE LOOP: save the editor's approve / reject decisions and report on them.

How the editor uses it:
  1. Open  <run>/ranked/promos.csv  in Excel / Google Sheets.
  2. In the column `editor_decision` write  approve  or  reject  (leave blank = not reviewed yet);
     optionally a short reason in `editor_note` (e.g. "spoils the Duke", "hook too slow").
  3. Save as CSV, then run:   python -m src.review --run outputs/FINAL_full_run

The decisions are saved to <run>/ranked/editor_decisions.json (keyed by clip, so they survive re-ranking and
are re-filled into promos.csv), and this prints the approval rate overall, in the top 10, and by audience,
plus the reasons for rejections: a direct measure of how useful the tool's ranking is to a real editor.
Next step (not built): feed approved promos back into the finder prompt as examples of this team's taste.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.judge import DECISIONS_FILE, load_decisions
from src import paths

ROOT = Path(__file__).resolve().parent.parent
VALID = {"approve", "reject"}


def read_csv(path: Path) -> dict:
    """{clip_id: {"decision", "note", "rank", "audience"}} for rows the editor filled in."""
    out, bad = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("editor_decision") or "").strip().lower()
            if not d:
                continue
            if d in ("approved", "yes", "y", "ok"):
                d = "approve"
            if d in ("rejected", "no", "n"):
                d = "reject"
            if d not in VALID:
                bad.append(f"rank {row.get('rank')}: '{row.get('editor_decision')}'")
                continue
            out[row["clip_id"]] = {"decision": d, "note": (row.get("editor_note") or "").strip(),
                                   "rank": int(row["rank"]), "audience": row.get("audience")}
    if bad:
        print("Ignored (write approve or reject):", ", ".join(bad))
    return out


def report(dec: dict, total: int) -> str:
    if not dec:
        return f"No decisions yet: 0 of {total} promos reviewed."
    appr = [d for d in dec.values() if d["decision"] == "approve"]
    top10 = [d for d in dec.values() if d["rank"] <= 10]
    lines = [f"Reviewed {len(dec)} of {total} promos: {len(appr)} approved ({len(appr) / len(dec):.0%})"]
    if top10:
        a10 = sum(d["decision"] == "approve" for d in top10)
        lines.append(f"  top 10: {a10}/{len(top10)} approved ({a10 / len(top10):.0%})")
    for aud in ("new_viewers", "retargeting"):
        g = [d for d in dec.values() if d["audience"] == aud]
        if g:
            lines.append(f"  {aud}: {sum(d['decision'] == 'approve' for d in g)}/{len(g)} approved")
    reasons = Counter(d["note"].lower() for d in dec.values() if d["decision"] == "reject" and d["note"])
    if reasons:
        lines.append("  rejection reasons: " + "; ".join(f"{r} ({n})" for r, n in reasons.most_common(8)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Save the editor's approve/reject decisions from promos.csv.")
    ap.add_argument("--run", default=str(paths.FINAL_RUN))
    a = ap.parse_args(argv)
    folder = Path(a.run) / "ranked"
    csv_path = folder / "promos.csv"
    if not csv_path.exists():
        raise SystemExit(f"No {csv_path}: run the pipeline first.")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        total = sum(1 for _ in csv.DictReader(f))
    decisions = {**load_decisions(folder), **read_csv(csv_path)}      # newest CSV edits win
    (folder / DECISIONS_FILE).write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report(decisions, total))
    print(f"Saved {folder / DECISIONS_FILE}")


if __name__ == "__main__":
    main()
