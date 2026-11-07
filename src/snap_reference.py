"""Snap the approximate reference-promo times to exact screenplay cut points.

Start and end -> nearest CAMERA shot boundary (any shot start or end) within +/- window seconds.
Prints the first and last spoken/action lines of each snapped segment so a human can confirm
the hook and cliffhanger are still inside. Use --write to save the exact times into the file.

Usage:
    python -m src.snap_reference                 # dry run: print only
    python -m src.snap_reference --write         # save start_exact / end_exact into the JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.parser import fmt_ts


def mmss(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def nearest(target: float, options: pd.Series, window: float) -> float | None:
    near = options[(options - target).abs() <= window]
    return None if near.empty else float(near.iloc[(near - target).abs().argmin()])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="test_data/reference_promos.json")
    ap.add_argument("--blocks", default="outputs/archive/step2_parsed_blocks/blocks.csv")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    blocks = pd.read_csv(a.blocks)
    for p in ref["promos"]:
        total = 0.0
        print(f"\n#{p['rank']}")
        for seg in p["segments"]:
            ep = blocks[blocks.episode == seg["episode"]]
            cams = ep[ep.kind == "camera"]
            # a cut point is ANY camera shot boundary (start or end), same as Episode.shot_cuts()
            cuts = pd.concat([cams.start, cams.end, pd.Series([0.0])]).drop_duplicates()
            s0 = mmss(seg.get("start_approx", seg["start"]))
            e0 = mmss(seg.get("end_approx", seg["end"]))
            s = nearest(s0, cuts, a.window)
            e = nearest(e0, cuts, a.window)
            s = s if s is not None else s0
            e = e if e is not None else e0
            total += e - s
            inside = ep[(ep.kind != "camera") & (ep.start >= s) & (ep.end <= e)]
            first = inside.iloc[0] if len(inside) else None
            last = inside.iloc[-1] if len(inside) else None
            print(f"  Ep {seg['episode']}: {seg['start']}-{seg['end']}  ->  {fmt_ts(s)}-{fmt_ts(e)}  ({e - s:.1f}s)")
            if first is not None:
                print(f"    first: [{first.id} {first.start_ts}] {first.kind} {first.character if isinstance(first.character, str) else ''}: {str(first.text)[:70]}")
                print(f"    last : [{last.id} {last.end_ts}] {last.kind} {last.character if isinstance(last.character, str) else ''}: {str(last.text)[:70]}")
            seg["start_approx"] = seg.get("start_approx", seg["start"])
            seg["end_approx"] = seg.get("end_approx", seg["end"])
            seg["start_exact"], seg["end_exact"] = fmt_ts(s), fmt_ts(e)
        flag = "OK" if 20 <= total <= 60 else "OUT OF 20-60s"
        print(f"  total {total:.1f}s  {flag}")
        p["duration_exact"] = round(total, 2)

    if a.write:
        Path(a.ref).write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved exact times to {a.ref}")


if __name__ == "__main__":
    main()
