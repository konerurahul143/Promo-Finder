"""Ranking tests (no AI). Run: python -m tests.test_rank"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.judge import overlap_s, rank  # noqa: E402


def P(segs, score, audience="new_viewers", flags=None, hook=4):
    s = [{"episode": e, "start": a, "end": b, "start_ts": str(a), "end_ts": str(b)} for e, a, b in segs]
    return {"segments": s, "duration": sum(b - a for _, a, b in segs), "status": "ok", "audience": audience,
            "flags": flags or [], "judge": {"score": score, "hook": hook, "cliffhanger": 4, "standalone": 3,
                                             "first_3s": "", "verdict": ""}}


def test_spoiler_always_retargeting():
    r = rank([P([(1, 0, 40)], 4.5, "new_viewers", ["spoiler"]), P([(2, 0, 40)], 3.0)])
    assert [p["audience"] for p in r] == ["new_viewers", "retargeting"]   # spoiler moved, ranked after
    assert "moved to retargeting" in r[1]["audience_note"]


def test_overlapping_cuts_grouped_keep_best():
    a = P([(3, 58, 97)], 4.5)                            # 39s
    b = P([(3, 133, 146), (4, 1, 45)], 3.75)             # shares 0s with a
    c = P([(3, 70, 110)], 3.0)                           # shares 27s of a's 39s -> same moment
    r = rank([a, b, c])
    assert len(r) == 2 and r[0]["judge"]["score"] == 4.5
    assert len(r[0]["alternatives"]) == 1 and r[0]["alternatives"][0]["score"] == 3.0


def test_small_overlap_kept_separate():
    a, b = P([(1, 0, 40)], 4.0), P([(1, 35, 75)], 3.5)  # share 5s = 12.5% < 30%
    assert len(rank([a, b])) == 2 and overlap_s(a, b) == 5


def test_failed_promos_never_ranked():
    bad = {**P([(1, 0, 40)], 5.0), "status": "failed"}
    assert rank([bad, P([(2, 0, 40)], 2.0)])[0]["judge"]["score"] == 2.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f(); print(f"PASS  {n}")
        except AssertionError as e:
            bad += 1; print(f"FAIL  {n} {e}")
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
