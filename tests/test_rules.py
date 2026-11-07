"""Rule-check tests (no AI, no Kuku data). Run: python -m tests.test_rules"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rules import check, merge_touching, snap_to_cuts  # noqa: E402


def promo(segs, hook_delay=0.0):
    s = [{"episode": e, "start": a, "end": b, "start_ts": str(a), "end_ts": str(b), "seconds": b - a,
          "start_id": "x", "end_id": "y"} for e, a, b in segs]
    return {"segments": s, "duration": sum(x["seconds"] for x in s), "hook_delay_s": hook_delay, "problems": []}


def test_good_promo_passes():
    assert check(promo([(1, 10, 50)])) == []


def test_too_long_and_too_short():
    assert any("too long" in e for e in check(promo([(1, 0, 68.7)])))
    assert any("too short" in e for e in check(promo([(1, 0, 12)])))


def test_hook_too_late_or_missing():
    assert any("hook lands" in e for e in check(promo([(1, 0, 40)], hook_delay=5.3)))
    assert any("not inside" in e for e in check(promo([(1, 0, 40)], hook_delay=None)))


def test_overlap_and_order():
    assert any("overlap" in e for e in check(promo([(1, 0, 20), (1, 15, 30)])))
    assert any("story order" in e for e in check(promo([(2, 0, 20), (1, 0, 20)])))


def test_cross_episode_ok():
    assert check(promo([(41, 80, 93), (42, 1, 29)])) == []


def test_snap_moves_start_back_and_end_forward():
    p = promo([(1, 10.3, 45.6)], hook_delay=0.5)
    q = snap_to_cuts(p, {1: [0.0, 9.8, 12.0, 45.0, 46.2, 90.0]})
    s = q["segments"][0]
    assert q["snapped"] and (s["start"], s["end"]) == (9.8, 46.2)
    assert q["hook_delay_s"] == 1.0            # start moved 0.5s earlier, so the hook is 0.5s later


def test_snap_kept_off_when_it_would_break_a_rule():
    p = promo([(1, 0.0, 59.5)])
    q = snap_to_cuts(p, {1: [0.0, 61.0]})      # end would move to 61s (> 60s)
    assert q["snapped"] is False and q["segments"][0]["end"] == 59.5


def test_merge_touching_pieces():
    q = merge_touching(promo([(1, 39.0, 60.1), (1, 60.1, 79.0)]))          # Ep 1 case: pieces touch
    assert q["merged"] == 1 and len(q["segments"]) == 1 and q["duration"] == 40.0


def test_no_merge_for_real_gap_or_other_episode():
    assert merge_touching(promo([(1, 0, 14), (1, 44, 80)]))["merged"] == 0    # 30s skipped on purpose
    assert merge_touching(promo([(41, 80, 93), (42, 0, 29)]))["merged"] == 0  # cross-episode stays 2 pieces


def test_max_gap_between_pieces():
    assert any("apart" in e for e in check(promo([(1, 0, 15), (1, 67, 80)])))      # 52s jump: collage
    assert not any("apart" in e for e in check(promo([(3, 58, 86), (3, 125.6, 146)])))  # ref #3: 39.6s is OK


def test_safety_flags_weighted():
    from src.parser import Block
    from src.safety import content_flags
    b = {"a": Block(id="a", episode=1, scene=1, kind="action", start=0, end=5,
                    text="Her bloodied hands, deep gashes in the flesh."),
         "b": Block(id="b", episode=2, scene=1, kind="action", start=0, end=5,
                    text="The Duchess lies in bed, breathing with passion for life.")}
    assert "graphic" in content_flags({"segments": [{"episode": 1, "start": 0, "end": 5}]}, b)
    assert content_flags({"segments": [{"episode": 2, "start": 0, "end": 5}]}, b) == {}   # weak words only


def _blocks():
    from src.parser import Block
    rows = [("E01-B001", "dialogue", 0.0, 2.0, "You lied to me!"),
            ("E01-B002", "dialogue", 2.5, 4.0, "I may leave this chapel..."),
            ("E01-B003", "dialogue", 4.2, 6.0, "...but not ashamed."),
            ("E01-B004", "action", 6.0, 8.0, "She turns away."),
            ("E01-B005", "dialogue", 30.0, 32.0, "Maya..."),          # trailing off, NOT a split sentence
            ("E01-B006", "action", 50.0, 52.0, "The door opens.")]
    return {i: Block(id=i, episode=1, scene=1, kind=k, start=a, end=b, text=t) for i, k, a, b, t in rows}


def _seg(a, b):
    return {"segments": [{"episode": 1, "start": a, "end": b}]}


def test_dead_air_flagged_only_when_long():
    from src.rules import content_errors
    errs = content_errors(_seg(0.0, 52.0), _blocks())          # 8.0 -> 30.0 = 22s of nothing
    assert any("dead air: 22.0s" in e for e in errs)
    assert not any("dead air" in e for e in content_errors(_seg(0.0, 8.0), _blocks()))


def test_mid_sentence_start_and_end():
    from src.rules import content_errors
    assert any("starts mid-sentence" in e for e in content_errors(_seg(4.2, 8.0), _blocks()))
    assert any("ends mid-sentence" in e for e in content_errors(_seg(0.0, 4.0), _blocks()))
    assert not any("mid-sentence" in e for e in content_errors(_seg(0.0, 6.0), _blocks()))
    assert not any("ends mid-sentence" in e for e in content_errors(_seg(28.0, 32.0), _blocks()))  # "Maya..."


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
