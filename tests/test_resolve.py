"""Tests for the core design rule: the AI cites block IDs, CODE turns them into times.
Uses the made-up screenplay from test_parser (no Kuku content, no AI calls). Run: python -m tests.test_resolve"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.finder import resolve  # noqa: E402
from src.show_summary import show_id  # noqa: E402
from tests.test_parser import FAKE_EP, _parse_text  # noqa: E402

EP = _parse_text(FAKE_EP)   # blocks: E07-B001 camera 0-5, B002 MAYA 1-3.5, B003 TOM 4-9.9, B004 STRANGER 10.5-12, B005 sfx 12-13


def test_times_come_from_blocks_not_the_ai():
    # the AI also sends a made-up "start": 99 - code must ignore it and use the block's real time
    raw = {"segments": [{"start_id": "E07-B002", "end_id": "E07-B003", "start": 99}],
           "hook_id": "E07-B002", "cliffhanger_id": "E07-B003"}
    p = resolve(EP, raw)
    s = p["segments"][0]
    assert (s["start"], s["end"], s["start_ts"], s["end_ts"]) == (1.0, 9.9, "00:01.00", "00:09.90")
    assert p["duration"] == 8.9 and p["hook_delay_s"] == 0.0 and p["problems"] == []


def test_invented_id_is_reported_not_crashed():
    p = resolve(EP, {"segments": [{"start_id": "E07-B999", "end_id": "E07-B003"}],
                     "hook_id": "E07-B002", "cliffhanger_id": "E07-B003"})
    assert p["segments"] == [] and any("unknown block id" in x for x in p["problems"])


def test_hook_delay_measured_by_code():
    p = resolve(EP, {"segments": [{"start_id": "E07-B002", "end_id": "E07-B005"}],
                     "hook_id": "E07-B004", "cliffhanger_id": "E07-B005"})
    assert p["hook_delay_s"] == 9.5          # hook (10.5s) - start (1.0s)


def test_each_show_gets_its_own_summary_id():
    other = _parse_text(FAKE_EP.replace("A made-up test episode.", "A different show."))
    assert show_id([EP]) != show_id([other]) and show_id([EP]) == show_id([EP])


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
