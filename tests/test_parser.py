"""Parser tests. Run:  python -m tests.test_parser   (no extra libraries needed; pytest also works)

Part A uses a small MADE-UP screenplay (no Kuku content), so it is safe to publish.
Part B checks the real input pack, and is skipped automatically if that folder isn't present.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parser import parse_episode, parse_folder, parse_speaker  # noqa: E402

FAKE_EP = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A made-up test episode.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. KITCHEN – DAY
Scene start: 00:00.00
Scene end: 00:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Wide shot of a kitchen.

**MAYA (Angry, Loud):**
Start: (00:01.00) | End: (00:03.50)
"You lied to me!"

**TOM (O.S.) (Quiet):**
Start: (00:04.00) | End: (00:09.90)
"I had no choice."

SCENE 02
Scene heading: EXT. STREET – NIGHT
Scene start: 00:10.00
Scene end: 00:20.00

Actions and dialogue:

**STRANGER (Leo) (Cold, Calm):**
Start: (00:10.50) | End: (00:12.00)
"Get in the car."

**SOUND FX:**
Start: (00:12.00) | End: (00:13.00)
A door slams.

--- SCRIPT END ---
"""

# Same episode, but with UPPERCASE field names (like Episode 6 of the real pack).
FAKE_EP_UPPER = (FAKE_EP.replace("Scene heading:", "SCENE HEADING:").replace("Scene start:", "SCENE START:")
                 .replace("Scene end:", "SCENE END:").replace("Actions and dialogue:", "ACTIONS AND DIALOGUE:"))


def _parse_text(text: str, name: str = "episode-007-test.txt"):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        p.write_text(text, encoding="utf-8")
        return parse_episode(p)


# ---------------- Part A: made-up screenplay ----------------
def test_basic_structure():
    ep = _parse_text(FAKE_EP)
    assert ep.episode == 7
    assert len(ep.scenes) == 2
    assert [b.kind for b in ep.blocks] == ["camera", "dialogue", "dialogue", "dialogue", "sfx"]
    assert ep.blocks[1].start == 1.0 and ep.blocks[1].end == 3.5
    assert ep.blocks[1].text == "You lied to me!"
    assert ep.issues == []


def test_uppercase_fields_parse_the_same():
    a, b = _parse_text(FAKE_EP), _parse_text(FAKE_EP_UPPER)
    assert len(b.scenes) == 2
    assert [(x.kind, x.start, x.text) for x in a.blocks] == [(x.kind, x.start, x.text) for x in b.blocks]


def test_scene_header_does_not_leak_into_text():
    """Regression test for the bug found on 2026-11-06 (50 blocks had scene headers in their text)."""
    ep = _parse_text(FAKE_EP)
    for b in ep.blocks:
        assert "Scene heading" not in b.text and "SCENE" not in b.text, b.text
    assert ep.blocks[2].text == "I had no choice."


def test_speaker_parsing():
    assert parse_speaker("TOM (O.S.) (Quiet):") == ("TOM", "Quiet", "O.S.", "")
    assert parse_speaker("MAYA (V.O.):") == ("MAYA", "", "V.O.", "")
    # identity hint: first of 2+ brackets is a single capitalised word -> real name
    assert parse_speaker("STRANGER (Leo) (Cold, Calm):") == ("STRANGER", "Cold, Calm", "", "LEO")
    # a single bracket is always an emotion, even if it is one capitalised word
    assert parse_speaker("MAYA (Furious):") == ("MAYA", "Furious", "", "")


def test_identity_sets_character():
    ep = _parse_text(FAKE_EP)
    stranger = [b for b in ep.blocks if b.speaker == "STRANGER"][0]
    assert stranger.character == "LEO" and stranger.emotion == "Cold, Calm"


def test_shot_cuts_include_start_and_end():
    ep = _parse_text(FAKE_EP)
    cuts = ep.shot_cuts()
    assert 0.0 in cuts and 5.0 in cuts and ep.duration in cuts


# ---------------- Part B: real input pack (skipped if absent) ----------------
from src.paths import DEFAULT_INPUT as REAL  # set PROMO_INPUT in .env; skipped if absent


def test_real_pack():
    if not REAL.exists():
        print("  (skipped: real input pack not found)")
        return
    aliases = ROOT / "config" / "character_aliases.json"     # show-specific + confidential: not in the repo
    eps = parse_folder(REAL, aliases)
    assert len(eps) == 50
    assert sum(len(e.blocks) for e in eps) == 3727
    assert all(not e.issues for e in eps)
    by = {e.episode: e for e in eps}
    line = [b for b in by[1].blocks if b.id == "E01-B027"][0]   # no script text in this file (confidential)
    assert (line.kind, line.start, line.end) == ("dialogue", 39.1, 41.2)
    assert len(by[6].scenes) == 2                       # Ep 6 uses UPPERCASE field names
    assert not any("Scene heading" in b.text for e in eps for b in e.blocks)
    if aliases.exists():                                # name-merging checks only when the name list is present
        assert line.character != line.speaker           # a label was merged to a character name
        seen = {}                                        # some label must map to different people by episode
        for e in eps:
            for x in e.blocks:
                if x.kind == "dialogue":
                    seen.setdefault(x.speaker, set()).add(x.character)
        assert any(len(c) > 1 for c in seen.values())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {n}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
