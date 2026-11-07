"""Screenplay parser: turns Kuku microdrama screenplay .txt files into clean, timed blocks.

Design rule: timestamps ALWAYS come from this parser, never from the LLM.
The LLM only ever refers to block IDs (e.g. "E01-B023"); code maps IDs back to times.

Usage:
    python -m src.parser --input "<folder of episode .txt files>" --out outputs/archive/step2_parsed_blocks
    python -m src.parser --input "<new show>" --aliases none      # new show: no name list needed

The character alias list (config/character_aliases.json) is OPTIONAL and show-specific.
Without it, every speaker keeps the label written in the script (e.g. "LADY IN GOLD GOWN"),
which is enough for finding promos. The list only merges labels for the same person; for a new
show it can be drafted by the AI from the speaker list and checked by a human in a few minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------- regexes (all case-insensitive: Episode 6 uses UPPERCASE field names) ----------
TS = r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?"
RE_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(.+)$", re.I | re.M)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.+)$", re.I | re.M)
RE_CONTINUITY = re.compile(r"^CONTINUITY NOTES:\s*(.+)$", re.I | re.M)
RE_SCENE = re.compile(
    r"^SCENE\s+(\d+)\s*\n\s*Scene heading:\s*(.+?)\s*\n\s*Scene start:\s*" + TS
    + r"\s*\n\s*Scene end:\s*" + TS,
    re.I | re.M,
)
RE_BLOCK_HEAD = re.compile(r"^\*\*(.+?)\*\*\s*$", re.M)
RE_BLOCK_TIME = re.compile(r"^\s*Start:\s*\(?\s*" + TS + r"\s*\)?\s*\|\s*End:\s*\(?\s*" + TS + r"\s*\)?\s*$", re.I)
RE_FILE_EP = re.compile(r"episode-(\d+)", re.I)
RE_SCENE_HEADER_LINE = re.compile(r"^(SCENE\s+\d+\s*$|Scene heading:|Scene start:|Scene end:|Actions and dialogue:)", re.I)

# Parenthetical markers that are delivery/position notes, not emotions.
POSITION_MARKERS = re.compile(r"^(o\.?s\.?|v\.?o\.?|off[- ]?screen|cont'?d|continu\w*|o\.s\. ?/ ?continues)", re.I)
NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def to_seconds(m: tuple) -> float:
    mins, secs, frac = m
    frac = frac or "0"
    return round(int(mins) * 60 + int(secs) + int(frac) / (10 ** len(frac)), 3)


def fmt_ts(sec: float) -> str:
    """Seconds -> 'MM:SS.cc' (same format as the screenplay)."""
    m, s = divmod(round(sec, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Block:
    id: str
    episode: int
    scene: int
    kind: str              # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""      # raw label as written, e.g. "LADY IN GOLD GOWN"
    character: str = ""    # canonical name after alias mapping, e.g. "ANNA"
    emotion: str = ""      # e.g. "Gloating, Smug"
    voice: str = ""        # "" | "V.O." | "O.S."

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    episode: int
    source_file: str
    logline: str
    continuity: str
    scenes: list[Scene] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes] + [b.end for b in self.blocks]
        return max(ends) if ends else 0.0

    def shot_cuts(self) -> list[float]:
        """Clean cut points = start/end of every CAMERA shot. Promos should start/end on these."""
        cuts = {0.0, self.duration}
        for b in self.blocks:
            if b.kind == "camera":
                cuts.update((b.start, b.end))
        return sorted(cuts)


# ---------- speaker label parsing ----------
def parse_speaker(label: str) -> tuple[str, str, str, str]:
    """Split a speaker label into (name, emotion, voice, identity).
    'THE BEARDED STRANGER (O.S.) (Deep, Resonant):' -> ('THE BEARDED STRANGER', 'Deep, Resonant', 'O.S.', '')
    'STRANGER (Leo) (Cold, Calculating):'  -> ('STRANGER', 'Cold, Calculating', '', 'LEO')
    General rule (works for any show): if a label has 2+ non-position brackets and the FIRST is a
    single capitalised word, that word is an identity hint (the character's real name)."""
    label = label.strip().rstrip(":").strip()
    parens = [p.strip() for p in re.findall(r"\(([^)]*)\)", label)]
    name = re.sub(r"\s*\([^)]*\)", "", label).strip()
    voice = ""
    notes = []
    for p in parens:
        if POSITION_MARKERS.match(p):
            voice = "V.O." if re.match(r"v\.?o", p, re.I) else "O.S."
        else:
            notes.append(p)
    identity = ""
    if len(notes) >= 2 and re.fullmatch(r"[A-Z][a-z]+", notes[0]):
        identity = notes.pop(0).upper()
    return name.upper(), ", ".join(notes), voice, identity


def load_aliases(path: Path | None) -> list[dict]:
    """Alias rules: [{"label": "WOMAN IN RED", "character": "ANNA", "episodes": [1, 5]}, ...]
    'episodes' is optional (inclusive range). Needed because the same label can mean
    different people in different episodes (e.g. one label = the father early on, the villain later)."""
    if not path or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(label: str, episode: int, aliases: list[dict]) -> str:
    for rule in aliases:
        if rule["label"].upper() == label:
            rng = rule.get("episodes")
            if not rng or rng[0] <= episode <= rng[1]:
                return rule["character"]
    return label


# ---------- main parse ----------
def parse_episode(path: Path, aliases: list[dict] | None = None) -> Episode:
    aliases = aliases or []
    raw = path.read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n")

    m = RE_FILE_EP.search(path.name)
    ep_num = int(m.group(1)) if m else None
    head = RE_EP_NUM.search(text)
    if head:
        n = re.search(r"\d+", head.group(1))
        if n and ep_num is not None and int(n.group()) != ep_num:
            pass  # filename prefix wins; recorded below
        ep_num = ep_num if ep_num is not None else (int(n.group()) if n else 0)

    def grab(rx):
        mm = rx.search(text)
        return mm.group(1).strip() if mm else ""

    ep = Episode(episode=ep_num, source_file=path.name,
                 logline=grab(RE_LOGLINE), continuity=grab(RE_CONTINUITY))

    for sm in RE_SCENE.finditer(text):
        ep.scenes.append(Scene(number=int(sm.group(1)), heading=sm.group(2).strip(),
                               start=to_seconds(sm.group(3, 4, 5)), end=to_seconds(sm.group(6, 7, 8))))
    if not ep.scenes:
        ep.issues.append("no scene headers parsed")

    heads = list(RE_BLOCK_HEAD.finditer(text))
    for i, hm in enumerate(heads):
        seg_end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[hm.end():seg_end]
        body = body.split("--- SCRIPT END ---")[0]
        lines = [ln for ln in body.strip().split("\n")]
        if not lines or not RE_BLOCK_TIME.match(lines[0]):
            ep.issues.append(f"block '{hm.group(1)}' has no timestamp line")
            continue
        tm = RE_BLOCK_TIME.match(lines[0])
        start, end = to_seconds(tm.group(1, 2, 3)), to_seconds(tm.group(4, 5, 6))
        # Block text ends where the next scene header begins (bug fix: previously the
        # header lines "Scene heading/start/end", "Actions and dialogue" leaked into the text).
        kept = []
        for ln in lines[1:]:
            s = ln.strip()
            if RE_SCENE_HEADER_LINE.match(s):
                break
            if s:
                kept.append(s)
        content = " ".join(kept)

        label = hm.group(1).strip().rstrip(":").strip()
        kind = NON_DIALOGUE.get(label.upper())
        scene_no = next((s.number for s in ep.scenes if s.start <= start < s.end), ep.scenes[-1].number if ep.scenes else 1)
        blk = Block(id=f"E{ep_num:02d}-B{len(ep.blocks) + 1:03d}", episode=ep_num, scene=scene_no,
                    kind=kind or "dialogue", start=start, end=end, text=content.strip('"').strip())
        if not kind:
            blk.speaker, blk.emotion, blk.voice, identity = parse_speaker(label)
            # An identity written in the script itself wins over the alias list.
            blk.character = identity or canonical(blk.speaker, ep_num, aliases)
        if end < start:
            ep.issues.append(f"{blk.id}: end before start")
        ep.blocks.append(blk)

    ep.blocks.sort(key=lambda b: (b.start, b.end))
    return ep


def parse_folder(folder: Path, aliases_path: Path | None = None) -> list[Episode]:
    aliases = load_aliases(aliases_path)
    files = sorted(folder.glob("*.txt"), key=lambda p: (int(RE_FILE_EP.search(p.name).group(1)) if RE_FILE_EP.search(p.name) else 10**6, p.name))
    return [parse_episode(f, aliases) for f in files]


# ---------- output helpers ----------
def episode_to_prompt_text(ep: Episode, include_camera: bool = False) -> str:
    """Compact text for the LLM: one line per block, tagged with its ID (the LLM must cite IDs, not times)."""
    out = [f"EPISODE {ep.episode}", f"LOGLINE: {ep.logline}", f"CONTINUITY: {ep.continuity}", ""]
    for b in ep.blocks:
        if b.kind == "camera" and not include_camera:
            continue
        t = f"[{b.id} {fmt_ts(b.start)}-{fmt_ts(b.end)}]"
        if b.kind == "dialogue":
            who = b.character + (f" {b.voice}" if b.voice else "")
            out.append(f"{t} {who} ({b.emotion}): {b.text}")
        else:
            out.append(f"{t} {b.kind.upper()}: {b.text}")
    return "\n".join(out)


def write_outputs(episodes: list[Episode], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "episode", "scene", "kind", "start", "end", "duration", "start_ts", "end_ts",
            "speaker", "character", "emotion", "voice", "text"]
    with open(out_dir / "blocks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ep in episodes:
            for b in ep.blocks:
                row = asdict(b) | {"duration": b.duration, "start_ts": fmt_ts(b.start), "end_ts": fmt_ts(b.end)}
                w.writerow({k: row[k] for k in cols})
    with open(out_dir / "episodes.json", "w", encoding="utf-8") as f:
        json.dump([{
            "episode": ep.episode, "source_file": ep.source_file, "duration": ep.duration,
            "logline": ep.logline, "continuity": ep.continuity,
            "scenes": [asdict(s) for s in ep.scenes], "n_blocks": len(ep.blocks),
            "issues": ep.issues,
        } for ep in episodes], f, indent=2)


def summary(episodes: list[Episode]) -> str:
    kinds: dict[str, int] = {}
    for ep in episodes:
        for b in ep.blocks:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
    durs = [ep.duration for ep in episodes]
    issues = [f"Ep {ep.episode}: {i}" for ep in episodes for i in ep.issues]
    lines = [
        f"Episodes parsed: {len(episodes)}",
        f"Blocks: {sum(kinds.values())} " + str(kinds),
        f"Episode length: min {min(durs):.1f}s, max {max(durs):.1f}s, avg {sum(durs)/len(durs):.1f}s, total {sum(durs)/60:.1f} min",
        f"Issues: {len(issues)}",
    ] + ["  " + i for i in issues]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Parse Kuku screenplay .txt files into timed blocks.")
    ap.add_argument("--input", required=True, help="Folder with episode .txt files")
    ap.add_argument("--out", default="outputs/archive/step2_parsed_blocks", help="Output folder")
    ap.add_argument("--aliases", default="config/character_aliases.json",
                    help="Optional, show-specific character alias rules; 'none' to skip")
    a = ap.parse_args(argv)
    alias_path = None if a.aliases.lower() == "none" else Path(a.aliases)
    episodes = parse_folder(Path(a.input), alias_path)
    if not episodes:
        sys.exit(f"No .txt files found in {a.input}")
    write_outputs(episodes, Path(a.out))
    if alias_path and alias_path.exists():
        print(f"Character names: merged using {alias_path}")
    else:
        print("Character names: no alias list, using speaker labels as written (fine for a new show)")
    print(summary(episodes))
    print(f"\nWrote {a.out}/blocks.csv and {a.out}/episodes.json")


if __name__ == "__main__":
    main()
