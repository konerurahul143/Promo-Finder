"""All output folder names in ONE place, so every command uses the same clear names.

outputs/
  FINAL_full_run/          the submitted full run of the original show (raw outputs for the brief)
  <show name>_promos/      every new show gets its own folder, named after its input folder (e.g. the interview)
  raw_ai_calls/            every AI call, unedited (prompt, reply, tokens, time)
  show_summaries/          one secrets list per show (editor reviews it)
  llm_cache/               cached AI answers (re-runs are free)
  archive/                 old runs, tests and development experiments, each with a descriptive name
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FINAL_RUN = OUTPUTS / "FINAL_full_run"
RAW_LOG = OUTPUTS / "raw_ai_calls" / "llm_calls.jsonl"
ARCHIVE = OUTPUTS / "archive"
SHOW_SUMMARIES = OUTPUTS / "show_summaries"
LLM_CACHE = OUTPUTS / "llm_cache"
def _env_value(name: str) -> str | None:
    """Read one value from the environment or the private .env file (never committed)."""
    import os
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


# Default show folder for commands run without --input. Set PROMO_INPUT in .env; the repo stays show-neutral.
DEFAULT_INPUT = Path(_env_value("PROMO_INPUT") or (ROOT / "data" / "show"))


def run_folder_for(input_folder: str | Path) -> Path:
    """New show -> outputs/<show folder name>_promos  (e.g. 'New Show S1' -> outputs/New_Show_S1_promos)."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", Path(input_folder).resolve().name).strip("_") or "show"
    return OUTPUTS / f"{name[:60]}_promos"
