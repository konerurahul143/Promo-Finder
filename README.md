# Promo Finder

Reads a microdrama's screenplays and finds the best **20–60 second promos** for TikTok / Reels / Shorts.
For each promo it gives the episode, the start and end of every segment (in order), the **hook**, the
**cliffhanger** and **one line on why it works**, then **ranks** them (post #1 first) with a reason.

**Design in one line:** the AI does the judgment (which moment hooks, where it's unresolved); code does
everything exact (timestamps, lengths, camera cuts, rules, ranking order). The AI never writes a timestamp:
it points to numbered script lines, and code turns them into exact times.

```
screenplays ──► parse (code) ──► show summary (AI, 1 call) ──► find promos per episode (AI)
            ──► cross-episode pairs (code builds, AI judges) ──► rule checks + repair loop (code ⇄ AI)
            ──► snap to camera cuts, safety flags (code) ──► judge, a different AI ──► rank + report (code)
```

## Quick start

Requirements: Python 3.10+ and two free API keys.

```bash
pip install -r requirements.txt
```

1. Get a free **Gemini** key at https://aistudio.google.com/apikey and a free **Groq** key at https://console.groq.com/keys
2. Copy `.env.example` to `.env` and paste both keys into `.env` (never into `.env.example`; `.env` is git-ignored).
3. Check the models answer:
   ```bash
   python -m src.llm
   python -m src.llm --profile groq_gptoss
   ```

## Web app

```bash
streamlit run app.py
```

Upload a show's episode `.txt` files (or one `.zip`), or on your own computer paste the folder path. Click
**Find promos** to get the ranked promos (segments, hook, cliffhanger, why, flags).

**The editor then reviews them in the app:** each promo card has **Approve / Reject** and a reason box; the top
shows how many are reviewed, approved and rejected, the approval rate and the rejection reasons. Download
**approved promos only** (for cutting the videos), or all promos with the decisions. On the saved run,
**Save decisions** writes them to the same file `python -m src.review` uses.
API keys: paste them in the sidebar, or set `GEMINI_API_KEY` / `GROQ_API_KEY` as Streamlit secrets or in `.env`.
Start with a few episodes: a full 50-episode show takes ~20 minutes on free tiers.

## Run it on a show (command line)

Put the show's episode files (`episode-001-....txt`, `episode-002-....txt`, …) in one folder, then:

```bash
python -m src.run --input "path/to/show folder" --aliases none
```

- Results go to **`outputs/<show folder name>_promos/`** (named after the show, so runs never mix). Use `--out` to choose another folder.
- `--aliases none`: no show-specific setup is needed. (`config/character_aliases.json` optionally merges speaker
  labels for the same person; it only applies to the original show.)
- A full 50-episode show takes about 20 minutes on free tiers. Re-runs are instant (AI answers are cached).
- If one episode fails (e.g. the model is overloaded), it's logged and the run continues.

### What you get (in `outputs/<show>_promos/`)

| File | What it is |
|---|---|
| `ranked/ranked.md` | **The editor's report.** One block per promo: episode, segments in order, hook, cliffhanger, why it works, why this rank, ⚠ flags, alternative cuts |
| `ranked/promos.csv` | The same, one row per promo, for Excel / Google Sheets |
| `run_summary.json` | Counts, rule pass rates, time, tokens, errors |
| `finder/`, `pairs/` | Every candidate promo, including failed ones and the reason |
| `outputs/raw_ai_calls/llm_calls.jsonl` | Every AI call, unedited (prompt, reply, tokens, time) |

## Human in the loop: the editor decides

The tool replaces *watching every episode*, not the editor's final call. The editor:

1. **Reviews the AI-written list of the show's secrets** once per show (~1 min); it decides what counts as a spoiler:
   ```bash
   python -m src.show_summary --input "path/to/show folder" --review
   ```
2. **Checks every ⚠ graphic / sexual flag** (shown with the words that triggered it).
3. **Approves or rejects promos:** in `ranked/promos.csv` write `approve` or `reject` in `editor_decision`
   (optional reason in `editor_note`), save, then:
   ```bash
   python -m src.review --run "outputs/<show>_promos"
   ```
   Decisions are saved (and survive re-ranking), and the approval rate (overall, top 10, by audience) and
   rejection reasons are reported: a direct measure of how useful the ranking is. Next step: feed approved
   promos back into the finder as examples of the team's taste.

## Swap the AI model

Edit one line in `config/model.yaml`: `active:` (the finder) or `judge:`. Any OpenAI-compatible API works
(Gemini, Groq, OpenAI, OpenRouter, a local Ollama, or an API you provide), plus Anthropic. To add one, copy a
profile (4 lines: provider, base_url, model, api_key_env) and add its key to `.env`. You can also swap for one
run without editing anything:

```bash
PROMO_MODEL=claude python -m src.run --input "path/to/show folder" --aliases none      # swap the finder
PROMO_JUDGE=groq_qwen python -m src.run --input "path/to/show folder" --aliases none   # swap the judge
```

Keep the finder and the judge as **different** models, so no model grades its own picks.

Model names change: list the live ones with the provider's `/models` endpoint if a name stops working.

## Quality checks (need the original show + `test_data/`)

```bash
python -m src.evaluate --run outputs/FINAL_full_run    # vs the 13 reference promos
python -m src.baselines --run outputs/FINAL_full_run   # vs random clips and episode endings (blind judge)
python -m src.plain_baseline                      # vs pasting the raw script into the same model
python -m src.judge --rerank outputs/FINAL_full_run    # re-rank saved scores with the current rules (no AI calls)
```

## Tests

No API key or show data needed (the real-data test is skipped if the show folder is absent):

```bash
python -m tests.test_parser
python -m tests.test_rules
python -m tests.test_resolve
python -m tests.test_rank
```

## Project layout

| Path | Role |
|---|---|
| `src/parser.py` | Screenplay `.txt` → timed blocks with IDs (handles format variations) |
| `src/show_summary.py` | One AI call per show: premise + secrets, for spoiler judgment (`--review` for the editor) |
| `src/finder.py` | One AI call per episode → candidate promos as block IDs (prompt versions kept for comparison) |
| `src/pairs.py` | Cross-episode promos: end of Ep N + start of Ep N+1 |
| `src/rules.py` | Hard rules (20–60s, hook ≤3s, no dead air, no mid-sentence cuts, max gaps) + repair loop + snapping |
| `src/safety.py` | Graphic / sexual flags found by code |
| `src/judge.py` | Independent judge model scores hook / cliffhanger / standalone; ranking + reports |
| `src/run.py` | Runs the whole pipeline on a show |
| `src/llm.py` | One `complete()` for every model: retries, fallback, cache, raw log, token + cost counts |
| `config/model.yaml` | Which models, settings, prices |

## Known limits

Reads text only (visual hooks live in camera descriptions); can end a clip before the payoff; spoiler labels
depend on the AI-written secrets list (an editor should review it); free tiers cap volume (daily quotas).
Details and measured numbers are in the solution doc.

## Output folders

| Folder | What it is |
|---|---|
| `outputs/FINAL_full_run/` | The submitted full run of the original show (50 episodes) |
| `outputs/<show>_promos/` | Each new show's results (e.g. the interview), named after its input folder |
| `outputs/raw_ai_calls/` | Every AI call, unedited |
| `outputs/show_summaries/` | One secrets list per show (editor reviews it) |
| `outputs/llm_cache/` | Cached AI answers: re-runs are instant and free |
| `outputs/archive/` | Older runs, tests and development experiments, each with a descriptive name |

## Confidentiality

The screenplays are the client's confidential content. They and everything derived from them (`outputs/`,
`test_data/`, `.txt` files, `.env`) are excluded by `.gitignore` and must never be committed.
