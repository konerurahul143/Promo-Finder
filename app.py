"""Promo Finder: web app.

    streamlit run app.py

Upload a show's episode screenplays (.txt files or a .zip), click "Find promos", get ranked promos with
segments, hook, cliffhanger and why, plus downloads. No screenplay content is stored in the app itself:
uploads go to a temporary folder for the session.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SAVED_RUN = ROOT / "outputs" / "FINAL_full_run"            # exists only on the developer's machine (git-ignored)

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")


# ---------------- keys: pasted in the sidebar > Streamlit secrets > local .env ----------------
def set_keys(gemini: str, groq: str) -> None:
    from src.llm import load_env
    load_env()
    for name, pasted in (("GEMINI_API_KEY", gemini), ("GROQ_API_KEY", groq)):
        if pasted.strip():
            os.environ[name] = pasted.strip()
        elif not os.environ.get(name):
            try:
                if name in st.secrets:
                    os.environ[name] = st.secrets[name]
            except Exception:
                pass


# ---------------- the pipeline (same functions as `python -m src.run`) ----------------
def run_pipeline(folder: Path, max_eps: int, use_pairs: bool, status) -> tuple[list, list, dict, dict]:
    from src import finder, judge, pairs, safety, show_summary, why
    from src.llm import USAGE, LLMError
    from src.parser import parse_folder

    t0 = time.time()
    eps = parse_folder(folder, None)
    if not eps:
        raise ValueError("No episode .txt files found in the upload.")
    chosen = sorted(eps, key=lambda e: e.episode)[:max_eps]
    by_num = {e.episode: e for e in eps}
    status.write(f"Parsed **{len(eps)} episodes** ({sum(len(e.blocks) for e in eps)} script lines). "
                 f"Running on {len(chosen)}.")

    status.write("Reading the show's loglines to learn its secrets (for spoiler flags)…")
    summary = show_summary.get_for(eps)

    jobs = [("ep", e) for e in chosen]
    if use_pairs:
        jobs += [("pair", e) for e in chosen if e.episode + 1 in by_num and by_num[e.episode + 1] in chosen]
    bar = st.progress(0.0, text="Finding promos…")
    promos, errors = [], []
    for i, (kind, e) in enumerate(jobs, 1):
        label = f"Episode {e.episode}" if kind == "ep" else f"Episodes {e.episode}→{e.episode + 1}"
        bar.progress(i / (len(jobs) + 1), text=f"{label} ({i}/{len(jobs)})")
        try:
            if kind == "ep":
                promos += finder.find_in_episode(e, summary=summary)
            else:
                res = pairs.judge_pair(e, by_num[e.episode + 1], summary=summary)
                if res.get("use"):
                    promos.append(res)
        except LLMError as err:
            errors.append(f"{label}: {str(err)[:160]}")

    usable = [p for p in promos if p.get("status") in ("ok", "fixed")]
    bar.progress(len(jobs) / (len(jobs) + 1), text=f"Judging {len(usable)} promos with a second AI…")
    blocks = {b.id: b for e in eps for b in e.blocks}
    judged = safety.apply(judge.judge(usable, blocks), blocks) if usable else []
    if judged:
        judged, _ = why.rewrite(judged, blocks)          # 'why it works' about the viewer
    ranked = judge.rank(judged)
    bar.progress(1.0, text="Done")

    stats = {"episodes": len(chosen), "candidates": len(promos),
             "passed_rules": len(usable), "ranked": len(ranked), "errors": errors,
             "minutes": round((time.time() - t0) / 60, 1), "ai_calls": USAGE["calls"], "cached": USAGE["cached"]}
    dropped = [p for p in promos if p.get("status") not in ("ok", "fixed")]
    return ranked, dropped, stats, summary


# ---------------- display ----------------
def segs_text(p: dict) -> str:
    return "  →  ".join(f"Ep{s['episode']} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])


CHOICES = ["Not reviewed", "Approve", "Reject"]


def current_decisions(ranked: list, saved: dict) -> dict:
    """HUMAN IN THE LOOP: the editor's approve/reject choices (from the widgets, else previously saved ones),
    in the same format as `python -m src.review` (keyed by clip_id)."""
    from src import judge
    out = {}
    for p in ranked:
        cid = judge.clip_id(p)
        prev = saved.get(cid, {})
        choice = st.session_state.get(f"dec_{cid}", prev.get("decision", "").capitalize() or "Not reviewed")
        note = st.session_state.get(f"note_{cid}", prev.get("note", ""))
        if choice in ("Approve", "Reject"):
            out[cid] = {"decision": choice.lower(), "note": note.strip(), "rank": p["rank"], "audience": p.get("audience")}
    return out


def show_results(ranked: list, dropped: list, stats: dict | None, summary: dict | None,
                 decisions_dir: Path | None = None) -> None:
    from src import judge

    if stats:
        c = st.columns(5)
        c[0].metric("Episodes", stats["episodes"])
        c[1].metric("Candidates", stats["candidates"])
        c[2].metric("Passed the rules", stats["passed_rules"])
        c[3].metric("Ranked promos", stats["ranked"])
        c[4].metric("Time", f"{stats['minutes']} min")
        for e in stats.get("errors", []):
            st.warning(e)
    if not ranked:
        st.info("No promos passed all the checks for these episodes.")
        return

    # ---- the editor's review: approve / reject each promo below; counts, save and downloads here ----
    saved = judge.load_decisions(decisions_dir) if decisions_dir else {}
    decisions = current_decisions(ranked, saved)
    n_app = sum(d["decision"] == "approve" for d in decisions.values())
    st.subheader("Editor review")
    r = st.columns(4)
    r[0].metric("Reviewed", f"{len(decisions)} / {len(ranked)}")
    r[1].metric("Approved", n_app)
    r[2].metric("Rejected", len(decisions) - n_app)
    r[3].metric("Approval rate", f"{n_app / len(decisions):.0%}" if decisions else "–")
    reasons = [d["note"] for d in decisions.values() if d["decision"] == "reject" and d["note"]]
    if reasons:
        st.caption("Rejection reasons: " + "; ".join(reasons[:8]))

    out = Path(tempfile.mkdtemp())
    (out / judge.DECISIONS_FILE).write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    judge.write_report(ranked, dropped, out)                       # CSV now carries the editor's decisions
    approved = [p for p in ranked if decisions.get(judge.clip_id(p), {}).get("decision") == "approve"]
    if approved:
        judge._write_csv(approved, out / "approved_promos.csv")
    d1, d2, d3 = st.columns(3)
    d1.download_button(f"⬇ Approved promos only ({len(approved)})",
                       (out / "approved_promos.csv").read_bytes() if approved else b"",
                       "approved_promos.csv", "text/csv", disabled=not approved)
    d2.download_button("⬇ All promos + decisions (CSV)", (out / "promos.csv").read_bytes(), "promos.csv", "text/csv")
    d3.download_button("⬇ Full report (ranked.md)", (out / "ranked.md").read_bytes(), "ranked.md", "text/markdown")
    if decisions_dir is not None and st.button("💾 Save decisions"):
        (decisions_dir / judge.DECISIONS_FILE).write_text(json.dumps({**saved, **decisions}, indent=2),
                                                         encoding="utf-8")
        st.success(f"Saved {len(decisions)} decisions to {decisions_dir.name}/{judge.DECISIONS_FILE}")

    st.info("**Before posting:** check every ⚠ graphic / sexual flag, and skim the show's secrets list below: "
            "spoiler labels come from it, and it is written by AI.")
    if summary:
        with st.expander("The show's secrets (what the tool treats as spoilers): check once"):
            for t in summary.get("major_twists", []):
                st.markdown(f"- **Ep {t.get('first_revealed_in_episode')}:** {t.get('twist')}")

    new = [p for p in ranked if p.get("audience") == "new_viewers"]
    tabs = st.tabs([f"For new viewers ({len(new)})", f"Re-targeting / spoilers ({len(ranked) - len(new)})"])
    for tab, group in zip(tabs, (new, [p for p in ranked if p not in new])):
        with tab:
            for p in group:
                j = p["judge"]
                eps = sorted({s["episode"] for s in p["segments"]})
                flags = f"  ⚠ {', '.join(p['flags'])}" if p.get("flags") else ""
                title = (f"#{p['rank']} · Ep {' → '.join(map(str, eps))} · {p['duration']:.0f}s · "
                         f"score {j['score']:.2f}{flags}")
                with st.expander(title, expanded=p["rank"] <= 3):
                    st.markdown(f"**Segments (in order):** {segs_text(p)}")
                    st.markdown(f"**Hook** ({p.get('hook_at', '?')}): {p.get('hook', '')}  \n"
                                f"> *\"{(p.get('hook_text') or '')[:160]}\"*")
                    st.markdown(f"**Cliffhanger** ({p.get('cliffhanger_at', '?')}): {p.get('cliffhanger', '')}  \n"
                                f"> *\"{(p.get('cliffhanger_text') or '')[:160]}\"*")
                    st.markdown(f"**Why it works:** {p.get('why', '')}")
                    st.markdown(f"**Why rank #{p['rank']}:** hook {j['hook']}/5 · cliffhanger {j['cliffhanger']}/5 · "
                                f"makes sense alone {j['standalone']}/5. *Judge: {j.get('verdict', '')}*")
                    for flag, words in (p.get("safety_evidence") or {}).items():
                        st.warning(f"⚠ {flag}: found by code: {', '.join(words[:8])}")
                    if p.get("audience_note"):
                        st.caption(p["audience_note"])
                    for a in p.get("alternatives", []):
                        st.caption(f"Alternative cut of the same moment: {segs_text(a)} ({a['duration']:.0f}s, "
                                   f"score {a['score']:.2f})")
                    # HUMAN IN THE LOOP: the editor's decision on this promo
                    cid = judge.clip_id(p)
                    prev = saved.get(cid, {})
                    default = prev.get("decision", "").capitalize() or "Not reviewed"
                    dc, nc = st.columns([2, 3])
                    choice = dc.radio("Editor decision", CHOICES, index=CHOICES.index(default), horizontal=True,
                                      key=f"dec_{cid}")
                    nc.text_input("Reason (optional)", value=prev.get("note", ""), key=f"note_{cid}",
                                  placeholder="e.g. spoils the twist / hook too slow / great opener")
                    if choice == "Approve" and p.get("flags"):
                        st.caption(f"⚠ Approved with flags ({', '.join(p['flags'])}): check the platform's rules.")


# ---------------- page ----------------
st.title("🎬 Promo Finder")
st.caption("Finds the best 20–60 second promos in a microdrama's screenplays: hook in the first 3 seconds, "
           "cliffhanger ending, makes sense to a stranger. The AI picks the moments; code checks every rule and "
           "computes every timestamp; a second AI ranks them.")

with st.sidebar:
    st.header("Settings")
    modes = ["Find promos in uploaded episodes"]
    if (SAVED_RUN / "ranked" / "ranked.json").exists():
        modes.append("View the saved full run (no AI calls)")
    mode = st.radio("Mode", modes)
    st.subheader("API keys")
    st.caption("Free keys: [Gemini](https://aistudio.google.com/apikey) · [Groq](https://console.groq.com/keys). "
               "Used only for this session, never stored.")
    gemini = st.text_input("Gemini API key", type="password")
    groq = st.text_input("Groq API key", type="password")
    st.subheader("Run options")
    max_eps = st.slider("Max episodes", 1, 50, 5, help="A full 50-episode show takes ~20 min on free tiers.")
    use_pairs = st.checkbox("Also find cross-episode promos (end of N + start of N+1)", value=True)

if mode.startswith("View"):
    ranked = json.loads((SAVED_RUN / "ranked" / "ranked.json").read_text(encoding="utf-8"))
    rs = json.loads((SAVED_RUN / "run_summary.json").read_text(encoding="utf-8"))
    # the summary that belongs to THIS show (bug fix: picking the alphabetically last file could show another show's)
    from src import paths
    from src.parser import parse_folder
    from src.show_summary import SUMMARY_DIR, show_id
    summary = None
    if paths.DEFAULT_INPUT.exists():
        f = SUMMARY_DIR / f"{show_id(parse_folder(paths.DEFAULT_INPUT, None))}.json"
        summary = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    st.subheader(f"Saved full run: {rs.get('episodes_run')} episodes")
    show_results(ranked, [], {"episodes": rs.get("episodes_run"), "candidates": rs.get("candidates"),
                              "passed_rules": rs["status"].get("ok", 0) + rs["status"].get("fixed", 0),
                              "ranked": len(ranked), "minutes": round(rs.get("wall_time_s", 0) / 60, 1),
                              "errors": rs.get("errors", [])}, summary, decisions_dir=SAVED_RUN / "ranked")
else:
    files = st.file_uploader("Episode screenplays: the .txt files (episode-001-….txt, …) or one .zip",
                             type=["txt", "zip"], accept_multiple_files=True)
    local = st.text_input("…or, when running on your own computer, the path of the folder with the episode files",
                          placeholder=r"C:\path\to\show folder")
    local_dir = Path(local.strip().strip('"')) if local.strip() else None
    if local_dir is not None and not local_dir.is_dir():
        st.warning("That folder doesn't exist on this computer.")
        local_dir = None
    if st.button("Find promos", type="primary", disabled=not (files or local_dir)):
        set_keys(gemini, groq)
        if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("GROQ_API_KEY"):
            st.error("Add a Gemini and a Groq API key in the sidebar (both are free).")
            st.stop()
        tmp = Path(tempfile.mkdtemp())
        if local_dir is not None and not files:
            for src_file in local_dir.glob("*.txt"):
                (tmp / src_file.name).write_bytes(src_file.read_bytes())
        for f in files or []:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    for info in z.infolist():
                        name = Path(info.filename).name               # flatten folders; no path tricks
                        if name.lower().endswith(".txt") and not info.is_dir():
                            (tmp / name).write_bytes(z.read(info))
            else:
                (tmp / Path(f.name).name).write_bytes(f.getvalue())
        with st.status("Working…", expanded=True) as status:
            try:
                ranked, dropped, stats, summary = run_pipeline(tmp, max_eps, use_pairs, status)
                status.update(label="Done", state="complete")
            except Exception as err:
                status.update(label="Failed", state="error")
                st.error(str(err))
                st.stop()
        st.session_state["result"] = (ranked, dropped, stats, summary)
    if "result" in st.session_state:
        show_results(*st.session_state["result"])
