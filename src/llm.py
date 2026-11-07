"""One function the whole tool uses to talk to an AI model: complete().

Swapping models = changing `active` in config/model.yaml (or passing profile="...").
No SDKs needed: plain HTTPS, so any OpenAI-compatible API (Gemini, Groq, OpenAI, OpenRouter,
Ollama, or an API Kuku provides) plus Anthropic's native API all work the same way.

Usage:
    from src.llm import complete
    text = complete([{"role": "user", "content": "Say hi"}])
    data = complete(messages, json_mode=True)        # returns parsed JSON (dict/list)
    python -m src.llm --profile mock                 # quick self-test
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from src import paths as _paths

# The Windows console (cp1252) crashes on characters AI text often contains (e.g. non-breaking hyphens,
# curly quotes). Every CLI imports this module, so make printing safe once, here.
for _stream in (__import__("sys").stdout, __import__("sys").stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "model.yaml"
RAW_LOG = _paths.RAW_LOG   # every unedited model reply (PDF: "raw outputs")
CACHE_DIR = _paths.LLM_CACHE               # same prompt + same model -> reuse answer


def _log_raw(record: dict) -> None:
    """Append one call (prompt, unedited reply, usage) to the raw log. Never edited afterwards."""
    RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cache_key(p: dict, messages: list[dict], json_mode: bool) -> str:
    blob = json.dumps({"model": p["model"], "base_url": p.get("base_url"), "temperature": p["temperature"],
                       "max_tokens": p["max_tokens"], "extra": p.get("extra"), "json_mode": json_mode,
                       "messages": messages},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cost(p: dict, usage: dict) -> float | None:
    pin, pout = p.get("price_in_per_m"), p.get("price_out_per_m")
    if pin is None or pout is None:
        return None
    return usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=value lines. Never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_profile(name: str | None = None, role: str = "active") -> dict:
    cfg = load_config()
    # Env overrides are per role (bug fix: PROMO_MODEL used to replace the judge too, so a swapped
    # finder would have graded its own picks). PROMO_MODEL = finder ("active"), PROMO_JUDGE = judge.
    env = os.environ.get("PROMO_MODEL") if role == "active" else os.environ.get("PROMO_JUDGE") if role == "judge" else None
    name = name or env or cfg[role]
    if name not in cfg["profiles"]:
        raise ValueError(f"Unknown model profile '{name}'. Options: {list(cfg['profiles'])}")
    return {"name": name, **cfg["defaults"], **cfg["profiles"][name]}


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    # Some APIs (e.g. Groq, behind Cloudflare) reject Python's default "Python-urllib" User-Agent
    # with HTTP 403 / error 1010, so send a plain client name.
    headers = {"User-Agent": "promo-finder/0.1", **headers}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Truncated(LLMError):
    """The model stopped because it hit max_tokens: the answer is incomplete."""


def _call(p: dict, messages: list[dict], json_mode: bool) -> tuple[str, dict]:
    if p["provider"] == "mock":
        return _mock(messages, json_mode), {"input_tokens": 0, "output_tokens": 0}

    key = os.environ.get(p["api_key_env"], "")
    if not key:
        raise LLMError(f"Missing API key: set {p['api_key_env']} in your .env file (see .env.example).")

    if p["provider"] == "openai_compatible":
        body = {"model": p["model"], "messages": messages,
                "temperature": p["temperature"], "max_tokens": p["max_tokens"]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(p.get("extra") or {})          # provider-specific options from model.yaml
        data = _post(p["base_url"].rstrip("/") + "/chat/completions",
                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                     body, p["timeout_s"])
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise Truncated(f"reply cut off at max_tokens={p['max_tokens']} (used {usage}); "
                            f"raise max_tokens or lower reasoning for profile '{p['name']}'")
        return choice["message"]["content"] or "", usage

    if p["provider"] == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        body = {"model": p["model"], "max_tokens": p["max_tokens"],
                "temperature": p["temperature"], "messages": rest}
        if system:
            body["system"] = system
        data = _post(p["base_url"].rstrip("/") + "/messages",
                     {"Content-Type": "application/json", "x-api-key": key,
                      "anthropic-version": "2023-06-01"}, body, p["timeout_s"])
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
        if data.get("stop_reason") == "max_tokens":
            raise Truncated(f"reply cut off at max_tokens={p['max_tokens']} (used {usage})")
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        return text, usage

    raise LLMError(f"Unknown provider '{p['provider']}'")


def parse_json(text: str):
    """Models sometimes wrap JSON in ```json fences or add a sentence; extract the JSON part."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            return json.loads(m.group(1))
        raise


# Running totals, so a run can report its usage and cost (for the cost section of the doc).
# cost_usd = what the live calls in this run would cost at paid prices (free tier: actually $0).
USAGE = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "cost_usd": 0.0}


def complete(messages: list[dict], json_mode: bool = False, profile: str | None = None,
             role: str = "active", tag: str = "", use_cache: bool = True):
    """Send messages to the configured model. Retries rate limits / server errors with backoff.
    Returns text, or parsed JSON when json_mode=True.
    tag: label for the raw log (e.g. "finder:ep01"). use_cache=False forces a fresh call
    (or set env PROMO_NO_CACHE=1)."""
    load_env()
    p = get_profile(profile, role)
    use_cache = use_cache and os.environ.get("PROMO_NO_CACHE") != "1" and p["provider"] != "mock"
    key = _cache_key(p, messages, json_mode)
    cache_file = CACHE_DIR / f"{key}.json"

    if use_cache and cache_file.exists():
        text = json.loads(cache_file.read_text(encoding="utf-8"))["text"]
        USAGE["cached"] += 1
        _log_raw({"time": dt.datetime.now().isoformat(timespec="seconds"), "tag": tag, "profile": p["name"],
                  "model": p["model"], "cached": True, "cache_key": key, "response": text})
        return parse_json(text) if json_mode else text

    last_err = None
    for attempt in range(p["max_retries"] + 1):
        t0 = time.time()
        try:
            text, usage = _call(p, messages, json_mode)
            secs = time.time() - t0
            cost = _cost(p, usage)
            USAGE["calls"] += 1
            USAGE["input_tokens"] += usage["input_tokens"]
            USAGE["output_tokens"] += usage["output_tokens"]
            USAGE["seconds"] += secs
            USAGE["cost_usd"] += cost or 0.0
            _log_raw({"time": dt.datetime.now().isoformat(timespec="seconds"), "tag": tag, "profile": p["name"],
                      "model": p["model"], "cached": False, "cache_key": key, "seconds": round(secs, 2),
                      "usage": usage, "cost_usd_if_paid": cost, "messages": messages, "response": text})
            result = parse_json(text) if json_mode else text   # raises on bad JSON -> retry, NOT cached
            if use_cache:  # cache only answers that parsed correctly (bug fix 2026-11-06)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"model": p["model"], "text": text}, ensure_ascii=False),
                                      encoding="utf-8")
            return result
        except Truncated as e:
            raise LLMError(f"[{p['name']}] {e}") from None   # retrying the same request won't help
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = repr(e)
        time.sleep(min(60, 2 ** attempt * 3))
    if p.get("fallback"):
        # Model is down/overloaded after all retries: switch to the backup profile (logged).
        _log_raw({"time": dt.datetime.now().isoformat(timespec="seconds"), "tag": tag, "profile": p["name"],
                  "model": p["model"], "event": "fallback", "to": p["fallback"], "error": last_err})
        print(f"  [{p['name']}] unavailable ({(last_err or '')[:40]}...), using fallback '{p['fallback']}'")
        return complete(messages, json_mode, profile=p["fallback"], tag=tag, use_cache=use_cache)
    raise LLMError(f"[{p['name']}] failed after retries: {last_err}")


def _mock(messages: list[dict], json_mode: bool) -> str:
    """Deterministic fake answer so the pipeline can be tested offline."""
    if json_mode:
        return json.dumps({"mock": True, "echo_chars": sum(len(m["content"]) for m in messages)})
    return "mock reply"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Self-test the configured model.")
    ap.add_argument("--profile", default=None, help="Profile from config/model.yaml (default: active)")
    a = ap.parse_args()
    p = get_profile(a.profile)
    print(f"Profile: {p['name']}  provider={p['provider']}  model={p['model']}")
    msg = [{"role": "system", "content": "Reply with JSON only."},
           {"role": "user", "content": 'Return {"ok": true, "model": "<your name>"}'}]
    print("JSON reply:", complete(msg, json_mode=True, profile=a.profile))
    print("Usage:", USAGE)
