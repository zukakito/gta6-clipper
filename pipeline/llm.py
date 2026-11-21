"""LLM access with automatic fallback between free providers.

Order: Groq (GROQ_API_KEY, fast free tier, text only) -> Gemini (GEMINI_API_KEY, can see images).
Model names change over time, so each provider's usable models are listed on first use.
A provider that is rate-limited/down on two calls in a row is skipped for the rest of the run
(oversized prompts and bad JSON don't count as outages).
"""
from __future__ import annotations

import base64
import json
import re
import time

import requests

from .common import env, log

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
GROQ = "https://api.groq.com/openai/v1"
_models: dict[str, list[str]] = {}
_fail: dict[str, int] = {"groq": 0, "gemini": 0}
MAX_PROMPT_CHARS = 18000   # ~5k tokens: fits Groq's free-tier per-minute token cap with room for the reply


def _providers() -> list[str]:
    out = []
    if env("GROQ_API_KEY") and _fail["groq"] < 2:
        out.append("groq")
    if env("GEMINI_API_KEY") and _fail["gemini"] < 2:
        out.append("gemini")
    return out


def available() -> bool:
    return bool(_providers())


def _list_gemini(key: str) -> list[str]:
    if env("GEMINI_MODEL"):
        return [env("GEMINI_MODEL")]
    try:
        r = requests.get(GEMINI + "/models", params={"key": key, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa: BLE001
        log.warning("gemini: cannot list models: %s", e)
        names = []

    def rank(n: str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0.0]
        return (n == "gemini-flash-latest", "lite" not in n, not any(t in n for t in ("preview", "exp", "thinking")), nums[0])
    flash = sorted((n for n in names if "flash" in n and not any(t in n for t in ("image", "tts", "live", "audio", "embedding"))),
                   key=rank, reverse=True)
    return flash[:3] or ["gemini-flash-latest"]


def _list_groq(key: str) -> list[str]:
    if env("GROQ_MODEL"):
        return [env("GROQ_MODEL")]
    preferred = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "moonshotai/kimi-k2-instruct"]
    try:
        r = requests.get(GROQ + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        have = {m["id"] for m in r.json().get("data", []) if m.get("active", True)}
        picks = [m for m in preferred if m in have]
        return picks[:3] or sorted(have)[:2]
    except Exception as e:  # noqa: BLE001
        log.warning("groq: cannot list models: %s", e)
        return preferred[:2]


def _json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None


def _call_gemini(model: str, prompt: str, want_json: bool, temperature: float, images):
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(im).decode()}} for im in (images or [])]
    parts.append({"text": prompt})
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": temperature}}
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    r = requests.post(f"{GEMINI}/models/{model}:generateContent", params={"key": env("GEMINI_API_KEY")}, json=body, timeout=25)
    return r, (lambda: r.json()["candidates"][0]["content"]["parts"][0]["text"])


def _call_groq(model: str, prompt: str, want_json: bool, temperature: float, images):
    msg = prompt + ("\n\nRespond with JSON only." if want_json else "")
    body = {"model": model, "temperature": temperature, "messages": [{"role": "user", "content": msg}]}
    if want_json:
        body["response_format"] = {"type": "json_object"}   # Groq JSON mode: always valid JSON
    r = requests.post(GROQ + "/chat/completions", headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"}, json=body, timeout=40)
    return r, (lambda: r.json()["choices"][0]["message"]["content"])


def _retry_after(r) -> float | None:
    try:
        h = getattr(r, "headers", {}) or {}
        v = h.get("retry-after") or h.get("Retry-After")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def ask(prompt: str, want_json: bool = False, temperature: float = 0.4, images: list[bytes] | None = None,
        retries: int = 2) -> str | dict | list | None:
    """Try each available provider/model in turn. Returns None if all fail."""
    if len(prompt) > MAX_PROMPT_CHARS:   # Groq free tier rejects big requests with 413 (tokens-per-minute cap)
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n...(truncated)"
    for prov in _providers():
        hard_fail = False
        key = env("GROQ_API_KEY" if prov == "groq" else "GEMINI_API_KEY")
        if prov not in _models:
            _models[prov] = (_list_groq if prov == "groq" else _list_gemini)(key)
            log.info("%s: model order %s", prov, _models[prov])
        call = _call_groq if prov == "groq" else _call_gemini
        for attempt in range(retries * len(_models[prov])):
            model = _models[prov][attempt % len(_models[prov])]
            try:
                r, text_of = call(model, prompt, want_json, temperature, images)
                if r.status_code == 413:   # this prompt is too big for the free tier - not an outage
                    log.warning("%s: %s says the request is too large (%d chars) - skipping it", prov, model, len(prompt))
                    break
                if r.status_code == 400 and "json" in r.text.lower():
                    log.warning("%s: %s produced invalid JSON - retrying", prov, model)
                    continue
                wait = _retry_after(r)
                if r.status_code == 429 and wait is not None and wait <= 40:   # per-minute limit: just wait it out
                    log.info("%s: %s rate-limited, waiting %.0fs", prov, model, wait)
                    time.sleep(wait + 1)
                    continue
                if r.status_code in (404, 429, 500, 502, 503):
                    log.warning("%s: %s returned %s %s", prov, model, r.status_code, r.text[:120])
                    hard_fail = True
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                text = text_of()
                out = _json(text) if want_json else text.strip()
                if want_json and out is None:
                    log.warning("%s: %s reply was not parseable JSON - retrying", prov, model)
                    continue
                _fail[prov] = 0
                return out
            except requests.RequestException as e:
                hard_fail = True
                log.warning("%s call failed on %s: %s", prov, model, str(e)[:200])
                time.sleep(2)
            except Exception as e:  # noqa: BLE001
                log.warning("%s call failed on %s: %s", prov, model, str(e)[:200])
                time.sleep(2)
        if not hard_fail:
            continue
        _fail[prov] += 1
        if _fail[prov] >= 2:
            log.warning("%s: failing repeatedly - skipped for the rest of this run", prov)
    return None


def can_see_images() -> bool:
    return "gemini" in _providers()
