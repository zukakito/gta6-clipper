"""Thin wrapper around the free Gemini API (Google AI Studio key).

Every caller has a non-LLM fallback, so the pipeline still runs without a key.
"""
from __future__ import annotations

import base64
import json
import re
import time

import requests

from .common import env, log

BASE = "https://generativelanguage.googleapis.com/v1beta"
URL = BASE + "/models/{model}:generateContent"
_model: str | None = env("GEMINI_MODEL")   # optional override; otherwise picked automatically


def _pick_model(key: str) -> str | None:
    """Pick the newest stable 'flash' model this key can use (model names change over time)."""
    try:
        r = requests.get(BASE + "/models", params={"key": key, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa: BLE001
        log.warning("gemini: cannot list models: %s", e)
        return None
    def rank(n: str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0.0]
        return (("flash" in n) and ("lite" not in n), not any(t in n for t in ("preview", "exp", "image", "tts", "live", "audio")), nums[0], -len(n))
    best = max(names, key=rank) if names else None
    log.info("gemini: using model %s", best)
    return best


def available() -> bool:
    return bool(env("GEMINI_API_KEY"))


def ask(prompt: str, want_json: bool = False, temperature: float = 0.4, retries: int = 3,
        images: list[bytes] | None = None) -> str | dict | list | None:
    key = env("GEMINI_API_KEY")
    if not key:
        return None
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(im).decode()}} for im in (images or [])]
    parts.append({"text": prompt})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature},
    }
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    global _model
    if _model is None:
        _model = _pick_model(key) or "gemini-flash-latest"
    for attempt in range(retries):
        try:
            r = requests.post(URL.format(model=_model), params={"key": key}, json=body, timeout=90)
            if r.status_code == 404 and attempt == 0:  # model retired -> re-pick once
                _model = _pick_model(key) or _model
                continue
            if r.status_code == 429:  # free-tier rate limit
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            if not want_json:
                return text.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001 - never let the LLM kill a run
            log.warning("gemini call failed (%s/%s): %s", attempt + 1, retries, e)
            time.sleep(5)
    return None
