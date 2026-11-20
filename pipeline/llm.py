"""Thin wrapper around the Gemini API (Google AI Studio key).

Every caller has a non-LLM fallback, so the pipeline still runs without a key.
Model names change over time, so the usable 'flash' models are listed from Google
on first use and tried in order; overloaded (503), rate-limited (429) or retired (404)
models fall through to the next one.
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
SKIP = ("image", "tts", "live", "audio", "embedding")
_models: list[str] | None = [env("GEMINI_MODEL")] if env("GEMINI_MODEL") else None   # optional override


def _rank_models(key: str) -> list[str]:
    try:
        r = requests.get(BASE + "/models", params={"key": key, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa: BLE001
        log.warning("gemini: cannot list models: %s", e)
        names = []

    def rank(n: str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0.0]
        return (n == "gemini-flash-latest",                    # Google's stable alias first
                "lite" not in n,
                not any(t in n for t in ("preview", "exp", "thinking")),
                nums[0], -len(n))

    flash = sorted((n for n in names if "flash" in n and not any(t in n for t in SKIP)), key=rank, reverse=True)
    out = flash[:4] or ["gemini-flash-latest"]
    log.info("gemini: model order %s", out)
    return out


def available() -> bool:
    return bool(env("GEMINI_API_KEY"))


def ask(prompt: str, want_json: bool = False, temperature: float = 0.4, retries: int = 4,
        images: list[bytes] | None = None) -> str | dict | list | None:
    global _models
    key = env("GEMINI_API_KEY")
    if not key:
        return None
    if _models is None:
        _models = _rank_models(key)
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(im).decode()}} for im in (images or [])]
    parts.append({"text": prompt})
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": temperature}}
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    for attempt in range(retries):
        model = _models[attempt % len(_models)]
        try:
            r = requests.post(URL.format(model=model), params={"key": key}, json=body, timeout=45)
            if r.status_code in (404, 429, 500, 503):   # retired / rate-limited / overloaded -> next model
                log.warning("gemini: %s returned %s, trying another model", model, r.status_code)
                if r.status_code == 404 and len(_models) > 1:
                    _models.remove(model)
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            if not want_json:
                return text.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001 - never let the LLM kill a run
            log.warning("gemini call failed (%s/%s) on %s: %s", attempt + 1, retries, model, e)
            time.sleep(3)
    return None
