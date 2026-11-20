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

MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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
    for attempt in range(retries):
        try:
            r = requests.post(URL.format(model=MODEL), params={"key": key}, json=body, timeout=90)
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
