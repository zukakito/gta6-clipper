"""Final reviewer: a vision model looks at frames of the RENDERED Short (hook and captions burned in) and checks
that the hook/title match what is actually on screen. This catches the mistakes a text-only judge can't see:
hooks about things that never appear, map/menu/loading stretches, dark or unreadable footage, non-GTA games.

Uses a vision model on Groq (same GROQ_API_KEY). Fails open: if the model is unavailable or out of quota the
clip is kept and marked "unreviewed", because every clip is still checked by hand before publishing.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
from pathlib import Path

import requests

from .common import log
from .llm import _json

GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = """You are the final quality reviewer for a YouTube Shorts channel that posts authorised GTA 6 livestream
clips for paid clipping campaigns. You see frames from the finished vertical Short (the hook text and captions are
burned in). Campaign reviewers reject misleading or weak clips, so be strict and honest."""

PROMPT = """Streamer: {streamer}
Hook shown on screen: {hook}
Title: {title}
What the editor says happens: {what}
Transcript of the cut: {transcript}

The {n} attached frames are taken at {times} of the Short.
Check:
1. Does what you see support the hook and title? (e.g. a hook about a tank, a heist, an explosion or a win needs
   that to be visible or clearly happening; walking or driving around does not support it.)
2. Is it GTA 6 gameplay (not another game, not a map, menu, inventory/weapon wheel, phone screen, loading or black
   screen)? If 2 or more frames show such a screen, gta_gameplay is false; if 1 does, lower the score by 2.
3. Is the picture clear enough to follow on a phone (not too dark, tiny or blurry)?
4. Anything unsafe for advertisers (nudity, graphic gore or blood pools, drug references, hitting or insulting
   someone as the punchline)? Does it reveal a story character's death (a spoiler)? Either makes brand_safe false.
5. Would a stranger scrolling Shorts understand and enjoy it? Score 1-10.

Return JSON only: {{"hook_matches": bool, "gta_gameplay": bool, "clear_picture": bool, "brand_safe": bool,
"score": number, "verdict": "keep" | "reject", "reason": str}}
Use verdict "reject" if any check fails or the score is below {min_score}."""


def _frames(path: Path, n: int) -> tuple[list[bytes], list[float]]:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    dur = float(r.stdout.strip() or 0)
    times = [dur * (i + 1) / (n + 1) for i in range(n)]
    out = []
    for t in times:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                            "-vf", "scale=432:-2", "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", "-"],
                           capture_output=True, timeout=60)
        if p.returncode == 0 and p.stdout:
            out.append(p.stdout)
    return out, times


def _words_text(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["w"] for w in words if start - 0.5 <= w["s"] <= end + 0.5)[:1500] or "(no speech)"


def review_clip(out: Path, meta: dict, words: list[dict], streamer: str, rcfg: dict) -> dict:
    """Returns {"verdict": "keep"|"reject"|"unreviewed", "reason": str, ...}."""
    if not rcfg.get("enabled", True):
        return {"verdict": "unreviewed", "reason": "review disabled"}
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"verdict": "unreviewed", "reason": "no GROQ_API_KEY"}
    n = int(rcfg.get("frames", 3))
    min_score = float(rcfg.get("min_score", 6))
    imgs, times = _frames(out, n)
    if not imgs:
        return {"verdict": "unreviewed", "reason": "could not grab frames"}
    total = times[-1] * (n + 1) / n if times else 0
    prompt = PROMPT.format(streamer=streamer, hook=meta.get("hook", ""), title=meta.get("title", ""),
                           what=meta.get("what_happens") or "(not given)",
                           transcript=_words_text(words, meta.get("start", 0), meta.get("end", 1e9)),
                           n=len(imgs), times=", ".join(f"{t:.0f}s" for t in times) + (f" (of {total:.0f}s)" if total else ""),
                           min_score=f"{min_score:g}")
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
        for b in imgs]
    for model in rcfg.get("models") or ["qwen/qwen3.8-27b"]:
        try:
            r = requests.post(GROQ_CHAT, headers={"Authorization": f"Bearer {key}"}, timeout=90, json={
                "model": model, "temperature": 0.2, "max_completion_tokens": 1500,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]})
        except requests.RequestException as e:
            log.warning("review: %s request failed: %s", model, str(e)[:200])
            continue
        if r.status_code != 200:
            log.warning("review: %s returned %s %s", model, r.status_code, r.text[:300])
            continue
        text = r.json()["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        res = _json(text)
        if not isinstance(res, dict) or "verdict" not in res:
            log.warning("review: %s gave no usable JSON: %s", model, text[:200])
            continue
        try:
            score = float(res.get("score"))
        except (TypeError, ValueError):
            score = 0.0
        failed = [k for k in ("hook_matches", "gta_gameplay", "clear_picture", "brand_safe") if res.get(k) is False]
        if failed or score < min_score:
            res["verdict"] = "reject"
            res["reason"] = (res.get("reason") or "") + (f" [failed: {', '.join(failed)}]" if failed else "") + \
                (f" [score {score:g} < {min_score:g}]" if score < min_score else "")
        res["model"] = model
        log.info("review: %s -> %s (score %s): %s", meta.get("title"), res["verdict"], res.get("score"),
                 str(res.get("reason"))[:200])
        return res
    return {"verdict": "unreviewed", "reason": "vision model unavailable"}
