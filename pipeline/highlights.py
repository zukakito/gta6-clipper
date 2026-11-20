"""Find the most clip-worthy moments in a VOD window.

Signals (all free, no video download needed for the first pass):
  1. Chat velocity: messages per 5 s from the VOD's chat replay (Twitch/YouTube live chat),
     weighted up for hype tokens (LMAO, KEKW, W, OMEGALUL, "?????", "CLIP IT"...).
  2. Audio loudness: RMS of the stream audio per second (streamer yelling / big moments).
Scores are z-normalised and combined; we return the top non-overlapping peaks.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from .common import WORK, log

HYPE = re.compile(r"(LMAO|LMFAO|KEKW|OMEGALUL|LUL|ICANT|W{1,}\b|\bL\b|CLIP|NAHH+|💀|😂|🤣|\?{3,}|HOLY|BRO)", re.I)
BIN = 5  # seconds per chat bin


def chat_series(url: str, start: float, end: float) -> np.ndarray | None:
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        return None
    n = int((end - start) // BIN) + 1
    series = np.zeros(n)
    try:
        chat = ChatDownloader().get_chat(url, start_time=int(start), end_time=int(end), message_types=["text_message"])
        count = 0
        for m in chat:
            t = m.get("time_in_seconds")
            if t is None:
                continue
            i = int((t - start) // BIN)
            if 0 <= i < n:
                msg = m.get("message") or ""
                series[i] += 1 + (1.5 if HYPE.search(msg) else 0)
            count += 1
        log.info("highlights: %d chat messages read", count)
        return series if count > 50 else None
    except Exception as e:  # noqa: BLE001
        log.warning("highlights: chat replay unavailable (%s)", e)
        return None


def audio_series(url: str, start: float, end: float, vod_key: str) -> np.ndarray | None:
    """Download only the audio for [start, end] and compute loudness per BIN seconds."""
    out = WORK / f"audio_{re.sub(r'[^a-z0-9]', '_', vod_key.lower())}_{int(start)}.m4a"
    try:
        if not out.exists():
            subprocess.run([
                "yt-dlp", "-q", "--no-warnings", "-f", "Audio_Only/bestaudio/worst",
                "--download-sections", f"*{int(start)}-{int(end)}", "-o", str(out), url,
            ], check=True, timeout=1800)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-ac", "1", "-ar", "4000", "-f", "s16le", "-"],
                             check=True, capture_output=True, timeout=900).stdout
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        per = 4000 * BIN
        n = len(x) // per
        if n < 10:
            return None
        rms = np.sqrt((x[: n * per].reshape(n, per) ** 2).mean(axis=1) + 1e-9)
        return 20 * np.log10(rms + 1e-9)
    except Exception as e:  # noqa: BLE001
        log.warning("highlights: audio analysis failed (%s)", e)
        return None
    finally:
        out.unlink(missing_ok=True)


def _z(a: np.ndarray) -> np.ndarray:
    a = np.convolve(a, np.ones(3) / 3, mode="same")  # light smoothing
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a * 0


def find_peaks(url: str, start: float, end: float, vod_key: str, k: int = 6, min_gap: float = 120) -> list[dict]:
    chat = chat_series(url, start, end)
    audio = audio_series(url, start, end, vod_key)
    if chat is None and audio is None:
        return []
    n = max(len(s) for s in (chat, audio) if s is not None)
    score = np.zeros(n)
    if chat is not None:
        score[: len(chat)] += 0.65 * _z(chat)
    if audio is not None:
        score[: len(audio)] += (0.35 if chat is not None else 1.0) * _z(audio)

    picks: list[dict] = []
    for i in np.argsort(score)[::-1]:
        t = start + i * BIN
        if score[i] < 1.0 or len(picks) >= k:
            break
        if any(abs(t - p["t"]) < min_gap for p in picks):
            continue
        # Chat reacts a few seconds after the moment, so the window leans earlier.
        picks.append({"t": float(t), "score": float(score[i]),
                      "win_start": max(start, t - 45), "win_end": min(end, t + 30)})
    log.info("highlights: %d peak(s) in %s [%d-%d]", len(picks), vod_key, start, end)
    return sorted(picks, key=lambda p: -p["score"])
