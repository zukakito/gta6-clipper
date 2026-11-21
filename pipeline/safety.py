"""Local, API-free visual safety gate. Runs on the GitHub runner itself, so it never "goes down".

Rejects a moment when:
  * any frame is classified NSFW (Falconsai/nsfw_image_detection, a small ViT that runs on CPU),
  * most frames are very dark (sleeping / camera off / empty room),
  * the picture barely changes (AFK, BRB or starting screens),
  * a large flat-colour blob covers the middle of the gameplay (streamer censor emoji / overlay).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .common import log

_clf = None
NSFW_THRESHOLD = 0.45


def _frames(src: Path, start: float, end: float, n: int = 8, w: int = 384) -> list[np.ndarray]:
    out = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                            "-vf", f"scale={w}:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, timeout=60)
        if r.returncode or not r.stdout:
            continue
        h = len(r.stdout) // (w * 3)
        out.append(np.frombuffer(r.stdout[: h * w * 3], dtype=np.uint8).reshape(h, w, 3))
    return out


def _nsfw_scores(frames: list[np.ndarray]) -> list[float]:
    global _clf
    try:
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        log.warning("safety: transformers not installed - NSFW check skipped")
        return []
    if _clf is None:
        _clf = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=-1)
    scores = []
    for f in frames:
        res = _clf(Image.fromarray(f))
        scores.append(next((r["score"] for r in res if r["label"].lower() == "nsfw"), 0.0))
    return scores


def _censor_blob(f: np.ndarray) -> bool:
    """True if the centre of the frame is dominated by one flat colour (e.g. a giant emoji over the game)."""
    h, w, _ = f.shape
    c = f[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3).astype(np.int16)
    q = (c // 24)  # coarse colour buckets
    _, counts = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
    top = counts.max() / len(q)
    sat = (c.max(axis=1) - c.min(axis=1)).mean()
    return top > 0.45 and sat > 60   # big, saturated, flat region (emoji yellow etc.)


def check(src: Path, start: float, end: float) -> tuple[bool, str]:
    frames = _frames(src, start, end)
    if len(frames) < 3:
        return False, "could not read frames"
    lum = [float(f.mean()) for f in frames]
    if sum(x < 35 for x in lum) >= len(lum) / 2:
        return False, f"too dark (mean luminance {np.mean(lum):.0f}) - idle/sleeping/camera off"
    motion = [float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean()) for i in range(1, len(frames))]
    if np.median(motion) < 2.0:
        return False, f"static picture (motion {np.median(motion):.1f}) - AFK/BRB screen"
    blobs = sum(_censor_blob(f) for f in frames)
    if blobs >= 2:
        return False, f"censor graphic covering the gameplay in {blobs} frames"
    scores = _nsfw_scores(frames)
    if not scores:
        return False, "NSFW classifier unavailable - refusing to pass unchecked footage"
    if max(scores) >= NSFW_THRESHOLD:
        return False, f"NSFW content detected (score {max(scores):.2f})"
    return True, f"ok (lum {np.mean(lum):.0f}, motion {np.median(motion):.1f}, nsfw {max(scores) if scores else 'n/a'})"
