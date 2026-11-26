"""Read on-screen text from a stream window with Tesseract OCR (free, runs on the GitHub runner).

GTA 6 shows a lot of what is happening as text: character subtitles at the bottom ("We got a tank straight ahead!")
and big centre messages (MISSION PASSED, WASTED, BUSTED). The clip judge is text-only, so giving it this text
lets it ground hooks in what the game itself says, not just the streamer's words.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .common import WORK, log

# Regions of a 16:9 stream frame, as (x, y, w, h) fractions: in-game subtitles, centre banner.
REGIONS = {"sub": (0.15, 0.80, 0.70, 0.17), "mid": (0.20, 0.36, 0.60, 0.28)}
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")


def available() -> bool:
    return shutil.which("tesseract") is not None


def _clean(line: str) -> str | None:
    line = re.sub(r"\s+", " ", line).strip(" |_-~=*·.,'\"")
    words = WORD.findall(line)
    if len(words) < 2:
        return None
    letters = sum(ch.isalpha() for ch in line)
    if letters < 0.6 * max(1, len(line.replace(" ", ""))):   # mostly symbols/noise
        return None
    return line[:140]


def _ocr(img: Path) -> list[str]:
    r = subprocess.run(["tesseract", str(img), "-", "--psm", "6", "-l", "eng"],
                       capture_output=True, text=True, timeout=30)
    return [c for c in (_clean(x) for x in r.stdout.splitlines()) if c]


def screen_text(src: Path, start: float = 0.0, end: float | None = None, step: float = 3.0,
                regions: dict | None = None, max_lines: int = 40) -> list[tuple[float, str]]:
    """[(seconds from file start, text)] for text visible on screen, deduplicated. Empty list on any failure."""
    if not available():
        return []
    regions = regions or REGIONS
    seek = ["-ss", f"{start:.2f}"] + (["-to", f"{end:.2f}"] if end else [])
    out: list[tuple[float, str]] = []
    try:
        with tempfile.TemporaryDirectory(dir=str(WORK)) as tmp:
            for key, (x, y, w, h) in regions.items():
                # crop the region, upscale x2, grayscale, keep bright text (GTA subtitles/messages are white/yellow)
                vf = (f"fps=1/{step},crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale=iw*2:-2,format=gray,"
                      f"lutyuv=y='if(gt(val,170),0,255)'")
                subprocess.run(["ffmpeg", "-v", "error", *seek, "-i", str(src), "-vf", vf,
                                f"{tmp}/{key}_%04d.png"], check=True, timeout=300)
                for img in sorted(Path(tmp).glob(f"{key}_*.png")):
                    t = start + (int(img.stem.split("_")[1]) - 1) * step
                    for line in _ocr(img):
                        out.append((round(t, 1), line))
    except Exception as e:  # noqa: BLE001
        log.warning("screen: OCR failed: %s", str(e)[:200])
        return []
    out.sort()
    dedup: list[tuple[float, str]] = []
    for t, line in out:   # the same subtitle stays up for several samples: keep its first sighting only
        if any(line.lower() == s.lower() and t - t0 <= 12 for t0, s in dedup[-8:]):
            continue
        dedup.append((t, line))
    log.info("screen: %d on-screen text lines", len(dedup))
    return dedup[:max_lines]


def as_prompt(lines: list[tuple[float, str]]) -> str:
    return "\n".join(f"[{t:6.1f}] {s}" for t, s in lines) or "(no readable on-screen text)"
