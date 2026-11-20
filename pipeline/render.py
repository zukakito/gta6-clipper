"""Download a moment, transcribe it, and render a captioned vertical Short (or a 16:9 long-form cut)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .common import FONTS, OUT, WORK, log

W, H = 1080, 1920
FONT = "Montserrat"          # falls back to DejaVu Sans if the font file is missing


# ---------------------------------------------------------------- download
def download_section(url: str, start: float, end: float, name: str) -> Path:
    out = WORK / f"src_{name}.mp4"
    if out.exists():
        return out
    subprocess.run([
        "yt-dlp", "-q", "--no-warnings",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start:.2f}-{end:.2f}",
        "-o", str(out), url,
    ], check=True, timeout=1800)
    return out


# ---------------------------------------------------------------- transcription
_model = None


def transcribe(path: Path, model_size: str = "small") -> list[dict]:
    """Return word-level transcript: [{'w': str, 's': float, 'e': float}] (times relative to file)."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper not installed; rendering without captions")
        return []
    import numpy as np

    if _model is None:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # Decode with ffmpeg ourselves (16 kHz mono float32). This avoids faster-whisper's PyAV
    # decoder, which breaks on newer PyAV releases ("unexpected keyword 'metadata_errors'").
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], check=True, capture_output=True, timeout=600).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return []
    segments, _ = _model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append({"w": t, "s": float(w.start), "e": float(w.end)})
    return words


# ---------------------------------------------------------------- captions (ASS)
def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _clean(s: str) -> str:
    s = re.sub(r"[^\w\s'!?.,&:#$%-]", "", s)  # libass can't draw colour emoji; drop them
    return s.replace("{", "").replace("}", "").strip()


def build_ass(words: list[dict], clip_start: float, clip_end: float, hook: str, tag: str,
              disclosure: bool, vertical: bool = True) -> str:
    pw, ph = (W, H) if vertical else (1920, 1080)
    cap_size = 78 if vertical else 64
    cap_y = int(ph * 0.70) if vertical else int(ph * 0.88)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {pw}
PlayResY: {ph}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{FONT},{cap_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Hook,{FONT},{66 if vertical else 56},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,1,0,0,0,100,100,0,0,3,18,0,8,70,70,{150 if vertical else 40},1
Style: Tag,{FONT},{54 if vertical else 44},&H0000E1FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,1,0,1,5,2,2,60,60,{230 if vertical else 30},1
Style: Disc,{FONT},34,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,8,0,7,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = clip_end - clip_start
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(dur)},Hook,,0,0,0,,{_clean(hook).upper()}")
    if tag:
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(dur)},Tag,,0,0,0,,{_clean(tag).upper()}")
    if disclosure:
        ev.append(f"Dialogue: 2,{_ts(0)},{_ts(3.5)},Disc,,0,0,0,,Paid partnership")

    # Word-by-word captions: chunks of up to 3 words, active word highlighted yellow.
    inwin = [w for w in words if w["e"] > clip_start and w["s"] < clip_end]
    chunks, cur = [], []
    for w in inwin:
        cur.append(w)
        if len(cur) == 3 or w["w"].endswith((".", "?", "!", ",")):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    for ch in chunks:
        for i, w in enumerate(ch):
            s = w["s"] - clip_start
            e = (ch[i + 1]["s"] if i + 1 < len(ch) else w["e"]) - clip_start
            parts = []
            for j, x in enumerate(ch):
                t = _clean(x["w"]).upper()
                parts.append("{\\c&H0000E1FF&}" + t + "{\\c&H00FFFFFF&}" if j == i else t)
            ev.append(f"Dialogue: 0,{_ts(s)},{_ts(max(e, s + 0.08))},Cap,,0,0,{0},,{{\\pos({pw // 2},{cap_y})}}" + " ".join(parts))
    return head + "\n".join(ev) + "\n"


# ---------------------------------------------------------------- ffmpeg compose
def _filter_vertical(layout: str, ass_path: Path) -> str:
    fontsdir = f":fontsdir={FONTS}" if FONTS.exists() else ""
    ass = f"ass={ass_path}{fontsdir}"
    if layout == "center_crop":
        return f"[0:v]scale=-2:{H},crop={W}:{H},setsar=1,fps=30,{ass}[v]"
    if layout == "zoom_blur":
        fg = f"scale=-2:1000,crop={W}:1000"
    else:  # fit_blur: whole frame visible (keeps streamer overlays/facecam)
        fg = f"scale={W}:-2"
    return (f"[0:v]split=2[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:3,eq=brightness=-0.10[bg];"
            f"[b]{fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-60,setsar=1,fps=30,{ass}[v]")


def render_short(src: Path, rel_start: float, rel_end: float, words: list[dict], hook: str, tag: str,
                 disclosure: bool, layout: str, name: str) -> Path:
    ass_path = WORK / f"{name}.ass"
    ass_path.write_text(build_ass(words, rel_start, rel_end, hook, tag, disclosure), encoding="utf-8")
    out = OUT / f"{name}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{rel_start:.2f}", "-to", f"{rel_end:.2f}", "-i", str(src),
        "-filter_complex", _filter_vertical(layout, ass_path), "-map", "[v]", "-map", "0:a?",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
    ], check=True, timeout=1800)
    # -ss before -i resets timestamps to 0, matching the ASS times built relative to rel_start.
    return out


def render_long(parts: list[tuple[Path, float, float, list[dict], str]], tag: str, name: str) -> Path:
    """Concatenate several moments into a 16:9 compilation with captions."""
    seg_files = []
    for i, (src, s, e, words, hook) in enumerate(parts):
        ass_path = WORK / f"{name}_{i}.ass"
        ass_path.write_text(build_ass(words, s, e, hook, tag, disclosure=(i == 0), vertical=False), encoding="utf-8")
        fontsdir = f":fontsdir={FONTS}" if FONTS.exists() else ""
        seg = WORK / f"{name}_{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", str(src),
            "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,ass={ass_path}{fontsdir}",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(seg),
        ], check=True, timeout=3600)
        seg_files.append(seg)
    lst = WORK / f"{name}_list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in seg_files))
    out = OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    return out


def probe_duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip() or 0)
