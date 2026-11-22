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
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/best", "-S", "res:1080,fps,br",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start:.2f}-{end:.2f}",
        "-o", str(out), url,
    ], check=True, timeout=1800)
    return out


# ---------------------------------------------------------------- transcription
_model = None


_models_by_size: dict = {}


def transcribe(path: Path, model_size: str = "small", start: float | None = None, end: float | None = None,
               prompt: str | None = None) -> list[dict]:
    """Return word-level transcript: [{'w': str, 's': float, 'e': float}] (times relative to file).
    With start/end only that part is transcribed (times still relative to the file)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper not installed; rendering without captions")
        return []
    import numpy as np

    if model_size not in _models_by_size:
        _models_by_size[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model = _models_by_size[model_size]
    # Decode with ffmpeg ourselves (16 kHz mono float32). This avoids faster-whisper's PyAV
    # decoder, which breaks on newer PyAV releases ("unexpected keyword 'metadata_errors'").
    seek = ["-ss", f"{start:.2f}", "-to", f"{end:.2f}"] if start is not None and end is not None else []
    raw = subprocess.run(["ffmpeg", "-v", "error", *seek, "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], check=True, capture_output=True, timeout=600).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return []
    segments, _ = _model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True,
                                    initial_prompt=prompt, condition_on_previous_text=False)
    off = start or 0.0
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append({"w": t, "s": float(w.start) + off, "e": float(w.end) + off})
    return words


def refine_words(path: Path, words: list[dict], start: float, end: float, model_size: str = "medium",
                 prompt: str | None = None) -> list[dict]:
    """Re-transcribe just the chosen cut with a bigger Whisper model for accurate captions."""
    try:
        better = transcribe(path, model_size, max(0.0, start - 0.5), end + 0.5, prompt=prompt)
    except Exception as e:  # noqa: BLE001
        log.warning("render: caption refine failed (%s) - keeping draft captions", e)
        return words
    if not better:
        return words
    return [w for w in words if w["e"] <= start - 0.5 or w["s"] >= end + 0.5] + better



# ---------------------------------------------------------------- facecam detection
def detect_facecam(src: Path, start: float, end: float) -> tuple[int, int, int, int] | None:
    """Find a static webcam overlay: a face that stays in the same place across frames
    (game characters move, a facecam doesn't). Returns (x, y, w, h) in source pixels."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:  # noqa: BLE001
        return None
    if cascade.empty():
        log.warning("render: face cascade not available - no facecam detection")
        return None
    cap = cv2.VideoCapture(str(src))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    hits = []
    for i in range(7):
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * (start + (end - start) * (i + 0.5) / 7))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(int(H0 * 0.05), int(H0 * 0.05)))
        hits.append([tuple(map(int, f)) for f in faces])
    cap.release()
    if len(hits) < 4:
        return None
    # cluster face centres; a facecam appears in >= 60% of frames within ~4% of frame width
    cands = [f for fr in hits for f in fr]
    best, best_n = None, 0
    for f in cands:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        n = sum(any(abs((g[0] + g[2] / 2) - cx) < W0 * 0.04 and abs((g[1] + g[3] / 2) - cy) < H0 * 0.04 for g in fr) for fr in hits)
        if n > best_n:
            best, best_n = f, n
    if not best or best_n < 0.6 * len(hits):
        return None
    x, y, w, h = best
    if H0 < 900 or w < W0 * 0.04:   # low-res source or a tiny webcam: upscaling it to 1080 px wide looks pixelated
        log.info("render: facecam too small to split (face %dpx in %dx%d) - using full-frame layout", w, W0, H0)
        return None
    # webcam box ~ 3.2x face width (at least 30% of the frame, so it is never upscaled > ~1.9x), clamped
    bw = min(W0, max(int(w * 3.2), int(W0 * 0.30)))
    bh = min(H0, int(bw * 640 / 1080))
    bx = max(0, min(W0 - bw, int(x + w / 2 - bw / 2)))
    by = max(0, min(H0 - bh, int(y + h / 2 - bh * 0.45)))
    return bx, by, bw, bh


# ---------------------------------------------------------------- dead-air trimming
def speech_segments(words: list[dict], start: float, end: float, max_gap: float = 2.0,
                    pad: float = 0.35) -> list[tuple[float, float]]:
    """Keep-segments (relative to start) that drop pauses longer than max_gap seconds."""
    ws = [w for w in words if w["e"] > start and w["s"] < end]
    if not ws:
        return [(0.0, end - start)]
    segs = []
    cur_s, cur_e = max(start, ws[0]["s"] - pad), min(end, ws[0]["e"] + pad)
    for w in ws[1:]:
        if w["s"] - cur_e > max_gap:
            segs.append((cur_s - start, cur_e - start))
            cur_s = max(start, w["s"] - pad)
        cur_e = min(end, max(cur_e, w["e"] + pad))
    segs.append((cur_s - start, cur_e - start))
    # always keep the very beginning (setup) and the reaction tail
    segs[0] = (0.0, segs[0][1])
    segs[-1] = (segs[-1][0], end - start)
    return segs


def remap_words(words: list[dict], start: float, segs: list[tuple[float, float]]) -> list[dict]:
    """Shift word times (absolute in the source window) onto the trimmed timeline (relative to 0)."""
    out, offset = [], 0.0
    for a, b in segs:
        for w in words:
            ws, we = w["s"] - start, w["e"] - start
            if a <= ws < b:
                out.append({"w": w["w"], "s": ws - a + offset, "e": min(we, b) - a + offset})
        offset += b - a
    return out


# ---------------------------------------------------------------- captions (ASS)
def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


_SWEAR = re.compile(r"(?i)\b(mother)?(f+u+c+k+\w*|sh+i+t+\w*|bitch\w*|cunts?|dicks?|pussy|pussies|nigg\w*|fag\w*|retard\w*)")


def _bleep(m: re.Match) -> str:
    w = m.group(0)
    return w[0] + "*" * (len(w) - 2) + w[-1] if len(w) > 2 else "*" * len(w)


def _clean(s: str) -> str:
    s = re.sub(r"[^\w\s'!?.,&:#$%/@*-]", "", s)  # libass can't draw colour emoji; drop them
    s = _SWEAR.sub(_bleep, s)                   # on-screen text stays advertiser-friendly
    return s.replace("{", "").replace("}", "").strip()


def build_ass(words: list[dict], clip_start: float, clip_end: float, hook: str, tag: str,
              disclosure: bool, vertical: bool = True, split: bool = False) -> str:
    pw, ph = (W, H) if vertical else (1920, 1080)
    cap_size = 84 if split else (78 if vertical else 64)
    cap_y = 1180 if split else (int(ph * 0.70) if vertical else int(ph * 0.88))
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
    if hook and split:   # split layout: hook sits on the seam between facecam and gameplay for the first 4 s
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(min(4.0, dur))},Hook,,0,0,0,,{{\\an5\\pos({pw // 2},640)}}{_clean(hook).upper()}")
    elif hook:
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
            e = min((ch[i + 1]["s"] if i + 1 < len(ch) else w["e"]) - clip_start, s + 1.2)  # never linger
            parts = []
            for j, x in enumerate(ch):
                t = _clean(x["w"]).upper()
                parts.append("{\\c&H0000E1FF&}" + t + "{\\c&H00FFFFFF&}" if j == i else t)
            ev.append(f"Dialogue: 0,{_ts(s)},{_ts(max(e, s + 0.08))},Cap,,0,0,{0},,{{\\pos({pw // 2},{cap_y})}}" + " ".join(parts))
    return head + "\n".join(ev) + "\n"


# ---------------------------------------------------------------- ffmpeg compose
def _filter_split(face: tuple[int, int, int, int], ass_path: Path, trim: str, overlay_img: Path | None) -> str:
    """Facecam (top 1080x640) + gameplay centre crop (bottom 1080x1280)."""
    fontsdir = f":fontsdir={FONTS}" if FONTS.exists() else ""
    fx, fy, fw, fh = face
    chain = (f"[0:v]{trim}split=2[a][b];"
             f"[a]crop={fw}:{fh}:{fx}:{fy},scale={W}:640,setsar=1[top];"
             f"[b]crop=ih*{W}/1280:ih:(iw-ow)/2:0,scale={W}:1280,setsar=1[bot];"
             f"[top][bot]vstack=inputs=2[st]")
    if overlay_img:
        chain += f";[1:v]scale={int(W * 0.30)}:-1[ov];[st][ov]overlay=W-w-24:664[st]"
    return chain + f";[st]fps=30,ass={ass_path}{fontsdir}[v]"


def _filter_vertical(layout: str, ass_path: Path, overlay_img: Path | None = None) -> str:
    fontsdir = f":fontsdir={FONTS}" if FONTS.exists() else ""
    ass = f"ass={ass_path}{fontsdir}"
    if overlay_img:   # campaign-supplied overlay (e.g. KreekCraft logo): top-right corner of the video area
        y0 = (H - int(W * 9 / 16)) // 2 - 60 + 16
        return (f"[0:v]split=2[a][b];"
                f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:3,eq=brightness=-0.10[bg];"
                f"[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2-60[base];"
                f"[1:v]scale={int(W * 0.28)}:-1[ov];[base][ov]overlay=W-w-24:{y0},setsar=1,fps=30,{ass}[v]")
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
                 disclosure: bool, layout: str, name: str, overlay_img: Path | None = None,
                 trim_silence: bool = True, min_len: float = 0.0) -> Path:
    out = OUT / f"{name}.mp4"
    ass_path = WORK / f"{name}.ass"
    extra = ["-i", str(overlay_img)] if overlay_img else []
    face = detect_facecam(src, rel_start, rel_end) if layout in ("auto", "split") else None
    segs = speech_segments(words, rel_start, rel_end) if trim_silence else [(0.0, rel_end - rel_start)]
    vtrim = atrim = ""
    orig_len = rel_end - rel_start
    total = sum(b - a for a, b in segs)
    # Trim dead air, but never below the campaign/Shorts minimum length (mostly-action clips are left whole).
    if len(segs) > 1 and total >= max(0.75 * (rel_end - rel_start), min_len):
        cut = WORK / f"{name}_cut.mp4"
        parts = ";".join(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}]"
                         for i, (a, b) in enumerate(segs))
        joins = "".join(f"[v{i}][a{i}]" for i in range(len(segs)))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{rel_start:.3f}", "-to", f"{rel_end:.3f}", "-i", str(src),
                        "-filter_complex", f"{parts};{joins}concat=n={len(segs)}:v=1:a=1[v][a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                        "-c:a", "aac", "-b:a", "192k", str(cut)], check=True, timeout=900)
        cap_words = remap_words(words, rel_start, segs)
        src, rel_start, rel_end = cut, 0.0, total
    else:
        segs, total = [(0.0, rel_end - rel_start)], rel_end - rel_start
        cap_words = [{**w, "s": w["s"] - rel_start, "e": w["e"] - rel_start} for w in words]
    ass_path.write_text(build_ass(cap_words, 0.0, total, hook, tag, disclosure, split=bool(face)), encoding="utf-8")
    if face:
        fc = _filter_split(face, ass_path, vtrim, overlay_img)
    else:
        lay = "zoom_blur" if layout == "auto" else layout
        fc = _filter_vertical(lay, ass_path, overlay_img).replace("[0:v]", f"[0:v]{vtrim}", 1)
    log.info("render: %s layout=%s facecam=%s segments=%d (%.1fs -> %.1fs)", name,
             "split" if face else layout, face, len(segs), orig_len, total)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{rel_start:.2f}", "-to", f"{rel_end:.2f}", "-i", str(src), *extra,
        "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
        "-af", f"{atrim}loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
    ], check=True, timeout=1800)
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


def fetch_overlay(url: str | None, key: str) -> Path | None:
    """Download a campaign overlay image once per run (cached in WORK)."""
    if not url:
        return None
    out = WORK / f"overlay_{key}.png"
    if not out.exists():
        import requests
        raw = WORK / f"overlay_{key}.src"
        raw.write_bytes(requests.get(url, timeout=60).content)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), str(out)], check=True, timeout=60)
    return out


def probe_duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip() or 0)
