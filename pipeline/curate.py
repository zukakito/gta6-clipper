"""Pick the exact cut and write title/description/tags that follow each campaign brief."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import llm
from .common import log

MIN_WORDS = 15   # a clip with less speech than this is almost never a real highlight

def frames(src: Path, start: float, end: float, n: int = 3) -> list[bytes]:
    out = []
    for i in range(n):
        t = start + (end - start) * (i + 1) / (n + 1)
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                            "-vf", "scale=640:-2", "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", "-"],
                           capture_output=True, timeout=60)
        if r.returncode == 0 and r.stdout:
            out.append(r.stdout)
    return out


def quality_gate(campaign: dict, src: Path, words: list[dict], start: float, end: float) -> tuple[bool, str]:
    """Speech check on the final cut (the visual check happens inside curate)."""
    spoken = [w for w in words if start <= w["s"] <= end]
    if len(spoken) < MIN_WORDS:
        return False, f"too little speech ({len(spoken)} words)"
    return True, "ok"


CURATE_PROMPT = """You edit YouTube Shorts from authorised GTA 6 livestream footage for a paid clipping campaign.

CAMPAIGN: {name}
STREAMER: {streamer}
MUST INCLUDE IN TITLE OR DESCRIPTION: {must}
CAMPAIGN RULES (obey strictly): {rules}

Below is a timestamped transcript (seconds from the start of the downloaded window, window length {dur:.0f}s).
A chat/audio spike happened near second {peak:.0f}.

{transcript}

If frames from this window are attached as images, look at them carefully.

Decide:
1. keep: false if ANY of these is true: the stream is idle/AFK/sleeping/a dark or empty room/BRB or starting screen;
   no GTA 6 gameplay is happening; the streamer is not part of the moment; sexual content, nudity or strip-club
   scenes (even if covered by an emoji or censor graphic); graphic gore; slurs/hate; real-world personal info;
   negative portrayal of the streamer; nothing interesting happens (no clear setup and payoff); or anything breaks
   the campaign rules. Otherwise true. Be strict - campaign reviewers reject weak clips.
2. start, end: the best self-contained cut between {minlen} and {maxlen} seconds. Start right before the
   setup (strong first 2 seconds), end right after the payoff/reaction. Use transcript times.
3. hook: on-screen headline, max 6 words, punchy and specific to what happens, no emojis, no clickbait lies,
   no spoilers of major story twists.
4. title: YouTube title under 70 chars, specific and curiosity-driven, includes the streamer's name, ends with " #shorts".
5. description: 2 short lines describing the moment + required names/hashtags + "Clip from {streamer}'s GTA 6 stream."
6. tags: 8-12 lowercase tags.
Return JSON: {{"keep": bool, "reason": str, "start": number, "end": number, "hook": str, "title": str, "description": str, "tags": [str]}}
"""


def _transcript_text(words: list[dict]) -> str:
    lines, cur, t0 = [], [], None
    for w in words:
        if t0 is None:
            t0 = w["s"]
        cur.append(w["w"])
        if w["w"].endswith((".", "?", "!")) or len(cur) >= 14:
            lines.append(f"[{t0:6.1f}] {' '.join(cur)}")
            cur, t0 = [], None
    if cur:
        lines.append(f"[{t0:6.1f}] {' '.join(cur)}")
    return "\n".join(lines) or "(no speech detected)"


def _ensure(text: str, required: list[str], limit: int) -> str:
    for r in required:
        if r.lower() not in text.lower():
            candidate = f"{text} {r}".strip()
            if len(candidate) <= limit:
                text = candidate
    return text


def streamer_name(campaign: dict) -> str:
    s = campaign.get("streamer") or campaign.get("overlay_text") or (campaign.get("must_include") or [campaign["name"]])[0]
    return s.title() if s.isupper() else s


def curate(campaign: dict, words: list[dict], window_len: float, peak_rel: float, cfg: dict,
           images: list[bytes] | None = None, require_llm: bool = True) -> dict:
    """One LLM call that picks the cut and writes metadata (and, with Gemini, also looks at the frames).
    Returns {"defer": True} when no LLM answered and require_llm is set - generic titles hurt reach."""
    streamer = streamer_name(campaign)
    must = campaign.get("must_include") or []
    minlen, maxlen = cfg["short_min_sec"], cfg["short_max_sec"]

    res = None
    if llm.available():
        res = llm.ask(CURATE_PROMPT.format(
            name=campaign["name"], streamer=streamer, must=", ".join(must) or "none",
            rules=(campaign.get("rules") or "none").strip(), dur=window_len, peak=peak_rel,
            transcript=_transcript_text(words), minlen=minlen, maxlen=maxlen), want_json=True, temperature=0.5,
            images=images)
    if require_llm and (not isinstance(res, dict) or "keep" not in res):
        return {"defer": True, "reason": "no LLM available for cut/title - retry next run"}

    if not isinstance(res, dict):
        # Heuristic fallback: centre a ~40 s cut so the spike lands ~60% in.
        length = min(maxlen, max(minlen, 40))
        start = max(0.0, peak_rel - 0.6 * length)
        res = {"keep": True, "reason": "heuristic", "start": start, "end": min(window_len, start + length),
               "hook": f"{streamer} in GTA 6", "title": f"{streamer} GTA 6 moment #shorts",
               "description": f"Clip from {streamer}'s GTA 6 stream.", "tags": ["gta 6", "gta vi", streamer.lower(), "gta 6 clips"]}

    # Clamp to valid ranges no matter what the model said.
    s = max(0.0, float(res.get("start", 0)))
    e = min(window_len, float(res.get("end", s + 40)))
    if e - s < minlen:
        e = min(window_len, s + minlen)
        s = max(0.0, e - minlen)
    if e - s > maxlen:
        e = s + maxlen
    res["start"], res["end"] = round(s, 2), round(e, 2)

    title = re.sub(r"\s*#shorts\s*", " ", res.get("title") or "", flags=re.I)
    title = re.sub(r"\s+", " ", title).strip()
    title = _ensure(title, [m for m in must if not m.startswith("#") and len(m) <= 20 and "." not in m], 90)
    res["title"] = (title[:90] + " #shorts").strip()
    desc = res.get("description") or ""
    res["description"] = _ensure(desc, must, 4800) + "\n\n#GTA6 #GTAVI #shorts"
    if campaign.get("paid_disclosure"):
        res["description"] = "Paid partnership / clipping campaign.\n" + res["description"]
    res["tags"] = list(dict.fromkeys([t.lower() for t in (res.get("tags") or [])] + ["gta 6", "gta vi"]))[:15]
    log.info("curate: keep=%s %s [%.1f-%.1f] %s", res.get("keep"), campaign["id"], s, e, res["title"])
    return res
