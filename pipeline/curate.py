"""Pick the exact cut and write title/description/tags that follow each campaign brief."""
from __future__ import annotations

import re

from . import llm
from .common import log

CURATE_PROMPT = """You edit YouTube Shorts from authorised GTA 6 livestream footage for a paid clipping campaign.

CAMPAIGN: {name}
STREAMER: {streamer}
MUST INCLUDE IN TITLE OR DESCRIPTION: {must}
CAMPAIGN RULES (obey strictly): {rules}

Below is a timestamped transcript (seconds from the start of the downloaded window, window length {dur:.0f}s).
A chat/audio spike happened near second {peak:.0f}.

{transcript}

Decide:
1. keep: false if this moment breaks any rule (streamer not the focus, negative portrayal of the streamer,
   slurs/hate, sexual content, real-world personal info, nothing happens) - otherwise true.
2. start, end: the best self-contained cut between {minlen} and {maxlen} seconds. Start right before the
   setup (strong first 2 seconds), end right after the payoff/reaction. Use transcript times.
3. hook: on-screen headline, max 7 words, punchy, no emojis, no clickbait lies, no spoilers of major story twists.
4. title: YouTube title under 90 chars that includes the streamer's name and ends with " #shorts".
   Include required hashtags if they fit.
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


def curate(campaign: dict, words: list[dict], window_len: float, peak_rel: float, cfg: dict) -> dict:
    streamer = campaign.get("overlay_text") or (campaign.get("must_include") or [campaign["name"]])[0]
    streamer = streamer.title() if streamer.isupper() else streamer
    must = campaign.get("must_include") or []
    minlen, maxlen = cfg["short_min_sec"], cfg["short_max_sec"]

    res = None
    if llm.available():
        res = llm.ask(CURATE_PROMPT.format(
            name=campaign["name"], streamer=streamer, must=", ".join(must) or "none",
            rules=(campaign.get("rules") or "none").strip(), dur=window_len, peak=peak_rel,
            transcript=_transcript_text(words), minlen=minlen, maxlen=maxlen), want_json=True, temperature=0.5)

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

    title = re.sub(r"\s+", " ", res.get("title") or "").strip()
    if "#shorts" not in title.lower():
        title = (title[:88] + " #shorts").strip()
    res["title"] = _ensure(title, [m for m in must if not m.startswith("#")], 100)[:100]
    desc = res.get("description") or ""
    res["description"] = _ensure(desc, must, 4800) + "\n\n#GTA6 #GTAVI #shorts"
    if campaign.get("paid_disclosure"):
        res["description"] = "Paid partnership / clipping campaign.\n" + res["description"]
    res["tags"] = list(dict.fromkeys([t.lower() for t in (res.get("tags") or [])] + ["gta 6", "gta vi"]))[:15]
    log.info("curate: keep=%s %s [%.1f-%.1f] %s", res.get("keep"), campaign["id"], s, e, res["title"])
    return res
