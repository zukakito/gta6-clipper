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

{vision}

Decide:
1. keep: false if ANY of these is true: the stream is idle/AFK/sleeping/a dark or empty room/BRB or starting screen;
   no GTA 6 gameplay is happening; the streamer is not part of the moment; sexual content, nudity or strip-club
   scenes (even if covered by an emoji or censor graphic); graphic gore; slurs/hate; real-world personal info;
   negative portrayal of the streamer; nothing interesting happens (no clear setup and payoff); the talk is
   mumbled, incoherent or needs outside context to follow; the moment is mostly a loading/fail/black screen; or anything breaks
   the campaign rules. Otherwise true. Be strict - campaign reviewers reject weak clips.
2. start, end: the best self-contained cut between {minlen} and {maxlen} seconds. Start right before the
   setup (strong first 2 seconds), end right after the payoff/reaction. Prefer 30-45 s so a new viewer gets the whole
   story: setup, the moment, and the streamer's reaction. Use transcript times.
3. hook: a clear one-line summary of the scene, max 6 words, that says concretely WHAT HAPPENS (who does what)
   (good: "He robs a store for $13", "Cops lock him inside the store", "Car flips off the bridge";
   bad: "Wait for it", "Crazy moment", "Epic car chase surprise", "Lost him, hilarious reaction",
   "Video tape mystery revealed"). Never use hype words (epic, hilarious, insane, crazy, wild, shock, mystery,
   surprise, revealed, chaos). No emojis, no clickbait, no spoilers of major story twists.
4. title: YouTube title under 70 chars, specific (what actually happens), includes the streamer's name, no hype
   words, ends with " #shorts".
5. description: 2 short lines describing the moment + required names/hashtags + "Clip from {streamer}'s GTA 6 stream."
6. tags: 8-12 lowercase tags.
Return JSON: {{"keep": bool, "reason": str, "start": number, "end": number, "hook": str, "title": str, "description": str, "tags": [str]}}
"""


VISION_NOTE = "Frames from this window are attached as images: look at them carefully."
BLIND_NOTE = ("You cannot see the video, only the transcript. A local check already confirmed the screen is not black "
              "or a loading screen, and idle/static footage is checked again after you, so do NOT reject just because the "
              "transcript doesn't describe "
              "gameplay - streamers talk to chat while they play. Reject for 'no gameplay' only if the talk makes it clear "
              "they are not playing (BRB, eating, sleeping, a long unrelated real-life story for the whole window). "
              "Keep moments where the streamer clearly reacts to something happening (shock, laughter, panic, a chase, "
              "a crash, a fail, a funny exchange).")

# Hard filters on what the model wrote (it sometimes ignores the prompt).
BANNED_TOPIC = re.compile(r"strip[\s-]?club|stripper|lap[\s-]?dance|pole[\s-]?danc|twerk|\bnud(?:e|ity)\b|naked|porn|onlyfans|\bsex(?:y|ual)?\b|brothel|escort", re.I)
HYPE = re.compile(r"\b(?:epic|hilarious|insane|crazy|wild|shock(?:ing|ed)?|mystery|surprise[sd]?|revealed|chaos|chaotic|"
                  r"unbelievable|you won'?t believe|gone wrong|must see|jaw[- ]dropping)\b", re.I)

REWRITE_PROMPT = """Rewrite the hook and title of this GTA 6 YouTube Short so they say concretely what happens.
Streamer: {streamer}
Current hook: {hook}
Current title: {title}
Transcript: {transcript}
Rules: hook max 6 words; title under 70 chars, includes "{streamer}", ends with " #shorts". Never use these words:
epic, hilarious, insane, crazy, wild, shock, mystery, surprise, revealed, chaos. No emojis, no clickbait.
Return JSON: {{"hook": str, "title": str}}"""


def _strip_hype(text: str) -> str:
    text = HYPE.sub("", text or "")
    text = re.sub(r"\s+([,!?:])", r"\1", re.sub(r"\s{2,}", " ", text))
    return text.strip(" ,:-!")


def _fix_wording(res: dict, words: list[dict], streamer: str) -> None:
    """Replace hype/clickbait hooks and titles with a concrete one-line summary."""
    if not (HYPE.search(res.get("hook") or "") or HYPE.search(res.get("title") or "")):
        return
    new = None
    if llm.available():
        new = llm.ask(REWRITE_PROMPT.format(streamer=streamer, hook=res.get("hook"), title=res.get("title"),
                                            transcript=_transcript_text(words)[:3000]), want_json=True, temperature=0.3)
    for k in ("hook", "title"):
        v = (new or {}).get(k) if isinstance(new, dict) else None
        res[k] = v if v and not HYPE.search(v) else _strip_hype(res.get(k) or "")
    log.info("curate: reworded hook/title -> %r / %r", res.get("hook"), res.get("title"))



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
            transcript=_transcript_text(words), minlen=minlen, maxlen=maxlen,
            vision=VISION_NOTE if images and llm.first_provider() == "gemini" else BLIND_NOTE), want_json=True, temperature=0.5,
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

    if not res.get("keep", True):   # rejected: the model often leaves start/end empty
        log.info("curate: keep=False %s: %s", campaign["id"], str(res.get("reason"))[:160])
        return res
    text = " ".join(str(res.get(k) or "") for k in ("title", "hook", "description"))
    if BANNED_TOPIC.search(text):   # the model sometimes keeps these despite the prompt
        log.info("curate: banned topic %s: %s", campaign["id"], res.get("title"))
        return {**res, "keep": False, "reason": f"banned topic (sexual content): {BANNED_TOPIC.search(text).group(0)}"}
    _fix_wording(res, words, streamer)
    # Clamp to valid ranges no matter what the model said.
    def num(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    s = max(0.0, num(res.get("start"), 0.0))
    e = min(window_len, num(res.get("end"), s + 40))
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
