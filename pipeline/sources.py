"""Find recent authorised VODs for each campaign's source channels (Twitch, Kick, YouTube)."""
from __future__ import annotations

import datetime as dt

import yt_dlp

import os

from .common import log, now

YDL_BASE = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
if os.environ.get("YTDLP_COOKIES_FILE"):          # optional: helps when YouTube asks "confirm you're not a bot"
    YDL_BASE["cookiefile"] = os.environ["YTDLP_COOKIES_FILE"]


def _listing_url(src: str) -> str:
    s = src.rstrip("/")
    if "twitch.tv" in s and "/videos" not in s:
        return s + "/videos?filter=archives&sort=time"
    if "kick.com" in s and "/videos" not in s:
        return s + "/videos"
    return s


def recent_vods(src: str, lookback_hours: int, limit: int = 4, since: str | None = None) -> list[dict]:
    """Return [{id, url, title, duration, started_at(datetime|None), is_live}] newest first."""
    url = _listing_url(src)
    with yt_dlp.YoutubeDL({**YDL_BASE, "extract_flat": "in_playlist", "playlistend": limit * 2}) as ydl:
        listing = ydl.extract_info(url, download=False) or {}
    entries = [e for e in (listing.get("entries") or []) if e][: limit * 2]

    cutoff = now() - dt.timedelta(hours=lookback_hours)
    if since:
        cutoff = max(cutoff, dt.datetime.fromisoformat(since).replace(tzinfo=dt.timezone.utc))

    out = []
    with yt_dlp.YoutubeDL(YDL_BASE) as ydl:
        for e in entries:
            vurl = e.get("url") or e.get("webpage_url")
            if not vurl:
                continue
            try:
                info = ydl.extract_info(vurl, download=False) or {}
            except Exception as ex:  # noqa: BLE001
                log.warning("sources: cannot read %s: %s", vurl, ex)
                continue
            if not info.get("id"):
                continue  # extraction failed (e.g. YouTube bot check) - skip quietly
            ts = info.get("timestamp") or info.get("release_timestamp")
            started = dt.datetime.fromtimestamp(ts, dt.timezone.utc) if ts else None
            if started and started < cutoff:
                continue
            if info.get("live_status") == "is_upcoming":
                continue
            out.append({
                "id": f'{info.get("extractor_key", "x").lower()}:{info.get("id")}',
                "url": info.get("webpage_url") or vurl,
                "title": info.get("title") or "",
                "duration": float(info.get("duration") or 0),
                "started_at": started.isoformat() if started else None,
                "is_live": info.get("live_status") == "is_live" or bool(info.get("is_live")),
                "extractor": info.get("extractor_key", ""),
            })
            if len(out) >= limit:
                break
    log.info("sources: %s -> %d recent VOD(s)", src, len(out))
    return out
