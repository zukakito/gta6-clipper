"""Find recent authorised VODs for each campaign's source channels (Twitch, Kick, YouTube)."""
from __future__ import annotations

import datetime as dt

import os
import re

import requests
import yt_dlp

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


def _hls_duration(info: dict) -> float:
    """Kick VODs often come back without a duration: add up the segment lengths of the HLS playlist."""
    fmts = [f for f in (info.get("formats") or []) if "m3u8" in (f.get("protocol") or "") or ".m3u8" in (f.get("url") or "")]
    if not fmts and (info.get("url") or "").find(".m3u8") > 0:
        fmts = [info]
    for f in sorted(fmts, key=lambda f: f.get("tbr") or 0)[:2]:
        try:
            r = requests.get(f["url"], headers=f.get("http_headers") or {}, timeout=30)
            r.raise_for_status()
            total = sum(float(x) for x in re.findall(r"#EXTINF:([\d.]+)", r.text))
            if total > 0:
                return total
        except Exception as ex:  # noqa: BLE001
            log.warning("sources: playlist duration failed for %s: %s", info.get("id"), str(ex)[:200])
    return 0.0


def recent_vods(src: str, lookback_hours: int, limit: int = 4, since: str | None = None) -> list[dict]:
    """Return [{id, url, title, duration, started_at(datetime|None), is_live}] newest first."""
    entries = []
    for url in dict.fromkeys([_listing_url(src), src.rstrip("/") + "/videos"]):
        try:
            with yt_dlp.YoutubeDL({**YDL_BASE, "ignoreerrors": False, "extract_flat": "in_playlist",
                                   "playlistend": limit * 2}) as ydl:
                listing = ydl.extract_info(url, download=False) or {}
            entries = [e for e in (listing.get("entries") or []) if e][: limit * 2]
        except Exception as ex:  # noqa: BLE001
            log.warning("sources: listing %s failed: %s", url, str(ex)[:300])
        if entries:
            break

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
                log.warning("sources: could not read VOD %s (bot check, sub-only or age gate?)", vurl)
                continue
            ts = info.get("timestamp") or info.get("release_timestamp")
            started = dt.datetime.fromtimestamp(ts, dt.timezone.utc) if ts else None
            if started and started < cutoff:
                continue
            if info.get("live_status") == "is_upcoming":
                continue
            duration = float(info.get("duration") or 0)
            is_live = info.get("live_status") == "is_live" or bool(info.get("is_live"))
            if duration <= 0 and not is_live:
                duration = _hls_duration(info)
                log.info("sources: %s had no duration; playlist says %.0fs", info.get("id"), duration)
            out.append({
                "id": f'{info.get("extractor_key", "x").lower()}:{info.get("id")}',
                "url": info.get("webpage_url") or vurl,
                "title": info.get("title") or "",
                "duration": duration,
                "started_at": started.isoformat() if started else None,
                "is_live": is_live,
                "extractor": info.get("extractor_key", ""),
                # Twitch VODs list the game played in each part ("chapters"); used to clip GTA segments only.
                "chapters": [[float(ch.get("start_time") or 0), float(ch.get("end_time") or 0), ch.get("title") or ""]
                             for ch in (info.get("chapters") or [])],
            })
            if len(out) >= limit:
                break
    log.info("sources: %s -> %d recent VOD(s)", src, len(out))
    return out
