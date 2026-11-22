"""Upload a rendered clip to YouTube.

publisher = youtube_api  -> your Google Cloud project (needs YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN).
                            Until Google approves the project's API audit, YouTube forces these uploads to private.
publisher = upload_post  -> Upload-Post free plan (UPLOAD_POST_API_KEY, UPLOAD_POST_USER). 10 uploads/month free.
publisher = dry_run      -> nothing is uploaded to YouTube; the clip is still stored privately in Google Drive.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .common import env, iso, load_state, log, now, parse_iso, save_state


class PublishError(RuntimeError):
    pass


def publish_time(p: dict) -> dt.datetime | None:
    """When a new Short should go public (UTC), or None to publish immediately.

    mode "slots" (default): one daily slot in New York time, chosen to suit the owner in India -
    weekdays 13:00 ET (about 23:30 IST: US lunchtime + UK evening), weekends 10:00 ET (about 20:30 IST: US
    weekend morning). Each clip goes into the next slot that is at least `lead_min` away; clips in the same
    slot go public `spacing_min` apart. The morning-digest workflow emails the campaign links at that time.
    mode "window": inside the window a clip goes public right away, otherwise at the next window start.
    """
    sch = p.get("schedule") or {}
    if not sch.get("enabled"):
        return None
    tz = ZoneInfo(sch.get("tz", "America/New_York"))
    earliest = (now() + dt.timedelta(minutes=sch.get("lead_min", 10))).astimezone(tz)
    hm = lambda s: tuple(map(int, s.split(":")))   # noqa: E731
    st = load_state("schedule.json", {})
    if sch.get("mode", "slots") == "slots":
        slots = sch.get("slots", {"weekday": "13:00", "weekend": "10:00"})
        for d in range(8):
            day = (earliest + dt.timedelta(days=d)).date()
            h, m = hm(slots["weekend" if day.weekday() >= 5 else "weekday"])
            base = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=tz)
            if base >= earliest:
                break
        base = base.astimezone(dt.timezone.utc)
        n = st.get("slot_count", 0) if st.get("slot") == iso(base) else 0
        when = base + dt.timedelta(minutes=sch.get("spacing_min", 5) * n)
        st.update(slot=iso(base), slot_count=n + 1, last_publish_at=iso(when))
        save_state("schedule.json", st)
        return when
    (h0, m0), (h1, m1) = hm(sch.get("window", ["11:00", "22:00"])[0]), hm(sch.get("window", ["11:00", "22:00"])[1])
    start = earliest.replace(hour=h0, minute=m0, second=0, microsecond=0)
    end = earliest.replace(hour=h1, minute=m1, second=0, microsecond=0)
    when = start if earliest < start else (start + dt.timedelta(days=1) if earliest > end else earliest)
    when = when.astimezone(dt.timezone.utc)
    last = parse_iso(st["last_publish_at"]) if st.get("last_publish_at") else None
    if last and when < last + dt.timedelta(minutes=p.get("min_minutes_between_uploads", 45)):
        when = last + dt.timedelta(minutes=p.get("min_minutes_between_uploads", 45))
    st["last_publish_at"] = iso(when)
    save_state("schedule.json", st)
    return when


def _youtube_api(path: Path, meta: dict, privacy: str, category: int, publish_at: dt.datetime | None = None) -> dict:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        token=None,
        refresh_token=env("YT_REFRESH_TOKEN"),
        client_id=env("YT_CLIENT_ID"),
        client_secret=env("YT_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
    )
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {"title": meta["title"], "description": meta["description"], "tags": meta["tags"],
                    "categoryId": str(category), "defaultLanguage": "en", "defaultAudioLanguage": "en"},
        "status": {"privacyStatus": "private" if publish_at else privacy, "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": False, **({"publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")} if publish_at else {})},
        "paidProductPlacementDetails": {"hasPaidProductPlacement": bool(meta.get("paid"))},
    }
    parts = "snippet,status,paidProductPlacementDetails"
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    try:
        req = yt.videos().insert(part=parts, body=body, media_body=media)
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
    except Exception as e:  # noqa: BLE001 - older API surfaces may reject the paid-placement part
        if "paidProductPlacementDetails" not in str(e):
            raise
        body.pop("paidProductPlacementDetails")
        media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
    vid = resp["id"]
    st = resp.get("status", {})
    status = st.get("privacyStatus")
    if publish_at and status == "private" and st.get("publishAt"):
        status = "scheduled"
        log.info("publish: %s scheduled to go public at %s", vid, st.get("publishAt"))
    elif privacy == "public" and status != "public":
        log.warning("YouTube set the video to %s - the API project is probably not audited yet", status)
    if meta.get("playlist"):
        _add_to_playlist(yt, vid, meta["playlist"])
    return {"video_id": vid, "url": f"https://www.youtube.com/shorts/{vid}", "privacy": status,
            **({"publish_at": iso(publish_at)} if status == "scheduled" else {})}


_playlists: dict[str, str] = {}


def _add_to_playlist(yt, video_id: str, title: str) -> None:
    """File the upload into the streamer's playlist (e.g. "Kai Cenat GTA 6 Moments"). Best effort:
    needs the youtube.force-ssl scope on the refresh token; costs ~51 quota units."""
    try:
        if not _playlists:
            req = yt.playlists().list(part="snippet", mine=True, maxResults=50)
            while req is not None:
                resp = req.execute()
                for it in resp.get("items", []):
                    _playlists[it["snippet"]["title"].strip().lower()] = it["id"]
                req = yt.playlists().list_next(req, resp)
        pid = _playlists.get(title.strip().lower())
        if not pid:
            log.info("playlist %r not found on the channel - skipping", title)
            return
        yt.playlistItems().insert(part="snippet", body={"snippet": {
            "playlistId": pid, "resourceId": {"kind": "youtube#video", "videoId": video_id}}}).execute()
        log.info("added %s to playlist %r", video_id, title)
    except Exception as e:  # noqa: BLE001
        log.warning("could not add %s to playlist %r (token may lack youtube.force-ssl scope): %s", video_id, title, str(e)[:200])


def _upload_post(path: Path, meta: dict, privacy: str, category: int) -> dict:
    key, user = env("UPLOAD_POST_API_KEY"), env("UPLOAD_POST_USER")
    if not key or not user:
        raise PublishError("UPLOAD_POST_API_KEY / UPLOAD_POST_USER not set")
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.upload-post.com/api/upload",
            headers={"Authorization": f"Apikey {key}"},
            data={"user": user, "platform[]": "youtube", "title": meta["title"],
                  "youtube_description": meta["description"], "youtube_tags": ",".join(meta["tags"]),
                  "youtube_visibility": privacy, "youtube_category_id": str(category)},
            files={"video": (path.name, f, "video/mp4")}, timeout=900)
    if r.status_code >= 300:
        raise PublishError(f"upload-post {r.status_code}: {r.text[:300]}")
    data = r.json()
    url = None
    for v in (data.get("results") or {}).values() if isinstance(data.get("results"), dict) else []:
        url = v.get("url") or url
    return {"video_id": None, "url": url, "privacy": privacy, "provider_response": data}


def publish(path: Path, meta: dict, settings: dict) -> dict | None:
    p = settings["posting"]
    mode = p.get("publisher", "dry_run")
    category = settings["channel"].get("default_category_id", 20)
    if mode == "dry_run":
        log.info("publish: DRY RUN - %s (%s)", meta["title"], path.name)
        return {"video_id": None, "url": None, "privacy": "dry_run"}
    if mode == "youtube_api":
        if not env("YT_REFRESH_TOKEN"):
            raise PublishError("YT_REFRESH_TOKEN not set - run tools/get_youtube_token.py once")
        privacy = p.get("privacy", "public")
        when = publish_time(p) if privacy == "public" else None
        return _youtube_api(path, meta, privacy, category, when)
    if mode == "upload_post":
        return _upload_post(path, meta, p.get("privacy", "public"), category)
    raise PublishError(f"unknown publisher {mode}")
