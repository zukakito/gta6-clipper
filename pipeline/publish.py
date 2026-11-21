"""Upload a rendered clip to YouTube.

publisher = youtube_api  -> your Google Cloud project (needs YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN).
                            Until Google approves the project's API audit, YouTube forces these uploads to private.
publisher = upload_post  -> Upload-Post free plan (UPLOAD_POST_API_KEY, UPLOAD_POST_USER). 10 uploads/month free.
publisher = dry_run      -> nothing is uploaded to YouTube; the clip is still stored privately in Google Drive.
"""
from __future__ import annotations

from pathlib import Path

import requests

from .common import env, log


class PublishError(RuntimeError):
    pass


def _youtube_api(path: Path, meta: dict, privacy: str, category: int) -> dict:
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
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False, "containsSyntheticMedia": False},
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
    status = resp.get("status", {}).get("privacyStatus")
    if privacy == "public" and status != "public":
        log.warning("YouTube set the video to %s - the API project is probably not audited yet", status)
    if meta.get("playlist"):
        _add_to_playlist(yt, vid, meta["playlist"])
    return {"video_id": vid, "url": f"https://www.youtube.com/shorts/{vid}", "privacy": status}


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
        return _youtube_api(path, meta, p.get("privacy", "public"), category)
    if mode == "upload_post":
        return _upload_post(path, meta, p.get("privacy", "public"), category)
    raise PublishError(f"unknown publisher {mode}")
