"""Private storage for finished clips in the channel owner's Google Drive.

Every clip that passes all checks gets its own folder (dates and times in India time):
    <drive_folder>/<YYYY-MM-DD>/<HH.MM> <Streamer> - <Title>/
        <Title>.mp4          the finished Short
        post-details.txt     title, description, tags, playlist - ready to copy into YouTube
        campaign.txt         campaign, where to submit, source VOD + timestamp, hook
so nothing is exposed through the public GitHub repo.

Uses the same OAuth client as the YouTube uploader but its own refresh token (DRIVE_REFRESH_TOKEN, scope drive.file):
Google won't grant YouTube (brand-account) and Drive scopes in one sign-in. drive.file only sees files this app created.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from .common import env, log

FOLDER_MIME = "application/vnd.google-apps.folder"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_svc = None
_folders: dict[tuple[str, str], str] = {}


def _service():
    global _svc
    if _svc is None:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(token=None, refresh_token=env("DRIVE_REFRESH_TOKEN") or env("YT_REFRESH_TOKEN"), client_id=env("YT_CLIENT_ID"),
                            client_secret=env("YT_CLIENT_SECRET"), token_uri="https://oauth2.googleapis.com/token")
        _svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _svc


def _folder(name: str, parent: str | None) -> str:
    key = (parent or "root", name)
    if key in _folders:
        return _folders[key]
    svc = _service()
    safe = name.replace("'", "\\'")
    q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed = false and '{parent or 'root'}' in parents"
    found = svc.files().list(q=q, fields="files(id)", pageSize=1, spaces="drive").execute().get("files", [])
    if found:
        fid = found[0]["id"]
    else:
        body = {"name": name, "mimeType": FOLDER_MIME, **({"parents": [parent]} if parent else {})}
        fid = svc.files().create(body=body, fields="id").execute()["id"]
    _folders[key] = fid
    return fid


def enabled() -> bool:
    return bool(env("DRIVE_REFRESH_TOKEN") and env("YT_CLIENT_ID") and env("YT_CLIENT_SECRET"))


def _clean(name: str, limit: int = 80) -> str:
    name = re.sub(r"#\w+", "", name or "")
    name = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", name)
    return re.sub(r"\s+", " ", name).strip(" .-")[:limit].strip() or "clip"


def _text(svc, name: str, body: str, parent: str) -> None:
    from googleapiclient.http import MediaInMemoryUpload

    svc.files().create(body={"name": name, "parents": [parent]},
                       media_body=MediaInMemoryUpload(body.encode("utf-8"), mimetype="text/plain"), fields="id").execute()


def save(clip: Path, meta: dict, campaign: dict, source: dict, root_name: str = "Zukakito Clips") -> str | None:
    """Upload the clip into its own folder with two text files. Returns the clip folder's Drive link, or None."""
    if not enabled():
        log.warning("drive: DRIVE_REFRESH_TOKEN not set - clip not stored")
        return None
    try:
        from googleapiclient.http import MediaFileUpload

        svc = _service()
        t = dt.datetime.now(IST)
        title = _clean(meta.get("title", ""))
        streamer = _clean(meta.get("streamer") or campaign.get("streamer") or campaign.get("name", ""), 30)
        day = _folder(t.strftime("%Y-%m-%d"), _folder(root_name, None))
        body = {"name": f"{t:%H.%M} {streamer} - {title}"[:120], "mimeType": FOLDER_MIME, "parents": [day]}
        folder = svc.files().create(body=body, fields="id,webViewLink").execute()

        media = MediaFileUpload(str(clip), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        req = svc.files().create(body={"name": f"{title}.mp4", "parents": [folder["id"]]}, media_body=media, fields="id")
        resp = None
        while resp is None:
            _, resp = req.next_chunk()

        _text(svc, "post-details.txt", "\n".join([
            "TITLE:", meta.get("title", ""), "",
            "DESCRIPTION:", meta.get("description", ""), "",
            "TAGS:", ", ".join(meta.get("tags") or []), "",
            f"PLAYLIST: {meta.get('playlist', '')}",
            f"PAID PROMOTION LABEL: {'yes' if meta.get('paid') else 'no'}",
            "",
            f"AUTO REVIEW: {(meta.get('review') or {}).get('verdict', 'n/a')} "
            f"(score {(meta.get('review') or {}).get('score', '-')}): {(meta.get('review') or {}).get('reason', '')}",
            f"WHAT HAPPENS: {meta.get('what_happens', '')}",
        ]), folder["id"])
        _text(svc, "campaign.txt", "\n".join([
            f"CAMPAIGN: {campaign.get('name', campaign.get('id'))}",
            f"SUBMIT AT: {campaign.get('join_url', '')}",
            f"SOURCE: {source.get('vod_url', '')} at {source.get('offset', 0):.0f}s",
            f"HOOK: {meta.get('hook', '')}",
            f"MADE: {t:%Y-%m-%d %H:%M} IST",
        ]), folder["id"])
        log.info("drive: stored %s -> %s", clip.name, folder.get("webViewLink"))
        return folder.get("webViewLink")
    except Exception as e:  # noqa: BLE001
        log.warning("drive: could not store %s (token may lack the drive.file scope): %s", clip.name, str(e)[:300])
        return None
