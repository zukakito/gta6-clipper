"""Private storage for finished clips in the channel owner's Google Drive.

Every clip that passes all checks is uploaded to
    <drive_folder>/<YYYY-MM-DD>/<clip>.mp4   (+ <clip>.txt with title, description, tags, campaign, source)
so nothing is exposed through the public GitHub repo.

Uses the same OAuth client and refresh token as the YouTube uploader; the token must include the
drive.file scope (run tools/get_youtube_token.py again). drive.file only lets the app see files it created.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from .common import env, log

FOLDER_MIME = "application/vnd.google-apps.folder"
_svc = None
_folders: dict[tuple[str, str], str] = {}


def _service():
    global _svc
    if _svc is None:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(token=None, refresh_token=env("YT_REFRESH_TOKEN"), client_id=env("YT_CLIENT_ID"),
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
    return bool(env("YT_REFRESH_TOKEN") and env("YT_CLIENT_ID") and env("YT_CLIENT_SECRET"))


def save(clip: Path, meta: dict, campaign: dict, source: dict, root_name: str = "Zukakito Clips") -> str | None:
    """Upload the clip + a sidecar text file. Returns the Drive link of the clip, or None if storage failed."""
    if not enabled():
        log.warning("drive: no Google credentials - clip not stored")
        return None
    try:
        from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload

        svc = _service()
        day = _folder(dt.datetime.utcnow().strftime("%Y-%m-%d"), _folder(root_name, None))
        media = MediaFileUpload(str(clip), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        req = svc.files().create(body={"name": clip.name, "parents": [day]}, media_body=media, fields="id,webViewLink")
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        note = "\n".join([
            f"TITLE: {meta.get('title', '')}",
            "",
            "DESCRIPTION:",
            meta.get("description", ""),
            "",
            f"TAGS: {', '.join(meta.get('tags') or [])}",
            f"PAID PROMOTION: {'yes' if meta.get('paid') else 'no'}",
            f"PLAYLIST: {meta.get('playlist', '')}",
            f"CAMPAIGN: {campaign.get('name', campaign.get('id'))}",
            f"SUBMIT AT: {campaign.get('join_url', '')}",
            f"SOURCE: {source.get('vod_url', '')} at {source.get('offset', 0):.0f}s",
            f"HOOK: {meta.get('hook', '')}",
        ])
        svc.files().create(body={"name": clip.with_suffix(".txt").name, "parents": [day]},
                           media_body=MediaInMemoryUpload(note.encode("utf-8"), mimetype="text/plain"),
                           fields="id").execute()
        log.info("drive: stored %s -> %s", clip.name, resp.get("webViewLink"))
        return resp.get("webViewLink")
    except Exception as e:  # noqa: BLE001
        log.warning("drive: could not store %s (token may lack the drive.file scope): %s", clip.name, str(e)[:300])
        return None
