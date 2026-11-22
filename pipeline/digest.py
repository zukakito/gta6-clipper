"""Morning email: every Short made since the last digest, ready to publish and submit from a phone.

Runs from .github/workflows/digest.yml at about 05:00 India time (18:30 New York - US prime time).
Sent with the Gmail API from the channel owner's own Gmail (scope gmail.send, secret GMAIL_REFRESH_TOKEN)
to the address in the DIGEST_TO secret, so no address is stored in this public repo.

For each clip the mail has: the private YouTube draft (open it in the YouTube app and set it to Public),
the campaign page to submit the link, the link to paste, and the title/description/tags to copy.
Before uploads are enabled (publisher dry_run) it links the clip's private Drive folder instead.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
from email.mime.text import MIMEText

from .common import campaigns, env, iso, load_state, log, now, parse_iso, save_state, settings

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _gmail():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(token=None, refresh_token=env("GMAIL_REFRESH_TOKEN"), client_id=env("YT_CLIENT_ID"),
                        client_secret=env("YT_CLIENT_SECRET"), token_uri="https://oauth2.googleapis.com/token")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _btn(url: str, label: str, color: str = "#cc0000") -> str:
    return (f'<a href="{html.escape(url)}" style="display:inline-block;margin:4px 6px 4px 0;padding:10px 14px;'
            f'background:{color};color:#fff;border-radius:8px;text-decoration:none;font-weight:600">{label}</a>')


def _card(i: int, p: dict, c: dict) -> str:
    e = html.escape
    made = parse_iso(p["posted_at"]).astimezone(IST).strftime("%d %b, %H:%M IST")
    pay = f'${c["cpm"]:.2f} per 1K views' if c.get("type") == "cpm" and c.get("cpm") else (
        f'${c.get("pool_usd", 0):,.0f} pool' if c.get("type") == "pool" else "")
    vid, short = p.get("video_id"), p.get("url")
    parts = [f'<div style="border:1px solid #ddd;border-radius:12px;padding:14px;margin:14px 0">',
             f'<div style="font-size:13px;color:#666">#{i} · {e(c.get("name", p["campaign_id"]))}'
             f'{" · " + e(pay) if pay else ""} · made {made}</div>',
             f'<div style="font-size:17px;font-weight:700;margin:6px 0 10px">{e(p.get("title", ""))}</div>']
    if vid:
        parts += [_btn(f"https://youtube.com/shorts/{vid}", "1. Open draft → set Public"),
                  _btn(f"https://studio.youtube.com/video/{vid}/edit", "Studio", "#555")]
    else:
        parts += [_btn(p.get("drive") or "https://drive.google.com", "1. Get video (Drive)", "#1a73e8"),
                  '<div style="font-size:13px;color:#666">Not uploaded yet: download it, upload in the YouTube app '
                  'as a Short, paste the title and description below, turn on "paid promotion".</div>']
    if c.get("join_url"):
        within = c.get("submit_within_min")
        parts += ["<div>" + _btn(c["join_url"], "2. Submit to campaign", "#0a7a3d") +
                  (f'<span style="font-size:13px;color:#b00020">within {within} min of publishing</span>' if within else "") +
                  "</div>"]
    if short:
        parts += [f'<div style="font-size:13px;margin-top:6px">Link to submit: <b>{e(short)}</b></div>']
    parts += ['<div style="font-size:12px;color:#666;margin-top:8px">Title, description, tags (copy):</div>',
              f'<pre style="white-space:pre-wrap;font-size:13px;background:#f6f6f6;padding:8px;border-radius:8px;margin:4px 0 0">'
              f'{e(p.get("title", ""))}\n\n{e(p.get("description", ""))}\n\n{e(", ".join(p.get("tags") or []))}</pre>',
              "</div>"]
    return "".join(parts)


def run(force: bool = False) -> None:
    cfg = settings()
    if not (cfg.get("digest") or {}).get("enabled", True) and not force:
        return
    to = env("DIGEST_TO")
    if not (to and env("GMAIL_REFRESH_TOKEN")):
        log.warning("digest: DIGEST_TO / GMAIL_REFRESH_TOKEN not set - no email sent")
        return
    st = load_state("digest.json", {})
    since = parse_iso(st["last_sent_at"]) if st.get("last_sent_at") else now() - dt.timedelta(days=1)
    camp = {c["id"]: c for c in campaigns(enabled_only=False)}
    items = [p for p in load_state("posted.json", []) if parse_iso(p["posted_at"]) > since]

    today = now().astimezone(IST).strftime("%a %d %b")
    if items:
        subject = f"{len(items)} GTA 6 Short{'s' if len(items) != 1 else ''} ready to publish · {today}"
        intro = ("Publish them one by one, about 10 minutes apart (it's US prime time now). "
                 "Right after each one goes Public, submit its link to the campaign.")
        if cfg["posting"].get("publisher") == "dry_run":
            intro = ("Uploads to YouTube aren't switched on yet, so these are in your private Drive. "
                     "Upload each in the YouTube app, then submit its link to the campaign.")
        body = "".join(_card(i + 1, p, camp.get(p["campaign_id"], {"name": p["campaign_id"]})) for i, p in enumerate(items))
    else:
        subject = f"No new GTA 6 Shorts this morning · {today}"
        intro = "The pipeline ran but no clip passed the quality checks since the last email."
        body = ""
    page = (f'<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:640px;margin:auto">'
            f'<h2 style="margin:0 0 6px">Zukakito · {html.escape(subject)}</h2>'
            f'<p style="color:#333">{html.escape(intro)}</p>{body}'
            f'<p style="font-size:12px;color:#888">Sent by the gta6-clipper pipeline.</p></div>')

    msg = MIMEText(page, "html", "utf-8")
    msg["to"], msg["subject"] = to, subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    _gmail().users().messages().send(userId="me", body={"raw": raw}).execute()
    st["last_sent_at"], st["last_count"] = iso(), len(items)
    save_state("digest.json", st)
    log.info("digest: sent %d clips", len(items))


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
