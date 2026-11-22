"""Publish-time email: the Shorts going public in this slot, with campaign links to submit from a phone.

.github/workflows/digest.yml runs this every 30 minutes; it only sends when uploaded clips have reached (or are
within 15 minutes of) their publish time and haven't been emailed yet - i.e. at the daily slot (about 23:30 IST on
weekdays, 20:30 IST on weekends; see posting.schedule). Nothing is sent while publisher is dry_run.
Sent with the Gmail API from the channel owner's own Gmail (scope gmail.send, secret GMAIL_REFRESH_TOKEN)
to the address in the DIGEST_TO secret, so no address is stored in this public repo.
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
    live = parse_iso(p.get("publish_at") or p["posted_at"]).astimezone(IST).strftime("%H:%M IST")
    pay = f'${c["cpm"]:.2f} per 1K views' if c.get("type") == "cpm" and c.get("cpm") else (
        f'${c.get("pool_usd", 0):,.0f} pool' if c.get("type") == "pool" else "")
    vid, short = p.get("video_id"), p.get("url")
    parts = [f'<div style="border:1px solid #ddd;border-radius:12px;padding:14px;margin:14px 0">',
             f'<div style="font-size:13px;color:#666">#{i} · {e(c.get("name", p["campaign_id"]))}'
             f'{" · " + e(pay) if pay else ""}</div>',
             f'<div style="font-size:17px;font-weight:700;margin:6px 0 4px">{e(p.get("title", ""))}</div>',
             f'<div style="font-size:14px;margin-bottom:8px">Goes public at <b>{live}</b> - submit right after.</div>']
    if vid:
        parts += [_btn(f"https://youtube.com/shorts/{vid}", "1. Check it's live"),
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


def _due(p: dict, horizon: dt.datetime) -> bool:
    if not p.get("video_id") or p.get("privacy") not in ("public", "scheduled"):
        return False
    return parse_iso(p.get("publish_at") or p["posted_at"]) <= horizon


def run(force: bool = False) -> None:
    cfg = settings()
    if not (cfg.get("digest") or {}).get("enabled", True) and not force:
        return
    if cfg["posting"].get("publisher") == "dry_run" and not force:
        log.info("digest: publisher is dry_run - nothing is published, no email")
        return
    to = env("DIGEST_TO")
    if not (to and env("GMAIL_REFRESH_TOKEN")):
        log.warning("digest: DIGEST_TO / GMAIL_REFRESH_TOKEN not set - no email sent")
        return
    st = load_state("digest.json", {})
    sent = set(st.get("sent", []))
    camp = {c["id"]: c for c in campaigns(enabled_only=False)}
    horizon = now() + dt.timedelta(minutes=15)
    items = [p for p in load_state("posted.json", []) if p.get("file") not in sent and _due(p, horizon)]
    items.sort(key=lambda p: p.get("publish_at") or p["posted_at"])
    if not items and not force:
        log.info("digest: nothing due")
        return

    when = now().astimezone(IST).strftime("%a %d %b")
    subject = (f"Submit {len(items)} GTA 6 Short{'s' if len(items) != 1 else ''} now · {when}" if items
               else f"Digest test · {when}")
    intro = ("These Shorts are going public on your channel now. As each one goes live, tap "
             "Submit to campaign and paste its link (most campaigns want it within 30 minutes).") if items else (
             "Test email: the pipeline can reach your inbox. Real ones arrive at the daily publish slot.")
    body = "".join(_card(i + 1, p, camp.get(p["campaign_id"], {"name": p["campaign_id"]})) for i, p in enumerate(items))
    page = (f'<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:640px;margin:auto">'
            f'<h2 style="margin:0 0 6px">Zukakito · {html.escape(subject)}</h2>'
            f'<p style="color:#333">{html.escape(intro)}</p>{body}'
            f'<p style="font-size:12px;color:#888">Sent by the gta6-clipper pipeline.</p></div>')

    msg = MIMEText(page, "html", "utf-8")
    msg["to"], msg["subject"] = to, subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    _gmail().users().messages().send(userId="me", body={"raw": raw}).execute()
    st["sent"] = (st.get("sent", []) + [p.get("file") for p in items])[-500:]
    st["last_sent_at"], st["last_count"] = iso(), len(items)
    save_state("digest.json", st)
    log.info("digest: sent %d clips", len(items))


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
