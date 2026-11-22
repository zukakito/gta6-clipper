"""Submit a posted clip's link to its campaign.

Automatic path: Whop API (POST /bounty_submissions) with a *user* OAuth token that has the
`bounty:submission:create` scope (WHOP_USER_TOKEN) and a `whop_bounty_id` on the campaign.
Whop's docs say account API keys cannot author submissions, so a plain API key will not work.

Anything that cannot be submitted automatically is written to state/pending_submissions.json with a
deadline. The Claude monitor submits those through your logged-in browser, and the README explains
how to do it by hand in under a minute.
"""
from __future__ import annotations

import datetime as dt

import requests

from .common import env, iso, load_state, log, now, parse_iso, save_state

WHOP_API = "https://api.whop.com/api/v1/bounty_submissions"


def _whop(campaign: dict, url: str) -> dict:
    token = env("WHOP_USER_TOKEN")
    r = requests.post(WHOP_API, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                      json={"bounty_id": campaign["whop_bounty_id"], "deliverable": {"urls": [url]}}, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"whop {r.status_code}: {r.text[:300]}")
    return r.json()


def submit(campaign: dict, post: dict) -> str:
    url = post.get("url")
    if not url or post.get("privacy") not in ("public", "scheduled", None):
        return "skipped"  # dry runs / private uploads are never submitted
    live_at = parse_iso(post["publish_at"]) if post.get("publish_at") else now()   # scheduled: count from going public
    if campaign.get("whop_bounty_id") and env("WHOP_USER_TOKEN") and live_at <= now():
        try:
            res = _whop(campaign, url)
            log.info("submit: whop accepted %s -> %s", url, res.get("id"))
            return "submitted"
        except Exception as e:  # noqa: BLE001
            log.warning("submit: whop API failed, queueing for browser submission: %s", e)
    pending = load_state("pending_submissions.json", [])
    pending.append({
        "campaign_id": campaign["id"], "campaign": campaign["name"], "join_url": campaign.get("join_url"),
        "video_url": url, "posted_at": iso(live_at), "submit_after": iso(live_at),
        "deadline": iso(live_at + dt.timedelta(minutes=int(campaign.get("submit_within_min", 30)))),
        "status": "pending",
    })
    save_state("pending_submissions.json", pending)
    return "queued"
