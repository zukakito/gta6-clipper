"""Scout: discover new GTA 6 clipping campaigns on public campaign boards.

Writes state/discovered.json (every campaign ever seen, with first/last seen
times and parsed details). New YouTube-eligible campaigns are flagged
`needs_review` so the Claude monitor (or you) can read the full brief and add
them to config/campaigns.yaml. Nothing here posts anything.
"""
from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

from . import llm
from .common import campaigns, iso, load_state, log, save_state, settings

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
CAMPAIGN_LINK = re.compile(r"(whop\.com/experiences/[^\s\"'<>]+/campaigns/[0-9a-f-]{36}|vyro\.com/[^\s\"'<>]*campaign[^\s\"'<>]*|clipping\.net/[^\s\"'<>]*campaign[^\s\"'<>]*)", re.I)


def fetch(url: str) -> tuple[str, list[str]]:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    links = sorted({m.group(0) for m in CAMPAIGN_LINK.finditer(r.text)})
    return text[:60000], links


EXTRACT_PROMPT = """You are reading a public listing of paid video-clipping campaigns.
Return a JSON array. One object per campaign that is about GTA 6 / GTA VI (or a GTA 6 streamer
marathon). Fields:
  name (string), host (string|null), board (content_rewards|vyro|clipster|clipstake|clipping_net|other),
  type ("cpm"|"pool"|"unknown"), cpm (number|null, USD per 1000 views), budget_left_usd (number|null),
  platforms (array of: youtube, tiktok, instagram, x, facebook), apply_first (bool),
  streamer (string|null), url (string|null - campaign link if visible), status ("open"|"paid_out"|"unknown").
Only include campaigns you can actually see in the text. No commentary.

LINKS FOUND ON PAGE:
{links}

PAGE TEXT:
{text}
"""


def key_for(c: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (c.get("name") or "").lower()).strip("-")[:80]


def run() -> dict:
    cfg = settings()["scout"]
    kw = [k.lower() for k in cfg["keywords"]]
    seen: dict = load_state("discovered.json", {})
    configured = {c["name"].lower() for c in campaigns(enabled_only=False)}
    new_items = []

    for board in cfg["boards"]:
        try:
            text, links = fetch(board)
        except Exception as e:  # noqa: BLE001
            log.warning("scout: could not read %s: %s", board, e)
            continue

        found = llm.ask(EXTRACT_PROMPT.format(links="\n".join(links[:200]), text=text), want_json=True, temperature=0) if llm.available() else None
        if not isinstance(found, list):
            # Fallback without an LLM: record raw campaign links containing GTA keywords nearby.
            found = []
            for ln in links:
                found.append({"name": ln, "url": "https://" + ln if not ln.startswith("http") else ln,
                              "platforms": [], "status": "unknown", "board": "other"})

        for c in found:
            name = (c.get("name") or "").strip()
            if not name:
                continue
            if llm.available() is False and not any(k in text.lower() for k in kw):
                continue
            k = key_for(c)
            entry = seen.get(k)
            if entry is None:
                entry = {**c, "first_seen": iso(), "source_board": board,
                         "needs_review": "youtube" in (c.get("platforms") or []) and name.lower() not in configured}
                seen[k] = entry
                new_items.append(entry)
                log.info("scout: NEW campaign %s (%s)", name, board)
            else:
                entry.update({kk: vv for kk, vv in c.items() if vv not in (None, [], "")})
            entry["last_seen"] = iso()

    save_state("discovered.json", seen)
    yt_open = [v for v in seen.values() if "youtube" in (v.get("platforms") or []) and v.get("status") != "paid_out"]
    summary = {"checked_at": iso(), "new": len(new_items), "known": len(seen), "youtube_open": len(yt_open),
               "needs_review": [v["name"] for v in seen.values() if v.get("needs_review")]}
    save_state("scout_summary.json", summary)
    log.info("scout summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
