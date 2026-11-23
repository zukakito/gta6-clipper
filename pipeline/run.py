"""One pipeline pass: scout -> find moments -> render -> upload -> submit. Safe to run every 30 min.

python -m pipeline.run                 # normal pass
python -m pipeline.run --scout-only    # only refresh campaign discovery
python -m pipeline.run --no-scout
"""
from __future__ import annotations

import argparse
import re
import datetime as dt
import time
import traceback

from . import curate, drive, highlights, llm, publish, render, safety, scout, sources, submit
from .common import OUT, campaigns, iso, load_state, log, now, parse_iso, save_state, settings, slug

RUN_BUDGET_SEC = 50 * 60
SCAN_CHUNK_SEC = 3600           # analyse at most 1 h of new VOD per VOD per run
FIRST_LOOK_SEC = 6 * 3600       # on first sight of a long marathon VOD, start 6 h from the end
MAX_LAG_SEC = 3 * 3600          # if we fall more than 3 h behind a live VOD, jump to the newest hour
LIVE_EDGE_SEC = 90              # don't analyse the last 90 s of a live VOD (still being written)


def _posted_today(posted: list[dict]) -> list[dict]:
    day_ago = now() - dt.timedelta(hours=24)
    return [p for p in posted if p.get("privacy") != "dry_run" and parse_iso(p["posted_at"]) > day_ago]


def _campaign_priority(c: dict) -> float:
    if c.get("type") == "pool":
        return 2.0 + (c.get("pool_usd", 0) / 10000)
    return float(c.get("cpm") or 0.5)


GTA_RE = re.compile(r"grand\s*theft\s*auto|\bgta\b|\bgta\s*(?:vi|6)\b", re.I)


def _gta_ranges(v: dict) -> list[tuple[float, float]] | None:
    """Time ranges of the VOD where GTA was being played, from its game chapters.
    None = the VOD has no chapter info (Kick, YouTube): allow everything and rely on the other checks."""
    chapters = v.get("chapters") or []
    if not chapters:
        return None
    end_default = v.get("duration") or 10 ** 9
    return [(a, b or end_default) for a, b, title in chapters if GTA_RE.search(title or "")]


def discover_moments(cfg: dict, vod_state: dict, candidates: list[dict], deadline: float) -> None:
    clip_cfg = cfg["clipping"]
    for c in sorted(campaigns(), key=_campaign_priority, reverse=True):
        for src in c.get("sources") or []:
            if time.time() > deadline:
                return
            try:
                vods = sources.recent_vods(src, clip_cfg["vod_lookback_hours"], since=c.get("source_since"))
            except Exception as e:  # noqa: BLE001
                log.warning("discover: %s failed: %s", src, e)
                continue
            for v in vods:
                st = vod_state.setdefault(v["id"], {"url": v["url"], "title": v["title"], "scanned_until": None})
                dur = v["duration"] or 0
                if dur < 120:
                    log.info("discover: skipping %s (duration %.0fs unknown/too short)", v["url"], dur)
                    continue
                edge = dur - (LIVE_EDGE_SEC if v["is_live"] else 0)
                start = st["scanned_until"] if st["scanned_until"] is not None else max(0.0, edge - FIRST_LOOK_SEC)
                if edge - start > MAX_LAG_SEC:   # fresh moments earn far more than old ones: skip stale backlog
                    start = edge - SCAN_CHUNK_SEC
                new = edge - start
                if new < 60 or (v["is_live"] and new < 600):
                    continue  # wait for at least 10 new minutes on live VODs
                end = min(edge, start + SCAN_CHUNK_SEC)
                gta = _gta_ranges(v)
                if gta == []:   # the VOD says which games were played and none of them is GTA
                    log.info("discover: %s has no GTA segment (%s) - skipping", v["url"],
                             ", ".join(sorted({ch[2] for ch in v.get("chapters") or []}))[:120])
                    st["scanned_until"], st["last_scan"] = end, iso()
                    save_state("vods.json", vod_state)
                    continue
                peaks = highlights.find_peaks(v["url"], start, end, v["id"], k=clip_cfg["highlights_per_vod"])
                if gta is not None:
                    kept = [p for p in peaks if any(a <= p["t"] <= b for a, b in gta)]
                    if len(kept) < len(peaks):
                        log.info("discover: dropped %d moment(s) outside GTA segments of %s", len(peaks) - len(kept), v["id"])
                    peaks = kept
                for p in peaks:
                    candidates.append({"campaign_id": c["id"], "vod_id": v["id"], "vod_url": v["url"],
                                       "vod_title": v["title"], **p, "found_at": iso(), "status": "new"})
                st["scanned_until"] = end
                st["last_scan"] = iso()
                save_state("vods.json", vod_state)
                save_state("candidates.json", candidates)


def produce(cfg: dict, candidates: list[dict], posted: list[dict], deadline: float) -> int:
    camp = {c["id"]: c for c in campaigns()}
    p_cfg, clip_cfg = cfg["posting"], cfg["clipping"]
    made = 0
    queue = [x for x in candidates if x["status"] == "new" and x["campaign_id"] in camp]
    # Freshest + strongest first; stale moments (>36 h) are dropped - launch-week clips age fast.
    queue.sort(key=lambda x: (-parse_iso(x["found_at"]).timestamp() // 3600, -x["score"]))
    for cand in queue:
        if time.time() > deadline or made >= p_cfg["max_uploads_per_run"]:
            break
        if p_cfg.get("require_llm", True) and not llm.available():
            log.warning("produce: no LLM provider is answering (quota/outage) - leaving %d moments queued", len(queue))
            break
        if parse_iso(cand["found_at"]) < now() - dt.timedelta(hours=36):
            cand["status"] = "stale"
            continue
        today = _posted_today(posted)
        if len(today) >= p_cfg["max_uploads_per_day"]:
            log.info("daily upload cap reached (%d)", len(today))
            break
        if today and (now() - max(parse_iso(p["posted_at"]) for p in today)).total_seconds() < p_cfg["min_minutes_between_uploads"] * 60:
            log.info("spacing uploads; next run will continue")
            break

        c = camp[cand["campaign_id"]]
        gap = clip_cfg.get("min_gap_same_vod_sec", 360)   # no near-duplicate clips from the same part of a stream
        if any(p.get("vod_url") == cand["vod_url"] and abs(float(p.get("vod_offset") or -1e9) - cand["t"]) < gap for p in posted):
            cand["status"], cand["reason"] = "rejected", f"too close to an existing clip from this VOD (<{gap}s)"
            continue
        if c.get("format") == "long":
            made += produce_long(c, candidates, posted, cfg)
            continue
        name = f'{c["id"]}_{slug(cand["vod_id"], 20)}_{int(cand["t"])}'
        try:
            src = render.download_section(cand["vod_url"], cand["win_start"], cand["win_end"], name)
            words = render.transcribe(src, clip_cfg["whisper_model"])
            if len(words) < curate.MIN_WORDS:   # cheap check first: idle/sleeping moments have no speech
                cand["status"], cand["reason"] = "rejected", f"too little speech ({len(words)} words)"
                log.info("rejected %s: no speech", name)
                continue
            win_len = render.probe_duration(src)
            ok, why = safety.gameplay_ok(src, 0, win_len, max_dead=0.4)   # cheap: skip black/loading windows before the LLM
            if not ok:
                cand["status"], cand["reason"] = "rejected", why
                log.info("rejected %s before curate: %s", name, why)
                continue
            meta = curate.curate(c, words, win_len, cand["t"] - cand["win_start"], clip_cfg,
                                 images=curate.frames(src, 0, win_len, n=4),
                                 require_llm=p_cfg.get("require_llm", True))
            if meta.get("defer"):
                cand["deferrals"] = cand.get("deferrals", 0) + 1
                if cand["deferrals"] >= 6:
                    cand["status"], cand["reason"] = "stale", meta["reason"]
                log.info("deferred %s: %s", name, meta["reason"])
                continue
            if not meta.get("keep", True):
                cand["status"] = "rejected"
                cand["reason"] = meta.get("reason")
                continue
            ok, why = curate.quality_gate(c, src, words, meta["start"], meta["end"])
            if ok:
                ok, why = safety.check(src, meta["start"], meta["end"])   # local NSFW / idle / censor check
            if ok:
                ok, why = safety.gameplay_ok(src, meta["start"], meta["end"])   # no long black/loading screens
            if not ok:
                cand["status"] = "rejected"
                cand["reason"] = why
                log.info("quality gate rejected %s: %s", name, why)
                continue
            words = render.refine_words(src, words, meta["start"], meta["end"], clip_cfg.get("caption_model", "medium"),
                                        prompt=f"{curate.streamer_name(c)} playing GTA 6.")
            out = render.render_short(src, meta["start"], meta["end"], words, meta["hook"],
                                      c.get("overlay_text") or "", bool(c.get("paid_disclosure")),
                                      c.get("layout") or clip_cfg["layout"], name,
                                      overlay_img=render.fetch_overlay(c.get("overlay_image_url"), c["id"]),
                                      min_len=clip_cfg["short_min_sec"])
            meta["paid"] = bool(c.get("paid_disclosure"))
            meta["playlist"] = f"{curate.streamer_name(c)} GTA 6 Moments"
            meta["streamer"] = curate.streamer_name(c)
            stored = drive.save(out, meta, c, {"vod_url": cand["vod_url"], "offset": cand["win_start"] + meta["start"]},
                                cfg.get("storage", {}).get("drive_folder", "Zukakito Clips"))
            res = publish.publish(out, meta, cfg)
            sub = submit.submit(c, res or {})
            rec = {"campaign_id": c["id"], "file": out.name, "title": meta["title"], "posted_at": iso(),
                   "vod_url": cand["vod_url"], "vod_offset": cand["win_start"] + meta["start"],
                   "submission": sub, "drive": stored, **(res or {})}
            posted.append(rec)
            cand["status"] = "posted" if res and res.get("privacy") != "dry_run" else "rendered"
            made += 1
            save_state("posted.json", posted)
            log.info("DONE %s -> %s (%s)", meta["title"], rec.get("url"), sub)
        except Exception as e:  # noqa: BLE001
            cand["status"] = "error"
            cand["error"] = f"{type(e).__name__}: {e}"[:500]
            log.error("produce failed for %s: %s\n%s", name, e, traceback.format_exc())
        finally:
            save_state("candidates.json", candidates)
    return made


def produce_long(c: dict, candidates: list[dict], posted: list[dict], cfg: dict) -> int:
    """Long-form campaigns: compile the best ~8-12 queued moments into one 16:9 video."""
    clip_cfg = cfg["clipping"]
    pool = sorted([x for x in candidates if x["campaign_id"] == c["id"] and x["status"] == "new"], key=lambda x: -x["score"])
    target = clip_cfg["long_target_min"] * 60
    if len(pool) < 6:
        return 0  # wait until enough moments are queued
    parts, total, used = [], 0.0, []
    for cand in pool:
        if total >= target:
            break
        name = f'{c["id"]}_L_{slug(cand["vod_id"], 20)}_{int(cand["t"])}'
        try:
            src = render.download_section(cand["vod_url"], cand["win_start"], cand["win_end"], name)
            words = render.transcribe(src, clip_cfg["whisper_model"])
            meta = curate.curate(c, words, render.probe_duration(src), cand["t"] - cand["win_start"],
                                 {**clip_cfg, "short_max_sec": 75}, images=curate.frames(src, 0, render.probe_duration(src), n=4))
            if meta.get("defer") or not meta.get("keep", True) or not curate.quality_gate(c, src, words, meta["start"], meta["end"])[0] \
                    or not safety.check(src, meta["start"], meta["end"])[0]:
                cand["status"] = "rejected"
                continue
            parts.append((src, meta["start"], meta["end"], words, meta["hook"]))
            total += meta["end"] - meta["start"]
            used.append(cand)
        except Exception as e:  # noqa: BLE001
            cand["status"], cand["error"] = "error", str(e)[:300]
    if total < target * 0.6:
        return 0
    streamer = curate.streamer_name(c)
    out = render.render_long(parts, streamer, f'{c["id"]}_long_{now():%Y%m%d%H%M}')
    must = " ".join(c.get("must_include") or [])
    meta = {"title": f"{streamer.title()}'s Best GTA 6 Moments ({now():%b %d})"[:100],
            "description": f"The best moments from {streamer.title()}'s GTA 6 stream. {must}\n\n#GTA6 #GTAVI",
            "tags": ["gta 6", "gta vi", streamer.lower(), "gta 6 funny moments"], "paid": bool(c.get("paid_disclosure"))}
    if c.get("paid_disclosure"):
        meta["description"] = "Paid partnership / clipping campaign.\n" + meta["description"]
    res = publish.publish(out, meta, cfg)
    if res and res.get("url"):
        res["url"] = res["url"].replace("/shorts/", "/watch?v=")
    sub = submit.submit(c, res or {})
    posted.append({"campaign_id": c["id"], "file": out.name, "title": meta["title"], "posted_at": iso(),
                   "format": "long", "submission": sub, **(res or {})})
    for cand in used:
        cand["status"] = "posted"
    save_state("posted.json", posted)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout-only", action="store_true")
    ap.add_argument("--no-scout", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + RUN_BUDGET_SEC
    cfg = settings()
    status = load_state("status.json", {})

    if not args.no_scout:
        last = status.get("last_scout")
        if args.scout_only or not last or (now() - parse_iso(last)).total_seconds() > cfg.get("scout", {}).get("interval_hours", 1) * 3600 - 300:
            try:
                status["scout"] = scout.run()
                status["last_scout"] = iso()
            except Exception as e:  # noqa: BLE001
                log.error("scout failed: %s", e)
    if args.scout_only:
        save_state("status.json", status)
        return

    vod_state = load_state("vods.json", {})
    candidates = load_state("candidates.json", [])
    posted = load_state("posted.json", [])

    made = produce(cfg, candidates, posted, min(deadline - 25 * 60, t0 + 15 * 60))   # render queued moments first, max 15 min
    discover_moments(cfg, vod_state, candidates, deadline - 20 * 60)
    made += produce(cfg, candidates, posted, deadline)

    # keep state files small
    candidates[:] = [x for x in candidates if x["status"] == "new" or parse_iso(x["found_at"]) > now() - dt.timedelta(days=3)]
    save_state("candidates.json", candidates)

    status.update({
        "last_run": iso(), "run_seconds": int(time.time() - t0), "made_this_run": made,
        "publisher": cfg["posting"]["publisher"],
        "enabled_campaigns": [c["id"] for c in campaigns()],
        "posted_24h": len(_posted_today(posted)), "posted_total": len([p for p in posted if p.get("privacy") != "dry_run"]),
        "queued_candidates": len([x for x in candidates if x["status"] == "new"]),
        "pending_submissions": len([p for p in load_state("pending_submissions.json", []) if p["status"] == "pending"]),
        "rendered_files": sorted(p.name for p in OUT.glob("*.mp4")),
    })
    save_state("status.json", status)
    log.info("run complete: %s", status)


if __name__ == "__main__":
    main()
