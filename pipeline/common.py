"""Shared helpers: paths, config, JSON state, logging."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
STATE = ROOT / "state"
WORK = Path(os.environ.get("CLIPPER_WORKDIR", ROOT / "work"))
OUT = WORK / "out"
FONTS = ROOT / "fonts"

for p in (STATE, WORK, OUT):
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clipper")


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime | None = None) -> str:
    return (t or now()).isoformat(timespec="seconds")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_yaml(name: str) -> dict:
    with open(CONFIG / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def settings() -> dict:
    return load_yaml("settings.yaml")


def campaigns(enabled_only: bool = True) -> list[dict]:
    data = load_yaml("campaigns.yaml")
    defaults = data.get("defaults", {})
    out = []
    for c in data.get("campaigns", []):
        merged = {**defaults, **c}
        if enabled_only and not merged.get("enabled"):
            continue
        out.append(merged)
    return out


def load_state(name: str, default):
    p = STATE / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("state file %s is corrupt; starting fresh", name)
        return default


def save_state(name: str, data) -> None:
    p = STATE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n]


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v if v else default
