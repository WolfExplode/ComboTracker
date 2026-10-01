"""
Wuthering Waves character library: every character's moves and the community's team rotations.

Sources (fetched by the app on demand, then cached on disk so the page works offline):
  - encore.moe JSON API: character list and each character's full kit (skill text, multipliers, icons)
      https://api.encore.moe/en/character          -> {"roleList": [{Id, Name, Element, WeaponType, RoleHeadIcon}]}
      https://api.encore.moe/en/character/<Id>     -> {..., "Skills": [{SkillType, SkillName, SkillDescribe, Icon,
                                                        SkillAttributes: [{attributeName, values: [lv1..lv10]}]}]}
  - AntoCrasher's calc compilation (Google Sheet): one block per main DPS listing team setups with author,
    DPS, video, and a link to a Google Doc tab holding the move-by-move rotation transcript.

Served to static/characters.html by ui_server.py under /api/ww/...
"""

from __future__ import annotations

import csv
import html as html_mod
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "Mozilla/5.0 (compatible; ComboTracker)"
ENCORE_API = "https://api.encore.moe/en/character"
ROTATION_SHEET_ID = "1mdl9J08N-0_j-U2zNP5OTHGBKprwmJEmz_Iy4IOJfPk"
ROTATION_SHEET_TAB = "2 Minute Calc Compilation"
ROTATION_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{ROTATION_SHEET_ID}/edit"

# How long a cached copy counts as fresh; stale copies are still used when the network is down.
CACHE_TTL_S = 7 * 24 * 3600

# encore.moe SkillType -> the key Wolf presses for it (shown as a badge on the Moves tab).
SKILL_KEYS = {
    "Normal Attack": "LMB",
    "Resonance Skill": "E",
    "Resonance Liberation": "R",
    "Intro Skill": "Swap in",
    "Outro Skill": "Swap out",
    "Tune Break": "F",
}

# Order the Moves tab lists skill types in; unknown types go last in API order.
SKILL_TYPE_ORDER = [
    "Normal Attack",
    "Resonance Skill",
    "Forte Circuit",
    "Resonance Liberation",
    "Intro Skill",
    "Outro Skill",
    "Tune Break",
    "Inherent Skill",
]


class LibraryError(RuntimeError):
    """A source couldn't be read and there is no cached copy to fall back on."""


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise LibraryError(f"{urllib.parse.urlsplit(url).netloc} answered HTTP {e.code}.") from e
    except urllib.error.URLError as e:
        raise LibraryError(f"Could not reach {urllib.parse.urlsplit(url).netloc}: {e.reason}") from e
    except TimeoutError as e:
        raise LibraryError(f"{urllib.parse.urlsplit(url).netloc} timed out.") from e


class Library:
    """Fetches and normalizes the sources, caching each result as JSON under cache_dir."""

    def __init__(self, cache_dir: Path, fetch=_http_get) -> None:
        self.cache_dir = Path(cache_dir)
        self._fetch = fetch
        self._lock = threading.Lock()

    def _cached(self, name: str, build, refresh: bool) -> dict[str, Any]:
        path = self.cache_dir / f"{name}.json"
        cached: dict[str, Any] | None = None
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = None
        if cached and not refresh and time.time() - cached.get("fetched_at", 0) < CACHE_TTL_S:
            return cached
        try:
            data = build()
        except LibraryError as e:
            if cached:
                return dict(cached, stale=True, error=str(e))
            raise
        data["fetched_at"] = time.time()
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        return data

    def roster(self, refresh: bool = False) -> dict[str, Any]:
        return self._cached("roster", lambda: parse_roster(json.loads(self._fetch(ENCORE_API))), refresh)

    def kit(self, char_id: int, refresh: bool = False) -> dict[str, Any]:
        char_id = int(char_id)
        return self._cached(
            f"kit_{char_id}", lambda: parse_kit(json.loads(self._fetch(f"{ENCORE_API}/{char_id}"))), refresh
        )

    def rotations(self, refresh: bool = False) -> dict[str, Any]:
        def build() -> dict[str, Any]:
            url = (
                f"https://docs.google.com/spreadsheets/d/{ROTATION_SHEET_ID}/gviz/tq?tqx=out:csv&sheet="
                + urllib.parse.quote(ROTATION_SHEET_TAB)
            )
            groups = parse_rotation_sheet(self._fetch(url))
            if not groups:
                raise LibraryError("The rotation sheet had no team rows. Its layout may have changed.")
            return {"source": ROTATION_SHEET_URL, "groups": groups}

        return self._cached("rotations", build, refresh)

    def transcript(self, doc_url: str, refresh: bool = False) -> dict[str, Any]:
        export = transcript_export_url(doc_url)
        if not export:
            raise LibraryError("That isn't a Google Docs link.")
        key = re.sub(r"[^A-Za-z0-9]+", "_", export.split("/d/", 1)[1])[:120]
        return self._cached(
            f"transcript_{key}", lambda: dict(parse_transcript(self._fetch(export)), source=doc_url), refresh
        )


# ---------------------------------------------------------------------------
# encore.moe
# ---------------------------------------------------------------------------

def _name_of(v: Any) -> str:
    return (v.get("Name") if isinstance(v, dict) else v) or ""


def parse_roster(data: dict[str, Any]) -> dict[str, Any]:
    """Character list. Rover appears once per gender; keep the first of each name."""
    seen: set[str] = set()
    chars = []
    for r in data.get("roleList") or []:
        name = (r.get("Name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        chars.append({
            "id": r.get("Id"),
            "name": name,
            "rarity": r.get("QualityId"),
            "element": _name_of(r.get("Element")),
            "weapon": _name_of(r.get("WeaponType")),
            "icon": r.get("RoleHeadIcon") or "",
        })
    chars.sort(key=lambda c: c["name"].lower())
    return {"source": ENCORE_API, "characters": chars}


_BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_BOLD = re.compile(r'<span[^>]*font-bold[^>]*>(.*?)</span>', re.I | re.S)


def clean_description(raw: str) -> str:
    """encore.moe skill HTML -> plain text. Bold spans (move names) become **text**; <br> becomes a newline."""
    s = _BR.sub("\n", raw or "")
    s = _BOLD.sub(lambda m: "**" + _TAG.sub("", m.group(1)).strip() + "**", s)
    s = html_mod.unescape(_TAG.sub("", s))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_kit(data: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for sk in data.get("Skills") or []:
        stype = (sk.get("SkillType") or "").strip()
        skills.append({
            "id": sk.get("SkillId"),
            "type": stype,
            "key": SKILL_KEYS.get(stype, ""),
            "name": (sk.get("SkillName") or "").strip(),
            "icon": sk.get("Icon") or "",
            "video": sk.get("SkillMedia") or "",
            "description": clean_description(sk.get("SkillDescribe") or ""),
            "multipliers": [
                {"name": (a.get("attributeName") or "").strip(), "values": [str(v) for v in a.get("values") or []]}
                for a in sk.get("SkillAttributes") or []
                if (a.get("attributeName") or "").strip()
            ],
        })
    order = {t: i for i, t in enumerate(SKILL_TYPE_ORDER)}
    skills.sort(key=lambda s: order.get(s["type"], len(order)))  # stable: keeps API order within a type
    name = data.get("Name")
    return {
        "id": data.get("Id"),
        "name": (name.get("Content") if isinstance(name, dict) else name) or "",
        "element": data.get("ElementName") or _name_of(data.get("Element")),
        "icon": data.get("RoleHeadIcon") or data.get("RoleHeadIconCircle") or "",
        "skills": skills,
        "source": f"{ENCORE_API}/{data.get('Id')}",
    }


# ---------------------------------------------------------------------------
# AntoCrasher rotation sheet
# ---------------------------------------------------------------------------

_BLOCK_TITLE = re.compile(r"^\s*AntoCrasher Calc\s*-\s*(.+?)\s+Team\b", re.I)
_TEAM_TAGS = re.compile(r"\s*\(([^)]*)\)\s*$")


def _thoughts(cells: list[str]) -> str:
    """'Personal Thoughts' text: either inside the same cell or in the next non-empty cell."""
    for i, c in enumerate(cells):
        if c.strip().lower().startswith("personal thoughts"):
            rest = c.strip()[len("personal thoughts"):].strip().strip('"').strip()
            if rest:
                return rest
            for nxt in cells[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
    return ""


def _team_members(team: str) -> list[str]:
    """'Zani, Phoebe S0R1, Rover S6R1 (Advanced Quickswap)' -> ['Zani', 'Phoebe', 'Rover']."""
    base = _TEAM_TAGS.sub("", team)
    out = []
    for part in base.split(","):
        words = [w for w in part.split() if not re.fullmatch(r"S\d+R\d+", w, re.I)]
        if words:
            out.append(" ".join(words))
    return out


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_rotation_sheet(csv_text: str) -> list[dict[str, Any]]:
    """Split the compilation into one group per main character with its team rows."""
    groups: list[dict[str, Any]] = []
    cols: dict[str, int] = {}
    group: dict[str, Any] | None = None
    for cells in csv.reader(io.StringIO(csv_text)):
        cells = [c.strip() for c in cells]
        joined = " ".join(cells)
        title = next((m for m in (_BLOCK_TITLE.match(c) for c in cells) if m), None)
        if title:
            group = {"character": title.group(1).strip(), "title": title.string.strip(), "thoughts": "", "teams": []}
            groups.append(group)
        if group is not None and not group["thoughts"]:
            group["thoughts"] = _thoughts(cells)
        if "Author" in cells:
            # Later blocks repeat a partial header (blank damage columns); keep earlier positions for those.
            cols = {**cols, **{c.lower(): i for i, c in enumerate(cells) if c}}
            continue
        if title or group is None or not cols or not joined.strip():
            continue

        def col(name: str) -> str:
            i = cols.get(name)
            return cells[i] if i is not None and i < len(cells) else ""

        team = next((c for c in cells[: cols.get("author", 1)] if c), "")
        dps = _num(col("dps"))
        if not team or dps is None:
            continue
        tags = _TEAM_TAGS.search(team)
        author = col("author")
        group["teams"].append({
            "team": team,
            "members": _team_members(team),
            "style": tags.group(1).strip() if tags else "",
            "author": "" if author in ("—", "-") else author,
            "setup": col("extra info"),
            "dps": dps,
            "total_damage": _num(col("total damage")),
            "rotation_time": _num(col("rotation time")),
            "relative": col("%"),
            "video": col("video showcase"),
            "transcript": col("rotation transcript"),
            "calc_sheet": col("calc sheet"),
        })
    return [g for g in groups if g["teams"]]


# ---------------------------------------------------------------------------
# Rotation transcripts (Google Docs)
# ---------------------------------------------------------------------------

def transcript_export_url(doc_url: str) -> str:
    """Docs link (optionally ?tab=t.xxx) -> its plain-text export for that tab."""
    m = re.match(r"https://docs\.google\.com/document/d/([A-Za-z0-9_-]+)", doc_url or "")
    if not m:
        return ""
    tab = urllib.parse.parse_qs(urllib.parse.urlsplit(doc_url).query).get("tab", [""])[0]
    url = f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt"
    return url + (f"&tab={urllib.parse.quote(tab)}" if tab else "")


_GLOSSARY = re.compile(r"^([a-z0-9]+)\s*=\s*(.+)$", re.I)
_ACTION_LINE = re.compile(r"^([A-Za-z][\w .'-]{0,30}):\s*(.+)$")


def parse_transcript(text: str) -> dict[str, Any]:
    """
    Transcript text -> glossary + sections of "<character>: move > move > swap" lines.

    The hub doc starts with an abbreviation list ("ba = basic attack"), then a title line
    ("Zani Phoebe Rover ADV QS Cyan 4NF Transcript") and sections ("Opener", "Rotation 1"...).
    """
    glossary: dict[str, str] = {}
    title = ""
    sections: list[dict[str, Any]] = []
    notes: list[str] = []
    for raw in (text or "").replace("﻿", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        g = _GLOSSARY.match(line)
        if g and not any(sec["steps"] for sec in sections):
            glossary[g.group(1).lower()] = g.group(2).strip()
            continue
        a = _ACTION_LINE.match(line)
        if a and (">" in a.group(2) or (sections and len(a.group(2).split()) <= 3)):
            if not sections:
                sections.append({"name": "", "steps": []})
            moves = [m.strip() for m in a.group(2).split(">") if m.strip()]
            sections[-1]["steps"].append({"character": a.group(1).strip(), "moves": moves})
            continue
        if line.lower().endswith("transcript") and not title:
            title = line
            continue
        if len(line) <= 40 and not line.endswith("."):
            sections.append({"name": line, "steps": []})
            continue
        notes.append(line)
    sections = [s for s in sections if s["steps"]]
    return {"title": title, "glossary": glossary, "sections": sections, "notes": notes[:20]}


# ---------------------------------------------------------------------------
# HTTP API used by static/characters.html
# ---------------------------------------------------------------------------

API_PREFIX = "/api/ww/"


def handle_api(library: Library, path: str) -> tuple[int, dict[str, Any]]:
    """
    Route one GET under /api/ww/. Add ?refresh=1 to bypass the cache.
      roster                      -> character list
      kit/<id>                    -> one character's moves
      rotations                   -> community team rotations grouped by main character
      transcript?url=<docs link>  -> one rotation's move-by-move transcript
    """
    parts = urllib.parse.urlsplit(path)
    query = urllib.parse.parse_qs(parts.query)
    refresh = query.get("refresh", [""])[0] in ("1", "true")
    route = parts.path[len(API_PREFIX):].strip("/")
    try:
        if route == "roster":
            return 200, library.roster(refresh)
        if route.startswith("kit/") and route[4:].isdigit():
            return 200, library.kit(int(route[4:]), refresh)
        if route == "rotations":
            return 200, library.rotations(refresh)
        if route == "transcript":
            return 200, library.transcript(query.get("url", [""])[0], refresh)
    except LibraryError as e:
        return 502, {"error": str(e)}
    except (ValueError, KeyError, TypeError) as e:
        return 502, {"error": f"Unexpected data from the source: {e}"}
    return 404, {"error": f"Unknown API route: {route}"}
