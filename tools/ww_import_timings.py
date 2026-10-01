"""
Build static/data/ww_timings.json (the Characters page's Timings tab) from a WuwaLAB scrape.

WuwaLAB (https://wuwalab.com) lists every ability's frame data on its character pages
(/characters/<slug>/abilities). The scrape is read from those rendered pages; this tool only
reshapes it. ComboTracker is about which buttons to press and when, so only timing columns are
kept: hits, total frames, cancel frame, no-swap window, time stop, motion stop, cooldown, hit
frames, the motion/time-stop zones on the frame strip, the chips WuwaLAB shows next to a name, the
move type, and Concerto (it decides when an Outro fires and when moves like Iuno's Absolute
Fullness are available). Damage columns (MV, scaling, energy, off-tune, forte) are dropped.

Two scrape shapes are read: the full one (characters.<slug>.abilities_total = {headers, rows},
every column read by its header name) and the first, flat one (characters.<slug>.abilities).

Each WuwaLAB character is matched to its entry in static/data/ww_characters.json by name, and the
output is keyed by that entry's id so the page can look it up for the selected character.

Usage:
  python tools/ww_import_timings.py path/to/wuwalab_full.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CHARACTERS = ROOT / "static" / "data" / "ww_characters.json"
OUT = ROOT / "static" / "data" / "ww_timings.json"

KEEP = ("section", "name", "tags", "genre", "hits", "frames", "cancel", "noswap", "tstop", "mstop", "cd", "concerto",
        "hit_frames", "zones")

# WuwaLAB names that don't share a word set with the encore.moe name.
ALIASES = {"xuanling": "Yangyang: Xuanling"}


def name_tokens(name: str) -> str:
    """'Rover: Spectro' -> 'rover spectro'; word order and punctuation ignored."""
    return " ".join(sorted(t for t in re.split(r"[^a-z]+", str(name).lower()) if t))


# WuwaLAB's priority-timeline column ("0:11, 81:2").
PRIORITY_TL = re.compile(r"^\d+:\d+")


def cooldown(raw: dict[str, Any]) -> int | None:
    """The scraped cd, or None when it can't be trusted.

    Pages with the extra Concerto/Energy/Off-tune/F1 columns came through shifted: the
    priority number landed in cd and the priority timeline in genre, or the timeline itself
    landed in cd. A cd is only real when genre holds an actual move type (BASIC, SKILL...).
    """
    cd, genre = raw.get("cd"), raw.get("genre")
    if not isinstance(cd, (int, float)) or isinstance(cd, bool):
        return None
    if not isinstance(genre, str) or PRIORITY_TL.match(genre):
        return None
    return int(cd)


def _num(v: Any) -> int | None:
    """8 / "81f" / "1,000" / "-10,000" -> int; "—" and blanks -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"(-?[\d,]+)f?", str(v or "").strip())
    return int(m.group(1).replace(",", "")) if m else None


def _zones(row: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    """Motion/time-stop bands on WuwaLAB's frame strip ("left: 7.4%; right: 60.5%") as frames."""
    out = []
    for t in row.get("timeline") or []:
        cls = str(t.get("cls") or "")
        kind = "ms" if "tl-zone--ms" in cls else "ts" if "tl-zone--ts" in cls else None
        if not kind or not frames:
            continue
        style = str(t.get("style") or "")
        left = re.search(r"left:\s*([\d.]+)%", style)
        right = re.search(r"right:\s*([\d.]+)%", style)
        if not left:
            continue
        a = round(float(left.group(1)) * frames / 100)
        b = round(frames - (float(right.group(1)) if right else 0) * frames / 100)
        out.append({"kind": kind, "from": a, "to": b})
    return out


def flat_row(row: dict[str, Any]) -> dict[str, Any]:
    """A full-scrape row ({section, name, tags, values, hits_detail, timeline}) in the flat shape."""
    v = row.get("values") or {}
    frames = _num(v.get("frames")) or 0
    section = re.sub(r"\s*\d+\s+ABILIT(Y|IES)\s*$", "", str(row.get("section") or ""), flags=re.I)
    out: dict[str, Any] = {"section": section, "name": row.get("name", ""), "tags": row.get("tags") or []}
    for k in ("hits", "frames", "cancel", "noswap", "tstop", "mstop", "cd", "concerto"):
        out[k] = _num(v.get(k))
    out["genre"] = str(v.get("genre") or "")
    out["hit_frames"] = [n for n in (_num(h.get("frame")) for h in row.get("hits_detail") or []) if n is not None]
    out["zones"] = _zones(row, frames)
    return out


def abilities_of(ch: dict[str, Any]) -> list[dict[str, Any]]:
    total = ch.get("abilities_total")
    if isinstance(total, dict):
        return [flat_row(r) for r in total.get("rows") or []]
    return list(ch.get("abilities") or [])


def ability(raw: dict[str, Any]) -> dict[str, Any]:
    out = {k: raw[k] for k in KEEP if k in raw}
    out["cd"] = cooldown(raw)
    out.setdefault("zones", [])
    g = out.get("genre")
    out["genre"] = g if isinstance(g, str) and not PRIORITY_TL.match(g) else ""
    out["tags"] = [str(t) for t in out.get("tags") or []]
    out.setdefault("hit_frames", [])
    return out


def build(scrape: dict[str, Any], characters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    by_tokens = {name_tokens(c["name"]): c for c in characters.values()}
    out: dict[str, Any] = {}
    unmatched: list[str] = []
    for slug, ch in sorted(scrape.get("characters", {}).items()):
        key = name_tokens(ALIASES.get(slug) or ch.get("name", slug))
        target = by_tokens.get(key)
        if not target:
            unmatched.append(ch.get("name", slug))
            continue
        out[str(target["id"])] = {
            "name": target["name"],
            "wuwalab_name": ch.get("name", slug),
            "url": ch.get("url") or f"https://wuwalab.com/characters/{slug}/abilities",
            "abilities": [ability(a) for a in abilities_of(ch)],
        }
    data = {
        "about": "Ability frame data from WuwaLAB (https://wuwalab.com): timing columns and Concerto "
                 "(WuwaLAB units, 100 = 1 point; 10,000 = full), no damage. "
                 "Built by tools/ww_import_timings.py; keyed by the ids in ww_characters.json.",
        "source": "https://wuwalab.com",
        "fetched_at": scrape.get("fetched_at", ""),
        "fps": 60,
        "characters": out,
    }
    return data, unmatched


def dump(data: dict[str, Any]) -> str:
    """Pretty at the character level, one ability per line, so updates diff cleanly."""
    lines = ["{"]
    for k in ("about", "source", "fetched_at", "fps"):
        lines.append(f"  {json.dumps(k)}: {json.dumps(data[k], ensure_ascii=False)},")
    lines.append('  "characters": {')
    chars = list(data["characters"].items())
    for i, (cid, c) in enumerate(chars):
        lines.append(f"    {json.dumps(cid)}: {{")
        for k in ("name", "wuwalab_name", "url"):
            lines.append(f"      {json.dumps(k)}: {json.dumps(c[k], ensure_ascii=False)},")
        lines.append('      "abilities": [')
        abl = c["abilities"]
        for j, a in enumerate(abl):
            lines.append(f"        {json.dumps(a, ensure_ascii=False)}{',' if j < len(abl) - 1 else ''}")
        lines.append("      ]")
        lines.append(f"    }}{',' if i < len(chars) - 1 else ''}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scrape", type=Path, help="WuwaLAB scrape JSON (characters -> slug -> abilities_total or abilities)")
    args = ap.parse_args(argv)

    scrape = json.loads(args.scrape.read_text(encoding="utf-8"))
    characters = json.loads(CHARACTERS.read_text(encoding="utf-8"))["characters"]
    data, unmatched = build(scrape, characters)
    OUT.write_text(dump(data), encoding="utf-8")
    total = sum(len(c["abilities"]) for c in data["characters"].values())
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(data['characters'])} characters, {total} abilities.")
    if unmatched:
        print(f"No match in ww_characters.json (skipped): {', '.join(unmatched)}", file=sys.stderr)
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
