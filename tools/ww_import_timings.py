"""
Build static/data/ww_timings.json (the Characters page's Timings tab) from a WuwaLAB scrape.

WuwaLAB (https://wuwalab.com) lists every ability's frame data on its character pages
(/characters/<slug>/abilities). The scrape is read from those rendered pages; this tool only
reshapes it. ComboTracker is about which buttons to press and when, so only timing columns are
kept: hits, total frames, cancel frame, no-swap window, time stop, motion stop, cooldown, hit
frames and the chips WuwaLAB shows next to a name. Damage columns are dropped.

Each WuwaLAB character is matched to its entry in static/data/ww_characters.json by name, and the
output is keyed by that entry's id so the page can look it up for the selected character.

Usage:
  python tools/ww_import_timings.py path/to/wuwalab_timings.json
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

KEEP = ("section", "name", "tags", "hits", "frames", "cancel", "noswap", "tstop", "mstop", "cd", "hit_frames")

# WuwaLAB names that don't share a word set with the encore.moe name.
ALIASES = {"xuanling": "Yangyang: Xuanling"}


def name_tokens(name: str) -> str:
    """'Rover: Spectro' -> 'rover spectro'; word order and punctuation ignored."""
    return " ".join(sorted(t for t in re.split(r"[^a-z]+", str(name).lower()) if t))


def ability(raw: dict[str, Any]) -> dict[str, Any]:
    out = {k: raw[k] for k in KEEP if k in raw}
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
            "abilities": [ability(a) for a in ch.get("abilities", [])],
        }
    data = {
        "about": "Ability frame data from WuwaLAB (https://wuwalab.com), timing columns only. "
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
    ap.add_argument("scrape", type=Path, help="WuwaLAB scrape JSON (characters -> slug -> abilities)")
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
