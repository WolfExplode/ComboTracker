"""
Draft static/data/ww_characters.json (the move data the app uses) from the raw encore.moe snapshot.

ComboTracker only cares about key inputs, so each character keeps just:
  - basic:      how many hits the Basic Attack chain has (A1, A2, ...), each hit's name,
                and whether holding LMB is a Heavy Attack or just keeps the chain going
  - moves:      attack and skill names with the input that casts them
  - chain_entry: where LMB picks the chain back up after another move
                ("LMB right after the Intro is Basic Attack Stage 3" -> {"Intro Skill": 3})
  - followups:  other "press X shortly after Y to cast Z" links from the kit text
No damage numbers, scaling, cooldowns or buff text.

Usage:
  python tools/ww_build_moves.py            # draft every character from data/encore_raw/
  python tools/ww_build_moves.py --force    # also re-draft characters marked "reviewed": true

Characters marked "reviewed": true were checked by hand and are kept as they are, so re-running
after a new raw download only fills in new characters. Get the raw snapshot with the Characters
page's "Download raw data" button (it saves to data/encore_raw/ and changes nothing else).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ww_library  # noqa: E402

RAW_DIR = ROOT / "data" / ww_library.RAW_DIR_NAME
OUT_PATH = ROOT / "static" / "data" / "ww_characters.json"

# encore.moe skill type -> the input that casts it in tracker notation
SKILL_INPUTS = {
    "Resonance Skill": "e",
    "Resonance Liberation": "r",
    "Intro Skill": "swap in",
    "Outro Skill": "swap out",
    "Tune Break": "f",
}
INPUT_WORDS = {"normal attack": "lmb", "resonance skill": "e", "resonance liberation": "r"}
# Move-name prefixes that are always player actions (vs. states/buffs such as "Majesty").
ACTION_PREFIX = re.compile(
    r"^(basic attack|heavy attack|mid-air|dodge counter|plunging|resonance skill|resonance liberation|"
    r"intro skill|outro skill|tune break|[\w' -]+ - (basic attack|heavy attack|dodge counter|mid-air attack))",
    re.I,
)

_HEADING = re.compile(r"^\*\*([^*]+)\*\*\s*$")
_FOLLOWUP = re.compile(
    r"(?:press|use|hold|tap)\s+(?:the\s+)?(?:\*\*)?(normal attack|resonance skill)(?:\*\*)?(?:\s+button)?(?:\s+again)?"
    r"\s+(?:shortly\s+|right\s+|immediately\s+)?(?:within a certain period(?: of time)?\s+)?after\s+"
    r"(?:casting|performing|using|triggering)\s+(?:resonance (?:skill|liberation)\s+|intro skill\s+)?"
    r"(this (?:skill|attack)|\*\*([^*]+)\*\*)"
    r"[^.]*?\bto\s+(?:cast|perform|trigger|chain into)\s+(?:resonance (?:skill|liberation)\s+|heavy attack\s+)?"
    r"(?:\*\*([^*]+)\*\*|([A-Z][\w' :-]*?Stage \d+))",
    re.I,
)
_STAGE = re.compile(r"\bStage\s*(\d+)\b", re.I)
_CHAIN = re.compile(r"up to (\d+) consecutive attacks", re.I)
_HOLD_CHAIN = re.compile(r"(?:or|and) hold(?: it down| down)?[^.]{0,40}?to perform up to \d+ consecutive", re.I)
_HOLD_SEQUENCE = re.compile(r"hold \*\*normal attack\*\* to cast \*\*basic attack stage", re.I)


def _headings(desc: str) -> list[str]:
    out = []
    for line in (desc or "").splitlines():
        m = _HEADING.match(line.strip())
        if m:
            name = m.group(1).strip()
            if name and name not in out:
                out.append(name)
    return out


def _is_move(name: str, mult_names: list[str], desc: str) -> bool:
    """A heading is a move (not a state or buff) if it deals damage, reads as an action, or is cast."""
    low = name.lower()
    if ACTION_PREFIX.match(name):
        return True
    if any(m.lower().startswith(low) for m in mult_names):
        return True
    return bool(re.search(r"(?:cast|perform)s?\s+(?:resonance (?:skill|liberation)\s+)?\*\*" + re.escape(name) + r"\*\*", desc, re.I))


def _basic_hit_names(base: str, mult_names: list[str], hits: int | None) -> list[str]:
    """['Stage 1', 'Stage 2', 'Stage 3 - Unremarkable / Commendable', ...] from the multiplier table."""
    if not hits:
        return []
    want = base.lower()
    per: dict[int, list[str]] = {}
    for m in mult_names:
        low = m.lower()
        if want not in ("basic attack",) and not low.startswith(want):
            continue
        if want == "basic attack" and not (low.startswith("basic attack") or low.startswith("stage ")):
            continue
        if want == "basic attack" and re.match(r"basic attack\s*-\s*\w", low):
            continue  # "Basic Attack - Tracing Forms ..." is another form's chain
        st = _STAGE.search(m) or re.search(rf"{re.escape(base)}\s+(\d+)\b", m, re.I)
        if not st:
            continue
        n = int(st.group(1))
        tail = m[st.end():].strip()
        tail = re.sub(r"\s*(DMG|Hold DMG|STA Cost.*)$", "", tail, flags=re.I).strip(" -:")
        per.setdefault(n, [])
        if tail and tail not in per[n]:
            per[n].append(tail)
    out = []
    for n in range(1, hits + 1):
        extra = per.get(n) or []
        out.append(f"Stage {n}" + (f" - {' / '.join(extra)}" if extra else ""))
    return out


def _chain_key(after: str) -> str:
    """
    The move a chain_entry follows, as the timeline sees it (static/ww_moves.js):
      intro (1/2/3 swap-in), skill (e), liberation (r), tune_break (f),
      heavy (hold lmb), dodge (rmb, lmb), midair (space, lmb). '' when it's something else.
    """
    low = after.lower()
    for prefix, key in (("intro skill", "intro"), ("resonance skill", "skill"), ("resonance liberation", "liberation"),
                        ("tune break", "tune_break")):
        if low == prefix:
            return key
    if re.match(r"^(?:[\w' :]+ - )?heavy attack", low) or low.startswith("heavy attack"):
        return "heavy"
    if "dodge counter" in low:
        return "dodge"
    if "mid-air attack" in low or "plunging" in low:
        return "midair"
    return ""


def _sections(skill: dict[str, Any]) -> list[tuple[str, str]]:
    """Split a skill's text at its bold headings: [(heading, text)]. Text before any heading uses the skill type."""
    text = (skill["description"] or "").replace("{Cus:Ipt,Touch=Tapping PC=Pressing Gamepad=Pressing}", "Press")
    out: list[tuple[str, str]] = []
    title, buf = skill["type"], []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if buf:
                out.append((title, "\n".join(buf)))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        out.append((title, "\n".join(buf)))
    return out


def _followups(skill: dict[str, Any], base_basic: str) -> list[dict[str, Any]]:
    out = []
    for title, text in _sections(skill):
        # "This skill" means the heading it sits under; the Intro/Skill/Liberation keep their type name.
        this = skill["type"] if skill["type"] in SKILL_INPUTS or title == skill["name"] else title
        for m in _FOLLOWUP.finditer(text):
            after = this if m.group(2).lower().startswith("this ") else m.group(3).strip()
            gives = (m.group(4) or m.group(5) or "").strip()
            item: dict[str, Any] = {"after": after, "press": INPUT_WORDS[m.group(1).lower()], "gives": gives}
            st = _STAGE.search(gives)
            if st and re.match(re.escape(base_basic) + r"\s+Stage\s*\d", gives, re.I) and item["press"] == "lmb":
                item["basic_stage"] = int(st.group(1))
            if item not in out:
                out.append(item)
    return out


def draft_character(raw: dict[str, Any], roster_row: dict[str, Any]) -> dict[str, Any]:
    kit = ww_library.parse_kit(raw)
    # Rover's kit lists some skills once per gender; keep one of each.
    skills, seen = [], set()
    for s in kit["skills"]:
        if (s["type"], s["name"]) in seen:
            continue
        seen.add((s["type"], s["name"]))
        skills.append(s)
    normal = next((s for s in skills if s["type"] == "Normal Attack"), None)

    basic: dict[str, Any] = {"name": "Basic Attack", "hits": None, "hit_names": [], "hold_lmb": "heavy"}
    moves: list[dict[str, Any]] = []
    if normal:
        desc = normal["description"]
        mult = [m["name"] for m in normal["multipliers"]]
        heads = _headings(desc)
        if heads and heads[0].lower().startswith(("basic attack", "moonring", "basic")) or (heads and "basic attack" in heads[0].lower()):
            basic["name"] = heads[0]
        m = _CHAIN.search(desc)
        if m:
            basic["hits"] = int(m.group(1))
        basic["hit_names"] = _basic_hit_names(basic["name"], mult, basic["hits"])
        if _HOLD_CHAIN.search(desc) or _HOLD_SEQUENCE.search(desc):
            basic["hold_lmb"] = "chain"
        # Short name the timeline uses for hold(lmb), e.g. "Heavy" -> "Zani Heavy".
        basic["hold_label"] = "Heavy" if basic["hold_lmb"] == "heavy" else ""
        for h in heads:
            if not _is_move(h, mult, desc):
                continue
            low = h.lower()
            if low == basic["name"].lower():
                inp = "lmb"
            elif "heavy attack" in low:
                inp = "hold(lmb)"
            elif "mid-air" in low or "plunging" in low:
                inp = "space, lmb"
            elif "dodge counter" in low:
                inp = "rmb, lmb"
            else:
                inp = "lmb"
            moves.append({"input": inp, "name": h})

    chain_entry: dict[str, int] = {}
    followups: list[dict[str, Any]] = []
    for s in skills:
        if s["type"] in SKILL_INPUTS:
            moves.append({"input": SKILL_INPUTS[s["type"]], "name": s["name"], "type": s["type"]})
        if s["type"] in ("Normal Attack", "Resonance Skill", "Forte Circuit", "Resonance Liberation", "Intro Skill"):
            mult = [m["name"] for m in s["multipliers"]]
            for h in _headings(s["description"]):
                if h == s["name"] or s["type"] == "Normal Attack":
                    continue
                if _is_move(h, mult, s["description"]):
                    moves.append({"input": "", "name": h, "type": s["type"]})
        for f in _followups(s, basic["name"]):
            if f in followups:
                continue
            followups.append(f)
            when = _chain_key(f["after"])
            if "basic_stage" in f and when and when not in chain_entry:
                chain_entry[when] = f["basic_stage"]

    return {
        "id": kit["id"],
        "name": kit["name"],
        "element": kit["element"] or roster_row.get("element", ""),
        "weapon": roster_row.get("weapon", ""),
        "rarity": roster_row.get("rarity"),
        "icon": kit["icon"] or roster_row.get("icon", ""),
        "reviewed": False,
        "basic": basic,
        "moves": moves,
        "chain_entry": chain_entry,
        "followups": followups,
        "notes": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-draft characters marked reviewed too")
    args = ap.parse_args()

    if not (RAW_DIR / "roster.json").exists():
        print(f"No raw data in {RAW_DIR}. Download it first (see this file's docstring).")
        return 1
    roster = ww_library.parse_roster(json.loads((RAW_DIR / "roster.json").read_text(encoding="utf-8")))
    existing: dict[str, Any] = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("characters", {})

    chars: dict[str, Any] = {}
    kept = drafted = 0
    for row in roster["characters"]:
        key = row["name"].lower()
        old = existing.get(key)
        if old and old.get("reviewed") and not args.force:
            chars[key] = old
            kept += 1
            continue
        raw_path = RAW_DIR / f"kit_{row['id']}.json"
        if not raw_path.exists():
            if old:
                chars[key] = old
            continue
        chars[key] = draft_character(json.loads(raw_path.read_text(encoding="utf-8")), row)
        drafted += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "about": "Key-input move data for ComboTracker, drafted from encore.moe by tools/ww_build_moves.py. "
                 "Entries with \"reviewed\": true were checked by hand; edit this file directly to fix or add details.",
        "characters": dict(sorted(chars.items())),
    }
    OUT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{OUT_PATH.relative_to(ROOT)}: {drafted} drafted, {kept} reviewed kept as-is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
