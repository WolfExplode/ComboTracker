"""
Add/update Wuthering Waves characters and teams in combos.json, scraping
icon URLs from wuthering.gg.

Why this exists
----------------
combos.json is normally owned by the running ComboTracker app (ui_server.py).
If the app is running and you edit combos.json on disk directly, the app's
own in-memory state doesn't know about your edit -- the next time it
autosaves (e.g. after any combo attempt) it overwrites your edit with its
stale state. This tool avoids that by talking to the live app over its
websocket API (the same messages the web UI sends) when the app is running,
and only falls back to a direct file edit when it isn't.

Icon field convention (matches existing entries in combos.json)
-----------------------------------------------------------------
  swap_image -> 100x100 head icon (iconrolehead150)
  lmb_image  -> normal attack icon (iconskill/SP_IconNor*.png)
  e          -> resonance skill icon (iconskill/*B1.png)
  r          -> resonance liberation icon (iconskill/*C1.png)
  q          -> NOT the character's own ultimate. It is the recommended
                4-cost "main" Echo icon shown on the character's build page
                (mstskill/T_MstSkil_<id>_UI.png). Characters that share a
                recommended echo (e.g. multiple Glacio DPS) will legitimately
                have the same q icon -- that's not a scraping bug.

Usage
-----
  # Dry run: just show what would be scraped for a character, don't save.
  python tools/ww_character_sync.py fetch suisui

  # Fetch + save a character (tries the live app first, falls back to
  # editing combos.json directly if the app isn't running).
  python tools/ww_character_sync.py add-character suisui
  python tools/ww_character_sync.py add-character lucilla --name Lucilla

  # Create/update a team (slot values are character keys, i.e. lowercase names)
  python tools/ww_character_sync.py add-team "Hiyuki Lucilla Suisui" hiyuki lucilla suisui

  # See what's currently saved
  python tools/ww_character_sync.py list
"""

from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMBOS_PATH = REPO_ROOT / "combos.json"
DEFAULT_WS_URI = "ws://localhost:8765"

USER_AGENT = "Mozilla/5.0 (compatible; ComboTracker-tools)"
IMG_HOST = "https://wuthering.gg"

# The site emits these as HTML-escaped relative paths (e.g. "/_ipx/q_70&amp;s_100x100/...").
# We unescape entities and prefix the host before matching to get real, usable URLs.
ICON_PATTERNS = {
    "swap_image": r"/_ipx/q_70&s_100x100/images/iconrolehead150/T_IconRoleHead150_\d+(?:_UI)?\.png",
    "lmb_image": r"/_ipx/q_70&s_32x32/images/iconskill/SP_IconNor[A-Za-z0-9]*\.png",
    "e": r"/_ipx/q_70&s_32x32/images/iconskill/[A-Za-z0-9_]+B1\.png",
    "r": r"/_ipx/q_70&s_32x32/images/iconskill/[A-Za-z0-9_]+C1\.png",
    "q": r"/_ipx/q_70&s_34x34/images/mstskill/T_MstSkil_\d+_UI\.png",
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def fetch_character_icons(slug: str) -> dict[str, Any]:
    """Scrape https://wuthering.gg/characters/<slug> for the standard icon set."""
    url = f"https://wuthering.gg/characters/{slug.strip().lower()}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}. Check the character slug.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach {url}: {e.reason}") from e

    html = html_mod.unescape(raw_html)

    result: dict[str, Any] = {"swap_image": "", "lmb_image": "", "ability_images": {"q": "", "e": "", "r": ""}}
    missing = []
    for field, pattern in ICON_PATTERNS.items():
        matches = _dedupe_preserve_order(re.findall(pattern, html))
        value = (IMG_HOST + matches[0]) if matches else ""
        if not value:
            missing.append(field)
        if field in ("q", "e", "r"):
            result["ability_images"][field] = value
        else:
            result[field] = value

    if missing:
        print(f"  warning: could not find {', '.join(missing)} on {url} -- fill in manually.", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Live-app sync (preferred) via the same websocket API static/app.js uses
# ---------------------------------------------------------------------------

async def _ws_send_and_wait(messages: list[dict], ws_uri: str) -> bool:
    """Send messages to a running ComboTracker instance. Returns True if connected."""
    try:
        import websockets
    except ImportError:
        return False

    try:
        async with websockets.connect(ws_uri, open_timeout=2) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # initial 'init' payload
            for msg in messages:
                await ws.send(json.dumps(msg))
                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=3)
                    reply_data = json.loads(reply)
                    if reply_data.get("type") == "status" and reply_data.get("color") == "fail":
                        print(f"  server rejected {msg.get('type')}: {reply_data.get('text')}", file=sys.stderr)
                except asyncio.TimeoutError:
                    pass  # no reply means no error, per the app's own protocol
        return True
    except Exception:
        return False


def sync_via_live_app(messages: list[dict], ws_uri: str) -> bool:
    return asyncio.run(_ws_send_and_wait(messages, ws_uri))


# ---------------------------------------------------------------------------
# Direct file fallback (only used when the app isn't running)
# ---------------------------------------------------------------------------

def _load_combos(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"combos.json not found at {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_combos(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_character_direct(path: Path, name: str, icons: dict[str, Any]) -> None:
    data = _load_combos(path)
    chars = data.setdefault("ww_characters", {})
    key = name.strip().lower()
    chars[key] = {
        "name": name.strip(),
        "swap_image": icons["swap_image"],
        "lmb_image": icons["lmb_image"],
        "ability_images": {k: v for k, v in icons["ability_images"].items() if v},
    }
    _save_combos(path, data)


def save_team_direct(path: Path, team_name: str, slots: tuple[str, str, str], team_id: str | None) -> str:
    data = _load_combos(path)
    teams = data.setdefault("ww_teams", {})
    tid = team_id
    if not tid:
        tid = next((t for t, tv in teams.items() if tv.get("name") == team_name), None) or uuid4().hex[:10]
    teams[tid] = {
        "name": team_name,
        "slot1": slots[0].strip().lower(),
        "slot2": slots[1].strip().lower(),
        "slot3": slots[2].strip().lower(),
    }
    data["ww_active_team_id"] = tid
    _save_combos(path, data)
    return tid


def delete_character_direct(path: Path, name: str) -> None:
    data = _load_combos(path)
    chars = data.get("ww_characters", {})
    key = name.strip().lower()
    if key not in chars:
        raise SystemExit(f"'{name}' not found in {path}.")
    ref_teams = [
        tv.get("name", tid)
        for tid, tv in data.get("ww_teams", {}).items()
        if key in (tv.get("slot1"), tv.get("slot2"), tv.get("slot3"))
    ]
    if ref_teams:
        raise SystemExit(f"'{name}' is used by team(s) {', '.join(ref_teams)}. Remove from those teams first.")
    del chars[key]
    _save_combos(path, data)


def delete_team_direct(path: Path, team_id: str) -> None:
    data = _load_combos(path)
    teams = data.get("ww_teams", {})
    if team_id not in teams:
        raise SystemExit(f"Team id '{team_id}' not found in {path}.")
    del teams[team_id]
    if data.get("ww_active_team_id") == team_id:
        data["ww_active_team_id"] = None
    _save_combos(path, data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> None:
    icons = fetch_character_icons(args.slug)
    print(json.dumps({"name": args.name or args.slug.capitalize(), **icons}, indent=2))


def cmd_add_character(args: argparse.Namespace) -> None:
    name = args.name or args.slug.capitalize()
    icons = fetch_character_icons(args.slug)
    if args.swap:
        icons["swap_image"] = args.swap
    if args.lmb:
        icons["lmb_image"] = args.lmb
    if args.q:
        icons["ability_images"]["q"] = args.q
    if args.e:
        icons["ability_images"]["e"] = args.e
    if args.r:
        icons["ability_images"]["r"] = args.r

    print(json.dumps({"name": name, **icons}, indent=2))

    if not args.no_ws:
        msg = {
            "type": "save_character",
            "name": name,
            "swap_image": icons["swap_image"],
            "lmb_image": icons["lmb_image"],
            "ability_images": icons["ability_images"],
        }
        if sync_via_live_app([msg], args.ws_uri):
            print(f"Saved '{name}' via the running app at {args.ws_uri}.")
            return
        print(f"No running app found at {args.ws_uri}; editing {args.file} directly.", file=sys.stderr)

    save_character_direct(args.file, name, icons)
    print(f"Saved '{name}' to {args.file}.")


def cmd_add_team(args: argparse.Namespace) -> None:
    slots = (args.slot1, args.slot2, args.slot3)

    if not args.no_ws:
        msg = {
            "type": "save_team",
            "team_id": args.team_id or "",
            "team_name": args.name,
            "slot1": slots[0],
            "slot2": slots[1],
            "slot3": slots[2],
        }
        if sync_via_live_app([msg], args.ws_uri):
            print(f"Saved team '{args.name}' via the running app at {args.ws_uri}.")
            return
        print(f"No running app found at {args.ws_uri}; editing {args.file} directly.", file=sys.stderr)

    tid = save_team_direct(args.file, args.name, slots, args.team_id)
    print(f"Saved team '{args.name}' ({tid}) to {args.file}.")


def cmd_delete_character(args: argparse.Namespace) -> None:
    if not args.no_ws:
        msg = {"type": "delete_character", "name": args.name}
        if sync_via_live_app([msg], args.ws_uri):
            print(f"Deleted '{args.name}' via the running app at {args.ws_uri}.")
            return
        print(f"No running app found at {args.ws_uri}; editing {args.file} directly.", file=sys.stderr)

    delete_character_direct(args.file, args.name)
    print(f"Deleted '{args.name}' from {args.file}.")


def cmd_delete_team(args: argparse.Namespace) -> None:
    if not args.no_ws:
        msg = {"type": "delete_team", "team_id": args.team_id}
        if sync_via_live_app([msg], args.ws_uri):
            print(f"Deleted team '{args.team_id}' via the running app at {args.ws_uri}.")
            return
        print(f"No running app found at {args.ws_uri}; editing {args.file} directly.", file=sys.stderr)

    delete_team_direct(args.file, args.team_id)
    print(f"Deleted team '{args.team_id}' from {args.file}.")


def cmd_list(args: argparse.Namespace) -> None:
    data = _load_combos(args.file)
    print("Characters:")
    for key, c in sorted(data.get("ww_characters", {}).items()):
        print(f"  {key:16s} {c.get('name', '')}")
    print("\nTeams:")
    for tid, t in data.get("ww_teams", {}).items():
        print(f"  {tid}  {t.get('name', ''):30s} [{t.get('slot1')}, {t.get('slot2')}, {t.get('slot3')}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_COMBOS_PATH, help="Path to combos.json (default: repo root)")
    parser.add_argument("--ws-uri", default=DEFAULT_WS_URI, help="Live app websocket URI (default: %(default)s)")
    parser.add_argument("--no-ws", action="store_true", help="Skip the live app and edit combos.json directly")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Scrape icon URLs for a character without saving anything")
    p_fetch.add_argument("slug", help="wuthering.gg character slug, e.g. 'suisui'")
    p_fetch.add_argument("--name", help="Display name (default: slug, capitalized)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_add_char = sub.add_parser("add-character", help="Scrape and save a character")
    p_add_char.add_argument("slug", help="wuthering.gg character slug, e.g. 'suisui'")
    p_add_char.add_argument("--name", help="Display name (default: slug, capitalized)")
    p_add_char.add_argument("--swap", help="Override the scraped swap_image URL")
    p_add_char.add_argument("--lmb", help="Override the scraped lmb_image URL")
    p_add_char.add_argument("--q", help="Override the scraped q ability image URL")
    p_add_char.add_argument("--e", help="Override the scraped e ability image URL")
    p_add_char.add_argument("--r", help="Override the scraped r ability image URL")
    p_add_char.set_defaults(func=cmd_add_character)

    p_add_team = sub.add_parser("add-team", help="Create or update a 3-slot team")
    p_add_team.add_argument("name", help="Team display name")
    p_add_team.add_argument("slot1")
    p_add_team.add_argument("slot2")
    p_add_team.add_argument("slot3")
    p_add_team.add_argument("--team-id", help="Update a specific existing team id instead of matching by name")
    p_add_team.set_defaults(func=cmd_add_team)

    p_del_char = sub.add_parser("delete-character", help="Delete a character (fails if a team still uses it)")
    p_del_char.add_argument("name", help="Character name (case-insensitive)")
    p_del_char.set_defaults(func=cmd_delete_character)

    p_del_team = sub.add_parser("delete-team", help="Delete a team by id (see 'list' for ids)")
    p_del_team.add_argument("team_id")
    p_del_team.set_defaults(func=cmd_delete_team)

    p_list = sub.add_parser("list", help="List saved characters and teams")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
