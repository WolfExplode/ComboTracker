"""
One-click character sync: add or refresh Wuthering Waves characters from wuthering.gg
(the websocket "sync_character" / "sync_all_characters" messages).
"""

from __future__ import annotations

import asyncio
from typing import Any

from ww_icons import fetch_character_icons


def _send_status(engine: Any, text: str, color: str) -> None:
    # Goes through the engine's emitter (not a direct websocket send) so it lands after the
    # init payload that saving a character broadcasts, instead of being overwritten by it.
    engine._send({"type": "status", "text": text, "color": color})


async def sync_character(engine: Any, name: str) -> None:
    """Add or refresh one character from wuthering.gg, keeping any icon the site doesn't have."""
    name = " ".join((name or "").split())
    if not name:
        _send_status(engine, "Type a character name first.", "fail")
        return
    _send_status(engine, f"Fetching {name} from wuthering.gg...", "neutral")
    try:
        icons = await asyncio.to_thread(fetch_character_icons, name)
    except RuntimeError as e:
        _send_status(engine, str(e), "fail")
        return

    existing = engine.ww.ww_characters.get(name.lower()) or {}
    old_abilities = existing.get("ability_images") or {}
    display_name = existing.get("name") or name.title()
    ok, err = engine.save_ww_character(
        name=display_name,
        swap_image=icons["swap_image"] or existing.get("swap_image", ""),
        lmb_image=icons["lmb_image"] or existing.get("lmb_image", ""),
        ability_images={k: v or old_abilities.get(k, "") for k, v in icons["ability_images"].items()},
    )
    if not ok:
        _send_status(engine, err or f"Could not save {display_name}.", "fail")
    elif icons["missing"]:
        _send_status(
            engine,
            f"Saved {display_name}, but wuthering.gg had no {', '.join(icons['missing'])} icon. Add it by hand.",
            "fail",
        )
    else:
        _send_status(engine, f"Saved {display_name} with icons from wuthering.gg.", "success")


async def refresh_all_characters(engine: Any) -> None:
    """Re-fetch icons for every saved character (fixes broken links after site updates)."""
    chars = [dict(c, key=k) for k, c in engine.ww.ww_characters.items() if isinstance(c, dict)]
    if not chars:
        _send_status(engine, "No characters saved yet.", "fail")
        return
    failed: list[str] = []
    for i, ch in enumerate(chars, 1):
        name = ch.get("name") or ch["key"]
        _send_status(engine, f"Refreshing icons {i}/{len(chars)}: {name}...", "neutral")
        try:
            icons = await asyncio.to_thread(fetch_character_icons, name)
        except RuntimeError:
            failed.append(name)
            continue
        old_abilities = ch.get("ability_images") or {}
        engine.save_ww_character(
            name=name,
            swap_image=icons["swap_image"] or ch.get("swap_image", ""),
            lmb_image=icons["lmb_image"] or ch.get("lmb_image", ""),
            ability_images={k: v or old_abilities.get(k, "") for k, v in icons["ability_images"].items()},
        )
    if failed:
        _send_status(
            engine,
            f"Refreshed {len(chars) - len(failed)} of {len(chars)}. Not found on wuthering.gg: {', '.join(failed)}.",
            "fail",
        )
    else:
        _send_status(engine, f"Refreshed icons for all {len(chars)} characters.", "success")
