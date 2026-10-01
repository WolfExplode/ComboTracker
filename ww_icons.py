"""
Wuthering Waves character icons scraped from wuthering.gg.

Used by the app's "Add from wuthering.gg" / "Refresh all icons" buttons and by
tools/ww_character_sync.py.

Icon field convention (matches existing entries in combos.json)
  swap_image -> 100x100 head icon (iconrolehead150)
  lmb_image  -> normal attack icon (iconskill/SP_IconNor*.png)
  e          -> resonance skill icon (iconskill/*B1.png)
  r          -> resonance liberation icon (iconskill/*C1.png)
  q          -> the recommended 4-cost main Echo icon from the build page
                (mstskill/T_MstSkil_<id>_UI.png); characters sharing an Echo
                legitimately share this icon.
"""

from __future__ import annotations

import html as html_mod
import re
import urllib.error
import urllib.request
from typing import Any

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


def character_slug(name: str) -> str:
    """wuthering.gg page slug for a character name: 'Phrolova' -> 'phrolova', 'Jiyan Zhe' -> 'jiyan-zhe'."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def parse_character_icons(raw_html: str) -> dict[str, Any]:
    """
    Pull the standard icon set out of a wuthering.gg character page.
    Returns {"swap_image", "lmb_image", "ability_images": {q,e,r}, "missing": [field, ...]}.
    """
    page = html_mod.unescape(raw_html)
    result: dict[str, Any] = {"swap_image": "", "lmb_image": "", "ability_images": {"q": "", "e": "", "r": ""}}
    missing: list[str] = []
    for field, pattern in ICON_PATTERNS.items():
        m = re.search(pattern, page)
        value = (IMG_HOST + m.group(0)) if m else ""
        if not value:
            missing.append(field)
        if field in ("q", "e", "r"):
            result["ability_images"][field] = value
        else:
            result[field] = value
    result["missing"] = missing
    return result


def fetch_character_icons(slug: str) -> dict[str, Any]:
    """Scrape https://wuthering.gg/characters/<slug>. Raises RuntimeError when the page can't be read."""
    url = f"{IMG_HOST}/characters/{character_slug(slug)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"wuthering.gg has no character page named '{slug}'. Check the spelling.") from e
        raise RuntimeError(f"{url} -> HTTP {e.code}.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach wuthering.gg: {e.reason}") from e

    result = parse_character_icons(raw_html)
    if not result["swap_image"] and len(result["missing"]) == len(ICON_PATTERNS):
        raise RuntimeError(f"No icons found on {url}. The site layout may have changed.")
    return result
