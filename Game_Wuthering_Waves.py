from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class WutheringWavesGame:
    """
    Wuthering Waves specific state + helpers.

    This module exists to keep `ComboTrackerEngine` focused on core combo parsing/tracking,
    while game-specific metadata (target game, WW teams/presets, WW combo->team mappings)
    lives in one place.
    """

    # Per-combo target game ("generic" | "wuthering_waves")
    combo_target_game: dict[str, str] = field(default_factory=dict)

    # Team presets (slot-based; characters are stored in ww_characters)
    # ww_teams: team_id -> {
    #   "name": str,
    #   "slot1": char_name_lower | "",
    #   "slot2": char_name_lower | "",
    #   "slot3": char_name_lower | "",
    # }
    ww_teams: dict[str, dict[str, Any]] = field(default_factory=dict)
    ww_active_team_id: str | None = None

    # Per-combo assigned team (when target_game = wuthering_waves)
    combo_ww_team: dict[str, str] = field(default_factory=dict)

    # Character library: name_lower -> { name, swap_image, lmb_image, ability_images: {e,q,r} }
    ww_characters: dict[str, Any] = field(default_factory=dict)

    # Global dash / RMB icon (shared across all teams)
    ww_dash_image: str = ""

    # Active character slot during an attempt ("1", "2", or "3").
    ww_active_character: str | None = None

    # Character slot keys (for WW and future games with swap slots)
    WW_CHARACTER_SLOTS = ("1", "2", "3")

    # -------------------------
    # Ender / combo-end policy (engine delegates here to avoid game-specific branches)
    # -------------------------

    def can_ender_drop_combo(self, engine: Any, input_name: str) -> bool:
        """
        True iff this key is allowed to end the combo.
        Only combo enders can end combos, and only when off cooldown.
        WW: pressing the current character slot (1/2/3) does not end the combo until you switch off.
        """
        if not (input_name or "").strip():
            return False
        if not engine._is_combo_ender(input_name):
            return False
        if engine._ender_on_cooldown(input_name):
            return False
        if not getattr(engine, "last_input_time", None) and getattr(engine, "current_index", 0) == 0:
            return False
        if self.get_target_game(getattr(engine, "active_combo_name", None) or "") == "wuthering_waves":
            key = (input_name or "").strip().lower()
            if key in self.WW_CHARACTER_SLOTS and self.ww_active_character and key == self.ww_active_character:
                return False
        return True

    def on_accepted_key(self, engine: Any, input_name: str) -> None:
        """When the correct key is 1/2/3 and game is WW, track active character for ender logic."""
        if self.get_target_game(getattr(engine, "active_combo_name", None) or "") != "wuthering_waves":
            return
        key = (input_name or "").strip().lower()
        if key in self.WW_CHARACTER_SLOTS:
            self.ww_active_character = key

    def get_target_game(self, combo_name: str) -> str:
        name = (combo_name or "").strip()
        g = str(self.combo_target_game.get(name, "generic") or "generic").strip().lower()
        return g if g in ("generic", "wuthering_waves") else "generic"

    def set_target_game(self, combo_name: str, target_game: str | None):
        name = (combo_name or "").strip()
        g = str(target_game or "").strip().lower()
        if not name:
            return
        if g in ("generic", "wuthering_waves"):
            self.combo_target_game[name] = g
        else:
            self.combo_target_game.pop(name, None)

    def apply_combo_team_assignment(self, combo_name: str, *, target_game: str, ww_team_id: str | None):
        """
        Apply per-combo WW team assignment.
        Expected to be called after `set_target_game()`.
        """
        name = (combo_name or "").strip()
        if not name:
            return
        g = str(target_game or "generic").strip().lower()
        if g == "wuthering_waves":
            tid = str(ww_team_id or "").strip()
            if tid and tid in self.ww_teams:
                self.combo_ww_team[name] = tid
                self.ww_active_team_id = tid
            else:
                self.combo_ww_team.pop(name, None)
        else:
            self.combo_ww_team.pop(name, None)

    def rename_combo(self, old_name: str, new_name: str):
        old = (old_name or "").strip()
        new = (new_name or "").strip()
        if not old or not new or old == new:
            return
        if old in self.combo_target_game and new not in self.combo_target_game:
            self.combo_target_game[new] = self.combo_target_game.pop(old)
        if old in self.combo_ww_team and new not in self.combo_ww_team:
            self.combo_ww_team[new] = self.combo_ww_team.pop(old)

    def delete_combo(self, name: str):
        cname = (name or "").strip()
        if not cname:
            return
        self.combo_target_game.pop(cname, None)
        self.combo_ww_team.pop(cname, None)

    # -------------------------
    # Character library helpers
    # -------------------------

    def _resolve_slot(self, slot_name: str) -> dict[str, Any] | None:
        key = (slot_name or "").strip().lower()
        if not key:
            return None
        return self.ww_characters.get(key)

    def _resolved_team_images(self, team_data: dict[str, Any]) -> tuple[dict, dict, dict]:
        """Returns (swap_images, lmb_images, ability_images) keyed by slot '1'/'2'/'3'."""
        swap: dict[str, str] = {}
        lmb: dict[str, str] = {}
        ability: dict[str, dict[str, str]] = {}
        for field_name, sk in zip(("slot1", "slot2", "slot3"), ("1", "2", "3")):
            char = self._resolve_slot(str(team_data.get(field_name, "") or ""))
            if not char:
                continue
            if char.get("swap_image"):
                swap[sk] = char["swap_image"]
            if char.get("lmb_image"):
                lmb[sk] = char["lmb_image"]
            ai = char.get("ability_images") or {}
            if isinstance(ai, dict):
                m = {a: str(v) for a, v in ai.items() if a in ("e", "q", "r") and str(v or "").strip()}
                if m:
                    ability[sk] = m
        return swap, lmb, ability

    # -------------------------
    # Editor payload helpers
    # -------------------------

    def editor_payload(self, combo_name: str, target_game_override: str | None = None) -> dict[str, Any]:
        """
        Build the WW-related section of the editor payload for the frontend.
        """
        name = (combo_name or "").strip()
        target_game = str(target_game_override).strip().lower() if target_game_override else self.get_target_game(name)

        # Build teams list with slot info
        ww_teams = []
        for tid, tv in self.ww_teams.items():
            if not isinstance(tv, dict):
                continue
            ww_teams.append({
                "id": str(tid),
                "name": str(tv.get("name", "") or "Team"),
                "slot1": str(tv.get("slot1", "") or ""),
                "slot2": str(tv.get("slot2", "") or ""),
                "slot3": str(tv.get("slot3", "") or ""),
            })
        ww_teams.sort(key=lambda x: (x.get("name") or "").lower())

        # Selected team: active team > combo assignment > none
        sel_team_id = ""
        if target_game == "wuthering_waves":
            if self.ww_active_team_id and self.ww_active_team_id in self.ww_teams:
                sel_team_id = self.ww_active_team_id
            elif name and name in self.combo_ww_team and self.combo_ww_team[name] in self.ww_teams:
                sel_team_id = self.combo_ww_team[name]

        team_name = ""
        team_slots = {"slot1": "", "slot2": "", "slot3": ""}
        team_swap_images: dict[str, str] = {}
        team_lmb_images: dict[str, str] = {}
        team_ability_images: dict[str, dict[str, str]] = {}
        if sel_team_id and sel_team_id in self.ww_teams:
            tv = self.ww_teams[sel_team_id]
            team_name = str(tv.get("name", "") or "")
            team_slots = {
                "slot1": str(tv.get("slot1", "") or ""),
                "slot2": str(tv.get("slot2", "") or ""),
                "slot3": str(tv.get("slot3", "") or ""),
            }
            team_swap_images, team_lmb_images, team_ability_images = self._resolved_team_images(tv)

        # Characters list sorted by display name
        ww_chars_list = []
        for key, cv in sorted(
            self.ww_characters.items(),
            key=lambda x: (x[1].get("name") or "").lower() if isinstance(x[1], dict) else "",
        ):
            if not isinstance(cv, dict):
                continue
            ww_chars_list.append({
                "name": str(cv.get("name", "") or key),
                "name_key": key,
                "swap_image": str(cv.get("swap_image", "") or ""),
                "lmb_image": str(cv.get("lmb_image", "") or ""),
                "ability_images": {
                    a: str(v)
                    for a, v in (cv.get("ability_images") or {}).items()
                    if a in ("e", "q", "r") and str(v or "").strip()
                },
            })

        return {
            "target_game": target_game,
            "ww_teams": ww_teams,
            "ww_team_id": sel_team_id,
            "ww_team_name": team_name,
            "ww_team_slots": team_slots,
            "ww_dash_image": self.ww_dash_image,
            "ww_team_dash_image": self.ww_dash_image,  # backward-compat alias for timeline
            "ww_team_swap_images": team_swap_images,
            "ww_team_lmb_images": team_lmb_images,
            "ww_team_ability_images": team_ability_images,
            "ww_characters": ww_chars_list,
        }

    # -------------------------
    # Character operations
    # -------------------------

    def save_character(
        self,
        name: str,
        swap_image: str,
        lmb_image: str,
        ability_images: Any,
    ) -> tuple[bool, str | None]:
        n = (name or "").strip()
        if not n:
            return False, "Please provide a character name."
        key = n.lower()
        ai: dict[str, str] = {}
        if isinstance(ability_images, dict):
            for a, v in ability_images.items():
                akey = str(a or "").strip().lower()
                if akey in ("e", "q", "r"):
                    url = str(v or "").strip()
                    if url:
                        ai[akey] = url
        self.ww_characters[key] = {
            "name": n,
            "swap_image": str(swap_image or "").strip(),
            "lmb_image": str(lmb_image or "").strip(),
            "ability_images": ai,
        }
        return True, None

    def delete_character(self, name: str) -> tuple[bool, str | None]:
        key = (name or "").strip().lower()
        if not key:
            return False, "No character name provided."
        if key not in self.ww_characters:
            return False, "Character not found."
        # Block deletion if any team references this character
        ref_teams: list[str] = []
        for tid, tv in self.ww_teams.items():
            if not isinstance(tv, dict):
                continue
            slots = [
                str(tv.get("slot1", "") or "").lower(),
                str(tv.get("slot2", "") or "").lower(),
                str(tv.get("slot3", "") or "").lower(),
            ]
            if key in slots:
                ref_teams.append(str(tv.get("name", "") or tid))
        if ref_teams:
            team_list = ", ".join(ref_teams)
            return False, f"Remove from these teams first: {team_list}"
        del self.ww_characters[key]
        return True, None

    # -------------------------
    # Team operations (called by engine)
    # -------------------------

    def set_active_ww_team(self, team_id: str):
        tid = str(team_id or "").strip()
        if tid and tid in self.ww_teams:
            self.ww_active_team_id = tid
            return
        self.ww_active_team_id = None

    def save_or_update_ww_team(
        self,
        *,
        team_id: str,
        team_name: str,
        slot1: str,
        slot2: str,
        slot3: str,
    ) -> tuple[bool, str | None, str]:
        """Returns (ok, err, resolved_team_id)."""
        tid = str(team_id or "").strip() or uuid4().hex[:10]
        name = str(team_name or "").strip() or "Team"
        self.ww_teams[tid] = {
            "name": name,
            "slot1": str(slot1 or "").strip().lower(),
            "slot2": str(slot2 or "").strip().lower(),
            "slot3": str(slot3 or "").strip().lower(),
        }
        self.ww_active_team_id = tid
        return True, None, tid

    def delete_ww_team(self, team_id: str) -> tuple[bool, str | None]:
        tid = str(team_id or "").strip()
        if not tid or tid not in self.ww_teams:
            return False, "Select a team to delete."

        del self.ww_teams[tid]
        for cname, ct in list(self.combo_ww_team.items()):
            if ct == tid:
                del self.combo_ww_team[cname]
        if self.ww_active_team_id == tid:
            self.ww_active_team_id = None
        return True, None


# -------------------------
# UI command glue (engine-coupled; caller holds engine._lock)
# -------------------------


def set_active_ww_team(engine, team_id: str) -> None:
    tid = str(team_id or "").strip()
    if tid and tid in engine.ww.ww_teams:
        engine.ww.set_active_ww_team(tid)
        engine.save_combos()
        engine._send({"type": "combo_data", **engine.get_editor_payload()})
        engine._send({"type": "timeline_update", "steps": engine.timeline_steps()})


def save_or_update_ww_team(
    engine,
    *,
    team_id: str | None,
    team_name: str | None,
    slot1: str,
    slot2: str,
    slot3: str,
) -> tuple[bool, str | None]:
    name = str(team_name or "").strip()
    if not name:
        return False, "Please provide a Team name."

    ok, err, _tid = engine.ww.save_or_update_ww_team(
        team_id=str(team_id or "").strip(),
        team_name=name,
        slot1=slot1,
        slot2=slot2,
        slot3=slot3,
    )
    if not ok:
        return False, err

    engine.save_combos()
    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def delete_ww_team(engine, team_id: str) -> tuple[bool, str | None]:
    ok, err = engine.ww.delete_ww_team(team_id)
    if not ok:
        return False, err
    engine.save_combos()
    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def select_team_stateless(engine, team_id: str, target_game: str) -> None:
    """Stateless team selection - doesn't persist, just updates UI."""
    tid = str(team_id or "").strip()
    game = str(target_game or "generic").strip().lower()

    if game == "wuthering_waves":
        engine.ww.set_active_ww_team(tid)
        engine._send({"type": "combo_data", **engine.get_editor_payload(target_game_override=game)})
        engine._send({"type": "timeline_update", "steps": engine.timeline_steps()})
        return

    engine.ww.set_active_ww_team("")
    engine._send({"type": "combo_data", **engine.get_editor_payload(target_game_override=game)})
    engine._send({"type": "timeline_update", "steps": engine.timeline_steps()})


def update_target_game_stateless(engine, target_game: str) -> None:
    """Stateless target game update - doesn't persist, just updates UI."""
    game = str(target_game or "generic").strip().lower()
    if game not in ("generic", "wuthering_waves"):
        game = "generic"

    payload = engine.get_editor_payload(target_game_override=game)
    payload["target_game"] = game
    engine._send({"type": "combo_data", **payload})


def save_ww_character_cmd(
    engine,
    *,
    name: str,
    swap_image: str,
    lmb_image: str,
    ability_images: Any,
) -> tuple[bool, str | None]:
    ok, err = engine.ww.save_character(name, swap_image, lmb_image, ability_images)
    if not ok:
        return False, err
    engine.save_combos()
    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def delete_ww_character_cmd(engine, *, name: str) -> tuple[bool, str | None]:
    ok, err = engine.ww.delete_character(name)
    if not ok:
        return False, err
    engine.save_combos()
    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def update_ww_dash_cmd(engine, *, dash_image: str) -> tuple[bool, str | None]:
    engine.ww.ww_dash_image = str(dash_image or "").strip()
    engine.save_combos()
    engine._send({"type": "combo_data", **engine.get_editor_payload()})
    engine._send({"type": "timeline_update", "steps": engine.timeline_steps()})
    return True, None
