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

    # Team presets
    # ww_teams: team_id -> {
    #   "name": str,
    #   "dash_image": str,  # RMB / dash (shared)
    #   "swap_images": {"1":..,"2":..,"3":..},
    #   "lmb_images": {"1":..,"2":..,"3":..},
    #   "ability_images": {"1":{"e":..,"q":..,"r":..}, ...}
    # }
    ww_teams: dict[str, dict[str, Any]] = field(default_factory=dict)
    ww_active_team_id: str | None = None

    # Per-combo assigned team (when target_game = wuthering_waves)
    combo_ww_team: dict[str, str] = field(default_factory=dict)

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
    # Editor payload helpers
    # -------------------------
    def editor_payload(self, combo_name: str) -> dict[str, Any]:
        """
        Build the WW-related section of the editor payload for the frontend.
        Returns keys:
        - target_game
        - ww_teams
        - ww_team_id
        - ww_team_name
        - ww_team_dash_image
        - ww_team_swap_images
        - ww_team_lmb_images
        - ww_team_ability_images
        """
        name = (combo_name or "").strip()
        target_game = self.get_target_game(name)

        ww_teams = []
        for tid, tv in self.ww_teams.items():
            if not isinstance(tv, dict):
                continue
            ww_teams.append({"id": str(tid), "name": str(tv.get("name", "") or "Team")})
        ww_teams.sort(key=lambda x: (x.get("name") or "").lower())

        # Selected team: combo assignment > active team > none
        sel_team_id = ""
        if target_game == "wuthering_waves":
            if name and name in self.combo_ww_team and self.combo_ww_team[name] in self.ww_teams:
                sel_team_id = self.combo_ww_team[name]
            elif self.ww_active_team_id and self.ww_active_team_id in self.ww_teams:
                sel_team_id = self.ww_active_team_id

        team_name = ""
        team_swap_images: dict[str, str] = {}
        team_lmb_images: dict[str, str] = {}
        team_dash_image = ""
        team_ability_images: dict[str, dict[str, str]] = {}
        if sel_team_id and sel_team_id in self.ww_teams:
            tv = self.ww_teams.get(sel_team_id) or {}
            team_name = str(tv.get("name", "") or "")
            team_dash_image = str(tv.get("dash_image", "") or "")
            si = tv.get("swap_images", {})
            if isinstance(si, dict):
                team_swap_images = {k: str(v) for k, v in si.items() if k in ("1", "2", "3") and str(v).strip()}
            li = tv.get("lmb_images", {})
            if isinstance(li, dict):
                team_lmb_images = {k: str(v) for k, v in li.items() if k in ("1", "2", "3") and str(v).strip()}
            ai = tv.get("ability_images", {})
            if isinstance(ai, dict):
                for ck, vv in ai.items():
                    if ck in ("1", "2", "3") and isinstance(vv, dict):
                        team_ability_images[ck] = {a: str(u) for a, u in vv.items() if a in ("e", "q", "r") and str(u).strip()}

        return {
            "target_game": target_game,
            "ww_teams": ww_teams,
            "ww_team_id": sel_team_id,
            "ww_team_name": team_name,
            "ww_team_dash_image": team_dash_image,
            "ww_team_swap_images": team_swap_images,
            "ww_team_lmb_images": team_lmb_images,
            "ww_team_ability_images": team_ability_images,
        }

    # -------------------------
    # Team operations (called by engine)
    # -------------------------
    def set_active_ww_team(self, team_id: str):
        tid = str(team_id or "").strip()
        if tid and tid in self.ww_teams:
            self.ww_active_team_id = tid

    def save_or_update_ww_team(
        self,
        *,
        team_id: str,
        team_name: str,
        dash_image: str | None,
        swap_images: Any | None,
        lmb_images: Any | None,
        ability_images: Any | None,
    ) -> tuple[bool, str | None, str]:
        """
        Returns (ok, err, resolved_team_id).
        """
        tid = str(team_id or "").strip() or uuid4().hex[:10]
        name = str(team_name or "").strip() or "Team"
        dash = str(dash_image or "").strip()

        # Sanitize swap/lmb maps
        swap: dict[str, str] = {}
        if isinstance(swap_images, dict):
            for k, v in swap_images.items():
                kk = str(k or "").strip()
                if kk not in ("1", "2", "3"):
                    continue
                url = str(v or "").strip()
                if url:
                    swap[kk] = url

        lmb: dict[str, str] = {}
        if isinstance(lmb_images, dict):
            for k, v in lmb_images.items():
                kk = str(k or "").strip()
                if kk not in ("1", "2", "3"):
                    continue
                url = str(v or "").strip()
                if url:
                    lmb[kk] = url

        abil: dict[str, dict[str, str]] = {}
        if isinstance(ability_images, dict):
            for ck, mapping in ability_images.items():
                c = str(ck or "").strip()
                if c not in ("1", "2", "3") or not isinstance(mapping, dict):
                    continue
                m: dict[str, str] = {}
                for akey, av in mapping.items():
                    a = str(akey or "").strip().lower()
                    if a not in ("e", "q", "r"):
                        continue
                    url = str(av or "").strip()
                    if url:
                        m[a] = url
                if m:
                    abil[c] = m

        self.ww_teams[tid] = {"name": name, "dash_image": dash, "swap_images": swap, "lmb_images": lmb, "ability_images": abil}
        self.ww_active_team_id = tid
        return True, None, tid

    def delete_ww_team(self, team_id: str) -> tuple[bool, str | None]:
        tid = str(team_id or "").strip()
        if not tid or tid not in self.ww_teams:
            return False, "Select a team to delete."

        del self.ww_teams[tid]
        # Remove any combo mappings pointing at this team
        for cname, ct in list(self.combo_ww_team.items()):
            if ct == tid:
                del self.combo_ww_team[cname]
        if self.ww_active_team_id == tid:
            self.ww_active_team_id = None
        return True, None

