from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _clean_counter_dict(obj: Any, *, key_norm) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        kk = key_norm(k)
        if not kk:
            continue
        vv = _as_int(v, 0)
        if vv > 0:
            out[str(kk)] = int(vv)
    return out


def load_engine_state(engine) -> None:
    """
    Load persisted state from `engine.save_path` into the engine instance.

    This function owns JSON schema compatibility, sanitization, and migrations.
    (Keeping it here avoids bloating `ComboTrackerEngine` with persistence concerns.)
    """
    try:
        if not engine.save_path.exists():
            return

        data = json.loads(engine.save_path.read_text(encoding="utf-8"))

        # Combos
        combos = data.get("combos", {})
        if isinstance(combos, dict):
            sanitized: dict[str, list[str]] = {}
            for name, seq in combos.items():
                if not isinstance(name, str) or not isinstance(seq, list):
                    continue
                sanitized[name] = [str(x).strip().lower() for x in seq if str(x).strip()]
            engine.combos = sanitized

        # Enders
        enders = data.get("combo_enders", {})
        parsed: dict[str, int] = {}
        if isinstance(enders, dict):
            for k, v in enders.items():
                key = str(k).strip().lower()
                if not key:
                    continue
                ms = _as_int(v, 0)
                parsed[key] = max(0, ms)
        elif isinstance(enders, list):
            for x in enders:
                key = str(x).strip().lower()
                if key:
                    parsed[key] = 0
        engine.combo_enders = parsed

        # Stats
        stats = data.get("combo_stats", {})
        if isinstance(stats, dict):
            cleaned: dict[str, dict[str, Any]] = {}
            for k, v in stats.items():
                name = str(k).strip()
                if not name or not isinstance(v, dict):
                    continue
                s = max(0, _as_int(v.get("success", 0), 0))
                f = max(0, _as_int(v.get("fail", 0), 0))
                best_raw = v.get("best_ms")
                best_ms = _as_int(best_raw, 0) if best_raw is not None else None
                total_success_ms = max(0, _as_int(v.get("total_success_ms", 0), 0))

                fail_by_step = _clean_counter_dict(v.get("fail_by_step", {}), key_norm=lambda kk: str(kk).strip())
                fail_by_expected = _clean_counter_dict(
                    v.get("fail_by_expected", {}),
                    key_norm=lambda kk: str(kk).strip().lower(),
                )
                fail_by_reason = _clean_counter_dict(
                    v.get("fail_by_reason", {}),
                    key_norm=lambda kk: str(kk).strip().lower(),
                )

                # Keep only recent fail events to cap file growth.
                fail_events = v.get("fail_events", [])
                fes: list[dict[str, Any]] = []
                if isinstance(fail_events, list):
                    for ev in fail_events[-100:]:
                        if not isinstance(ev, dict):
                            continue
                        fes.append(
                            {
                                "ts": int(ev.get("ts", 0) or 0),
                                "attempt": max(0, _as_int(ev.get("attempt", 0), 0)),
                                "step_index": max(0, _as_int(ev.get("step_index", 0), 0)),
                                "expected": str(ev.get("expected", "") or ""),
                                "actual": str(ev.get("actual", "") or ""),
                                "reason": str(ev.get("reason", "") or ""),
                                "elapsed_ms": (_as_int(ev.get("elapsed_ms"), 0) if ev.get("elapsed_ms") is not None else None),
                            }
                        )

                cleaned[name] = {
                    "success": s,
                    "fail": f,
                    "best_ms": best_ms,
                    "total_success_ms": total_success_ms,
                    "fail_by_step": fail_by_step,
                    "fail_by_expected": fail_by_expected,
                    "fail_by_reason": fail_by_reason,
                    "fail_events": fes,
                }
            engine.combo_stats = cleaned

        # Optional: per-combo expected execution time (ms)
        exp = data.get("combo_expected_ms", {})
        expected_ms: dict[str, int] = {}
        if isinstance(exp, dict):
            for k, v in exp.items():
                name = str(k).strip()
                if not name:
                    continue
                ms = _as_int(v, -1)
                if ms > 0:
                    expected_ms[name] = ms
        engine.combo_expected_ms = expected_ms

        # Optional: per-combo user difficulty (0..10)
        ud = data.get("combo_user_difficulty", {})
        user_diff: dict[str, float] = {}
        if isinstance(ud, dict):
            for k, v in ud.items():
                name = str(k).strip()
                if not name:
                    continue
                try:
                    d = float(v)
                except Exception:
                    continue
                if 0.0 <= d <= 10.0:
                    user_diff[name] = d
        engine.combo_user_difficulty = user_diff

        # Optional: per-combo step display mode ("icons" | "images")
        dm = data.get("combo_step_display_mode", {})
        display_mode: dict[str, str] = {}
        if isinstance(dm, dict):
            for k, v in dm.items():
                name = str(k).strip()
                if not name:
                    continue
                mode = str(v or "").strip().lower()
                if mode in ("icons", "images"):
                    display_mode[name] = mode
        engine.combo_step_display_mode = display_mode

        # Optional: per-combo key images mapping
        ki = data.get("combo_key_images", {})
        key_images: dict[str, dict[str, str]] = {}
        if isinstance(ki, dict):
            for k, v in ki.items():
                name = str(k).strip()
                if not name or not isinstance(v, dict):
                    continue
                m: dict[str, str] = {}
                for kk, vv in v.items():
                    key = str(kk).strip().lower()
                    url = str(vv).strip()
                    if not key or not url:
                        continue
                    m[key] = url
                if m:
                    key_images[name] = m
        engine.combo_key_images = key_images

        # ---- Wuthering Waves / target game ----
        # target game per combo
        tg = data.get("combo_target_game", {})
        if isinstance(tg, dict):
            for k, v in tg.items():
                name = str(k).strip()
                if not name:
                    continue
                g = str(v or "").strip().lower()
                if g in ("generic", "wuthering_waves"):
                    engine.ww.combo_target_game[name] = g

        # WW teams
        teams = data.get("ww_teams", {})
        ww_teams: dict[str, dict[str, Any]] = {}
        if isinstance(teams, dict):
            for tid, tv in teams.items():
                team_id = str(tid).strip()
                if not team_id or not isinstance(tv, dict):
                    continue
                name = str(tv.get("name", "") or "").strip() or "Team"
                dash_image = str(tv.get("dash_image", "") or "").strip()
                swap_raw = tv.get("swap_images", {})
                swap_images: dict[str, str] = {}
                if isinstance(swap_raw, dict):
                    for kk, vv in swap_raw.items():
                        k = str(kk or "").strip()
                        if k not in ("1", "2", "3"):
                            continue
                        url = str(vv or "").strip()
                        if url:
                            swap_images[k] = url
                lmb_raw = tv.get("lmb_images", {})
                lmb_images: dict[str, str] = {}
                if isinstance(lmb_raw, dict):
                    for kk, vv in lmb_raw.items():
                        k = str(kk or "").strip()
                        if k not in ("1", "2", "3"):
                            continue
                        url = str(vv or "").strip()
                        if url:
                            lmb_images[k] = url
                abil_raw = tv.get("ability_images", {})
                ability_images: dict[str, dict[str, str]] = {}
                if isinstance(abil_raw, dict):
                    for ck, mapping in abil_raw.items():
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
                            ability_images[c] = m
                ww_teams[team_id] = {
                    "name": name,
                    "dash_image": dash_image,
                    "swap_images": swap_images,
                    "lmb_images": lmb_images,
                    "ability_images": ability_images,
                }
        engine.ww.ww_teams = ww_teams

        active_team = str(data.get("ww_active_team_id") or "").strip()
        engine.ww.ww_active_team_id = active_team if active_team in engine.ww.ww_teams else None

        combo_team = data.get("combo_ww_team", {})
        combo_ww_team: dict[str, str] = {}
        if isinstance(combo_team, dict):
            for k, v in combo_team.items():
                cname = str(k).strip()
                tid = str(v).strip()
                if not cname or not tid:
                    continue
                if tid in engine.ww.ww_teams:
                    combo_ww_team[cname] = tid
        engine.ww.combo_ww_team = combo_ww_team

        # Migration: old per-combo ww ability images -> teams (so presets don't vanish)
        legacy = data.get("combo_ww_ability_images", {})
        if isinstance(legacy, dict):
            for combo_name, mapping in legacy.items():
                cname = str(combo_name).strip()
                if not cname or not isinstance(mapping, dict):
                    continue
                per_char: dict[str, dict[str, str]] = {}
                for ck, cm in mapping.items():
                    c = str(ck or "").strip()
                    if c not in ("1", "2", "3") or not isinstance(cm, dict):
                        continue
                    m: dict[str, str] = {}
                    for akey, av in cm.items():
                        a = str(akey or "").strip().lower()
                        if a not in ("e", "q", "r"):
                            continue
                        url = str(av or "").strip()
                        if url:
                            m[a] = url
                    if m:
                        per_char[c] = m
                if not per_char:
                    continue

                team_id = uuid4().hex[:10]
                swap_images: dict[str, str] = {}
                lmb_images: dict[str, str] = {}
                dash_image = ""
                try:
                    km = (key_images.get(cname) or {})
                    if isinstance(km, dict):
                        for sk in ("1", "2", "3"):
                            url = str(km.get(sk, "") or "").strip()
                            if url:
                                swap_images[sk] = url
                        dash_image = str(km.get("rmb", "") or "").strip()
                        lmb_any = str(km.get("lmb", "") or "").strip()
                        if lmb_any:
                            for sk in ("1", "2", "3"):
                                lmb_images[sk] = lmb_any
                except Exception:
                    pass

                engine.ww.ww_teams[team_id] = {
                    "name": f"Imported: {cname}",
                    "dash_image": dash_image,
                    "swap_images": swap_images,
                    "lmb_images": lmb_images,
                    "ability_images": per_char,
                }
                engine.ww.combo_ww_team[cname] = team_id
                if engine.ww.ww_active_team_id is None:
                    engine.ww.ww_active_team_id = team_id

        last_active = data.get("last_active_combo")
        if last_active in engine.combos:
            engine.set_active_combo(str(last_active), emit=False)
    except Exception:
        logger.exception("Failed to load engine state; resetting to safe defaults")
        # Best-effort: if load fails, reset to a safe empty state.
        engine.combos = {}
        engine.combo_stats = {}
        engine.combo_enders = {}
        engine.combo_expected_ms = {}
        engine.combo_user_difficulty = {}
        engine.combo_step_display_mode = {}
        engine.combo_key_images = {}
        engine.ww.combo_target_game = {}
        engine.ww.ww_teams = {}
        engine.ww.ww_active_team_id = None
        engine.ww.combo_ww_team = {}
        engine.active_combo_name = None
        engine.active_combo_tokens = []
        engine.active_combo_steps = []


def save_engine_state(engine) -> None:
    """
    Persist the engine state to `engine.save_path`.
    """
    try:
        engine.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "last_active_combo": engine.active_combo_name,
            "combos": dict(engine.combos),
            "combo_enders": dict(engine.combo_enders),
            "combo_stats": dict(engine.combo_stats),
            "combo_expected_ms": dict(engine.combo_expected_ms),
            "combo_user_difficulty": dict(engine.combo_user_difficulty),
            "combo_step_display_mode": dict(engine.combo_step_display_mode),
            "combo_key_images": dict(engine.combo_key_images),
            "combo_target_game": dict(engine.ww.combo_target_game),
            "ww_teams": dict(engine.ww.ww_teams),
            "ww_active_team_id": engine.ww.ww_active_team_id,
            "combo_ww_team": dict(engine.ww.combo_ww_team),
        }
        engine.save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save engine state")

