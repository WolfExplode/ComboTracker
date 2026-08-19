from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def fresh_combo_stats() -> dict[str, Any]:
    """Return a new combo-stats dict (single source of truth for shape)."""
    return {
        "success": 0,
        "fail": 0,
        "best_ms": None,
        "total_success_ms": 0,
        "fail_by_step": {},
        "fail_by_expected": {},
        "fail_by_reason": {},
        "fail_events": [],
    }


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
    Load persisted state from `engine.state_store` into the engine instance.

    This function owns JSON schema compatibility, sanitization, and migrations.
    (Keeping it here avoids bloating `ComboTrackerEngine` with persistence concerns.)
    """
    try:
        data = engine.state_store.load()
        if data is None:
            return

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

        # Soft enders (~key:2s): do not drop combo when pressed during hold
        soft = data.get("combo_enders_soft", [])
        if isinstance(soft, list):
            engine.combo_enders_soft = {str(x).strip().lower() for x in soft if str(x).strip()}
        else:
            engine.combo_enders_soft = set()

        # Transcribe valid keys (comma-separated string)
        tvk = data.get("transcribe_valid_keys")
        if isinstance(tvk, str):
            engine.transcribe_valid_keys = tvk.strip()
        else:
            engine.transcribe_valid_keys = getattr(engine, "transcribe_valid_keys", "") or ""

        tsk = data.get("transcribe_start_key")
        if isinstance(tsk, str) and tsk.strip():
            engine.transcribe_start_key = tsk.strip().lower()
        else:
            engine.transcribe_start_key = getattr(engine, "transcribe_start_key", "f") or "f"

        tst = data.get("transcribe_strip_wait_under_enabled")
        if isinstance(tst, bool):
            engine.transcribe_strip_wait_under_enabled = tst
        elif tst is None:
            engine.transcribe_strip_wait_under_enabled = False
        else:
            engine.transcribe_strip_wait_under_enabled = bool(tst)

        tsms = data.get("transcribe_strip_wait_under_ms")
        if isinstance(tsms, str):
            engine.transcribe_strip_wait_under_ms = tsms.strip()
        elif isinstance(tsms, (int, float)):
            engine.transcribe_strip_wait_under_ms = str(int(tsms))
        elif tsms is None:
            engine.transcribe_strip_wait_under_ms = "0"
        else:
            engine.transcribe_strip_wait_under_ms = str(tsms).strip()

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

        # Per-combo demo video URL (e.g. YouTube)
        dv = data.get("combo_demo_video", {})
        demo_video: dict[str, str] = {}
        if isinstance(dv, dict):
            for k, v in dv.items():
                name = str(k).strip()
                url = str(v or "").strip()
                if name and url:
                    demo_video[name] = url
        engine.combo_demo_video = demo_video

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

        # WW character library
        chars_raw = data.get("ww_characters", {})
        ww_characters: dict[str, Any] = {}
        if isinstance(chars_raw, dict):
            for ckey, cv in chars_raw.items():
                key = str(ckey or "").strip().lower()
                if not key or not isinstance(cv, dict):
                    continue
                ai: dict[str, str] = {}
                ai_raw = cv.get("ability_images", {})
                if isinstance(ai_raw, dict):
                    for a, v in ai_raw.items():
                        akey = str(a or "").strip().lower()
                        if akey in ("e", "q", "r"):
                            url = str(v or "").strip()
                            if url:
                                ai[akey] = url
                ww_characters[key] = {
                    "name": str(cv.get("name", "") or key),
                    "swap_image": str(cv.get("swap_image", "") or "").strip(),
                    "lmb_image": str(cv.get("lmb_image", "") or "").strip(),
                    "ability_images": ai,
                }
        engine.ww.ww_characters = ww_characters

        # Global dash image
        engine.ww.ww_dash_image = str(data.get("ww_dash_image", "") or "").strip()

        # WW teams (new slot-based format; migrates from old embedded-image format)
        teams = data.get("ww_teams", {})
        ww_teams: dict[str, dict[str, Any]] = {}
        if isinstance(teams, dict):
            for tid, tv in teams.items():
                team_id = str(tid).strip()
                if not team_id or not isinstance(tv, dict):
                    continue
                team_name = str(tv.get("name", "") or "").strip() or "Team"

                if "slot1" in tv or "slot2" in tv or "slot3" in tv:
                    # New format
                    ww_teams[team_id] = {
                        "name": team_name,
                        "slot1": str(tv.get("slot1", "") or "").strip().lower(),
                        "slot2": str(tv.get("slot2", "") or "").strip().lower(),
                        "slot3": str(tv.get("slot3", "") or "").strip().lower(),
                    }
                else:
                    # Old format: migrate embedded swap/lmb/ability per slot into characters
                    dash_old = str(tv.get("dash_image", "") or "").strip()
                    if dash_old and not engine.ww.ww_dash_image:
                        engine.ww.ww_dash_image = dash_old
                    swap_old = tv.get("swap_images") or {}
                    lmb_old = tv.get("lmb_images") or {}
                    abil_old = tv.get("ability_images") or {}
                    slot_chars: dict[str, str] = {}
                    for slot in ("1", "2", "3"):
                        has_swap = isinstance(swap_old, dict) and str(swap_old.get(slot, "") or "").strip()
                        has_lmb = isinstance(lmb_old, dict) and str(lmb_old.get(slot, "") or "").strip()
                        has_abil = (
                            isinstance(abil_old, dict)
                            and isinstance(abil_old.get(slot), dict)
                            and any(str(v or "").strip() for v in abil_old[slot].values())
                        )
                        if has_swap or has_lmb or has_abil:
                            char_display = f"{team_name} {slot}"
                            char_key = char_display.lower()
                            suffix = 0
                            while char_key in engine.ww.ww_characters:
                                suffix += 1
                                char_key = f"{char_display.lower()}_{suffix}"
                                char_display = f"{team_name} {slot}_{suffix}"
                            slot_ai: dict[str, str] = {}
                            if isinstance(abil_old, dict) and isinstance(abil_old.get(slot), dict):
                                for a, v in abil_old[slot].items():
                                    akey = str(a or "").strip().lower()
                                    if akey in ("e", "q", "r"):
                                        url = str(v or "").strip()
                                        if url:
                                            slot_ai[akey] = url
                            engine.ww.ww_characters[char_key] = {
                                "name": char_display,
                                "swap_image": str(swap_old.get(slot, "") or "").strip() if isinstance(swap_old, dict) else "",
                                "lmb_image": str(lmb_old.get(slot, "") or "").strip() if isinstance(lmb_old, dict) else "",
                                "ability_images": slot_ai,
                            }
                            slot_chars[slot] = char_key
                        else:
                            slot_chars[slot] = ""
                    ww_teams[team_id] = {
                        "name": team_name,
                        "slot1": slot_chars.get("1", ""),
                        "slot2": slot_chars.get("2", ""),
                        "slot3": slot_chars.get("3", ""),
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

        # Migration: old per-combo ww ability images -> character library + teams
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
                slot_chars_legacy: dict[str, str] = {}
                for slot in ("1", "2", "3"):
                    slot_abil = per_char.get(slot, {})
                    char_display = f"Imported: {cname} {slot}"
                    char_key = char_display.lower()
                    suffix = 0
                    while char_key in engine.ww.ww_characters:
                        suffix += 1
                        char_key = f"{char_display.lower()}_{suffix}"
                    engine.ww.ww_characters[char_key] = {
                        "name": char_display,
                        "swap_image": "",
                        "lmb_image": "",
                        "ability_images": slot_abil,
                    }
                    slot_chars_legacy[slot] = char_key

                engine.ww.ww_teams[team_id] = {
                    "name": f"Imported: {cname}",
                    "slot1": slot_chars_legacy.get("1", ""),
                    "slot2": slot_chars_legacy.get("2", ""),
                    "slot3": slot_chars_legacy.get("3", ""),
                }
                engine.ww.combo_ww_team[cname] = team_id
                if engine.ww.ww_active_team_id is None:
                    engine.ww.ww_active_team_id = team_id

        no_fail = data.get("no_fail_mode")
        engine.no_fail_mode = bool(no_fail) if no_fail is not None else getattr(engine, "no_fail_mode", False)

        macro_start = data.get("macro_start_key")
        if isinstance(macro_start, str) and macro_start.strip():
            engine.macro_start_key = macro_start.strip().lower()
        else:
            engine.macro_start_key = getattr(engine, "macro_start_key", "f8") or "f8"

        macro_stop = data.get("macro_stop_key")
        if isinstance(macro_stop, str) and macro_stop.strip():
            engine.macro_stop_key = macro_stop.strip().lower()
        else:
            engine.macro_stop_key = getattr(engine, "macro_stop_key", "f9") or "f9"

        macro_spam_interval = data.get("macro_spam_interval_ms")
        if macro_spam_interval in (None, ""):
            engine.macro_spam_interval_ms = None
        else:
            try:
                iv = int(macro_spam_interval)
            except Exception:
                fallback = getattr(engine, "macro_spam_interval_ms", 100)
                if fallback in (None, ""):
                    iv = 100
                else:
                    iv = int(fallback)
            engine.macro_spam_interval_ms = iv if iv > 0 else None

        last_active = data.get("last_active_combo")
        if last_active in engine.combos:
            engine.set_active_combo(str(last_active), emit=False)
    except Exception:
        logger.exception("Failed to load engine state; resetting to safe defaults")
        # Best-effort: if load fails, reset to a safe empty state.
        engine.combos = {}
        engine.combo_stats = {}
        engine.combo_enders = {}
        engine.combo_enders_soft = set()
        engine.combo_expected_ms = {}
        engine.combo_user_difficulty = {}
        engine.combo_step_display_mode = {}
        engine.combo_key_images = {}
        engine.combo_demo_video = {}
        engine.ww.combo_target_game = {}
        engine.ww.ww_characters = {}
        engine.ww.ww_dash_image = ""
        engine.ww.ww_teams = {}
        engine.ww.ww_active_team_id = None
        engine.ww.combo_ww_team = {}
        engine.active_combo_name = None
        engine.active_combo_tokens = []
        engine.runtime_steps = []


def save_engine_state(engine) -> bool:
    """
    Persist the engine state through `engine.state_store`.
    """
    try:
        payload = {
            "version": 1,
            "last_active_combo": engine.active_combo_name,
            "no_fail_mode": getattr(engine, "no_fail_mode", False),
            "macro_start_key": getattr(engine, "macro_start_key", "f8") or "f8",
            "macro_stop_key": getattr(engine, "macro_stop_key", "f9") or "f9",
            "macro_spam_interval_ms": (
                None if getattr(engine, "macro_spam_interval_ms", None) in (None, "")
                else int(getattr(engine, "macro_spam_interval_ms", 100))
            ),
            "combos": dict(engine.combos),
            "combo_enders": dict(engine.combo_enders),
            "combo_enders_soft": list(getattr(engine, "combo_enders_soft", set())),
            "transcribe_valid_keys": getattr(engine, "transcribe_valid_keys", "") or "",
            "transcribe_start_key": getattr(engine, "transcribe_start_key", "f") or "f",
            "transcribe_strip_wait_under_enabled": bool(
                getattr(engine, "transcribe_strip_wait_under_enabled", False)
            ),
            "transcribe_strip_wait_under_ms": getattr(engine, "transcribe_strip_wait_under_ms", "0") or "",
            "combo_stats": dict(engine.combo_stats),
            "combo_expected_ms": dict(engine.combo_expected_ms),
            "combo_user_difficulty": dict(engine.combo_user_difficulty),
            "combo_step_display_mode": dict(engine.combo_step_display_mode),
            "combo_key_images": dict(engine.combo_key_images),
            "combo_demo_video": dict(getattr(engine, "combo_demo_video", {})),
            "combo_target_game": dict(engine.ww.combo_target_game),
            "ww_characters": dict(engine.ww.ww_characters),
            "ww_dash_image": engine.ww.ww_dash_image,
            "ww_teams": {
                tid: {
                    "name": tv.get("name", "Team"),
                    "slot1": tv.get("slot1", ""),
                    "slot2": tv.get("slot2", ""),
                    "slot3": tv.get("slot3", ""),
                }
                for tid, tv in engine.ww.ww_teams.items()
                if isinstance(tv, dict)
            },
            "ww_active_team_id": engine.ww.ww_active_team_id,
            "combo_ww_team": dict(engine.ww.combo_ww_team),
        }
        engine.state_store.save(payload)
        return True
    except Exception:
        logger.exception("Failed to save engine state")
        return False
