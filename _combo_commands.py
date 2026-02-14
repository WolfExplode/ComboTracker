"""
UI command handlers: apply/save/delete combo, new combo, clear history.
Internal module: call only from ComboTrackerEngine while holding the engine lock.
"""

from __future__ import annotations

from typing import Any

from persistence import fresh_combo_stats


def apply_enders_from_text(engine, raw: str) -> tuple[bool, str | None]:
    raw = (raw or "").strip()
    if not raw:
        engine.combo_enders = {}
        setattr(engine, "combo_enders_soft", set())
        engine.save_combos()
        return True, None

    parsed: dict[str, int] = {}
    soft_keys: set[str] = set()
    for token in engine.split_inputs(raw):
        t = token.strip()
        if not t:
            continue
        if ":" in t:
            k, v = t.split(":", 1)
            raw_key = k.strip().lower()
            # ~key:2s = soft ender (does not drop combo when pressed during hold)
            if raw_key.startswith("~"):
                raw_key = raw_key[1:].strip().lower()
                if raw_key:
                    soft_keys.add(raw_key)
            key = raw_key
            val = (v or "").strip().lower()
            if not key:
                continue
            # Require explicit "s" suffix for seconds: key:2s, key:0.2s
            if val.endswith("s"):
                val = val[:-1].strip()
            else:
                return False, f"Ender cooldown must be in seconds with 's' suffix, e.g. {key}:2s or {key}:0.2s"
            try:
                sec = float(val)
            except ValueError:
                return False, f"Invalid timing for '{key}'. Use seconds with 's', e.g. {key}:2s"
            parsed[key] = max(0, int(sec * 1000))
        else:
            key = t.strip().lower()
            if key.startswith("~"):
                key = key[1:].strip().lower()
                if key:
                    soft_keys.add(key)
            if key:
                parsed[key] = 0

    engine.combo_enders = parsed
    engine.combo_enders_soft = soft_keys
    engine.save_combos()
    return True, None


def save_or_update_combo(
    engine,
    *,
    name: str,
    inputs: str,
    enders: str,
    expected_time: str | None = None,
    user_difficulty: str | None = None,
    step_display_mode: str | None = None,
    key_images: Any | None = None,
    target_game: str | None = None,
    ww_team_id: str | None = None,
) -> tuple[bool, str | None]:
    name = (name or "").strip()
    keys_str = (inputs or "").strip()
    if not name or not keys_str:
        return False, "Please fill in Name and Inputs."

    ok, err = apply_enders_from_text(engine, enders)
    if not ok:
        return False, err

    expected_ms = None
    expected_raw = (expected_time or "").strip()
    if expected_raw:
        expected_ms = engine._parse_expected_time_ms(expected_raw)
        if expected_ms is None:
            return False, "Invalid Expected time. Examples: 1.05s or 1050ms"

    user_diff_val = None
    ud_raw = (user_difficulty or "").strip()
    if ud_raw:
        try:
            user_diff_val = float(ud_raw)
        except Exception:
            return False, "Invalid Your difficulty. Use a number from 0 to 10."
        if not (0.0 <= user_diff_val <= 10.0):
            return False, "Invalid Your difficulty. Use a number from 0 to 10."

    input_list = [k.strip().lower() for k in engine.split_inputs(keys_str) if k.strip()]
    if not input_list:
        return False, "Please provide at least one input."

    old_name = engine.active_combo_name if engine.active_combo_name in engine.combos else None
    if old_name and name != old_name:
        if old_name in engine.combo_stats and name not in engine.combo_stats:
            engine.combo_stats[name] = engine.combo_stats.pop(old_name)
        engine.combos[name] = input_list
        if old_name != name and old_name in engine.combos:
            del engine.combos[old_name]
        if old_name in engine.combo_expected_ms:
            del engine.combo_expected_ms[old_name]
        if old_name in engine.combo_user_difficulty:
            del engine.combo_user_difficulty[old_name]
        if old_name in engine.combo_step_display_mode and name not in engine.combo_step_display_mode:
            engine.combo_step_display_mode[name] = engine.combo_step_display_mode.pop(old_name)
        if old_name in engine.combo_key_images and name not in engine.combo_key_images:
            engine.combo_key_images[name] = engine.combo_key_images.pop(old_name)
        engine.ww.rename_combo(old_name, name)
    else:
        # Same name or new combo: if steps changed, clear all history for this combo
        old_list = engine.combos.get(name)
        if old_list is not None and old_list != input_list:
            engine.combo_stats[name] = fresh_combo_stats()
        engine.combos[name] = input_list

    if expected_ms is not None:
        engine.combo_expected_ms[name] = int(expected_ms)
    else:
        engine.combo_expected_ms.pop(name, None)

    if user_diff_val is not None:
        engine.combo_user_difficulty[name] = float(user_diff_val)
    else:
        engine.combo_user_difficulty.pop(name, None)

    mode_raw = (step_display_mode or "").strip().lower()
    if mode_raw in ("icons", "images"):
        engine.combo_step_display_mode[name] = mode_raw
    else:
        engine.combo_step_display_mode.pop(name, None)

    cleaned_imgs: dict[str, str] = {}
    if isinstance(key_images, dict):
        for k, v in key_images.items():
            key = str(k).strip().lower()
            url = str(v).strip()
            if not key or not url:
                continue
            cleaned_imgs[key] = url
    if cleaned_imgs:
        engine.combo_key_images[name] = cleaned_imgs
    else:
        if isinstance(key_images, dict):
            engine.combo_key_images.pop(name, None)

    g_raw = str(target_game or "").strip().lower()
    engine.ww.set_target_game(name, g_raw)
    engine.ww.apply_combo_team_assignment(name, target_game=engine.ww.get_target_game(name), ww_team_id=ww_team_id)

    engine._ensure_combo_stats(name)
    engine.set_active_combo(name, emit=False)
    engine.save_combos()

    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def delete_combo(engine, name: str) -> tuple[bool, str | None]:
    name = (name or "").strip()
    if not name or name not in engine.combos:
        return False, "Select a combo to delete."

    del engine.combos[name]
    if name in engine.combo_stats:
        del engine.combo_stats[name]
    if name in engine.combo_expected_ms:
        del engine.combo_expected_ms[name]
    if name in engine.combo_user_difficulty:
        del engine.combo_user_difficulty[name]
    if name in engine.combo_step_display_mode:
        del engine.combo_step_display_mode[name]
    if name in engine.combo_key_images:
        del engine.combo_key_images[name]
    engine.ww.delete_combo(name)

    if engine.active_combo_name == name:
        engine.active_combo_name = None
        engine.active_combo_tokens = []
        engine.runtime_steps = []
        engine.reset_tracking()

    engine.save_combos()
    engine._send({"type": "init", **engine.init_payload()})
    return True, None


def new_combo(engine) -> None:
    engine.active_combo_name = None
    engine.active_combo_tokens = []
    engine.runtime_steps = []
    engine.reset_tracking()
    engine._send({"type": "init", **engine.init_payload()})


def clear_history_and_stats(engine) -> None:
    engine.reset_tracking()
    if engine.active_combo_name:
        engine.combo_stats[engine.active_combo_name] = fresh_combo_stats()
        engine.save_combos()
    engine._send({"type": "clear_results"})
    engine._send({"type": "stat_update", "stats": engine.stats_text()})
    engine._send({"type": "fail_update", "fail_by_step": engine.failures_by_step()})
    engine._send({"type": "timeline_update", "steps": engine.timeline_steps()})
    st = engine.get_status()
    engine._send({"type": "status", "text": st.text, "color": st.color})
