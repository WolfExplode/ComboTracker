from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Status:
    text: str
    color: str  # ready|recording|success|fail|wait|neutral


def stats_text(engine) -> str:
    name = engine.active_combo_name
    if not name:
        return "Stats: —"
    engine._ensure_combo_stats(name)
    s = int(engine.combo_stats[name].get("success", 0))
    f = int(engine.combo_stats[name].get("fail", 0))
    pct = engine._format_percent(s, f)

    best = engine.combo_stats[name].get("best_ms", None)
    avg = engine._combo_avg_ms(name)

    # Hardest steps (top 2)
    hardest = ""
    by_step = engine.combo_stats[name].get("fail_by_step", {})
    if isinstance(by_step, dict) and by_step:
        pairs: list[tuple[int, int]] = []
        for k, v in by_step.items():
            try:
                idx = int(k)
                cnt = int(v)
            except Exception:
                continue
            if cnt <= 0:
                continue
            pairs.append((cnt, idx))
        pairs.sort(reverse=True)
        parts: list[str] = []
        for cnt, idx in pairs[:2]:
            label = "—"
            if 0 <= idx < len(engine.active_combo_steps):
                label = engine._expected_label_for_step(engine.active_combo_steps[idx])
            parts.append(f"#{idx+1}:{label} ({cnt})")
        if parts:
            hardest = " | Hardest: " + ", ".join(parts)

    return (
        f"Stats: {s} success / {f} fail ({pct})"
        f" | Best: {engine._format_ms_brief(best)} | Avg: {engine._format_ms_brief(avg)}"
        f"{hardest}"
    )


def failures_by_reason(engine) -> dict[str, int]:
    name = engine.active_combo_name
    if not name:
        return {}
    engine._ensure_combo_stats(name)
    by_reason = engine.combo_stats[name].get("fail_by_reason", {})
    if not isinstance(by_reason, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in by_reason.items():
        reason = str(k).strip() or "unknown"
        try:
            cnt = int(v)
        except Exception:
            cnt = 0
        if cnt > 0:
            out[reason] = cnt
    return out


def min_time_text(engine) -> str:
    if not engine.active_combo_steps:
        return "Fastest possible: —"
    min_ms = engine.calc_min_combo_time_ms(engine.active_combo_steps)
    return f"Fastest possible: {engine._format_ms(min_ms)}"


def practical_apm(engine) -> float | None:
    """
    Practical APM uses user-entered expected execution time (ms) for the active combo.
    """
    name = engine.active_combo_name
    if not name or not engine.active_combo_steps:
        return None
    expected_ms = engine.combo_expected_ms.get(name)
    if expected_ms is None or expected_ms <= 0:
        return None
    press_count, _hold_count, _actions = engine._count_combo_actions(engine.active_combo_steps)
    if press_count <= 0:
        return None
    return (60000.0 / float(expected_ms)) * float(press_count)


def theoretical_max_apm(engine) -> float | None:
    """
    Theoretical max APM uses the fastest-possible combo time (sum of waits + holds).
    """
    if not engine.active_combo_name or not engine.active_combo_steps:
        return None
    min_ms = engine.calc_min_combo_time_ms(engine.active_combo_steps)
    if min_ms <= 0:
        return None
    press_count, _hold_count, _actions = engine._count_combo_actions(engine.active_combo_steps)
    if press_count <= 0:
        return None
    return (60000.0 / float(min_ms)) * float(press_count)


def apm_text(engine) -> str:
    apm = practical_apm(engine)
    if apm is None:
        return "Practical APM: —"
    return f"Practical APM: {apm:.1f}"


def apm_max_text(engine) -> str:
    apm = theoretical_max_apm(engine)
    if apm is None:
        return "Theoretical max APM: —"
    return f"Theoretical max APM: {apm:.1f}"


def difficulty_score_10(engine) -> float | None:
    """
    Returns a 0..10 score (float) or None if there's no active combo.
    """
    if not engine.active_combo_steps or not engine.active_combo_name:
        return None

    # --- Keys camp (Practical APM + combo length) ---
    apm = practical_apm(engine) or 0.0
    _press_count, _hold_count, actions = engine._count_combo_actions(engine.active_combo_steps)

    # --- Normalization / scaling constants ---
    apm_norm = engine._clamp01(apm / 200.0)
    actions_norm = engine._clamp01(float(actions) / 8.0)

    keys = (0.6 * apm_norm) + (0.4 * actions_norm)

    # --- Timing camp (wait + hold + simple variation points) ---
    wait_scores: list[float] = []
    hold_scores: list[float] = []
    for s in engine.active_combo_steps:
        if not isinstance(s, dict):
            continue
        if s.get("wait_ms") is not None:
            try:
                wait_scores.append(engine._wait_triangle_score(int(s.get("wait_ms") or 0)))
            except Exception:
                continue
        elif s.get("hold_ms") is not None:
            try:
                hold_scores.append(engine._hold_score(int(s.get("hold_ms") or 0)))
            except Exception:
                continue

    has_wait = 1.0 if wait_scores else 0.0
    has_hold = 1.0 if hold_scores else 0.0
    wait_avg = (sum(wait_scores) / len(wait_scores)) if wait_scores else 0.0
    hold_avg = (sum(hold_scores) / len(hold_scores)) if hold_scores else 0.0

    wait_w = 1.0
    hold_w = 1.5
    denom = (wait_w * has_wait) + (hold_w * has_hold)
    timing_base = 0.0 if denom <= 0 else ((wait_avg * wait_w * has_wait) + (hold_avg * hold_w * has_hold)) / denom

    var_points = engine._timing_variation_points()
    K = 1.0
    var_norm = engine._clamp01(1.0 - (2.718281828 ** (-float(var_points) / K)))

    timing = (0.3 * engine._clamp01(timing_base)) + (0.7 * var_norm)

    combined = (0.45 * keys) + (0.55 * timing)
    return round(10.0 * engine._clamp01(combined), 1)


def difficulty_text(engine) -> str:
    d = difficulty_score_10(engine)
    if d is None:
        return "Difficulty: —"
    return f"Difficulty: {d:.1f} / 10"


def user_difficulty_value(engine) -> float | None:
    name = engine.active_combo_name
    if not name:
        return None
    d = engine.combo_user_difficulty.get(name)
    if d is None:
        return None
    try:
        d_f = float(d)
    except Exception:
        return None
    if 0.0 <= d_f <= 10.0:
        return d_f
    return None


def user_difficulty_text(engine) -> str:
    d = user_difficulty_value(engine)
    if d is None:
        return "Your difficulty: —"
    return f"Your difficulty: {d:g} / 10"


def get_editor_payload(engine) -> dict[str, Any]:
    name = engine.active_combo_name or ""
    inputs = ", ".join(engine.active_combo_tokens) if engine.active_combo_tokens else ""

    enders = ""
    if engine.combo_enders:
        parts: list[str] = []
        for k in sorted(engine.combo_enders.keys()):
            ms = int(engine.combo_enders[k])
            if ms > 0:
                parts.append(f"{k}:{ms/1000.0:.3g}")
            else:
                parts.append(k)
        enders = ", ".join(parts)

    expected = ""
    if name:
        ms = engine.combo_expected_ms.get(name)
        if ms is not None:
            expected = engine._format_ms_brief(ms)
    user_diff = ""
    if name:
        d = engine.combo_user_difficulty.get(name)
        if d is not None:
            # Keep it friendly for editing (no trailing .0)
            user_diff = f"{d:g}"

    mode = "icons"
    if name:
        m = str(engine.combo_step_display_mode.get(name, "icons") or "icons").strip().lower()
        if m in ("icons", "images"):
            mode = m
    key_images = {}
    if name:
        m = engine.combo_key_images.get(name)
        if isinstance(m, dict):
            # shallow copy for safety
            key_images = dict(m)

    ww_payload = engine.ww.editor_payload(name)
    return {
        "name": name,
        "inputs": inputs,
        "enders": enders,
        "expected_time": expected,
        "user_difficulty": user_diff,
        "step_display_mode": mode,
        "key_images": key_images,
        **ww_payload,
    }


def get_status(engine) -> Status:
    if not engine.active_combo_steps:
        return Status("Status: Select a combo to start", "neutral")

    step = engine._active_step()
    if not step:
        return Status("Status: Select a combo to start", "neutral")

    if engine.current_index == 0:
        # Any-order group can start with any option.
        if step.get("group_presses") is not None:
            opts = [str(x or "").strip().upper() for x in (step.get("group_presses") or [])]
            opts = [o for o in opts if o]
            pw_meta = step.get("group_pw_meta")
            if isinstance(pw_meta, dict) and pw_meta:
                pw_keys = []
                for _sig, meta in pw_meta.items():
                    k = str((meta or {}).get("input") or "").strip().upper()
                    if k:
                        pw_keys.append(k)
                if pw_keys:
                    opts = pw_keys + opts
            mw = step.get("group_mandatory_wait")
            if isinstance(mw, dict):
                mw_for = str(mw.get("wait_for") or "").strip().upper()
                if mw_for:
                    opts = [mw_for] + opts
            if opts:
                quoted = ", ".join([f"'{o}'" for o in opts])
                return Status(f"Ready! Press {quoted} to start.", "ready")
            return Status("Ready! Press the first input to start.", "ready")

        start_key = str(step.get("input") or "").upper()
        if step.get("hold_ms") is None:
            return Status(f"Ready! Press '{start_key}' to start.", "ready")
        return Status(
            f"Ready! Hold '{start_key}' for {int(step.get('hold_ms') or 0)}ms to start.",
            "ready",
        )

    if engine.wait_in_progress:
        req = engine._format_hold_requirement(int(engine.wait_required_ms or 0))
        # If this is a mandatory animation lock, make it explicit.
        mode = "soft"
        try:
            s = engine._active_step()
            if isinstance(s, dict) and s.get("wait_ms") is not None:
                mode = str(s.get("wait_mode") or "soft").strip().lower() or "soft"
        except Exception:
            mode = "soft"
        if mode == "mandatory":
            return Status(f"Animation lock ≥ {req} (inputs ignored)...", "wait")
        return Status(f"Waiting ≥ {req}...", "wait")
    if engine.hold_in_progress:
        req = engine._format_hold_requirement(int(engine.hold_required_ms or 0))
        inp = str(engine.hold_expected_input or "").upper()
        return Status(f"Holding '{inp}' (≥ {req}). Release OR press next input to continue...", "recording")
    return Status("Recording...", "recording")


def timeline_steps(engine) -> list[dict[str, Any]]:
    steps = []
    i = 0
    arr = engine.active_combo_steps or []
    # When idle after a success, keep the timeline fully completed/green until a new attempt starts.
    cur = engine.current_index
    try:
        if (
            int(cur) == 0
            and engine._ui_last_success_combo
            and engine._ui_last_success_combo == engine.active_combo_name
            and int(engine._ui_last_success_steps_len or 0) == len(arr)
        ):
            cur = len(arr)
    except Exception:
        cur = engine.current_index
    while i < len(arr):
        s = arr[i]
        idx = i
        mark = engine.step_marks.get(idx)
        if s.get("group_presses") is not None:
            done_counts = s.get("group_done_counts")
            done_counts = done_counts if isinstance(done_counts, dict) else {}
            pw_done_counts = s.get("group_pw_done_counts")
            pw_done_counts = pw_done_counts if isinstance(pw_done_counts, dict) else {}
            pw_need_counts = s.get("group_pw_need_counts")
            pw_need_counts = pw_need_counts if isinstance(pw_need_counts, dict) else {}
            pw_meta = s.get("group_pw_meta")
            pw_meta = pw_meta if isinstance(pw_meta, dict) else {}
            pw_active = bool(s.get("group_pw_active"))
            pw_sig_active = str(s.get("group_pw_sig") or "").strip()
            hold_done_counts = s.get("group_hold_done_counts")
            hold_done_counts = hold_done_counts if isinstance(hold_done_counts, dict) else {}
            hold_need_counts = s.get("group_hold_need_counts")
            hold_need_counts = hold_need_counts if isinstance(hold_need_counts, dict) else {}
            hold_meta = s.get("group_hold_meta")
            hold_meta = hold_meta if isinstance(hold_meta, dict) else {}
            hold_active = bool(s.get("group_hold_active"))
            hold_sig_active = str(s.get("group_hold_sig") or "").strip()
            mw_done = bool(s.get("group_wait_done"))
            mw_active = bool(s.get("group_wait_active"))
            mw = s.get("group_mandatory_wait")
            mw_for = ""
            mw_ms = None
            if isinstance(mw, dict):
                mw_for = str(mw.get("wait_for") or "").strip().lower()
                mw_ms = mw.get("wait_ms")

            order = s.get("group_order")
            order_list = order if isinstance(order, list) else []

            items_payload: list[dict[str, Any]] = []
            done_count = 0
            total = 0
            seen_press: dict[str, int] = {}  # key -> occurrences (for press duplicates)
            seen_pw: dict[str, int] = {}  # sig -> occurrences (for press_wait duplicates)
            seen_hold: dict[str, int] = {}  # sig -> occurrences (for hold duplicates)

            # Build items in the exact order written in the combo.
            for it in order_list:
                if not isinstance(it, dict):
                    continue
                kind = str(it.get("kind") or "")
                if kind == "anim_wait":
                    total += 1
                    dur = int(it.get("wait_ms") or 0)
                    wf = str(it.get("wait_for") or "").strip().lower()
                    comp = mw_done or (idx < cur)
                    act = (idx == cur) and mw_active
                    if comp:
                        done_count += 1
                    items_payload.append(
                        {
                            "type": "wait",
                            "mode": "mandatory",
                            "wait_for": wf,
                            "duration": dur,
                            "active": act,
                            "completed": comp,
                        }
                    )
                elif kind == "press_wait":
                    total += 1
                    sig = str(it.get("sig") or "").strip()
                    meta = pw_meta.get(sig) if isinstance(pw_meta, dict) else None
                    inp = str((meta or {}).get("input") or it.get("input") or "").strip().lower()
                    dur = int((meta or {}).get("wait_ms") or it.get("wait_ms") or 0)
                    # occurrence counting for duplicates of same sig:
                    seen_pw[sig] = int(seen_pw.get(sig, 0)) + 1
                    occ = int(seen_pw.get(sig, 1))
                    comp = (int(pw_done_counts.get(sig, 0)) >= occ) or (idx < cur)
                    act = (idx == cur) and pw_active and (pw_sig_active == sig)
                    if comp:
                        done_count += 1
                    items_payload.append(
                        {
                            "type": "press_wait",
                            "input": inp,
                            "duration": dur,
                            "active": act,
                            "completed": comp,
                        }
                    )
                elif kind == "press":
                    total += 1
                    inp = str(it.get("input") or "").strip().lower()
                    seen_press[inp] = int(seen_press.get(inp, 0)) + 1
                    occ = int(seen_press.get(inp, 1))
                    comp = (int(done_counts.get(inp, 0)) >= occ) or (idx < cur)
                    if comp:
                        done_count += 1
                    items_payload.append(
                        {
                            "type": "press",
                            "input": inp,
                            "duration": 0,
                            "active": False,
                            "completed": comp,
                        }
                    )
                elif kind == "hold":
                    total += 1
                    sig = str(it.get("sig") or "").strip()
                    key = str(it.get("input") or "").strip().lower()
                    dur = int(it.get("hold_ms") or 0)
                    seen_hold[sig] = int(seen_hold.get(sig, 0)) + 1
                    occ = int(seen_hold.get(sig, 1))
                    comp = (int(hold_done_counts.get(sig, 0)) >= occ) or (idx < cur)
                    act = (idx == cur) and hold_active and (hold_sig_active == sig)
                    if comp:
                        done_count += 1
                    items_payload.append(
                        {
                            "type": "hold",
                            "input": key,
                            "duration": dur,
                            "active": act,
                            "completed": comp,
                        }
                    )

            # Fallback (shouldn't happen): derive a stable order.
            if not items_payload:
                total = 0
                done_count = 0
                if mw_ms is not None:
                    total += 1
                    comp = mw_done or (idx < cur)
                    if comp:
                        done_count += 1
                    items_payload.append(
                        {
                            "type": "wait",
                            "mode": "mandatory",
                            "wait_for": mw_for,
                            "duration": int(mw_ms),
                            "active": (idx == engine.current_index) and mw_active,
                            "completed": comp,
                        }
                    )
                # Add press-wait items (if any) then plain presses.
                for sig, meta in (pw_meta or {}).items():
                    try:
                        need_n = int(pw_need_counts.get(sig, 0) or 0)
                    except Exception:
                        need_n = 0
                    key = str((meta or {}).get("input") or "").strip().lower()
                    dur = int((meta or {}).get("wait_ms") or 0)
                    for occ in range(1, max(0, need_n) + 1):
                        total += 1
                        comp = (int(pw_done_counts.get(sig, 0)) >= occ) or (idx < cur)
                        if comp:
                            done_count += 1
                        items_payload.append(
                            {
                                "type": "press_wait",
                                "input": key,
                                "duration": dur,
                                "active": (idx == cur) and pw_active and (pw_sig_active == str(sig)),
                                "completed": comp,
                            }
                        )
                # Plain press counts fallback: if need_counts exist, represent repeats.
                need_counts = s.get("group_press_need_counts")
                need_counts = need_counts if isinstance(need_counts, dict) else {}
                for inp, cnt in need_counts.items():
                    key = str(inp or "").strip().lower()
                    if not key:
                        continue
                    try:
                        n = int(cnt or 0)
                    except Exception:
                        n = 0
                    for occ in range(1, max(0, n) + 1):
                        total += 1
                        comp = (int(done_counts.get(key, 0)) >= occ) or (idx < cur)
                        if comp:
                            done_count += 1
                        items_payload.append({"type": "press", "input": key, "duration": 0, "active": False, "completed": comp})
                # Hold fallback (if any)
                for sig, cnt in hold_need_counts.items():
                    try:
                        n = int(cnt or 0)
                    except Exception:
                        n = 0
                    meta = hold_meta.get(sig) if isinstance(hold_meta, dict) else None
                    key = str((meta or {}).get("input") or "").strip().lower()
                    dur = int((meta or {}).get("hold_ms") or 0)
                    for occ in range(1, max(0, n) + 1):
                        total += 1
                        comp = (int(hold_done_counts.get(sig, 0)) >= occ) or (idx < cur)
                        if comp:
                            done_count += 1
                        items_payload.append(
                            {
                                "type": "hold",
                                "input": key,
                                "duration": dur,
                                "active": (idx == cur) and hold_active and (hold_sig_active == str(sig)),
                                "completed": comp,
                            }
                        )

            steps.append(
                {
                    "type": "group",
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                    "items": items_payload,
                    "progress": {"done": int(done_count), "total": int(total)},
                }
            )
            i += 1
            continue

        # Collapse "press X" followed by "mandatory wait for X" into a single displayed tile,
        # so wait(1, 0.5) doesn't show as "1" then "1 (animation time ...)".
        try:
            if (
                i + 1 < len(arr)
                and isinstance(s, dict)
                and s.get("wait_ms") is None
                and s.get("hold_ms") is None
                and arr[i + 1].get("wait_ms") is not None
                and str(arr[i + 1].get("wait_mode") or "").strip().lower() == "mandatory"
                and str(arr[i + 1].get("wait_for") or "").strip().lower() == str(s.get("input") or "").strip().lower()
            ):
                wait_step = arr[i + 1]
                wait_idx = i + 1
                wait_mark = engine.step_marks.get(wait_idx) or mark
                steps.append(
                    {
                        "type": "wait",
                        "input": None,
                        "duration": int(wait_step.get("wait_ms") or 0),
                        "mode": "mandatory",
                        "wait_for": str(wait_step.get("wait_for") or ""),
                        "active": (cur == idx) or (cur == wait_idx),
                        "completed": cur > wait_idx,
                        "mark": wait_mark,
                    }
                )
                i += 2
                continue
        except Exception:
            pass

        # Collapse "press X" followed immediately by a wait gate into a single displayed tile.
        try:
            if (
                i + 1 < len(arr)
                and isinstance(s, dict)
                and s.get("wait_ms") is None
                and s.get("hold_ms") is None
                and s.get("group_presses") is None
                and isinstance(arr[i + 1], dict)
                and arr[i + 1].get("wait_ms") is not None
                and str(arr[i + 1].get("wait_mode") or "soft").strip().lower() in ("soft", "hard")
            ):
                press_inp = str(s.get("input") or "").strip().lower()
                w = arr[i + 1]
                wait_idx = i + 1
                w_mark = engine.step_marks.get(wait_idx) or mark
                steps.append(
                    {
                        "type": "press_wait",
                        "input": press_inp,
                        "duration": int(w.get("wait_ms") or 0),
                        "mode": str(w.get("wait_mode") or "soft"),
                        "active": (cur == idx) or (cur == wait_idx),
                        "completed": cur > wait_idx,
                        "mark": w_mark,
                    }
                )
                i += 2
                continue
        except Exception:
            pass

        if s.get("wait_ms") is not None:
            steps.append(
                {
                    "type": "wait",
                    "input": None,
                    "duration": int(s.get("wait_ms") or 0),
                    "mode": str(s.get("wait_mode") or "soft"),
                    "wait_for": str(s.get("wait_for") or ""),
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                }
            )
        elif s.get("hold_ms") is not None:
            steps.append(
                {
                    "type": "hold",
                    "input": str(s.get("input") or ""),
                    "duration": int(s.get("hold_ms") or 0),
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                }
            )
        else:
            steps.append(
                {
                    "type": "press",
                    "input": str(s.get("input") or ""),
                    "duration": 0,
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                }
            )
        i += 1
    return steps


def init_payload(engine) -> dict[str, Any]:
    st = get_status(engine)
    return {
        "type": "init",
        "combos": sorted(engine.combos.keys()),
        "active_combo": engine.active_combo_name,
        "status": {"text": st.text, "color": st.color},
        "stats": stats_text(engine),
        "min_time": min_time_text(engine),
        "difficulty": difficulty_text(engine),
        "difficulty_value": difficulty_score_10(engine),
        "user_difficulty": user_difficulty_text(engine),
        "user_difficulty_value": user_difficulty_value(engine),
        "apm": apm_text(engine),
        "apm_max": apm_max_text(engine),
        "timeline": timeline_steps(engine),
        "failures": failures_by_reason(engine),
        "editor": get_editor_payload(engine),
    }

