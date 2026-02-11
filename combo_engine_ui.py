from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from timeline_model import TimelineSteps

import combo_analytics
from states import (
    GroupState,
    HoldState,
    PressState,
    SequenceState,
    WaitState,
)


@dataclass
class Status:
    text: str
    color: str  # ready|recording|success|fail|wait|neutral


def stats_text(engine) -> str:
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return "Stats: —"
    summary = combo_analytics.combo_stats_summary(engine)
    s = summary["success"]
    f = summary["fail"]
    pct = engine._format_percent(s, f)
    best = summary["best_ms"]
    avg = summary["avg_ms"]
    return (
        f"Stats: {s} success / {f} fail ({pct})"
        f" | Best: {engine._format_ms_brief(best)} | Avg: {engine._format_ms_brief(avg)}"
    )


def failures_by_step(engine) -> dict[str, int]:
    return combo_analytics.failures_by_step(engine)


def failures_by_reason(engine) -> dict[str, int]:
    return combo_analytics.failures_by_reason(engine)


def min_time_text(engine) -> str:
    if not getattr(engine, "runtime_steps", None):
        return "Fastest possible: —"
    min_ms = combo_analytics.min_combo_time_ms(engine)
    return f"Fastest possible: {engine._format_ms(min_ms)}"


def theoretical_max_apm(engine) -> float | None:
    """Theoretical max APM uses the fastest-possible combo time (sum of waits + holds)."""
    return combo_analytics.theoretical_max_apm(engine)


def apm_text(engine) -> str:
    apm = combo_analytics.practical_apm(engine)
    if apm is None:
        return "Practical APM: —"
    return f"Practical APM: {apm:.1f}"


def apm_max_text(engine) -> str:
    apm = theoretical_max_apm(engine)
    if apm is None:
        return "Theoretical max APM: —"
    return f"Theoretical max APM: {apm:.1f}"


def difficulty_text(engine) -> str:
    d = combo_analytics.difficulty_score_10(engine)
    if d is None:
        return "Difficulty: —"
    return f"Difficulty: {d:.1f} / 10"


def user_difficulty_value(engine) -> float | None:
    return combo_analytics.user_difficulty_value(engine)


def user_difficulty_text(engine) -> str:
    d = user_difficulty_value(engine)
    if d is None:
        return "Your difficulty: —"
    return f"Your difficulty: {d:g} / 10"


def get_editor_payload(engine, target_game_override: str | None = None) -> dict[str, Any]:
    name = engine.active_combo_name or ""
    inputs = ", ".join(engine.active_combo_tokens) if engine.active_combo_tokens else ""

    enders = ""
    if engine.combo_enders:
        parts: list[str] = []
        for k in sorted(engine.combo_enders.keys()):
            ms = int(engine.combo_enders[k])
            if ms > 0:
                sec = ms / 1000.0
                # Format with explicit "s" for seconds (e.g. 2s, 0.2s, 3s)
                parts.append(f"{k}:{sec:g}s")
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

    ww_payload = engine.ww.editor_payload(name, target_game_override=target_game_override)
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


def _group_start_options(step: GroupState) -> list[str]:
    """Collect allowed start keys for a group at index 0 (press, press_wait first key, anim_wait, hold, sequence first key)."""
    opts: list[str] = []
    for item in step.items:
        if item.kind == "press" and isinstance(item.state, PressState):
            k = str(item.state.expected or "").strip().upper()
            if k and k not in opts:
                opts.append(k)
        elif item.kind == "press_wait" and isinstance(item.state, SequenceState) and item.state.steps:
            first = item.state.steps[0]
            if isinstance(first, PressState):
                k = str(first.expected or "").strip().upper()
                if k and k not in opts:
                    opts.append(k)
        elif item.kind == "anim_wait" and isinstance(item.state, WaitState):
            k = str(item.state.wait_for or "").strip().upper()
            if k and k not in opts:
                opts.insert(0, k)
        elif item.kind == "hold" and isinstance(item.state, HoldState):
            k = str(item.state.expected or "").strip().upper()
            if k and k not in opts:
                opts.append(k)
        elif item.kind == "sequence" and isinstance(item.state, SequenceState):
            for s in item.state.steps:
                if isinstance(s, WaitState):
                    continue
                if isinstance(s, PressState):
                    k = str(s.expected or "").strip().upper()
                    if k and k not in opts:
                        opts.append(k)
                    break
    return opts


def get_status(engine) -> Status:
    if not engine.runtime_steps:
        return Status("Status: Select a combo to start", "neutral")

    step = engine._active_step()
    if not step:
        return Status("Status: Select a combo to start", "neutral")

    if engine.current_index == 0:
        if isinstance(step, GroupState):
            opts = _group_start_options(step)
            if opts:
                quoted = ", ".join([f"'{o}'" for o in opts])
                return Status(f"Ready! Press {quoted} to start.", "ready")
            return Status("Ready! Press the first input to start.", "ready")
        if isinstance(step, PressState):
            start_key = str(step.expected or "").upper()
            return Status(f"Ready! Press '{start_key}' to start.", "ready")
        if isinstance(step, HoldState):
            start_key = str(step.expected or "").upper()
            return Status(
                f"Ready! Hold '{start_key}' for {int(step.required_ms or 0)}ms to start.",
                "ready",
            )
        if isinstance(step, (SequenceState, WaitState)):
            return Status("Ready! Press the first input to start.", "ready")

    if engine.wait_in_progress:
        req = engine._format_hold_requirement(int(engine.wait_required_ms or 0))
        mode = "soft"
        try:
            s = engine._active_step()
            if isinstance(s, WaitState):
                mode = str(s.mode or "soft").strip().lower() or "soft"
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


def _wait_progress(wait_state: WaitState, now: float | None) -> int | None:
    """Return 0–100 progress for an in-progress wait, or None."""
    if now is None or not getattr(wait_state, "in_progress", False):
        return None
    started = getattr(wait_state, "started_at", 0.0)
    required_ms = getattr(wait_state, "required_ms", 0) or 1
    elapsed_ms = (now - started) * 1000.0
    pct = (elapsed_ms / required_ms) * 100.0
    return int(max(0, min(100, pct)))


def _render_sequence_items(
    *,
    seq: SequenceState,
    parent_is_active: bool,
    parent_is_completed: bool,
    now: float | None,
) -> list[dict[str, Any]]:
    """
    Render items inside a SequenceState for the UI.

    Rules (single source of truth):
    - Collapse anim-wait composite: press(x) + wait(mode=mandatory, wait_for=x) -> a single wait tile.
    - Collapse plain press+wait: press(x) + wait(mode=soft|hard, no wait_for) -> a single press_wait tile.
    """
    seq_items: list[dict[str, Any]] = []
    j = 0
    steps_list = seq.steps or []
    while j < len(steps_list):
        sub = steps_list[j]
        nxt = steps_list[j + 1] if (j + 1) < len(steps_list) else None

        # Collapse: press(x) + mandatory wait_for x  -> one anim-wait tile
        if (
            isinstance(sub, PressState)
            and isinstance(nxt, WaitState)
            and (nxt.mode == "mandatory")
            and (str(nxt.wait_for or "").strip().lower() == str(sub.expected or "").strip().lower())
        ):
            is_active = bool(seq.started and parent_is_active and (seq.current_index in (j, j + 1)))
            is_completed = bool((seq.started and seq.current_index > (j + 1)) or parent_is_completed)
            wait_dict = {
                "type": "wait",
                "mode": "mandatory",
                "wait_for": nxt.wait_for or "",
                "duration": nxt.required_ms,
                "active": is_active,
                "completed": is_completed,
            }
            prog = _wait_progress(nxt, now)
            if prog is not None:
                wait_dict["progress"] = prog
            seq_items.append(wait_dict)
            j += 2
            continue

        # Collapse: press(x) + plain wait (soft/hard, no wait_for) -> one press_wait tile
        if (
            isinstance(sub, PressState)
            and isinstance(nxt, WaitState)
            and (nxt.mode in ("soft", "hard"))
            and (not str(nxt.wait_for or "").strip())
        ):
            is_active = bool(seq.started and parent_is_active and (seq.current_index in (j, j + 1)))
            is_completed = bool((seq.started and seq.current_index > (j + 1)) or parent_is_completed)
            pw = {
                "type": "press_wait",
                "input": sub.expected,
                "duration": nxt.required_ms,
                "mode": nxt.mode,
                "active": is_active,
                "completed": is_completed,
            }
            prog = _wait_progress(nxt, now)
            if prog is not None:
                pw["progress"] = prog
            seq_items.append(pw)
            j += 2
            continue

        is_active = bool(seq.started and parent_is_active and (j == seq.current_index))
        is_completed = bool((seq.started and j < seq.current_index) or parent_is_completed)

        if isinstance(sub, WaitState):
            wait_dict = {
                "type": "wait",
                "mode": sub.mode,
                "wait_for": sub.wait_for or "",
                "duration": sub.required_ms,
                "active": is_active,
                "completed": is_completed,
            }
            prog = _wait_progress(sub, now)
            if prog is not None:
                wait_dict["progress"] = prog
            seq_items.append(wait_dict)
        elif isinstance(sub, HoldState):
            seq_items.append(
                {
                    "type": "hold",
                    "input": sub.expected,
                    "duration": sub.required_ms,
                    "active": is_active,
                    "completed": is_completed,
                }
            )
        else:
            inp = sub.expected if isinstance(sub, PressState) else ""
            d = {
                "type": "press",
                "input": inp,
                "duration": 0,
                "active": is_active,
                "completed": is_completed,
            }
            if getattr(sub, "optional", False):
                d["optional"] = True
                d["was_skipped"] = getattr(sub, "was_skipped", False)
            seq_items.append(d)
        j += 1

    return seq_items


def _timeline_steps_from_runtime(engine, now: float | None = None) -> TimelineSteps:
    """Build timeline payload from engine.runtime_steps (state objects).
    Each step dict includes step_indices: list of runtime_steps indices this tile represents
    (one index for normal steps, two when press+wait are merged). fail_by_step is keyed by
    runtime index, so the frontend uses step_indices to look up fail counts correctly."""
    arr = engine.runtime_steps
    if not arr:
        return []
    steps: list[dict[str, Any]] = []
    cur = engine.current_index
    try:
        if (
            int(cur) == 0
            and getattr(engine, "_ui_last_success_combo", None)
            and engine._ui_last_success_combo == engine.active_combo_name
            and int(getattr(engine, "_ui_last_success_steps_len", 0) or 0) == len(arr)
        ):
            cur = len(arr)
    except Exception:
        cur = engine.current_index

    i = 0
    while i < len(arr):
        step = arr[i]
        idx = i
        mark = engine.step_marks.get(idx)

        match step:
            case GroupState():
                items_payload: list[dict[str, Any]] = []
                done_count = 0
                total = 0
                for item in step.items:
                    comp = item.completed_count >= item.required_count
                    if comp:
                        done_count += 1
                    total += 1
                    if item.kind == "press" and isinstance(item.state, PressState):
                        p = {
                            "type": "press",
                            "input": item.state.expected,
                            "duration": 0,
                            "active": False,
                            "completed": comp,
                        }
                        if getattr(item.state, "optional", False):
                            p["optional"] = True
                            p["was_skipped"] = getattr(item.state, "was_skipped", False)
                        items_payload.append(p)
                    elif item.kind == "hold" and isinstance(item.state, HoldState):
                        act = step.active_item is item and item.kind == "hold"
                        items_payload.append({
                            "type": "hold",
                            "input": item.state.expected,
                            "duration": item.state.required_ms,
                            "active": act,
                            "completed": comp,
                        })
                    elif item.kind == "anim_wait" and isinstance(item.state, WaitState):
                        anim_w = {
                            "type": "wait",
                            "mode": "mandatory",
                            "wait_for": item.state.wait_for or "",
                            "duration": item.state.required_ms,
                            "active": (idx == cur) and step.wait_active,
                            "completed": comp,
                        }
                        prog = _wait_progress(item.state, now)
                        if prog is not None:
                            anim_w["progress"] = prog
                        items_payload.append(anim_w)
                    elif item.kind == "press_wait" and isinstance(item.state, SequenceState):
                        seq = item.state
                        inp = ""
                        dur = 0
                        if len(seq.steps) >= 2 and isinstance(seq.steps[0], PressState):
                            inp = seq.steps[0].expected
                        if len(seq.steps) >= 2 and isinstance(seq.steps[1], WaitState):
                            dur = seq.steps[1].required_ms
                        pw_active = step.active_item is item
                        items_payload.append({
                            "type": "press_wait",
                            "input": inp,
                            "duration": dur,
                            "active": (idx == cur) and pw_active,
                            "completed": comp,
                        })
                    elif item.kind == "sequence" and isinstance(item.state, SequenceState):
                        seq = item.state
                        parent_is_active = bool((idx == cur) and (step.active_item is item))
                        parent_is_completed = bool(idx < cur)
                        seq_items = _render_sequence_items(
                            seq=seq,
                            parent_is_active=parent_is_active,
                            parent_is_completed=parent_is_completed,
                            now=now,
                        )
                        seq_id = getattr(item.state, "_legacy_seq_id", f"gseq_{idx}_{len(items_payload)}")
                        items_payload.append({
                            "type": "sequence",
                            "sequence_id": seq_id,
                            "items": seq_items,
                            "active": (idx == cur) and (step.active_item is item),
                            "completed": comp,
                            "progress": {"done": seq.current_index if seq.started else 0, "total": len(seq.steps)},
                        })
                steps.append({
                    "type": "group",
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                    "items": items_payload,
                    "progress": {"done": int(done_count), "total": int(total)},
                    "step_indices": [idx],
                })
                i += 1
                continue

            case SequenceState():
                # Top-level press+wait (key with interruptible animation): emit single press_wait tile
                if (
                    len(step.steps) == 2
                    and isinstance(step.steps[0], PressState)
                    and isinstance(step.steps[1], WaitState)
                    and step.steps[1].mode in ("soft", "hard")
                ):
                    sub, nxt = step.steps[0], step.steps[1]
                    pw = {
                        "type": "press_wait",
                        "input": sub.expected,
                        "duration": nxt.required_ms,
                        "mode": nxt.mode,
                        "active": idx == cur,
                        "completed": idx < cur,
                        "mark": mark,
                        "step_indices": [idx],
                    }
                    prog = _wait_progress(nxt, now)
                    if prog is not None:
                        pw["progress"] = prog
                    if getattr(step, "optional", False):
                        pw["optional"] = True
                        pw["was_skipped"] = getattr(step, "was_skipped", False)
                    steps.append(pw)
                    i += 1
                    continue
                seq_items = _render_sequence_items(
                    seq=step,
                    parent_is_active=bool(idx == cur),
                    parent_is_completed=bool(idx < cur),
                    now=now,
                )
                steps.append({
                    "type": "sequence",
                    "active": idx == cur and step.started,
                    "completed": idx < cur,
                    "mark": mark,
                    "items": seq_items,
                    "progress": {"done": step.current_index if step.started else 0, "total": len(step.steps)},
                    "step_indices": [idx],
                })
                i += 1
                continue

            case PressState():
                # Press + mandatory wait (anim lock) only: soft/hard wait is always merged in parser
                if i + 1 < len(arr):
                    nxt = arr[i + 1]
                    if isinstance(nxt, WaitState) and nxt.mode == "mandatory" and (nxt.wait_for or "").strip().lower() == (step.expected or "").strip().lower():
                        wait_mark = engine.step_marks.get(i + 1) or mark
                        steps.append({
                            "type": "wait",
                            "input": None,
                            "duration": nxt.required_ms,
                            "mode": "mandatory",
                            "wait_for": nxt.wait_for or "",
                            "active": (cur == idx) or (cur == i + 1),
                            "completed": cur > i + 1,
                            "mark": wait_mark,
                            "step_indices": [idx, i + 1],
                        })
                        i += 2
                        continue
                press_payload = {
                    "type": "press",
                    "input": step.expected,
                    "duration": 0,
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                    "step_indices": [idx],
                }
                if getattr(step, "optional", False):
                    press_payload["optional"] = True
                    press_payload["was_skipped"] = getattr(step, "was_skipped", False)
                steps.append(press_payload)
                i += 1
                continue

            case WaitState():
                w = {
                    "type": "wait",
                    "input": None,
                    "duration": step.required_ms,
                    "mode": step.mode,
                    "wait_for": step.wait_for or "",
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                    "step_indices": [idx],
                }
                prog = _wait_progress(step, now)
                if prog is not None:
                    w["progress"] = prog
                steps.append(w)
                i += 1
                continue

            case HoldState():
                steps.append({
                    "type": "hold",
                    "input": step.expected,
                    "duration": step.required_ms,
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                    "step_indices": [idx],
                })
                i += 1
                continue

            case _:
                i += 1
                continue
    return steps


def timeline_steps(engine, now: float | None = None) -> list[dict[str, Any]]:
    """Build timeline payload for the frontend from engine.runtime_steps.
    If now is provided, in-progress waits get a numeric progress (0–100) for the fill animation.
    """
    return _timeline_steps_from_runtime(engine, now)


def init_payload(engine) -> dict[str, Any]:
    st = get_status(engine)
    return {
        "type": "init",
        "combos": sorted(engine.combos.keys()),
        "active_combo": engine.active_combo_name,
        "no_fail_mode": getattr(engine, "no_fail_mode", False),
        "status": {"text": st.text, "color": st.color},
        "stats": stats_text(engine),
        "min_time": min_time_text(engine),
        "difficulty": difficulty_text(engine),
        "difficulty_value": combo_analytics.difficulty_score_10(engine),
        "user_difficulty": user_difficulty_text(engine),
        "user_difficulty_value": user_difficulty_value(engine),
        "apm": apm_text(engine),
        "apm_max": apm_max_text(engine),
        "timeline": timeline_steps(engine),
        "fail_by_step": failures_by_step(engine),
        "editor": get_editor_payload(engine),
        "transcribe_valid_keys": getattr(engine, "transcribe_valid_keys", "") or "",
    }


# ---------------------------------------------------------------------------
# UI-only helpers for the engine (kept here to avoid polluting core engine logic)
# ---------------------------------------------------------------------------

def find_active_in_progress_wait(step: Any) -> WaitState | None:
    """
    Return the currently-active WaitState (including nested) if it is in progress.
    Used for client-side wait progress animation (wait_begin/wait_end).
    """
    try:
        if step is None:
            return None
        if isinstance(step, WaitState):
            return step if (step.in_progress and not step.completed) else None
        if isinstance(step, SequenceState):
            try:
                idx = int(getattr(step, "current_index", 0) or 0)
            except Exception:
                idx = 0
            if 0 <= idx < len(step.steps):
                return find_active_in_progress_wait(step.steps[idx])
            return None
        if isinstance(step, GroupState):
            # Animation lock wait within group
            if getattr(step, "wait_active", False):
                for item in (getattr(step, "items", None) or []):
                    if getattr(item, "kind", "") == "anim_wait":
                        ws = getattr(item, "state", None)
                        if isinstance(ws, WaitState) and ws.in_progress and not ws.completed:
                            return ws
            # Active item can be a sequence containing waits
            active_item = getattr(step, "active_item", None)
            if active_item is not None:
                return find_active_in_progress_wait(getattr(active_item, "state", None))
            return None
    except Exception:
        return None
    return None


def wait_animation_events(engine, *, active_step: Any, prev_sig: tuple | None) -> tuple[tuple | None, list[dict[str, Any]]]:
    """
    Determine if we should emit wait_begin/wait_end for a nested in-progress wait.

    Returns (new_sig, events) where:
    - new_sig is a stable signature of the current nested wait (or None)
    - events is a list of websocket payload dicts to emit
    """
    try:
        ws = find_active_in_progress_wait(active_step)
        if ws is None:
            if prev_sig is not None:
                return None, [{"type": "wait_end"}]
            return None, []

        # If this is the engine's top-level wait, the engine already manages wait_begin/end.
        try:
            if getattr(engine, "wait", None) is ws and bool(getattr(engine, "wait_in_progress", False)):
                return prev_sig, []
        except Exception:
            pass

        sig = (
            id(ws),
            float(getattr(ws, "started_at", 0.0) or 0.0),
            int(getattr(ws, "required_ms", 0) or 0),
        )
        if sig == prev_sig:
            return prev_sig, []

        mode = str(getattr(ws, "mode", "soft") or "soft").strip().lower() or "soft"
        wait_for = str(getattr(ws, "wait_for", "") or "")
        ev = {
            "type": "wait_begin",
            "required_ms": int(getattr(ws, "required_ms", 0) or 0),
            "mode": mode,
            "wait_for": wait_for,
        }
        return sig, [ev]
    except Exception:
        return prev_sig, []


def ui_signature_for_step(step: Any) -> Any:
    """
    Stable signature of a step's user-visible completion state.
    Must NOT include timestamps or continuously-changing values.

    Used to send timeline_update only when state *actually changes* (e.g. nested wait completes).
    """
    try:
        if step is None:
            return None

        if isinstance(step, PressState):
            return ("press", str(step.expected or ""), bool(getattr(step, "completed", False)))

        if isinstance(step, HoldState):
            return (
                "hold",
                str(step.expected or ""),
                int(getattr(step, "required_ms", 0) or 0),
                bool(getattr(step, "in_progress", False)),
                bool(getattr(step, "completed", False)),
            )

        if isinstance(step, WaitState):
            return (
                "wait",
                str(getattr(step, "mode", "") or ""),
                str(getattr(step, "wait_for", "") or ""),
                int(getattr(step, "required_ms", 0) or 0),
                bool(getattr(step, "in_progress", False)),
                bool(getattr(step, "completed", False)),
            )

        if isinstance(step, SequenceState):
            try:
                idx = int(getattr(step, "current_index", 0) or 0)
            except Exception:
                idx = 0
            cur_sig = None
            if 0 <= idx < len(step.steps):
                cur_sig = ui_signature_for_step(step.steps[idx])
            return ("seq", bool(getattr(step, "started", False)), idx, cur_sig)

        if isinstance(step, GroupState):
            active_sig = ""
            try:
                ai = getattr(step, "active_item", None)
                if ai is not None:
                    active_sig = str(getattr(ai, "signature", "") or "")
            except Exception:
                active_sig = ""

            items_sig: list[Any] = []
            for it in (getattr(step, "items", None) or []):
                try:
                    items_sig.append(
                        (
                            str(getattr(it, "kind", "") or ""),
                            str(getattr(it, "signature", "") or ""),
                            int(getattr(it, "completed_count", 0) or 0),
                            int(getattr(it, "required_count", 1) or 1),
                            ui_signature_for_step(getattr(it, "state", None)),
                        )
                    )
                except Exception:
                    items_sig.append(("?", "", 0, 1, None))

            return (
                "group",
                bool(getattr(step, "wait_active", False)),
                active_sig,
                tuple(items_sig),
            )

        return ("unknown", type(step).__name__)
    except Exception:
        return ("unknown", "error")