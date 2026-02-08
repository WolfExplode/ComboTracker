from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    parts = [f"#{idx+1}:{label} ({cnt})" for idx, label, cnt in summary["hardest"]]
    hardest = " | Hardest: " + ", ".join(parts) if parts else ""
    return (
        f"Stats: {s} success / {f} fail ({pct})"
        f" | Best: {engine._format_ms_brief(best)} | Avg: {engine._format_ms_brief(avg)}"
        f"{hardest}"
    )


def failures_by_reason(engine) -> dict[str, int]:
    return combo_analytics.failures_by_reason(engine)


def min_time_text(engine) -> str:
    if not getattr(engine, "runtime_steps", None):
        return "Fastest possible: —"
    min_ms = combo_analytics.min_combo_time_ms(engine)
    return f"Fastest possible: {engine._format_ms(min_ms)}"


def practical_apm(engine) -> float | None:
    """Practical APM uses user-entered expected execution time (ms) for the active combo."""
    return combo_analytics.practical_apm(engine)


def theoretical_max_apm(engine) -> float | None:
    """Theoretical max APM uses the fastest-possible combo time (sum of waits + holds)."""
    return combo_analytics.theoretical_max_apm(engine)


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
    """Returns a 0..10 score (float) or None if there's no active combo."""
    return combo_analytics.difficulty_score_10(engine)


def difficulty_text(engine) -> str:
    d = difficulty_score_10(engine)
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


def _timeline_steps_from_runtime(engine) -> list[dict[str, Any]]:
    """Build timeline payload from engine.runtime_steps (state objects)."""
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
                        items_payload.append({
                            "type": "press",
                            "input": item.state.expected,
                            "duration": 0,
                            "active": False,
                            "completed": comp,
                        })
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
                        items_payload.append({
                            "type": "wait",
                            "mode": "mandatory",
                            "wait_for": item.state.wait_for or "",
                            "duration": item.state.required_ms,
                            "active": (idx == cur) and step.wait_active,
                            "completed": comp,
                        })
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
                        seq_items = []
                        for seq_i, sub in enumerate(seq.steps):
                            is_active = seq.started and (idx == cur) and (step.active_item is item) and (seq_i == seq.current_index)
                            is_completed = (seq.started and seq_i < seq.current_index) or (idx < cur)
                            if isinstance(sub, WaitState):
                                seq_items.append({
                                    "type": "wait",
                                    "mode": sub.mode,
                                    "wait_for": sub.wait_for or "",
                                    "duration": sub.required_ms,
                                    "active": is_active,
                                    "completed": is_completed,
                                })
                            elif isinstance(sub, HoldState):
                                seq_items.append({
                                    "type": "hold",
                                    "input": sub.expected,
                                    "duration": sub.required_ms,
                                    "active": is_active,
                                    "completed": is_completed,
                                })
                            else:
                                inp = sub.expected if isinstance(sub, PressState) else ""
                                seq_items.append({
                                    "type": "press",
                                    "input": inp,
                                    "duration": 0,
                                    "active": is_active,
                                    "completed": is_completed,
                                })
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
                })
                i += 1
                continue

            case SequenceState():
                seq_items = []
                for seq_i, sub in enumerate(step.steps):
                    is_active = step.started and (idx == cur) and (seq_i == step.current_index)
                    is_completed = (step.started and seq_i < step.current_index) or (idx < cur)
                    if isinstance(sub, WaitState):
                        seq_items.append({
                            "type": "wait",
                            "mode": sub.mode,
                            "wait_for": sub.wait_for or "",
                            "duration": sub.required_ms,
                            "active": is_active,
                            "completed": is_completed,
                        })
                    elif isinstance(sub, HoldState):
                        seq_items.append({
                            "type": "hold",
                            "input": sub.expected,
                            "duration": sub.required_ms,
                            "active": is_active,
                            "completed": is_completed,
                        })
                    else:
                        inp = sub.expected if isinstance(sub, PressState) else ""
                        seq_items.append({
                            "type": "press",
                            "input": inp,
                            "duration": 0,
                            "active": is_active,
                            "completed": is_completed,
                        })
                steps.append({
                    "type": "sequence",
                    "active": idx == cur and step.started,
                    "completed": idx < cur,
                    "mark": mark,
                    "items": seq_items,
                    "progress": {"done": step.current_index if step.started else 0, "total": len(step.steps)},
                })
                i += 1
                continue

            case PressState():
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
                        })
                        i += 2
                        continue
                    if isinstance(nxt, WaitState) and nxt.mode in ("soft", "hard"):
                        w_mark = engine.step_marks.get(i + 1) or mark
                        steps.append({
                            "type": "press_wait",
                            "input": step.expected,
                            "duration": nxt.required_ms,
                            "mode": nxt.mode,
                            "active": (cur == idx) or (cur == i + 1),
                            "completed": cur > i + 1,
                            "mark": w_mark,
                        })
                        i += 2
                        continue
                steps.append({
                    "type": "press",
                    "input": step.expected,
                    "duration": 0,
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                })
                i += 1
                continue

            case WaitState():
                steps.append({
                    "type": "wait",
                    "input": None,
                    "duration": step.required_ms,
                    "mode": step.mode,
                    "wait_for": step.wait_for or "",
                    "active": idx == cur,
                    "completed": idx < cur,
                    "mark": mark,
                })
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
                })
                i += 1
                continue

            case _:
                i += 1
                continue
    return steps


def timeline_steps(engine) -> list[dict[str, Any]]:
    """Build timeline payload for the frontend from engine.runtime_steps."""
    return _timeline_steps_from_runtime(engine)


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