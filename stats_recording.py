"""
Stats recording: success/fail counts, fail detail, ensure combo_stats shape.
Mutates engine.combo_stats; calls engine.save_combos() and engine._emit_stats_and_fail().
"""

from __future__ import annotations

import time
from typing import Any

import step_introspection
from persistence import fresh_combo_stats


def ensure_combo_stats(engine: Any, name: str) -> None:
    """Ensure engine.combo_stats[name] exists with all keys from fresh_combo_stats()."""
    if not name:
        return
    fresh = fresh_combo_stats()
    if name not in engine.combo_stats or not isinstance(engine.combo_stats.get(name), dict):
        engine.combo_stats[name] = dict(fresh)
    else:
        for k, v in fresh.items():
            engine.combo_stats[name].setdefault(k, v)


def combo_avg_ms(engine: Any, name: str) -> float | None:
    """Average success completion time in ms for the given combo, or None."""
    ensure_combo_stats(engine, name)
    s = int(engine.combo_stats[name].get("success", 0) or 0)
    total = int(engine.combo_stats[name].get("total_success_ms", 0) or 0)
    if s <= 0 or total <= 0:
        return None
    return total / float(s)


def format_percent(success: int, fail: int) -> str:
    """Format success/(success+fail) as a percentage string."""
    total = success + fail
    if total <= 0:
        return "—"
    return f"{(success / total) * 100:.1f}%"


def record_fail_detail(
    engine: Any,
    *,
    step_index: int,
    expected: str,
    actual: str,
    reason: str,
    elapsed_ms: float | None,
) -> None:
    """Update engine.combo_stats for the active combo with one fail event detail."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return
    ensure_combo_stats(engine, name)

    by_step = engine.combo_stats[name].get("fail_by_step", {})
    if not isinstance(by_step, dict):
        by_step = {}
    key_step = str(max(0, int(step_index)))
    by_step[key_step] = int(by_step.get(key_step, 0) or 0) + 1
    engine.combo_stats[name]["fail_by_step"] = by_step

    by_exp = engine.combo_stats[name].get("fail_by_expected", {})
    if not isinstance(by_exp, dict):
        by_exp = {}
    exp_key = (expected or "—").strip().lower()
    by_exp[exp_key] = int(by_exp.get(exp_key, 0) or 0) + 1
    engine.combo_stats[name]["fail_by_expected"] = by_exp

    by_reason = engine.combo_stats[name].get("fail_by_reason", {})
    if not isinstance(by_reason, dict):
        by_reason = {}
    r = (reason or "unknown").strip().lower() or "unknown"
    by_reason[r] = int(by_reason.get(r, 0) or 0) + 1
    engine.combo_stats[name]["fail_by_reason"] = by_reason

    ev = {
        "ts": int(time.time()),
        "attempt": int(getattr(engine, "attempt_counter", 0) or 0),
        "step_index": int(step_index),
        "expected": str(expected or ""),
        "actual": str(actual or ""),
        "reason": str(reason or ""),
        "elapsed_ms": int(round(float(elapsed_ms))) if elapsed_ms is not None else None,
    }
    events = engine.combo_stats[name].get("fail_events", [])
    if not isinstance(events, list):
        events = []
    events.append(ev)
    if len(events) > 100:
        events = events[-100:]
    engine.combo_stats[name]["fail_events"] = events


def record_combo_success(engine: Any, completion_ms: float | int | None = None) -> None:
    """Record a successful combo completion; update stats, save, emit UI."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return
    runtime_steps = getattr(engine, "runtime_steps", None) or []
    engine._ui_last_success_combo = name
    engine._ui_last_success_steps_len = len(runtime_steps)
    ensure_combo_stats(engine, name)
    engine.combo_stats[name]["success"] += 1

    if completion_ms is None and getattr(engine, "start_time", 0):
        completion_ms = (engine._now() - engine.start_time) * 1000.0
    try:
        ms = int(round(float(completion_ms))) if completion_ms is not None else None
    except Exception:
        ms = None
    if ms is not None and ms > 0:
        total = int(engine.combo_stats[name].get("total_success_ms", 0) or 0)
        engine.combo_stats[name]["total_success_ms"] = total + ms
        best = engine.combo_stats[name].get("best_ms", None)
        try:
            best_i = int(best) if best is not None else None
        except Exception:
            best_i = None
        if best_i is None or ms < best_i:
            engine.combo_stats[name]["best_ms"] = ms

    engine.save_combos()
    engine._emit_stats_and_fail()


def record_combo_fail(
    engine: Any,
    *,
    actual: str | None = None,
    expected_step_index: int | None = None,
    expected_label: str | None = None,
    reason: str | None = None,
    elapsed_ms: float | None = None,
) -> None:
    """Record a combo failure; update stats, save, emit UI."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return
    if getattr(engine, "attempt_counter", 0) <= 0:
        return

    engine._ui_last_success_combo = None
    engine._ui_last_success_steps_len = 0

    ensure_combo_stats(engine, name)
    engine.combo_stats[name]["fail"] += 1

    idx = getattr(engine, "current_index", 0) if expected_step_index is None else expected_step_index
    try:
        idx_i = int(idx)
    except Exception:
        idx_i = 0
    exp = expected_label
    if not exp:
        step = engine._active_step() if hasattr(engine, "_active_step") else None
        exp = step_introspection.expected_label_for_step(step) if step else "—"

    record_fail_detail(
        engine,
        step_index=idx_i,
        expected=str(exp or "—"),
        actual=str(actual or ""),
        reason=str(reason or ""),
        elapsed_ms=elapsed_ms,
    )

    engine.save_combos()
    engine._emit_stats_and_fail()
