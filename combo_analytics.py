"""
Read-only analytics for combo stats, APM, difficulty, and min-time.
All functions accept an engine (duck-typed) and never mutate it.
"""

from __future__ import annotations

from typing import Any

from states import (
    GroupState,
    HoldState,
    PressState,
    SequenceState,
    WaitState,
)


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _step_time_ms(s: Any) -> int:
    """Sum of wait + hold ms for one runtime step."""
    if isinstance(s, WaitState):
        return int(s.required_ms or 0)
    if isinstance(s, HoldState):
        return int(s.required_ms or 0)
    if isinstance(s, PressState):
        return 0
    if isinstance(s, SequenceState):
        return sum(_step_time_ms(x) for x in s.steps)
    if isinstance(s, GroupState):
        return sum(_step_time_ms(item.state) for item in s.items)
    return 0


def min_combo_time_ms(engine) -> int:
    """Fastest possible combo time (sum of waits + holds) in ms."""
    steps = getattr(engine, "runtime_steps", None) or []
    total = 0
    for s in steps:
        total += _step_time_ms(s)
    return max(0, int(total))


def _count_step_actions(s: Any) -> tuple[int, int]:
    """(press_count, hold_count) for one StepState."""
    if isinstance(s, PressState):
        return (1, 0)
    if isinstance(s, HoldState):
        return (0, 1)
    if isinstance(s, WaitState):
        return (0, 0)
    if isinstance(s, SequenceState):
        p = h = 0
        for sub in s.steps:
            pp, hh = _count_step_actions(sub)
            p += pp
            h += hh
        return (p, h)
    if isinstance(s, GroupState):
        p = h = 0
        for item in s.items:
            pp, hh = _count_step_actions(item.state)
            n = int(getattr(item, "required_count", 1) or 1)
            p += pp * n
            h += hh * n
        return (p, h)
    return (0, 0)


def count_combo_actions(engine) -> tuple[int, int, int]:
    """Returns (press_count, hold_count, total_actions)."""
    steps = getattr(engine, "runtime_steps", None) or []
    press = 0
    hold = 0
    for s in steps:
        pp, hh = _count_step_actions(s)
        press += pp
        hold += hh
    return press, hold, press + hold


def practical_apm(engine) -> float | None:
    """Uses user-entered expected execution time (ms) for the active combo."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return None
    steps = getattr(engine, "runtime_steps", None) or []
    if not steps:
        return None
    expected_ms = getattr(engine, "combo_expected_ms", {}).get(name)
    if expected_ms is None or expected_ms <= 0:
        return None
    press_count, _, _ = count_combo_actions(engine)
    if press_count <= 0:
        return None
    return (60000.0 / float(expected_ms)) * float(press_count)


def theoretical_max_apm(engine) -> float | None:
    """Uses the fastest-possible combo time (sum of waits + holds)."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return None
    steps = getattr(engine, "runtime_steps", None) or []
    if not steps:
        return None
    min_ms = min_combo_time_ms(engine)
    if min_ms <= 0:
        return None
    press_count, _, _ = count_combo_actions(engine)
    if press_count <= 0:
        return None
    return (60000.0 / float(min_ms)) * float(press_count)


def _wait_triangle_score(wait_ms: int) -> float:
    """
    Triangle-shaped wait difficulty:
    - peak at 350ms; 0 at 0ms and at >=600ms.
    """
    try:
        w = float(wait_ms)
    except Exception:
        return 0.0
    if w <= 0.0:
        return 0.0
    if w >= 600.0:
        return 0.0
    if w < 350.0:
        return _clamp01(w / 350.0)
    return _clamp01((600.0 - w) / 250.0)


def _hold_score(hold_ms: int) -> float:
    """Hold difficulty: monotonic, saturating."""
    try:
        h = float(hold_ms)
    except Exception:
        return 0.0
    if h <= 0.0:
        return 0.0
    H = 350.0
    return _clamp01(1.0 - (2.718281828 ** (-h / H)))


def _timing_variation_points(engine) -> int:
    """
    +1 per distinct non-micro wait (ignores >=600ms); +1 per distinct hold;
    micro waits (<=60ms): +1 if exactly one.
    """
    micro_thresh = 60
    waits_micro_count = 0
    non_micro_waits: set[int] = set()
    holds: set[int] = set()

    def process_wait_hold(w_ms: int | None, h_ms: int | None):
        nonlocal waits_micro_count
        if w_ms is not None:
            if w_ms <= 0 or w_ms >= 600:
                return
            if w_ms <= micro_thresh:
                waits_micro_count += 1
            else:
                non_micro_waits.add(w_ms)
        elif h_ms is not None and h_ms > 0:
            holds.add(h_ms)

    def process_step(s: Any):
        if isinstance(s, WaitState):
            process_wait_hold(s.required_ms, None)
        elif isinstance(s, HoldState):
            process_wait_hold(None, s.required_ms)
        elif isinstance(s, PressState):
            pass
        elif isinstance(s, SequenceState):
            for sub in s.steps:
                process_step(sub)
        elif isinstance(s, GroupState):
            for item in s.items:
                process_step(item.state)

    for s in getattr(engine, "runtime_steps", None) or []:
        process_step(s)

    micro_bonus = 1 if waits_micro_count == 1 else 0
    return int(len(non_micro_waits) + len(holds) + micro_bonus)


def _collect_wait_hold_scores(steps: list[Any]) -> tuple[list[float], list[float]]:
    """Collect wait_triangle and hold scores from runtime steps (StepState only)."""
    wait_scores: list[float] = []
    hold_scores: list[float] = []

    def process(s: Any):
        if isinstance(s, WaitState):
            try:
                wait_scores.append(_wait_triangle_score(int(s.required_ms or 0)))
            except Exception:
                pass
        elif isinstance(s, HoldState):
            try:
                hold_scores.append(_hold_score(int(s.required_ms or 0)))
            except Exception:
                pass
        elif isinstance(s, SequenceState):
            for sub in s.steps:
                process(sub)
        elif isinstance(s, GroupState):
            for item in s.items:
                process(item.state)

    for s in steps:
        process(s)
    return wait_scores, hold_scores


def difficulty_score_10(engine) -> float | None:
    """Returns a 0..10 difficulty score or None if no active combo."""
    name = getattr(engine, "active_combo_name", None)
    steps = getattr(engine, "runtime_steps", None) or []
    if not steps or not name:
        return None

    apm = practical_apm(engine) or 0.0
    _press, _hold, actions = count_combo_actions(engine)
    apm_norm = _clamp01(apm / 200.0)
    actions_norm = _clamp01(float(actions) / 8.0)
    keys = (0.6 * apm_norm) + (0.4 * actions_norm)

    wait_scores, hold_scores = _collect_wait_hold_scores(steps)
    has_wait = 1.0 if wait_scores else 0.0
    has_hold = 1.0 if hold_scores else 0.0
    wait_avg = (sum(wait_scores) / len(wait_scores)) if wait_scores else 0.0
    hold_avg = (sum(hold_scores) / len(hold_scores)) if hold_scores else 0.0
    wait_w, hold_w = 1.0, 1.5
    denom = (wait_w * has_wait) + (hold_w * has_hold)
    timing_base = 0.0 if denom <= 0 else ((wait_avg * wait_w * has_wait) + (hold_avg * hold_w * has_hold)) / denom

    var_points = _timing_variation_points(engine)
    var_norm = _clamp01(1.0 - (2.718281828 ** (-float(var_points) / 1.0)))
    timing = (0.3 * _clamp01(timing_base)) + (0.7 * var_norm)
    combined = (0.45 * keys) + (0.55 * timing)
    return round(10.0 * _clamp01(combined), 1)


def user_difficulty_value(engine) -> float | None:
    """User-entered difficulty (0..10) for the active combo."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return None
    d = getattr(engine, "combo_user_difficulty", {}).get(name)
    if d is None:
        return None
    try:
        d_f = float(d)
    except Exception:
        return None
    if 0.0 <= d_f <= 10.0:
        return d_f
    return None


def combo_stats_summary(engine) -> dict[str, Any]:
    """
    Read-only stats summary for the active combo.
    Keys: success, fail, best_ms, avg_ms, hardest (list of (idx, label, cnt)).
    Treats missing stats as 0/None; does not call _ensure_combo_stats.
    """
    name = getattr(engine, "active_combo_name", None)
    stats = getattr(engine, "combo_stats", {})
    runtime_steps = getattr(engine, "runtime_steps", None) or []
    expected_label = getattr(engine, "_expected_label_for_step", None)

    if not name:
        return {"success": 0, "fail": 0, "best_ms": None, "avg_ms": None, "hardest": []}

    rec = stats.get(name)
    if not isinstance(rec, dict):
        return {"success": 0, "fail": 0, "best_ms": None, "avg_ms": None, "hardest": []}

    success = int(rec.get("success", 0) or 0)
    fail = int(rec.get("fail", 0) or 0)
    best_ms = rec.get("best_ms")
    total_ms = int(rec.get("total_success_ms", 0) or 0)
    avg_ms = (total_ms / float(success)) if success > 0 and total_ms > 0 else None

    hardest: list[tuple[int, str, int]] = []
    by_step = rec.get("fail_by_step")
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
        for cnt, idx in pairs[:2]:
            label = "—"
            if expected_label and 0 <= idx < len(runtime_steps):
                label = expected_label(runtime_steps[idx])
            hardest.append((idx, label, cnt))

    return {
        "success": success,
        "fail": fail,
        "best_ms": best_ms,
        "avg_ms": avg_ms,
        "hardest": hardest,
    }


def failures_by_reason(engine) -> dict[str, int]:
    """Read-only fail-by-reason counts for the active combo."""
    name = getattr(engine, "active_combo_name", None)
    if not name:
        return {}
    stats = getattr(engine, "combo_stats", {})
    rec = stats.get(name)
    if not isinstance(rec, dict):
        return {}
    by_reason = rec.get("fail_by_reason", {})
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
