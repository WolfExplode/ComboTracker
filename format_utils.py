"""
Time and display formatting and duration parsing.
Used by combo_engine and _combo_commands; logic lives in one place.
"""

from __future__ import annotations

from parser import _parse_duration


def format_ms(ms: int) -> str:
    ms = int(ms)
    if ms % 1000 == 0:
        return f"{ms//1000:d}s ({ms}ms)"
    return f"{ms/1000.0:.4g}s ({ms}ms)"


def format_ms_brief(ms: float | int | None) -> str:
    if ms is None:
        return "—"
    try:
        ms_i = int(round(float(ms)))
    except Exception:
        return "—"
    if ms_i < 1000:
        return f"{ms_i}ms"
    return f"{ms_i/1000.0:.3g}s"


def format_hold_requirement(hold_ms: int | None) -> str:
    if hold_ms is None:
        return ""
    if hold_ms % 1000 == 0:
        return f"{hold_ms // 1000:d}s"
    return f"{hold_ms / 1000.0:.3g}s"


def parse_expected_time_ms(raw: str | None) -> int | None:
    """Parse user-entered expected time (e.g. 1.05s or 1050ms) to milliseconds."""
    raw = (raw or "").strip().lower()
    if not raw:
        return None
    ms = _parse_duration(raw)
    if ms is None or ms <= 0:
        return None
    return int(ms)
