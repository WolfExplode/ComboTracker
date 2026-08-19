"""Low-overhead timing records and summaries for one macro playback run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


@dataclass(frozen=True)
class MacroTimingSample:
    order: int
    kind: str
    key: str
    planned_offset_ms: float
    wake_lateness_ms: float
    dispatch_start_lateness_ms: float
    output_duration_ms: float
    dispatch_complete_lateness_ms: float


@dataclass(frozen=True)
class MacroTimingProfile:
    request_to_clock_start_ms: float
    request_to_first_dispatch_ms: float | None
    samples: tuple[MacroTimingSample, ...]

    def summary(self) -> dict[str, Any]:
        start_lateness = [s.dispatch_start_lateness_ms for s in self.samples]
        completion_lateness = [s.dispatch_complete_lateness_ms for s in self.samples]
        output_duration = [s.output_duration_ms for s in self.samples]
        interval_error = [
            abs(current.dispatch_start_lateness_ms - previous.dispatch_start_lateness_ms)
            for previous, current in zip(self.samples, self.samples[1:])
        ]

        deadline_groups: dict[float, list[MacroTimingSample]] = {}
        output_groups: dict[str, list[float]] = {}
        for sample in self.samples:
            deadline_groups.setdefault(round(sample.planned_offset_ms, 6), []).append(sample)
            output_groups.setdefault(f"{sample.key}.{sample.kind}", []).append(
                sample.output_duration_ms
            )

        first_at_deadline = [group[0] for group in deadline_groups.values()]
        later_at_deadline = [sample for group in deadline_groups.values() for sample in group[1:]]
        scheduler_lateness = [sample.dispatch_start_lateness_ms for sample in first_at_deadline]
        collision_lateness = [sample.dispatch_start_lateness_ms for sample in later_at_deadline]
        output_by_input = {
            name: {"event_count": len(values), **_distribution(values)}
            for name, values in sorted(output_groups.items())
        }
        return {
            "event_count": len(self.samples),
            "request_to_clock_start_ms": self.request_to_clock_start_ms,
            "request_to_first_dispatch_ms": self.request_to_first_dispatch_ms,
            "dispatch_start_lateness_ms": _distribution(start_lateness),
            "scheduler_lateness_ms": _distribution(scheduler_lateness),
            "same_deadline_lateness_ms": _distribution(collision_lateness),
            "dispatch_complete_lateness_ms": _distribution(completion_lateness),
            "output_duration_ms": _distribution(output_duration),
            "output_duration_by_input_ms": output_by_input,
            "interval_error_ms": _distribution(interval_error),
            "deadline_analysis": {
                "deadline_count": len(deadline_groups),
                "collision_deadline_count": sum(
                    len(group) > 1 for group in deadline_groups.values()
                ),
                "later_collision_event_count": len(later_at_deadline),
            },
            "late_event_counts": {
                "over_1ms": sum(value > 1.0 for value in start_lateness),
                "over_2ms": sum(value > 2.0 for value in start_lateness),
                "over_5ms": sum(value > 5.0 for value in start_lateness),
            },
            "scheduler_late_event_counts": {
                "over_1ms": sum(value > 1.0 for value in scheduler_lateness),
                "over_2ms": sum(value > 2.0 for value in scheduler_lateness),
                "over_5ms": sum(value > 5.0 for value in scheduler_lateness),
            },
        }

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = self.summary()
        if include_events:
            payload["events"] = [asdict(sample) for sample in self.samples]
        return payload


class MacroTimingCollector:
    """Collect timestamps in the hot path; calculate statistics after playback."""

    def __init__(self, *, requested_at: float, clock_started_at: float) -> None:
        self._requested_ns = int(requested_at * 1_000_000_000)
        self._clock_started_ns = int(clock_started_at * 1_000_000_000)
        self._samples: list[MacroTimingSample] = []

    def record(
        self,
        *,
        order: int,
        kind: str,
        key: str,
        planned_offset_s: float,
        woke_ns: int,
        dispatch_started_ns: int,
        dispatch_completed_ns: int,
    ) -> None:
        deadline_ns = self._clock_started_ns + int(planned_offset_s * 1_000_000_000)
        self._samples.append(
            MacroTimingSample(
                order=order,
                kind=kind,
                key=key,
                planned_offset_ms=planned_offset_s * 1000.0,
                wake_lateness_ms=(woke_ns - deadline_ns) / 1_000_000.0,
                dispatch_start_lateness_ms=(dispatch_started_ns - deadline_ns) / 1_000_000.0,
                output_duration_ms=(dispatch_completed_ns - dispatch_started_ns) / 1_000_000.0,
                dispatch_complete_lateness_ms=(dispatch_completed_ns - deadline_ns) / 1_000_000.0,
            )
        )

    def finish(self) -> MacroTimingProfile:
        first_dispatch_ms = None
        if self._samples:
            first = self._samples[0]
            first_dispatch_ms = (
                (self._clock_started_ns - self._requested_ns) / 1_000_000.0
                + first.planned_offset_ms
                + first.dispatch_start_lateness_ms
            )
        return MacroTimingProfile(
            request_to_clock_start_ms=(self._clock_started_ns - self._requested_ns) / 1_000_000.0,
            request_to_first_dispatch_ms=first_dispatch_ms,
            samples=tuple(self._samples),
        )
