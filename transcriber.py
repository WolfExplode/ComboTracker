"""Accurate raw input recording and post-run combo transcription."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal


def _parse_valid_keys(keys_str: str) -> set[str]:
    return {
        part.strip().lower()
        for part in (keys_str or "").split(",")
        if part.strip()
    }


def _format_seconds(seconds: float) -> str:
    """Format a duration at millisecond precision without trailing zeroes."""
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    return f"{milliseconds / 1000.0:.3f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class RecordedInputEvent:
    sequence: int
    key: str
    phase: Literal["down", "up"]
    at_s: float
    synthetic: bool = False


@dataclass
class _RecordedAction:
    key: str
    started_at: float
    ended_at: float
    start_sequence: int
    end_sequence: int
    is_hold: bool
    children: list["_RecordedAction"] = field(default_factory=list)


class Transcriber:
    """Capture physical edges first, then compile the completed timeline."""

    def __init__(
        self,
        *,
        start_key: str = "f",
        merge_threshold_s: float = 0.2,
        hold_threshold_s: float = 0.20,
        min_wait_s: float = 0.1,
        on_stop: Callable[[str], None] | None = None,
    ):
        self.start_key = start_key.lower()
        # Retained for configuration compatibility. Accurate capture no longer
        # merges distinct physical clicks; repeat-downs are filtered per key.
        self.merge_threshold_s = merge_threshold_s
        self.hold_threshold_s = hold_threshold_s
        self.min_wait_s = min_wait_s
        self.on_stop = on_stop

        self._lock = threading.RLock()
        self._valid_keys: set[str] = set()
        self._state = "idle"
        self._events: list[RecordedInputEvent] = []
        self._pressed: dict[str, RecordedInputEvent] = {}
        self._sequence = 0
        self._ignored_repeat_downs = 0
        self._ignored_unmatched_ups = 0
        self._last_recording: dict | None = None

    def set_valid_keys(self, keys_str: str) -> None:
        with self._lock:
            self._valid_keys = _parse_valid_keys(keys_str)

    def set_min_wait_s(self, seconds: float) -> None:
        try:
            value = max(0.0, float(seconds))
        except (TypeError, ValueError):
            value = 0.0
        with self._lock:
            self.min_wait_s = value

    def is_valid_key(self, key: str) -> bool:
        with self._lock:
            return (key or "").strip().lower() in self._valid_keys

    def is_recording(self) -> bool:
        with self._lock:
            return self._state == "recording"

    def start(self, started_at: float | None = None) -> None:
        del started_at  # The first accepted physical edge anchors the recording.
        with self._lock:
            self._state = "recording"
            self._events = []
            self._pressed = {}
            self._sequence = 0
            self._ignored_repeat_downs = 0
            self._ignored_unmatched_ups = 0
            self._last_recording = None

    def stop(self, stopped_at: float | None = None) -> str:
        with self._lock:
            if self._state != "recording":
                return ""
            close_at = self._close_timestamp(stopped_at)
            self._close_pressed_inputs(close_at)
            transcript, diagnostics = self._compile(self._events)
            self._last_recording = self._recording_payload(transcript, diagnostics, self._events)
            self._state = "idle"
            self._events = []
            self._pressed = {}
            callback = self.on_stop

        if callback:
            callback(transcript)
        return transcript

    def last_recording(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._last_recording)

    def current_transcript(self) -> str:
        with self._lock:
            events = list(self._events)
            if self._pressed:
                close_at = self._close_timestamp(None)
                sequence = self._sequence
                for down in sorted(self._pressed.values(), key=lambda item: item.sequence):
                    sequence += 1
                    events.append(
                        RecordedInputEvent(
                            sequence=sequence,
                            key=down.key,
                            phase="up",
                            at_s=max(close_at, down.at_s),
                            synthetic=True,
                        )
                    )
            transcript, _diagnostics = self._compile(events)
            return transcript

    def key_down(self, key: str, t: float) -> None:
        key = (key or "").strip().lower()
        with self._lock:
            if self._state != "recording" or not key or key not in self._valid_keys:
                return
            if key in self._pressed:
                self._ignored_repeat_downs += 1
                return
            event = self._new_event(key, "down", t)
            self._events.append(event)
            self._pressed[key] = event

    def key_up(self, key: str, t: float) -> None:
        key = (key or "").strip().lower()
        with self._lock:
            if self._state != "recording" or not key or key not in self._valid_keys:
                return
            down = self._pressed.pop(key, None)
            if down is None:
                self._ignored_unmatched_ups += 1
                return
            self._events.append(self._new_event(key, "up", max(float(t), down.at_s)))

    def _new_event(
        self,
        key: str,
        phase: Literal["down", "up"],
        at_s: float,
        *,
        synthetic: bool = False,
    ) -> RecordedInputEvent:
        self._sequence += 1
        return RecordedInputEvent(
            sequence=self._sequence,
            key=key,
            phase=phase,
            at_s=float(at_s),
            synthetic=synthetic,
        )

    def _close_timestamp(self, requested: float | None) -> float:
        latest = max((event.at_s for event in self._events), default=0.0)
        if requested is None:
            return latest
        return max(latest, float(requested))

    def _close_pressed_inputs(self, close_at: float) -> None:
        for down in sorted(self._pressed.values(), key=lambda item: item.sequence):
            self._events.append(
                self._new_event(
                    down.key,
                    "up",
                    max(close_at, down.at_s),
                    synthetic=True,
                )
            )
        self._pressed.clear()

    def _compile(self, events: list[RecordedInputEvent]) -> tuple[str, dict]:
        actions = self._pair_actions(events)
        parents, crossing_overlaps = self._build_containment_tree(actions)
        top_level = [action for action in actions if id(action) not in parents]
        top_level.sort(key=lambda action: (action.started_at, action.start_sequence))
        origin = top_level[0].started_at if top_level else 0.0
        tokens, _cursor = self._render_sequence(top_level, origin)
        diagnostics = {
            "event_count": len(events),
            "action_count": len(actions),
            "hold_count": sum(action.is_hold for action in actions),
            "nested_action_count": len(parents),
            "crossing_overlap_count": crossing_overlaps,
            "ignored_repeat_downs": self._ignored_repeat_downs,
            "ignored_unmatched_ups": self._ignored_unmatched_ups,
        }
        return ", ".join(tokens), diagnostics

    def _pair_actions(self, events: list[RecordedInputEvent]) -> list[_RecordedAction]:
        active: dict[str, RecordedInputEvent] = {}
        actions: list[_RecordedAction] = []
        for event in sorted(events, key=lambda item: (item.at_s, item.sequence)):
            if event.phase == "down":
                active.setdefault(event.key, event)
                continue
            down = active.pop(event.key, None)
            if down is None:
                continue
            duration = max(0.0, event.at_s - down.at_s)
            actions.append(
                _RecordedAction(
                    key=event.key,
                    started_at=down.at_s,
                    ended_at=max(event.at_s, down.at_s),
                    start_sequence=down.sequence,
                    end_sequence=event.sequence,
                    is_hold=duration >= self.hold_threshold_s,
                )
            )
        actions.sort(key=lambda action: (action.started_at, action.start_sequence))
        return actions

    @staticmethod
    def _contains(parent: _RecordedAction, child: _RecordedAction) -> bool:
        if not parent.is_hold or parent is child:
            return False
        if parent.started_at > child.started_at or child.ended_at > parent.ended_at:
            return False
        if parent.started_at == child.started_at and parent.ended_at == child.ended_at:
            return parent.start_sequence < child.start_sequence
        return parent.started_at < child.started_at or child.ended_at < parent.ended_at

    def _parent_for(
        self,
        action: _RecordedAction,
        actions: list[_RecordedAction],
    ) -> _RecordedAction | None:
        candidates = [candidate for candidate in actions if self._contains(candidate, action)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                candidate.ended_at - candidate.started_at,
                -candidate.start_sequence,
            ),
        )

    def _build_containment_tree(
        self,
        actions: list[_RecordedAction],
    ) -> tuple[dict[int, _RecordedAction], int]:
        parents: dict[int, _RecordedAction] = {}
        for action in actions:
            action.children = []
        for action in actions:
            parent = self._parent_for(action, actions)
            if parent is not None:
                parents[id(action)] = parent
                parent.children.append(action)
        for action in actions:
            action.children.sort(key=lambda child: (child.started_at, child.start_sequence))

        crossings = 0
        for index, left in enumerate(actions):
            for right in actions[index + 1 :]:
                overlaps = left.started_at < right.ended_at and right.started_at < left.ended_at
                if overlaps and not self._contains(left, right) and not self._contains(right, left):
                    crossings += 1
        return parents, crossings

    def _render_sequence(
        self,
        actions: list[_RecordedAction],
        cursor: float,
    ) -> tuple[list[str], float]:
        tokens: list[str] = []
        for action in actions:
            wait_s = max(0.0, action.started_at - cursor)
            if wait_s > 0 and wait_s + 1e-9 >= self.min_wait_s:
                tokens.append(f"wait:{_format_seconds(wait_s)}s")

            if action.is_hold:
                duration = _format_seconds(action.ended_at - action.started_at)
                if action.children:
                    body, _body_cursor = self._render_sequence(action.children, action.started_at)
                    tokens.append(f"hold({action.key}, {duration}s, {{{', '.join(body)}}})")
                else:
                    tokens.append(f"hold({action.key}, {duration}s)")
                cursor = max(cursor, action.ended_at)
            else:
                tokens.append(action.key)
                cursor = max(cursor, action.started_at)
        return tokens, cursor

    def _recording_payload(
        self,
        transcript: str,
        diagnostics: dict,
        events: list[RecordedInputEvent],
    ) -> dict:
        ordered = sorted(events, key=lambda item: (item.at_s, item.sequence))
        origin = ordered[0].at_s if ordered else 0.0
        return {
            "schema_version": 1,
            "transcript": transcript,
            "settings": {
                "hold_threshold_ms": round(self.hold_threshold_s * 1000.0, 3),
                "min_wait_ms": round(self.min_wait_s * 1000.0, 3),
            },
            "diagnostics": copy.deepcopy(diagnostics),
            "events": [
                {
                    "sequence": event.sequence,
                    "key": event.key,
                    "phase": event.phase,
                    "offset_ms": round((event.at_s - origin) * 1000.0, 3),
                    "synthetic": event.synthetic,
                }
                for event in ordered
            ],
        }
