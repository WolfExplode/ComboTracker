"""
Auto-transcription: record key down/up with timestamps and produce combo syntax.
Used when transcribe mode is on: start key is configurable (default f), stop with Esc; output goes to Inputs field.
"""

from __future__ import annotations

from typing import Callable


def _parse_valid_keys(keys_str: str) -> set[str]:
    """Parse comma-separated keys, normalize to lowercase, return set."""
    out: set[str] = set()
    for part in (keys_str or "").split(","):
        k = part.strip().lower()
        if k:
            out.add(k)
    return out


class Transcriber:
    """
    Records key_down/key_up with timestamps and builds combo syntax string.
    State: idle | recording. Only key_down/key_up when recording (and key in valid_keys).
    """

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
        self.merge_threshold_s = merge_threshold_s
        self.hold_threshold_s = hold_threshold_s
        self.min_wait_s = min_wait_s
        self.on_stop = on_stop

        self._valid_keys: set[str] = set()
        self._state: str = "idle"  # idle | recording
        self._tokens: list[str] = []

        # Run: same key within merge_threshold gets merged into one key
        self._run_key: str | None = None
        self._run_start: float = 0.0
        self._run_key_down_count: int = 0
        # After key_up we keep this so next key_down can emit wait (next_down - last_run_start)
        self._last_run_start: float = 0.0

    def set_valid_keys(self, keys_str: str) -> None:
        self._valid_keys = _parse_valid_keys(keys_str or "")

    def is_valid_key(self, key: str) -> bool:
        return (key or "").strip().lower() in self._valid_keys

    def is_recording(self) -> bool:
        return self._state == "recording"

    def start(self) -> None:
        self._state = "recording"
        self._tokens = []
        self._run_key = None
        self._run_start = 0.0
        self._run_key_down_count = 0
        self._last_run_start = 0.0

    def stop(self) -> str:
        transcript = self._build_transcript()
        self._state = "idle"
        self._tokens = []
        self._run_key = None
        if self.on_stop and transcript is not None:
            self.on_stop(transcript)
        return transcript or ""

    def _emit_wait(self, duration_s: float) -> None:
        if duration_s < self.min_wait_s:
            return
        # don't emit a wait immediately after a hold
        if self._tokens and self._tokens[-1].startswith("hold("):
            return
        sec = round(duration_s, 2)
        self._tokens.append(f"wait:{sec}s")

    def _emit_key(self, key: str) -> None:
        self._tokens.append(key)

    def _emit_hold(self, key: str, duration_s: float) -> None:
        sec = round(duration_s, 2)
        self._tokens.append(f"hold({key}, {sec}s)")

    def _build_transcript(self) -> str:
        if self._run_key and self._run_key_down_count >= 1:
            self._tokens.append(self._run_key)
        return ", ".join(self._tokens)

    def key_down(self, key: str, t: float) -> None:
        if self._state != "recording":
            return
        key = (key or "").strip().lower()
        if not key or key not in self._valid_keys:
            return

        # Wait = time from first key-down of previous run to this key-down
        if self._run_key is not None:
            wait_duration = t - self._run_start
            if key == self._run_key:
                if t - self._run_start <= self.merge_threshold_s:
                    self._run_key_down_count += 1
                    return
                # Same key but past threshold: emit current run as key, wait, then new run
                self._emit_key(self._run_key)
                if wait_duration >= self.min_wait_s:
                    self._emit_wait(wait_duration)
                self._run_start = t
                self._run_key_down_count = 1
                return
            # Different key: flush current run as key (no key_up so not hold), then wait, then new run
            self._emit_key(self._run_key)
            if wait_duration >= self.min_wait_s:
                self._emit_wait(wait_duration)
            self._run_key = key
            self._run_start = t
            self._last_run_start = t
            self._run_key_down_count = 1
            return

        # No current run; maybe we have a previous run (just had key_up)
        if self._run_key is None and self._last_run_start > 0:
            wait_duration = t - self._last_run_start
            if wait_duration >= self.min_wait_s:
                self._emit_wait(wait_duration)

        self._run_key = key
        self._run_start = t
        self._last_run_start = t
        self._run_key_down_count = 1

    def key_up(self, key: str, t: float) -> None:
        if self._state != "recording":
            return
        key = (key or "").strip().lower()
        if not key or key not in self._valid_keys:
            return
        if self._run_key != key:
            return

        duration_s = t - self._run_start
        if duration_s >= self.hold_threshold_s:
            self._emit_hold(key, duration_s)
        else:
            self._emit_key(key)
        self._last_run_start = self._run_start
        self._run_key = None
