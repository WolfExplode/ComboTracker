import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from Game_Wuthering_Waves import (
    WutheringWavesGame,
    set_active_ww_team,
    save_or_update_ww_team,
    delete_ww_team,
    select_team_stateless,
    update_target_game_stateless,
)
import combo_analytics
import combo_engine_ui as ui
from combo_engine_ui import Status
from persistence import load_engine_state, save_engine_state

import _combo_commands as combo_commands
import input_normalization
from parser import expanded_ast_from_tokens
import format_utils
import stats_recording
import step_introspection
from combo_engine_ui_adapter import UIAdapter
from states import (
    AcceptResult,
    CompleteResult,
    FailResult,
    GroupState,
    HoldState,
    IgnoreResult,
    PressState,
    SequenceState,
    WaitState,
    build_runtime_state,
)

logger = logging.getLogger(__name__)


class ComboTrackerEngine:
    """
    Headless combo tracker:
    - Owns combos + stats persistence
    - Owns state machine (press/hold/wait + ender cooldown)
    - Emits UI events via a callback (WebSocket, etc.)
    """

    def __init__(self):
        # Engine is mutated from multiple threads:
        # - pynput keyboard/mouse callbacks
        # - ui_server tick thread (wait completion without input)
        self._lock = threading.RLock()
        # --- Data & State ---
        self.combos: dict[str, list[str]] = {}
        self.active_combo_name: str | None = None
        self.active_combo_tokens: list[str] = []
        self.runtime_steps: list[Any] = []  # list[StepState]

        self.current_index = 0

        # Shortcuts to active step when it's a hold or wait (updated on advance)
        self.hold: HoldState | None = None
        self.wait: WaitState | None = None
        self.start_time = 0.0
        self.last_input_time = 0.0
        self.attempt_counter = 0

        self.hold_in_progress = False
        self.hold_expected_input: str | None = None
        self.hold_started_at = 0.0
        self.hold_required_ms: int | None = None
        # Max hold duration observed this step (for drop message: "but held for Xms")
        self._hold_max_held_ms: float = 0.0

        self.wait_in_progress = False
        self.wait_started_at = 0.0
        self.wait_until = 0.0
        self.wait_required_ms: int | None = None

        self.currently_pressed: set[str] = set()

        # Per-attempt visual annotations for the timeline UI.
        # step_index -> mark string (e.g. "ok", "early", "missed", "wrong")
        self.step_marks: dict[int, str] = {}
        # For soft waits: track if the *next expected input* was pressed during the wait window.
        # wait_step_index -> set(inputs pressed too early for that gate)
        self.wait_early_inputs: dict[int, set[str]] = {}
        # So we only send wait_begin once per group mandatory wait (UI animates progress).
        self.ui_adapter = UIAdapter()

        # Generalized hold buffer: after any advance, if the next step is a hold and that key
        # is already in currently_pressed, we start the hold immediately (no separate buffer state).

        # Combo enders: key -> cooldown_ms (0 = no cooldown; wrong press drops immediately)
        self.combo_enders: dict[str, int] = {}
        # Soft enders: keys that do not drop the combo when pressed during a hold step (~key:2s)
        self.combo_enders_soft: set[str] = set()
        # Auto-transcribe: comma-separated valid keys (persisted like combo_enders)
        self.transcribe_valid_keys: str = ""
        # Key that begins a transcription when auto-transcribe is on (normalized token, e.g. f, space, lmb)
        self.transcribe_start_key: str = "f"
        # When enabled and transcribe_strip_wait_under_ms is non-empty, omit wait: below that many ms (legacy default 100)
        self.transcribe_strip_wait_under_enabled: bool = True
        self.transcribe_strip_wait_under_ms: str = "0"
        self.last_success_input: str | None = None
        # Per-ender cooldown: key -> time.perf_counter() when that key's cooldown ends.
        # When the user correctly presses an ender key, we set cooldown for that key so
        # re-pressing it during a wait/hold doesn't drop the combo until cooldown expires.
        self._ender_cooldown_until: dict[str, float] = {}

        # UI helper: after a successful completion we reset current_index back to 0 (ready for next attempt),
        # but we still want the timeline to stay fully "completed" (green) until the next attempt begins.
        self._ui_last_success_combo: str | None = None
        self._ui_last_success_steps_len: int = 0

        # No-fail (practice) mode: on failure, mark step red and advance instead of resetting combo.
        self.no_fail_mode: bool = False

        # Stats
        self.combo_stats: dict[str, dict[str, Any]] = {}
        # Per-combo metadata (kept minimal on purpose)
        # - expected_ms: user-entered typical execution time (used for Practical APM / difficulty)
        self.combo_expected_ms: dict[str, int] = {}
        # - user_difficulty: user-entered difficulty rating (0..10)
        self.combo_user_difficulty: dict[str, float] = {}

        # Optional: per-combo step display configuration
        # - combo_step_display_mode: "icons" (default) or "images"
        self.combo_step_display_mode: dict[str, str] = {}
        # - combo_key_images: combo_name -> { key -> image_url }
        self.combo_key_images: dict[str, dict[str, str]] = {}
        # - combo_demo_video: combo_name -> video URL (e.g. YouTube link for embedded demo)
        self.combo_demo_video: dict[str, str] = {}

        # Game-specific state (kept out of the core combo engine logic)
        self.ww = WutheringWavesGame()

        # Emission
        self._emit: Callable[[dict[str, Any]], None] | None = None

        # Persistence
        self.data_dir = self._get_data_dir()
        self.save_path = self.data_dir / "combos.json"

        # Load persisted state
        self.load_combos()

    # -------------------------
    # Backwards-compatible accessors (WW fields used throughout the engine)
    # -------------------------
    # These properties keep older code paths working while the WW logic is moved into
    # `Game_Wuthering_Waves.py`.

    @property
    def combo_target_game(self) -> dict[str, str]:
        return self.ww.combo_target_game

    @combo_target_game.setter
    def combo_target_game(self, value: dict[str, str]):
        self.ww.combo_target_game = value

    @property
    def ww_teams(self) -> dict[str, dict[str, Any]]:
        return self.ww.ww_teams

    @ww_teams.setter
    def ww_teams(self, value: dict[str, dict[str, Any]]):
        self.ww.ww_teams = value

    @property
    def ww_active_team_id(self) -> str | None:
        return self.ww.ww_active_team_id

    @ww_active_team_id.setter
    def ww_active_team_id(self, value: str | None):
        self.ww.ww_active_team_id = value

    @property
    def combo_ww_team(self) -> dict[str, str]:
        return self.ww.combo_ww_team

    @combo_ww_team.setter
    def combo_ww_team(self, value: dict[str, str]):
        self.ww.combo_ww_team = value

    @property
    def ww_active_character(self) -> str | None:
        return self.ww.ww_active_character

    # -------------------------
    # Emission helpers
    # -------------------------

    def set_emitter(self, emit_func: Callable[[dict[str, Any]], None] | None):
        """Set an event emitter callback. It must be thread-safe."""
        self._emit = emit_func

    def _send(self, msg: dict[str, Any]):
        if self._emit:
            try:
                self._emit(msg)
            except Exception:
                # Never let UI plumbing crash input processing
                logger.debug("Emitter raised while sending message", exc_info=True)

    def _emit_stats_and_fail(self):
        self._send({"type": "stat_update", "stats": self.stats_text()})
        self._send({"type": "fail_update", "fail_by_step": self.failures_by_step()})

    # -------------------------
    # Normalization helpers (delegate to input_normalization)
    # -------------------------

    def normalize_key(self, key) -> str:
        return input_normalization.normalize_key(key)

    def normalize_mouse(self, button) -> str:
        return input_normalization.normalize_mouse(button)

    def split_inputs(self, keys_str: str):
        return input_normalization.split_inputs(keys_str or "")

    def calc_min_combo_time_ms(self, steps: list[Any] | None) -> int:
        """Fastest possible combo time in ms. Delegates to combo_analytics."""
        import combo_analytics
        total = sum(combo_analytics._step_time_ms(s) for s in (steps or []))
        return max(0, int(total))

    def _format_ms(self, ms: int) -> str:
        return format_utils.format_ms(ms)

    def _format_ms_brief(self, ms: float | int | None) -> str:
        return format_utils.format_ms_brief(ms)

    def _format_hold_requirement(self, hold_ms: int) -> str:
        return format_utils.format_hold_requirement(hold_ms)

    def _expected_label_for_step(self, step: Any) -> str:
        return step_introspection.expected_label_for_step(step)

    def _start_keys_for_step(self, step: Any) -> set[str]:
        return step_introspection.start_keys_for_step(step)

    def _step_accepts_input(self, step: Any, input_name: str) -> bool:
        return step_introspection.step_accepts_input(step, input_name)

    def _find_next_step_index_for_input(self, input_name: str, *, start_index: int) -> int | None:
        """Look ahead for the next non-wait step that matches input_name."""
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return None
        try:
            for j in range(max(0, int(start_index)), len(self.runtime_steps)):
                s = self.runtime_steps[j]
                if isinstance(s, WaitState):
                    continue
                if self._step_accepts_input(s, input_name):
                    return j
        except Exception:
            return None
        return None

    def _find_prev_step_index_for_input(self, input_name: str, *, end_index: int) -> int | None:
        """Look backward for the most recent non-wait step that matches input_name."""
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return None
        try:
            end = max(0, int(end_index))
        except Exception:
            end = 0
        try:
            for j in range(min(end, len(self.runtime_steps)) - 1, -1, -1):
                s = self.runtime_steps[j]
                if isinstance(s, WaitState):
                    continue
                if self._step_accepts_input(s, input_name):
                    return j
        except Exception:
            return None
        return None

    def _mark_step(self, step_index: int, mark: str):
        """
        Set a per-attempt mark for a step (for UI coloring).
        Later marks overwrite earlier ones (e.g. wait can go from "early" -> "ok").
        """
        try:
            idx = int(step_index)
        except Exception:
            return
        if idx < 0:
            return
        m = str(mark or "").strip().lower()
        if not m:
            return
        self.step_marks[idx] = m

    def _reset_attempt_marks(self):
        self.step_marks = {}
        self.wait_early_inputs = {}

    def _next_non_wait_step_index(self, *, start_index: int) -> int | None:
        """Return the next step index >= start_index that is not a wait step."""
        try:
            for j in range(max(0, int(start_index)), len(self.runtime_steps)):
                if not isinstance(self.runtime_steps[j], WaitState):
                    return j
        except Exception:
            return None
        return None

    def _maybe_complete_combo_if_trailing_wait(self, *, now: float, total_ms: float) -> bool:
        """
        If the next expected step is a wait gate *and there are no further non-wait steps after it*,
        then the wait is effectively a no-op. In that case, complete the combo immediately.

        This avoids "hanging" on a trailing wait like: e, q, wait(r, 3.65s)
        (there's nothing left to time-gate).
        """
        try:
            step = self._active_runtime_step()
            if not isinstance(step, WaitState):
                return False
            # If there is any real action after this wait, it is not trailing.
            if self._next_non_wait_step_index(start_index=int(self.current_index) + 1) is not None:
                return False
        except Exception:
            return False

        self._on_combo_completed(total_ms)
        return True

    # -------------------------
    # Persistence
    # -------------------------

    def _get_data_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def load_combos(self):
        load_engine_state(self)

    def save_combos(self):
        save_engine_state(self)

    # -------------------------
    # Stats helpers
    # -------------------------

    def _ensure_combo_stats(self, name: str) -> None:
        stats_recording.ensure_combo_stats(self, name)

    def _combo_avg_ms(self, name: str):
        return stats_recording.combo_avg_ms(self, name)

    def _format_percent(self, success: int, fail: int) -> str:
        return stats_recording.format_percent(success, fail)

    def stats_text(self):
        return ui.stats_text(self)

    def failures_by_step(self) -> dict[str, int]:
        return ui.failures_by_step(self)

    def failures_by_reason(self) -> dict[str, int]:
        return ui.failures_by_reason(self)

    def min_time_text(self) -> str:
        return ui.min_time_text(self)

    def _parse_expected_time_ms(self, raw: str | None) -> int | None:
        return format_utils.parse_expected_time_ms(raw)

    def _count_combo_actions(self, steps: list[Any] | None) -> tuple[int, int, int]:
        """Returns (press_count, hold_count, total_actions). Delegates to combo_analytics."""
        import combo_analytics
        if steps is None or steps is self.runtime_steps:
            return combo_analytics.count_combo_actions(self)
        press, hold = 0, 0
        for s in steps:
            pp, hh = combo_analytics._count_step_actions(s)
            press += pp
            hold += hh
        return press, hold, press + hold

    def practical_apm(self) -> float | None:
        return combo_analytics.practical_apm(self)

    def theoretical_max_apm(self) -> float | None:
        return ui.theoretical_max_apm(self)

    def apm_text(self) -> str:
        return ui.apm_text(self)

    def apm_max_text(self) -> str:
        return ui.apm_max_text(self)

    def difficulty_score_10(self) -> float | None:
        return combo_analytics.difficulty_score_10(self)

    def difficulty_text(self) -> str:
        return ui.difficulty_text(self)

    def user_difficulty_value(self) -> float | None:
        return ui.user_difficulty_value(self)

    def user_difficulty_text(self) -> str:
        return ui.user_difficulty_text(self)

    # -------------------------
    # UI state snapshots
    # -------------------------

    def get_editor_payload(self, target_game_override: str | None = None) -> dict[str, Any]:
        return ui.get_editor_payload(self, target_game_override=target_game_override)

    def get_status(self) -> Status:
        return ui.get_status(self)

    def timeline_steps(self) -> list[dict[str, Any]]:
        return ui.timeline_steps(self, time.perf_counter())

    def init_payload(self) -> dict[str, Any]:
        return ui.init_payload(self)

    # -------------------------
    # Combo ender logic
    # -------------------------

    def _is_combo_ender(self, input_name: str) -> bool:
        return input_name in self.combo_enders

    def _ender_cooldown_ms(self, input_name: str) -> int:
        """Cooldown duration in ms for this ender key (0 = no cooldown)."""
        try:
            return int(self.combo_enders.get(input_name, 0))
        except Exception:
            return 0

    def _start_ender_cooldown(self, input_name: str, now: float) -> None:
        """When the user correctly presses an ender key, start that key's cooldown."""
        if not (input_name or "").strip() or not self._is_combo_ender(input_name):
            return
        ms = self._ender_cooldown_ms(input_name)
        if ms <= 0:
            return
        self._ender_cooldown_until[(input_name or "").strip().lower()] = now + (ms / 1000.0)

    def _ender_on_cooldown(self, input_name: str) -> bool:
        """True if this ender key was recently used correctly and is still on cooldown (ignore press)."""
        if not (input_name or "").strip() or not self._is_combo_ender(input_name):
            return False
        now = time.perf_counter()
        until = self._ender_cooldown_until.get((input_name or "").strip().lower(), 0.0)
        return now <= until

    # -------------------------
    # Commands from UI
    # -------------------------

    def apply_enders_from_text(self, raw: str) -> tuple[bool, str | None]:
        with self._lock:
            return combo_commands.apply_enders_from_text(self, raw)

    def save_or_update_combo(
        self,
        *,
        name: str,
        inputs: str,
        enders: str,
        expected_time: str | None = None,
        user_difficulty: str | None = None,
        step_display_mode: str | None = None,
        key_images: Any | None = None,
        demo_video: str | None = None,
        target_game: str | None = None,
        ww_team_id: str | None = None,
    ) -> tuple[bool, str | None]:
        with self._lock:
            return combo_commands.save_or_update_combo(
                self,
                name=name,
                inputs=inputs,
                enders=enders,
                expected_time=expected_time,
                user_difficulty=user_difficulty,
                step_display_mode=step_display_mode,
                key_images=key_images,
                demo_video=demo_video,
                target_game=target_game,
                ww_team_id=ww_team_id,
            )

    def delete_combo(self, name: str) -> tuple[bool, str | None]:
        with self._lock:
            return combo_commands.delete_combo(self, name)

    # -------------------------
    # Wuthering Waves teams (presets) - delegate to Game_Wuthering_Waves
    # -------------------------

    def set_active_ww_team(self, team_id: str):
        with self._lock:
            set_active_ww_team(self, team_id)

    def save_or_update_ww_team(
        self,
        *,
        team_id: str | None,
        team_name: str | None,
        dash_image: str | None,
        swap_images: Any | None,
        lmb_images: Any | None,
        ability_images: Any | None,
    ) -> tuple[bool, str | None]:
        with self._lock:
            return save_or_update_ww_team(
                self,
                team_id=team_id,
                team_name=team_name,
                dash_image=dash_image,
                swap_images=swap_images,
                lmb_images=lmb_images,
                ability_images=ability_images,
            )

    def delete_ww_team(self, team_id: str) -> tuple[bool, str | None]:
        with self._lock:
            return delete_ww_team(self, team_id)

    def select_team_stateless(self, team_id: str, target_game: str):
        with self._lock:
            select_team_stateless(self, team_id, target_game)

    def update_target_game_stateless(self, target_game: str):
        with self._lock:
            update_target_game_stateless(self, target_game)

    def new_combo(self):
        with self._lock:
            combo_commands.new_combo(self)

    def send_transcription_result(self, transcript: str) -> None:
        """Push transcript to client: inputs field + parsed steps / timeline (used for live updates and when recording stops)."""
        with self._lock:
            self._send({"type": "transcription_result", "inputs": transcript or ""})
            tokens = [k.strip().lower() for k in self.split_inputs(transcript or "") if k.strip()]
            ast_list = expanded_ast_from_tokens(tokens)
            self.active_combo_tokens = tokens
            self.runtime_steps = [build_runtime_state(node) for node in ast_list]
            self.reset_tracking()
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def clear_history_and_stats(self):
        with self._lock:
            combo_commands.clear_history_and_stats(self)

    def set_active_combo(self, name: str, *, emit: bool = True):
        with self._lock:
            name = (name or "").strip()
            if name not in self.combos:
                self.active_combo_name = None
                self.active_combo_tokens = []
                self.runtime_steps = []
                self.reset_tracking()
                if emit:
                    self._send({"type": "init", **self.init_payload()})
                return

            self.active_combo_name = name
            self.active_combo_tokens = self.combos[name]
            # Build runtime state objects from AST (expanded: wait(r,t) -> press + wait)
            ast_list = expanded_ast_from_tokens(self.active_combo_tokens)
            self.runtime_steps = [build_runtime_state(node) for node in ast_list]
            self._ensure_combo_stats(name)
            
            # Restore saved WW active team when selecting a combo
            if self.ww.get_target_game(name) == "wuthering_waves":
                saved_team = self.ww.combo_ww_team.get(name)
                self.ww.ww_active_team_id = saved_team
            else:
                self.ww.ww_active_team_id = None

            self.reset_tracking()
            self.save_combos()

            if emit:
                st = self.get_status()
                self._send({"type": "combo_data", **self.get_editor_payload()})
                self._send({"type": "min_time", "text": self.min_time_text()})
                self._send(
                    {
                        "type": "difficulty_update",
                        "text": self.difficulty_text(),
                        "value": self.difficulty_score_10(),
                    }
                )
                self._send(
                    {
                        "type": "user_difficulty_update",
                        "text": self.user_difficulty_text(),
                        "value": self.user_difficulty_value(),
                    }
                )
                self._send({"type": "apm_update", "text": self.apm_text()})
                self._send({"type": "apm_max_update", "text": self.apm_max_text()})
                self._emit_stats_and_fail()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                self._send({"type": "status", "text": st.text, "color": st.color})
                self._send({"type": "combo_list", "combos": sorted(self.combos.keys()), "active": self.active_combo_name})

    # -------------------------
    # Core state machine
    # -------------------------

    def reset_tracking(self):
        self.current_index = 0
        self.start_time = 0.0
        self.last_input_time = 0.0
        self.attempt_counter = 0
        self.last_success_input = None
        self._ender_cooldown_until.clear()
        self.ww.ww_active_character = None
        self._ui_last_success_combo = None
        self._ui_last_success_steps_len = 0
        self._reset_attempt_marks()
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()

    def _active_step(self):
        """Current step (StepState). Alias for _active_runtime_step for backward compatibility."""
        return self._active_runtime_step()

    def _active_runtime_step(self):
        """Current step as StepState (for new match dispatch)."""
        if 0 <= self.current_index < len(self.runtime_steps):
            return self.runtime_steps[self.current_index]
        return None

    def _insert_attempt_separator(self):
        self.attempt_counter += 1
        name = self.active_combo_name or "Combo"
        # New attempt → clear any per-step failure coloring from the previous attempt.
        self._reset_attempt_marks()
        # New attempt → stop showing the previous "success snapshot" (fully green timeline).
        self._ui_last_success_combo = None
        self._ui_last_success_steps_len = 0
        self._send({"type": "attempt_start", "name": name, "attempt": self.attempt_counter})

    def record_hit(self, label: str, split_ms: float | str, total_ms: float | str):
        # Keep formatting consistent with HTML table
        if isinstance(split_ms, (float, int)):
            split = f"{float(split_ms):.1f}"
        else:
            split = str(split_ms)
        if isinstance(total_ms, (float, int)):
            total = f"{float(total_ms):.1f}"
        else:
            total = str(total_ms)
        self._send({"type": "hit", "input": label, "split_ms": split, "total_ms": total})

    def _reset_hold_state(self):
        # If we were showing a hold indicator in the UI, clear it.
        if self.hold_in_progress:
            self._send({"type": "hold_end"})
        self.hold_in_progress = False
        self.hold_expected_input = None
        self.hold_started_at = 0.0
        self.hold_required_ms = None
        # _hold_max_held_ms is NOT cleared here so release-too-early keeps the max for the drop message

    def _reset_wait_state(self):
        # If we were showing a wait indicator in the UI, clear it.
        if self.wait_in_progress:
            self._send({"type": "wait_end"})
        self.wait_in_progress = False
        self.wait_started_at = 0.0
        self.wait_until = 0.0
        self.wait_required_ms = None

    def _reset_group_state(self):
        """Clear per-attempt progress: reset all runtime steps and update shortcuts."""
        try:
            if self.hold or self.wait:
                if self.wait:
                    self._send({"type": "wait_end"})
                if self.hold:
                    self._send({"type": "hold_end"})
            for step in self.runtime_steps:
                step.reset()
            self._update_shortcuts()
        except Exception:
            pass

    def _update_shortcuts(self) -> None:
        """Set self.hold and self.wait from current runtime step."""
        step = self._active_runtime_step()
        self.hold = step if isinstance(step, HoldState) else None
        self.wait = step if isinstance(step, WaitState) else None

    def _reset_to_start(self) -> None:
        """Reset index and step state after a combo completion (ready for next attempt)."""
        self.current_index = 0
        # Stop the ~50Hz tick loop from continuously emitting timeline frames while idle.
        # (A new attempt will re-initialize timing on the first accepted input.)
        self.start_time = 0.0
        self.last_input_time = 0.0
        try:
            self.currently_pressed.clear()
        except Exception:
            pass
        self._ender_cooldown_until.clear()
        self.ww.ww_active_character = None
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()
        self.ui_adapter.reset()

    def _on_combo_completed(self, total_ms: float) -> None:
        """Record success, reset to start, and notify UI. Single place for combo completion."""
        self.record_combo_success(total_ms)
        # Snapshot timeline while steps are still in completed state; after _reset_to_start()
        # all step state is cleared, so the UI would lose the "completed" styling.
        steps_snapshot = self.timeline_steps()
        self._reset_to_start()
        total_s = total_ms / 1000.0
        self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete! Completed in {total_s:.2f} seconds", "color": "success"})
        self._send({"type": "timeline_update", "steps": steps_snapshot})

    def cancel_attempt(self) -> None:
        """Cancel current attempt and reset to beginning (e.g. on Esc). No fail recorded."""
        with self._lock:
            if not self.runtime_steps:
                return
            self._reset_to_start()
            st = self.get_status()
            self._send({"type": "status", "text": st.text, "color": st.color})
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def _reset_after_fail(self) -> None:
        """Full reset after a combo failure: clear index, time, step state, and notify UI."""
        # Snapshot timeline while the failed step is still marked (step_marks), so the UI
        # can show the red outline on the failed step before we clear everything.
        steps_snapshot = self.timeline_steps()
        self.current_index = 0
        self.start_time = 0.0
        self.last_input_time = 0.0
        self._ender_cooldown_until.clear()
        self.ww.ww_active_character = None
        self._hold_max_held_ms = 0.0
        self.ui_adapter.reset()
        for s in self.runtime_steps:
            s.reset()
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()
        self._reset_attempt_marks()
        self._update_shortcuts()
        self._send({"type": "timeline_update", "steps": steps_snapshot})

    def _active_nested_mandatory_wait(self) -> WaitState | None:
        """
        Return an in-progress mandatory WaitState if the current step contains one (including nested).
        """
        try:
            step = self._active_runtime_step()
            ws = ui.find_active_in_progress_wait(step)
            if isinstance(ws, WaitState) and ws.in_progress and not ws.completed and (ws.mode == "mandatory"):
                return ws
        except Exception:
            return None
        return None

    def _get_expected_hold_key(self) -> str | None:
        """
        If the current step is (or starts with) a hold, return that hold's expected key.
        Used by generalized hold buffer: after any advance, we can start a hold immediately
        if that key is already in currently_pressed.
        """
        try:
            step = self._active_runtime_step()
            if step is None:
                return None
            if isinstance(step, HoldState):
                return (step.expected or "").strip().lower() or None
            if isinstance(step, SequenceState) and getattr(step, "steps", None):
                idx = getattr(step, "current_index", 0) or 0
                if 0 <= idx < len(step.steps):
                    inner = step.steps[idx]
                    if isinstance(inner, HoldState):
                        return (inner.expected or "").strip().lower() or None
                if step.steps and isinstance(step.steps[0], HoldState):
                    return (step.steps[0].expected or "").strip().lower() or None
            if isinstance(step, GroupState):
                for item in getattr(step, "items", []) or []:
                    if getattr(item, "completed_count", 0) >= getattr(item, "required_count", 1):
                        continue
                    if item.kind == "hold" and isinstance(item.state, HoldState):
                        key = (item.state.expected or "").strip().lower()
                        if key and key in self.currently_pressed:
                            return key
        except Exception:
            pass
        return None

    def _apply_buffered_hold_if_possible(self, now: float) -> None:
        """
        Generalized hold buffer: after any advance, if the next step is a hold and that key
        is already held (in currently_pressed), start the hold immediately.
        Works for top-level holds and for holds that are the first/current action in a sequence.
        """
        try:
            expected = self._get_expected_hold_key()
            if not expected or expected not in self.currently_pressed:
                return
            self._process_press_unlocked(expected)
        except Exception:
            return

    def _ensure_attempt_started(self, now: float) -> None:
        """On first accepted input of an attempt, start timing and send Recording status."""
        if not self.last_input_time and self.current_index == 0:
            self._insert_attempt_separator()
            self.start_time = now
            self.last_input_time = now
            self._send({"type": "status", "text": "Recording...", "color": "recording"})

    def _advance_step(self, now: float) -> None:
        """Advance to next step; update shortcuts, maybe start wait, send timeline."""
        self.current_index += 1
        self._update_shortcuts()
        self._maybe_start_wait_step()
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
        self.ui_adapter.set_active_step_signature(self)

    def _mark_current_and_skip(self, now: float, mark: str | None = None) -> None:
        """No-fail mode: mark current step as failed and advance without ending the combo."""
        idx = int(self.current_index)
        if idx not in self.step_marks:
            self._mark_step(idx, mark if mark else "wrong")
        step = self._active_runtime_step()
        if step is None:
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return
        if step.skip_and_advance(now):
            self._advance_step(now)
            self._sync_wait_animation_ui(now)
            self._send({"type": "status", "text": "Missed - continuing", "color": "fail"})
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            if self.current_index >= len(self.runtime_steps):
                self._send({"type": "status", "text": "Practice run finished", "color": "fail"})
                self._reset_to_start()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            else:
                self._apply_buffered_hold_if_possible(now)
        else:
            self._send({"type": "status", "text": "Missed - continuing", "color": "fail"})
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def set_no_fail_mode(self, enabled: bool) -> None:
        """Enable or disable no-fail (practice) mode."""
        self.no_fail_mode = bool(enabled)

    def _send_combo_dropped(self, full_status_text: str, now: float) -> None:
        """Send one message for both status display and attempt log (single source of truth).
        Uses same row shape as 'hit' (input, split_ms, total_ms) so the frontend can treat
        both as attempt-log rows; combo_dropped also updates the status line.
        Timings are computed from engine state before reset (same format as success rows)."""
        total_ms = (now - self.start_time) * 1000.0 if self.start_time else 0.0
        split_ms = (now - self.last_input_time) * 1000.0 if self.last_input_time else 0.0
        split_s = f"{float(split_ms):.1f}" if isinstance(split_ms, (float, int)) else str(split_ms)
        total_s = f"{float(total_ms):.1f}" if isinstance(total_ms, (float, int)) else str(total_ms)
        self._send({
            "type": "combo_dropped",
            "input": full_status_text,
            "split_ms": split_s,
            "total_ms": total_s,
            "color": "fail",
            "fail": True,
        })

    def _fail_combo(self, reason: str, now: float, *, expected_label: str | None = None, actual: str | None = None, waited_ms: float | None = None) -> None:
        """Record failure, reset all steps, send status and timeline."""
        # Mark the failed step so the timeline snapshot (sent in _reset_after_fail) shows red outline.
        if int(self.current_index) not in self.step_marks:
            self._mark_step(int(self.current_index), "wrong")
        msg_reason = reason or "fail"
        if "hold" in msg_reason.lower():
            if self.hold_in_progress:
                held_ms = (now - self.hold_started_at) * 1000
                self._hold_max_held_ms = max(self._hold_max_held_ms, held_ms)
            if self._hold_max_held_ms > 0:
                msg_reason += f" but held for {int(self._hold_max_held_ms)}ms"
        if actual is not None:
            msg_reason += f" but got {actual}"
            if waited_ms is not None:
                msg_reason += f" after {int(waited_ms)}ms"
            elif getattr(self, "wait_started_at", 0) and self.wait_started_at > 0:
                msg_reason += f" after {int((now - self.wait_started_at) * 1000)}ms"
        full_text = "Combo Dropped (" + msg_reason + ")"
        self._send_combo_dropped(full_text, now)
        elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
        self.record_combo_fail(
            actual=actual or reason,
            expected_step_index=int(self.current_index),
            expected_label=expected_label or self._expected_label_for_step(self._active_step()),
            reason=reason,
            elapsed_ms=elapsed_ms,
        )
        self._reset_after_fail()

    def _only_ender_can_drop(self, input_name: str) -> bool:
        return self.ww.can_ender_drop_combo(self, input_name)

    def _is_soft_ender(self, input_name: str) -> bool:
        """True if this key is a soft ender (~key:2s); does not drop combo when pressed during hold."""
        return (input_name or "").strip().lower() in getattr(self, "combo_enders_soft", set())

    def _ender_can_drop_now(self, input_name: str) -> bool:
        """True if this key can drop the combo in the current context. Soft enders do not drop during hold."""
        if not self._only_ender_can_drop(input_name):
            return False
        if self.hold_in_progress and self._is_soft_ender(input_name):
            return False
        return True

    def _check_ender_fail(self, input_name: str, now: float) -> None:
        """If key is a combo ender (and not on cooldown), drop the combo."""
        if not self._ender_can_drop_now(input_name):
            return
        self._mark_step(int(self.current_index), "missed")
        expected = str(self._expected_label_for_step(self._active_runtime_step()) or "").strip().lower()
        actual = str(input_name or "").strip().lower()
        if self.no_fail_mode:
            self._mark_current_and_skip(now, mark="missed")
            return
        if self.wait_in_progress and self.wait_started_at:
            waited_s = (now - self.wait_started_at)
            reason = f"only waited for {waited_s:.2f}s, combo ended by {actual}"
        else:
            reason = f"combo ended by {actual}"
        self._fail_combo(reason, now, actual=actual, expected_label=expected)

    def _start_hold(self, input_name: str, required_ms: int, now: float):
        self.hold_in_progress = True
        self.hold_expected_input = input_name
        self.hold_started_at = now
        self.hold_required_ms = required_ms
        # Keep _hold_max_held_ms across retries (multiple short releases) so drop message shows max
        self._send({"type": "hold_begin", "input": str(input_name or ""), "required_ms": int(required_ms)})
        st = self.get_status()
        self._send({"type": "status", "text": st.text, "color": st.color})
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def _start_wait(self, required_ms: int):
        self.wait_in_progress = True
        self.wait_started_at = float(self.last_input_time or time.perf_counter())
        self.wait_required_ms = required_ms
        self.wait_until = self.wait_started_at + (required_ms / 1000.0)
        # Tell the UI to animate a visible wait progress bar (similar to holds).
        # Mode may be soft|hard|mandatory (mandatory = animation lock; inputs ignored).
        try:
            step = self._active_runtime_step()
            mode = "soft"
            wait_for = ""
            if isinstance(step, WaitState):
                mode = str(step.mode or "soft").strip().lower() or "soft"
                wait_for = str(step.wait_for or "")
            self._send({"type": "wait_begin", "required_ms": int(required_ms), "mode": mode, "wait_for": wait_for})
        except Exception:
            self._send({"type": "wait_begin", "required_ms": int(required_ms), "mode": "soft", "wait_for": ""})
        st = self.get_status()
        self._send({"type": "status", "text": st.text, "color": st.color})
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def _complete_wait(self, now: float, *, fail: bool, reason: str | None = None):
        required_ms = int(self.wait_required_ms or 0)
        waited_ms = max(0.0, (now - self.wait_started_at) * 1000)
        req_s = self._format_hold_requirement(required_ms) if required_ms else "?"
        # For display, include mode when relevant
        mode = "soft"
        step = self._active_runtime_step()
        try:
            if isinstance(step, WaitState):
                mode = str(step.mode or "soft").strip().lower() or "soft"
        except Exception:
            mode = "soft"
        prefix = "wait-hard" if mode == "hard" else "wait"
        if mode == "mandatory":
            prefix = "anim-wait"
        label = f"{prefix} (≥ {req_s}, {waited_ms:.0f}ms)"
        total_ms = (now - self.start_time) * 1000 if self.start_time else 0.0

        if fail:
            if reason:
                label += f" [{reason}]"
            self._send_combo_dropped("Combo Dropped (Too Early)", now)
            elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
            self.record_combo_fail(
                actual=str(reason or ""),
                expected_step_index=int(self.current_index),
                expected_label=self._expected_label_for_step(self._active_runtime_step()),
                reason="too early",
                elapsed_ms=elapsed_ms,
            )
            self._reset_after_fail()
            return False

        split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0.0
        self.record_hit(label, split_ms, total_ms)
        self.last_input_time = now
        self.current_index += 1
        self._reset_wait_state()
        self._update_shortcuts()
        self.ui_adapter.set_active_step_signature(self)
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
        # If a mandatory wait (possibly nested) just ended and the user was already holding
        # the next hold key, start the hold immediately (game-like input buffer).
        self._apply_buffered_hold_if_possible(now)

        # If a wait step was (accidentally) the last step, don't get stuck past the end.
        if self.current_index >= len(self.runtime_steps):
            self._on_combo_completed(total_ms)
        return True

    def _maybe_start_wait_step(self):
        step = self._active_runtime_step()
        if not step or not isinstance(step, WaitState):
            return
        if not self.wait_in_progress:
            step.start(time.perf_counter())
            self._start_wait(int(step.required_ms))

    def _sync_wait_animation_ui(self, now: float) -> None:
        """UI hook: nested wait begin/end (delegated to UI adapter)."""
        try:
            self.ui_adapter.sync_wait_animation(self, emit=self._send)
        except Exception:
            return

    def _complete_hold(self, now: float, *, auto: bool):
        step = self._active_runtime_step()
        if not isinstance(step, HoldState):
            return False

        target_input = str(step.expected or "")
        target_hold_ms = int(step.required_ms or 0)

        held_ms = (now - self.hold_started_at) * 1000
        self._hold_max_held_ms = max(self._hold_max_held_ms, held_ms)
        ok = held_ms >= float(target_hold_ms)

        req_s = self._format_hold_requirement(target_hold_ms)
        split_ms = (now - self.last_input_time) * 1000 if self.current_index != 0 else 0.0
        total_ms = (now - self.start_time) * 1000 if self.start_time else 0.0

        label = f"{target_input} (hold ≥ {req_s}, {held_ms:.0f}ms)"
        if auto:
            label += " [auto]"

        if ok:
            self.record_hit(label, split_ms, total_ms)
            self.last_input_time = now
            self.last_success_input = target_input
            self._start_ender_cooldown(target_input, now)
            self.ww.on_accepted_key(self, target_input)
            step.complete()

            # If this hold was gated by a wait right before it, mark that wait green (timing satisfied).
            if self.current_index > 0:
                prev = self.runtime_steps[self.current_index - 1]
                if isinstance(prev, WaitState):
                    self._mark_step(self.current_index - 1, "ok")

            # Advance using the standard path so shortcuts update correctly.
            self._hold_max_held_ms = 0.0
            self._advance_step(now)
            if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                return True
            if self.current_index >= len(self.runtime_steps):
                self._on_combo_completed(total_ms)
        else:
            drop_text = "Combo Dropped (Hold Incomplete)"
            if self._hold_max_held_ms > 0:
                drop_text += f" but held for {int(self._hold_max_held_ms)}ms"
            self._send_combo_dropped(drop_text, now)
            elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
            self.record_combo_fail(
                actual=f"released @ {held_ms:.0f}ms",
                expected_step_index=int(self.current_index),
                expected_label=self._expected_label_for_step(step),
                reason="hold incomplete",
                elapsed_ms=elapsed_ms,
            )
            self._reset_after_fail()
            return False

        self._reset_hold_state()
        return ok

    def _record_fail_detail(
        self,
        *,
        step_index: int,
        expected: str,
        actual: str,
        reason: str,
        elapsed_ms: float | None,
    ) -> None:
        stats_recording.record_fail_detail(
            self,
            step_index=step_index,
            expected=expected,
            actual=actual,
            reason=reason,
            elapsed_ms=elapsed_ms,
        )

    def record_combo_success(self, completion_ms: float | int | None = None) -> None:
        stats_recording.record_combo_success(self, completion_ms)

    def record_combo_fail(
        self,
        *,
        actual: str | None = None,
        expected_step_index: int | None = None,
        expected_label: str | None = None,
        reason: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        stats_recording.record_combo_fail(
            self,
            actual=actual,
            expected_step_index=expected_step_index,
            expected_label=expected_label,
            reason=reason,
            elapsed_ms=elapsed_ms,
        )

    # -------------------------
    # Input processing (called from pynput)
    # -------------------------

    def process_press(self, input_name: str):
        # Thread-safe wrapper
        with self._lock:
            return self._process_press_unlocked(input_name)

    def _process_press_unlocked(self, input_name: str):
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return

        # Only a genuine key-down (new press) can end the combo; key repeat (already held) must not.
        press_is_repeat = input_name in self.currently_pressed
        self.currently_pressed.add(input_name)
        if not self.runtime_steps:
            return

        now = time.perf_counter()

        # During a nested mandatory wait (animation lock), ignore the key for progression.
        # currently_pressed already has it; when the wait ends and we advance,
        # _apply_buffered_hold_if_possible will start the hold if the next step is that key.
        if self._active_nested_mandatory_wait() is not None:
            return

        # 1) Active wait
        if self.wait and self.wait.in_progress:
            result = self.wait.process_press(input_name, now)
            if isinstance(result, CompleteResult):
                self._complete_wait(now, fail=False)
                return self._process_press_unlocked(input_name)
            if isinstance(result, FailResult):
                if self.no_fail_mode:
                    self._mark_step(int(self.current_index), "early")
                    self._reset_wait_state()
                    step = self._active_runtime_step()
                    if step is not None and step.skip_and_advance(now):
                        self._advance_step(now)
                        self._sync_wait_animation_ui(now)
                    self._send({"type": "status", "text": "Missed - continuing", "color": "fail"})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    if self.current_index >= len(self.runtime_steps):
                        self._send({"type": "status", "text": "Practice run finished", "color": "fail"})
                        self._reset_to_start()
                        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    else:
                        self._apply_buffered_hold_if_possible(now)
                else:
                    # Only enders can end combos; early press with non-ender is ignored.
                    # Key repeat (already held) does not end the combo.
                    if not press_is_repeat and self._ender_can_drop_now(input_name):
                        self._check_ender_fail(input_name, now)
                return
            # Ender during wait: use same gate as everywhere else. Key repeat does not end combo.
            if not press_is_repeat and self.wait.mode in ("soft", "hard") and self._ender_can_drop_now(input_name):
                if self.no_fail_mode:
                    self._mark_step(int(self.current_index), "wrong")
                    self._reset_wait_state()
                    step = self._active_runtime_step()
                    if step is not None and step.skip_and_advance(now):
                        self._advance_step(now)
                        self._sync_wait_animation_ui(now)
                    self._send({"type": "status", "text": "Missed - continuing", "color": "fail"})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    if self.current_index >= len(self.runtime_steps):
                        self._send({"type": "status", "text": "Practice run finished", "color": "fail"})
                        self._reset_to_start()
                        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    else:
                        self._apply_buffered_hold_if_possible(now)
                else:
                    self._check_ender_fail(input_name, now)
            return

        # 2) Active hold
        if self.hold and self.hold.in_progress:
            if input_name == self.hold.expected:
                return
            if self.hold.check_complete(now):
                self._complete_hold(now, auto=True)
                return self._process_press_unlocked(input_name)
            # Only enders can end combos (same gate as everywhere). Key repeat does not end combo.
            if not press_is_repeat and self._ender_can_drop_now(input_name):
                self._check_ender_fail(input_name, now)
                return

            # Forgiving hold: only drop if they pressed the next step's key AND it's an ender (new press only).
            next_step = self.runtime_steps[int(self.current_index) + 1] if (int(self.current_index) + 1) < len(self.runtime_steps) else None
            next_keys = self._start_keys_for_step(next_step)
            if not press_is_repeat and next_keys and (input_name in next_keys) and self._ender_can_drop_now(input_name):
                if self.no_fail_mode:
                    self._mark_current_and_skip(now)
                    self._sync_wait_animation_ui(now)
                    return
                self._fail_combo("hold incomplete", now, actual=f"pressed {input_name}")
            # Otherwise ignore (random keys or next key that isn't an ender).
            return

        # 3) Dispatch to current step (polymorphic: GroupState, SequenceState, PressState, HoldState, WaitState all implement process_press)
        step = self._active_runtime_step()
        if step is None:
            return

        result = step.process_press(input_name, now)

        if isinstance(result, AcceptResult):
            self._ensure_attempt_started(now)
            if result.record_hit:
                split_ms = (now - self.last_input_time) * 1000.0 if self.current_index != 0 else 0.0
                total_ms = (now - self.start_time) * 1000.0 if self.start_time else 0.0
                self.record_hit(input_name, split_ms, total_ms)
            self.last_input_time = now
            self.last_success_input = input_name
            self._start_ender_cooldown(input_name, now)
            self.ww.on_accepted_key(self, input_name)
            # If this press started a nested wait (e.g. press_wait inside a group/sequence),
            # kick off the client-side wait animation immediately.
            self._sync_wait_animation_ui(now)
            if result.advance:
                self._advance_step(now)
                self._sync_wait_animation_ui(now)
                # If a mandatory wait ended and we advanced into a hold step that is already held,
                # apply the buffered hold immediately.
                self._apply_buffered_hold_if_possible(now)
                if self.current_index >= len(self.runtime_steps):
                    self._on_combo_completed((now - self.start_time) * 1000.0 if self.start_time else 0.0)
                    return
                if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                    return
                # Important: do NOT re-process the same input against the next step.
                # The input was already consumed by the current step; re-processing can
                # incorrectly treat it as an "ender during wait" (e.g., `e, wait:0.4`).
                return
            if isinstance(step, HoldState):
                self._start_hold(step.expected, step.required_ms, now)
            elif isinstance(step, GroupState):
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return
        if isinstance(result, FailResult):
            if self.no_fail_mode:
                mark = "early" if "early" in (result.reason or "").lower() else "wrong"
                self._mark_current_and_skip(now, mark=mark)
                self._sync_wait_animation_ui(now)
                return
            # Only enders can end combos; wrong/early key that isn't an ender is ignored. Key repeat does not end combo.
            if not press_is_repeat and self._ender_can_drop_now(input_name):
                self._check_ender_fail(input_name, now)
            return
        if isinstance(result, CompleteResult):
            self._advance_step(now)
            self._sync_wait_animation_ui(now)
            if self.current_index >= len(self.runtime_steps):
                self._on_combo_completed((now - self.start_time) * 1000.0 if self.start_time else 0.0)
                return
            if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                return
            return self._process_press_unlocked(input_name)
        # During group mandatory wait (animation lock), ignore all keys including enders.
        if isinstance(step, GroupState) and step.wait_active:
            return
        # Input didn't match current step (IgnoreResult). Only enders can end combos. Key repeat does not end combo.
        if isinstance(result, IgnoreResult):
            if not self.last_input_time and self.current_index == 0:
                return
            # Optional press step: pressing the next step's key skips the optional without dropping.
            if isinstance(step, PressState) and getattr(step, "optional", False):
                next_idx = int(self.current_index) + 1
                if next_idx < len(self.runtime_steps):
                    next_step = self.runtime_steps[next_idx]
                    next_keys = self._start_keys_for_step(next_step)
                    if next_keys and input_name in next_keys:
                        step.was_skipped = True
                        step.completed = True
                        self._advance_step(now)
                        self._sync_wait_animation_ui(now)
                        return self._process_press_unlocked(input_name)
            # Optional sequence step (e.g. -e, wait:0.80s): pressing next step's key skips the whole sequence.
            if isinstance(step, SequenceState) and getattr(step, "optional", False):
                next_idx = int(self.current_index) + 1
                if next_idx < len(self.runtime_steps):
                    next_step = self.runtime_steps[next_idx]
                    next_keys = self._start_keys_for_step(next_step)
                    if next_keys and input_name in next_keys:
                        step.was_skipped = True
                        self._advance_step(now)
                        self._sync_wait_animation_ui(now)
                        return self._process_press_unlocked(input_name)
            # Optional-key grace: optional key is valid in its own slot and through the entire next step.
            # If we're 1 or 2 steps past a skipped optional and user presses that optional's key, accept it (don't drop).
            for prev_idx in (self.current_index - 1, self.current_index - 2):
                if prev_idx < 0:
                    continue
                prev_step = self.runtime_steps[prev_idx]
                opt_key = step_introspection.optional_step_key(prev_step)
                if opt_key is None or not getattr(prev_step, "was_skipped", False) or input_name != opt_key:
                    continue
                prev_step.was_skipped = False
                return
            if not press_is_repeat and self._ender_can_drop_now(input_name):
                expected = self._expected_label_for_step(step) or "?"
                self._mark_step(int(self.current_index), "missed")
                if self.no_fail_mode:
                    self._mark_current_and_skip(now, mark="missed")
                    self._sync_wait_animation_ui(now)
                    return
                # If current step is a sequence with an active inner wait, pass how long they waited
                waited_ms_val: float | None = None
                if isinstance(step, SequenceState) and 0 <= step.current_index < len(step.steps):
                    inner = step.steps[step.current_index]
                    if isinstance(inner, WaitState) and inner.started_at > 0:
                        waited_ms_val = (now - inner.started_at) * 1000
                self._fail_combo(f"expected {expected}", now, actual=input_name, expected_label=expected, waited_ms=waited_ms_val)
            return
        return


    def process_release(self, input_name: str):
        # Thread-safe wrapper
        with self._lock:
            return self._process_release_unlocked(input_name)

    def _process_release_unlocked(self, input_name: str):
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return

        self.currently_pressed.discard(input_name)

        if not self.runtime_steps:
            return

        step = self._active_runtime_step()
        if step is None:
            return

        now = time.perf_counter()
        result = step.process_release(input_name, now)

        # Track max hold duration when release was too short (for drop message later)
        if isinstance(step, HoldState) and input_name == str(step.expected or "") and self.hold_in_progress:
            if isinstance(result, IgnoreResult):
                held_ms = (now - self.hold_started_at) * 1000
                self._hold_max_held_ms = max(self._hold_max_held_ms, held_ms)

        # Forgiving holds: releasing too early should end the UI "hold" indicator,
        # but should NOT drop the combo.
        if isinstance(step, HoldState) and input_name == str(step.expected or ""):
            try:
                if self.hold_in_progress and (not bool(getattr(step, "in_progress", False))) and (not bool(getattr(step, "completed", False))):
                    self._reset_hold_state()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            except Exception:
                pass

        if isinstance(result, AcceptResult):
            if result.advance:
                if isinstance(step, HoldState):
                    self._complete_hold(now, auto=False)
                else:
                    self._advance_step(now)
                    self._sync_wait_animation_ui(now)
                    if self.current_index >= len(self.runtime_steps):
                        total_ms = (now - self.start_time) * 1000.0 if self.start_time else 0.0
                        self._on_combo_completed(total_ms)
                    elif self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                        pass
                    else:
                        st = self.get_status()
                        self._send({"type": "status", "text": st.text, "color": st.color})
            else:
                # Group-internal hold completed; notify UI
                if isinstance(step, GroupState):
                    self._send({"type": "hold_end"})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return
        if isinstance(result, FailResult):
            if self.no_fail_mode:
                self._mark_current_and_skip(now)
                self._sync_wait_animation_ui(now)
                return
            # Only enders can end combos; releasing hold key too early drops only if that key is an ender.
            if self._ender_can_drop_now(input_name):
                self._fail_combo(result.reason, now, actual="released (hold too short)")
            self._sync_wait_animation_ui(now)
            return

    def tick(self):
        # Thread-safe wrapper
        with self._lock:
            return self._tick_unlocked()

    def _tick_unlocked(self):
        """
        Advance time-based steps (waits / group internal waits) without requiring another input event.
        This allows wait tiles to complete/turn green automatically when the timer elapses.

        Why this exists:
        - `process_press()` only runs when the player presses something.
        - But we want waits to "finish" in the UI even if the player pauses and does not press the next key.
        - `ui_server.py` runs a lightweight tick loop (~50Hz) that calls `engine.tick()`.

        This method intentionally does **not** start a combo; it only advances timers for an already-started attempt.
        """
        try:
            if not self.runtime_steps:
                return
            if not self.start_time or not self.last_input_time:
                return

            now = time.perf_counter()
            self._maybe_start_wait_step()
            # Keep nested wait animations in sync (without spamming timeline frames).
            self._sync_wait_animation_ui(now)

            step = self._active_runtime_step()
            if step is None:
                return

            # Hold duration satisfied without release (e.g. user never let go): complete and advance so next step (e.g. mandatory wait) runs automatically
            if isinstance(step, HoldState) and self.hold_in_progress and self.hold and self.hold.check_complete(now):
                self._complete_hold(now, auto=True)
                self._sync_wait_animation_ui(now)
                return

            result = step.tick(now)

            if isinstance(result, CompleteResult) or (
                isinstance(result, AcceptResult) and result.advance
            ):
                if isinstance(step, WaitState):
                    self._complete_wait(now, fail=False)
                    st = self.get_status()
                    self._send({"type": "status", "text": st.text, "color": st.color})
                else:
                    self._advance_step(now)
                    self._sync_wait_animation_ui(now)
                    if self.current_index >= len(self.runtime_steps):
                        total_ms = (now - self.start_time) * 1000.0 if self.start_time else 0.0
                        self._on_combo_completed(total_ms)
                    elif self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                        pass
                    else:
                        st = self.get_status()
                        self._send({"type": "status", "text": st.text, "color": st.color})
                # State advanced; signature is handled by _advance_step / _complete_wait / resets.
                return

            # No top-level advance, but nested state may have changed (e.g. wait inside a SequenceState
            # finished, enabling the next press). Emit a single timeline_update on that transition.
            self.ui_adapter.maybe_emit_timeline_update_on_change(self, emit=self._send)
            # Also, if a nested mandatory wait just ended and the next step is a hold that is already held,
            # start it immediately.
            self._apply_buffered_hold_if_possible(now)
        except Exception:
            return