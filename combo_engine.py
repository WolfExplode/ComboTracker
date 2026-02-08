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
import combo_engine_ui as ui
from combo_engine_ui import Status
from persistence import load_engine_state, save_engine_state

import combo_commands
import input_normalization
from parser import expanded_ast_from_tokens
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
    - Owns state machine (press/hold/wait + ender-grace)
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
        self._group_wait_begin_sent: bool = False

        # Combo enders: key -> grace_ms (0 means no grace; wrong press drops immediately)
        self.combo_enders: dict[str, int] = {}
        self.last_success_input: str | None = None

        # UI helper: after a successful completion we reset current_index back to 0 (ready for next attempt),
        # but we still want the timeline to stay fully "completed" (green) until the next attempt begins.
        self._ui_last_success_combo: str | None = None
        self._ui_last_success_steps_len: int = 0

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

    # -------------------------
    # Normalization helpers (delegate to input_normalization)
    # -------------------------

    def normalize_key(self, key) -> str:
        return input_normalization.normalize_key(key)

    def normalize_mouse(self, button) -> str:
        return input_normalization.normalize_mouse(button)

    def split_inputs(self, keys_str: str):
        return input_normalization.split_inputs(keys_str or "")

    def _parse_duration(self, raw: str):
        token = (raw or "").lower().strip()
        if not token:
            return None

        if token.endswith("ms"):
            token = token[:-2].strip()
            multiplier = 1
        elif token.endswith("s"):
            token = token[:-1].strip()
            multiplier = 1000
        else:
            multiplier = 1000 if "." in token else 1

        try:
            value = float(token)
        except ValueError:
            return None

        millis = value * multiplier
        if millis <= 0:
            return None
        return int(millis)

    def calc_min_combo_time_ms(self, steps: list[Any] | None) -> int:
        """Fastest possible combo time in ms. Delegates to combo_analytics."""
        import combo_analytics
        total = sum(combo_analytics._step_time_ms(s) for s in (steps or []))
        return max(0, int(total))

    def _format_ms(self, ms: int):
        ms = int(ms)
        if ms % 1000 == 0:
            return f"{ms//1000:d}s ({ms}ms)"
        return f"{ms/1000.0:.3g}s ({ms}ms)"

    def _format_ms_brief(self, ms: float | int | None):
        if ms is None:
            return "—"
        try:
            ms_i = int(round(float(ms)))
        except Exception:
            return "—"
        if ms_i < 1000:
            return f"{ms_i}ms"
        return f"{ms_i/1000.0:.3g}s"

    def _format_hold_requirement(self, hold_ms: int):
        if hold_ms is None:
            return ""
        if hold_ms % 1000 == 0:
            return f"{hold_ms // 1000:d}s"
        return f"{hold_ms / 1000.0:.3g}s"

    def _expected_label_for_step(self, step: Any) -> str:
        """Return display label for a StepState (used by UI and fail reporting)."""
        if step is None:
            return "—"
        if isinstance(step, PressState):
            return (step.expected or "").strip().lower() or "—"
        if isinstance(step, HoldState):
            h = step.required_ms
            inp = (step.expected or "").strip().lower()
            return f"hold({inp},≥{h}ms)" if inp else f"hold(≥{h}ms)"
        if isinstance(step, WaitState):
            w = step.required_ms
            if step.mode == "hard":
                return f"wait-hard(≥{w}ms)"
            if step.mode == "mandatory":
                k = (step.wait_for or "").strip().lower()
                return f"anim-wait({k},≥{w}ms)" if k else f"anim-wait(≥{w}ms)"
            return f"wait(≥{w}ms)"
        if isinstance(step, SequenceState):
            labels = []
            for s in step.steps:
                if isinstance(s, WaitState):
                    w = s.required_ms
                    if s.mode == "hard":
                        labels.append(f"wait-hard({w}ms)")
                    elif s.mode == "mandatory":
                        k = (s.wait_for or "").strip().lower()
                        labels.append(f"{k}+wait({w}ms)" if k else f"wait({w}ms)")
                    else:
                        labels.append(f"wait({w}ms)")
                elif isinstance(s, HoldState):
                    labels.append(f"hold({s.expected},{s.required_ms}ms)")
                else:
                    labels.append((getattr(s, "expected", "") or "").strip().lower() or "?")
            return f"seq({' → '.join(labels)})" if labels else "seq(—)"
        if isinstance(step, GroupState):
            opts = []
            for item in step.items:
                if item.kind == "press" and isinstance(item.state, PressState):
                    opts.append((item.state.expected or "").strip().lower())
                elif item.kind == "hold" and isinstance(item.state, HoldState):
                    opts.append((item.state.expected or "").strip().lower())
                elif item.kind == "press_wait" and isinstance(item.state, SequenceState) and len(item.state.steps) >= 2 and isinstance(item.state.steps[0], PressState):
                    opts.append((item.state.steps[0].expected or "").strip().lower())
                elif item.kind == "anim_wait" and isinstance(item.state, WaitState) and item.state.wait_for:
                    opts.append((item.state.wait_for or "").strip().lower())
                elif item.kind == "sequence" and isinstance(item.state, SequenceState) and item.state.steps:
                    first = item.state.steps[0]
                    if isinstance(first, PressState):
                        opts.append((first.expected or "").strip().lower())
                    elif isinstance(first, WaitState) and first.wait_for:
                        opts.append((first.wait_for or "").strip().lower())
            opts = [o for o in opts if o]
            return f"any-order({'|'.join(opts)})" if opts else "any-order(—)"
        return "—"

    def _step_accepts_input(self, step: Any, input_name: str) -> bool:
        """True if this step could accept input_name (for find next/prev)."""
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return False
        if isinstance(step, WaitState):
            return False
        if isinstance(step, PressState):
            return (step.expected or "").strip().lower() == input_name
        if isinstance(step, HoldState):
            return (step.expected or "").strip().lower() == input_name
        if isinstance(step, GroupState):
            for item in step.items:
                if item.kind == "press" and isinstance(item.state, PressState) and (item.state.expected or "").strip().lower() == input_name:
                    return True
                if item.kind == "hold" and isinstance(item.state, HoldState) and (item.state.expected or "").strip().lower() == input_name:
                    return True
                if item.kind == "press_wait" and isinstance(item.state, SequenceState) and len(item.state.steps) >= 2 and isinstance(item.state.steps[0], PressState) and (item.state.steps[0].expected or "").strip().lower() == input_name:
                    return True
                if item.kind == "anim_wait" and isinstance(item.state, WaitState) and (item.state.wait_for or "").strip().lower() == input_name:
                    return True
                if item.kind == "sequence" and isinstance(item.state, SequenceState) and item.state.steps:
                    first = item.state.steps[0]
                    if isinstance(first, PressState) and (first.expected or "").strip().lower() == input_name:
                        return True
                    if isinstance(first, WaitState) and (first.wait_for or "").strip().lower() == input_name:
                        return True
            return False
        if isinstance(step, SequenceState) and step.steps:
            first = step.steps[0]
            if isinstance(first, PressState) and (first.expected or "").strip().lower() == input_name:
                return True
            if isinstance(first, HoldState) and (first.expected or "").strip().lower() == input_name:
                return True
        return False

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

    def _ensure_combo_stats(self, name: str):
        if not name:
            return
        if name not in self.combo_stats or not isinstance(self.combo_stats.get(name), dict):
            self.combo_stats[name] = {
                "success": 0,
                "fail": 0,
                "best_ms": None,
                "total_success_ms": 0,
                "fail_by_step": {},
                "fail_by_expected": {},
                "fail_by_reason": {},
                "fail_events": [],
            }
        else:
            self.combo_stats[name].setdefault("success", 0)
            self.combo_stats[name].setdefault("fail", 0)
            self.combo_stats[name].setdefault("best_ms", None)
            self.combo_stats[name].setdefault("total_success_ms", 0)
            self.combo_stats[name].setdefault("fail_by_step", {})
            self.combo_stats[name].setdefault("fail_by_expected", {})
            self.combo_stats[name].setdefault("fail_by_reason", {})
            self.combo_stats[name].setdefault("fail_events", [])

    def _combo_avg_ms(self, name: str):
        self._ensure_combo_stats(name)
        s = int(self.combo_stats[name].get("success", 0) or 0)
        total = int(self.combo_stats[name].get("total_success_ms", 0) or 0)
        if s <= 0 or total <= 0:
            return None
        return total / float(s)

    def _format_percent(self, success: int, fail: int):
        total = success + fail
        if total <= 0:
            return "—"
        return f"{(success / total) * 100:.1f}%"

    def stats_text(self):
        return ui.stats_text(self)

    def failures_by_reason(self) -> dict[str, int]:
        return ui.failures_by_reason(self)

    def min_time_text(self) -> str:
        return ui.min_time_text(self)

    def _parse_expected_time_ms(self, raw: str | None) -> int | None:
        raw = (raw or "").strip().lower()
        if not raw:
            return None
        ms = self._parse_duration(raw)
        if ms is None:
            return None
        if ms <= 0:
            return None
        return int(ms)

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
        return ui.practical_apm(self)

    def theoretical_max_apm(self) -> float | None:
        return ui.theoretical_max_apm(self)

    def apm_text(self) -> str:
        return ui.apm_text(self)

    def apm_max_text(self) -> str:
        return ui.apm_max_text(self)

    def difficulty_score_10(self) -> float | None:
        return ui.difficulty_score_10(self)

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
        return ui.timeline_steps(self)

    def init_payload(self) -> dict[str, Any]:
        return ui.init_payload(self)

    # -------------------------
    # Combo ender logic
    # -------------------------

    def _is_combo_ender(self, input_name: str) -> bool:
        return input_name in self.combo_enders

    def _ender_grace_for(self, input_name: str) -> int:
        try:
            return int(self.combo_enders.get(input_name, 0))
        except Exception:
            return 0

    def _within_ender_grace(self, input_name: str) -> bool:
        grace_ms = self._ender_grace_for(input_name)
        if not grace_ms or grace_ms <= 0:
            return False
        if not self.last_input_time:
            return False
        now = time.perf_counter()
        return ((now - self.last_input_time) * 1000) <= float(grace_ms)

    def _should_ignore_ender_miss(self, input_name: str) -> bool:
        return (input_name == self.last_success_input) and self._within_ender_grace(input_name)

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
                self._send({"type": "stat_update", "stats": self.stats_text()})
                self._send({"type": "fail_update", "failures": self.failures_by_reason()})
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
            if self._group_wait_begin_sent:
                self._send({"type": "wait_end"})
            self._group_wait_begin_sent = False
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
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()

    def _on_combo_completed(self, total_ms: float) -> None:
        """Record success, reset to start, and notify UI. Single place for combo completion."""
        self.record_combo_success(total_ms)
        self._reset_to_start()
        self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"})
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def _reset_after_fail(self) -> None:
        """Full reset after a combo failure: clear index, time, step state, and notify UI."""
        self.current_index = 0
        self.start_time = 0.0
        self.last_input_time = 0.0
        for s in self.runtime_steps:
            s.reset()
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()
        self._reset_attempt_marks()
        self._update_shortcuts()
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

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

    def _fail_combo(self, reason: str, now: float, *, expected_label: str | None = None, actual: str | None = None) -> None:
        """Record failure, reset all steps, send status and timeline."""
        self._send({"type": "status", "text": "Combo Dropped (" + (reason or "fail") + ")", "color": "fail"})
        elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
        self.record_combo_fail(
            actual=actual or reason,
            expected_step_index=int(self.current_index),
            expected_label=expected_label or self._expected_label_for_step(self._active_step()),
            reason=reason,
            elapsed_ms=elapsed_ms,
        )
        self._reset_after_fail()

    def _check_ender_fail(self, input_name: str, now: float) -> None:
        """If key is a combo ender (and not in grace), drop the combo."""
        if not self._is_combo_ender(input_name) or self._should_ignore_ender_miss(input_name):
            return
        # Don't drop on ender before the attempt has started (no accepted input yet).
        if not self.last_input_time and self.current_index == 0:
            return
        self._mark_step(int(self.current_index), "missed")
        expected = str(self._expected_label_for_step(self._active_runtime_step()) or "").strip().lower()
        actual = str(input_name or "").strip().lower()
        self.record_hit(f"{actual} (Exp: {expected}) [ender]", "FAIL", "FAIL")
        self._fail_combo("Combo Ender", now, actual=actual, expected_label=expected)

    def _start_hold(self, input_name: str, required_ms: int, now: float):
        self.hold_in_progress = True
        self.hold_expected_input = input_name
        self.hold_started_at = now
        self.hold_required_ms = required_ms
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
            self.record_hit(label, "FAIL", "FAIL")
            self._send({"type": "status", "text": "Combo Dropped (Too Early)", "color": "fail"})
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
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

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

    def _complete_hold(self, now: float, *, auto: bool):
        step = self._active_runtime_step()
        if not isinstance(step, HoldState):
            return False

        target_input = str(step.expected or "")
        target_hold_ms = int(step.required_ms or 0)

        held_ms = (now - self.hold_started_at) * 1000
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

            # If this hold was gated by a wait right before it, mark that wait green (timing satisfied).
            if self.current_index > 0:
                prev = self.runtime_steps[self.current_index - 1]
                if isinstance(prev, WaitState):
                    self._mark_step(self.current_index - 1, "ok")

            self.current_index += 1
            if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                return True
            self._maybe_start_wait_step()

            if self.current_index >= len(self.runtime_steps):
                self._on_combo_completed(total_ms)
        else:
            self.record_hit(label, "FAIL", "FAIL")
            self._send({"type": "status", "text": "Combo Dropped (Hold Too Short)", "color": "fail"})
            elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
            self.record_combo_fail(
                actual=f"released @ {held_ms:.0f}ms",
                expected_step_index=int(self.current_index),
                expected_label=self._expected_label_for_step(step),
                reason="hold too short",
                elapsed_ms=elapsed_ms,
            )
            self._reset_after_fail()
            return False

        self._reset_hold_state()
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
        return ok

    def _record_fail_detail(
        self,
        *,
        step_index: int,
        expected: str,
        actual: str,
        reason: str,
        elapsed_ms: float | None,
    ):
        name = self.active_combo_name
        if not name:
            return
        self._ensure_combo_stats(name)

        by_step = self.combo_stats[name].get("fail_by_step", {})
        if not isinstance(by_step, dict):
            by_step = {}
        key_step = str(max(0, int(step_index)))
        by_step[key_step] = int(by_step.get(key_step, 0) or 0) + 1
        self.combo_stats[name]["fail_by_step"] = by_step

        by_exp = self.combo_stats[name].get("fail_by_expected", {})
        if not isinstance(by_exp, dict):
            by_exp = {}
        exp_key = (expected or "—").strip().lower()
        by_exp[exp_key] = int(by_exp.get(exp_key, 0) or 0) + 1
        self.combo_stats[name]["fail_by_expected"] = by_exp

        by_reason = self.combo_stats[name].get("fail_by_reason", {})
        if not isinstance(by_reason, dict):
            by_reason = {}
        r = (reason or "unknown").strip().lower() or "unknown"
        by_reason[r] = int(by_reason.get(r, 0) or 0) + 1
        self.combo_stats[name]["fail_by_reason"] = by_reason

        ev = {
            "ts": int(time.time()),
            "attempt": int(self.attempt_counter or 0),
            "step_index": int(step_index),
            "expected": str(expected or ""),
            "actual": str(actual or ""),
            "reason": str(reason or ""),
            "elapsed_ms": int(round(float(elapsed_ms))) if elapsed_ms is not None else None,
        }
        events = self.combo_stats[name].get("fail_events", [])
        if not isinstance(events, list):
            events = []
        events.append(ev)
        if len(events) > 100:
            events = events[-100:]
        self.combo_stats[name]["fail_events"] = events

    def record_combo_success(self, completion_ms: float | int | None = None):
        if not self.active_combo_name:
            return
        # Snapshot for UI: keep the timeline fully green until the next attempt begins.
        self._ui_last_success_combo = self.active_combo_name
        self._ui_last_success_steps_len = len(self.runtime_steps or [])
        self._ensure_combo_stats(self.active_combo_name)
        self.combo_stats[self.active_combo_name]["success"] += 1

        if completion_ms is None and self.start_time:
            completion_ms = (time.perf_counter() - self.start_time) * 1000.0
        try:
            ms = int(round(float(completion_ms))) if completion_ms is not None else None
        except Exception:
            ms = None
        if ms is not None and ms > 0:
            total = int(self.combo_stats[self.active_combo_name].get("total_success_ms", 0) or 0)
            self.combo_stats[self.active_combo_name]["total_success_ms"] = total + ms
            best = self.combo_stats[self.active_combo_name].get("best_ms", None)
            try:
                best_i = int(best) if best is not None else None
            except Exception:
                best_i = None
            if best_i is None or ms < best_i:
                self.combo_stats[self.active_combo_name]["best_ms"] = ms

        self.save_combos()
        self._send({"type": "stat_update", "stats": self.stats_text()})
        self._send({"type": "fail_update", "failures": self.failures_by_reason()})

    def record_combo_fail(
        self,
        *,
        actual: str | None = None,
        expected_step_index: int | None = None,
        expected_label: str | None = None,
        reason: str | None = None,
        elapsed_ms: float | None = None,
    ):
        if not self.active_combo_name:
            return
        if self.attempt_counter <= 0:
            return

        # Failure should clear any previous "success snapshot" so we don't show a fully green timeline at idle.
        self._ui_last_success_combo = None
        self._ui_last_success_steps_len = 0

        self._ensure_combo_stats(self.active_combo_name)
        self.combo_stats[self.active_combo_name]["fail"] += 1

        idx = self.current_index if expected_step_index is None else expected_step_index
        try:
            idx_i = int(idx)
        except Exception:
            idx_i = 0
        exp = expected_label
        if not exp:
            step = self._active_step()
            exp = self._expected_label_for_step(step) if step else "—"

        self._record_fail_detail(
            step_index=idx_i,
            expected=str(exp or "—"),
            actual=str(actual or ""),
            reason=str(reason or ""),
            elapsed_ms=elapsed_ms,
        )

        self.save_combos()
        self._send({"type": "stat_update", "stats": self.stats_text()})
        self._send({"type": "fail_update", "failures": self.failures_by_reason()})

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

        self.currently_pressed.add(input_name)
        if not self.runtime_steps:
            return

        now = time.perf_counter()

        # 1) Active wait
        if self.wait and self.wait.in_progress:
            result = self.wait.process_press(input_name, now)
            if isinstance(result, CompleteResult):
                self._complete_wait(now, fail=False)
                return self._process_press_unlocked(input_name)
            if isinstance(result, FailResult):
                self._complete_wait(now, fail=True, reason=result.reason)
                return
            if self.wait.mode in ("soft", "hard") and self._is_combo_ender(input_name):
                if not self._should_ignore_ender_miss(input_name):
                    self._complete_wait(now, fail=True, reason=f"{input_name} (ender) during wait")
            return

        # 2) Active hold
        if self.hold and self.hold.in_progress:
            if input_name == self.hold.expected:
                return
            if self.hold.check_complete(now):
                self._complete_hold(now, auto=True)
                return self._process_press_unlocked(input_name)
            self._fail_combo("hold too short", now, actual="wrong key / released early")
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
            if result.advance:
                self._advance_step(now)
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
                if step.wait_active and not self._group_wait_begin_sent:
                    for item in step.items:
                        if (
                            item.kind == "anim_wait"
                            and isinstance(item.state, WaitState)
                            and item.state.in_progress
                        ):
                            self._send({
                                "type": "wait_begin",
                                "required_ms": int(item.state.required_ms),
                                "mode": "mandatory",
                                "wait_for": str(item.state.wait_for or ""),
                            })
                            self._group_wait_begin_sent = True
                            break
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return
        if isinstance(result, FailResult):
            self._fail_combo(result.reason, now)
            return
        if isinstance(result, CompleteResult):
            self._advance_step(now)
            if self.current_index >= len(self.runtime_steps):
                self._on_combo_completed((now - self.start_time) * 1000.0 if self.start_time else 0.0)
                return
            if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                return
            return self._process_press_unlocked(input_name)
        # During group mandatory wait (animation lock), ignore all keys including enders.
        if isinstance(step, GroupState) and step.wait_active:
            return
        # Input didn't match current step (IgnoreResult) — fail with actual reason (e.g. "expected e"), not "Combo Ender".
        if isinstance(result, IgnoreResult):
            # If the attempt hasn't started yet (no accepted input), ignore stray keys.
            # This prevents "dropping" a combo before the player has actually started it.
            if not self.last_input_time and self.current_index == 0:
                return
            # If it's NOT a combo ender, ignore it. Combo enders are the only "wrong" inputs
            # that should drop an in-progress attempt.
            if not self._is_combo_ender(input_name):
                return
            # Combo ender grace:
            # If the pressed key is configured as an ender *and* it's the same as the last
            # successfully accepted input, allow a short re-press window where we ignore it.
            # Example: combo `q, e, r` with ender `q:2` → `q` then `q` within 2s should NOT drop.
            if self._should_ignore_ender_miss(input_name):
                return
            expected = self._expected_label_for_step(step) or "?"
            actual = input_name
            self._mark_step(int(self.current_index), "missed")
            # Attempt Log rows are driven by "hit" messages; record the ender as a FAIL.
            self.record_hit(f"{actual} (Exp: {expected}) [wrong key]", "FAIL", "FAIL")
            self._fail_combo(f"expected {expected}", now, actual=input_name, expected_label=expected)
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

        if isinstance(result, AcceptResult):
            if result.advance:
                if isinstance(step, HoldState):
                    self._complete_hold(now, auto=False)
                else:
                    self._advance_step(now)
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
            self._fail_combo(result.reason, now, actual="released (hold too short)")
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

            step = self._active_runtime_step()
            if step is None:
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
                    if isinstance(step, GroupState) and self._group_wait_begin_sent:
                        self._send({"type": "wait_end"})
                        self._group_wait_begin_sent = False
                    self._advance_step(now)
                    if self.current_index >= len(self.runtime_steps):
                        total_ms = (now - self.start_time) * 1000.0 if self.start_time else 0.0
                        self._on_combo_completed(total_ms)
                    elif self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=(now - self.start_time) * 1000.0 if self.start_time else 0.0):
                        pass
                    else:
                        st = self.get_status()
                        self._send({"type": "status", "text": st.text, "color": st.color})
        except Exception:
            return