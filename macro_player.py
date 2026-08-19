"""
MacroPlayer: replays a parsed combo token list by emitting real keyboard/mouse events.
Uses pynput.keyboard.Controller and pynput.mouse.Controller.

Design:
- Runs in a daemon thread; at most one playback at a time.
- Interruptible at any point via a threading.Event (Esc hard-stop).
- Compiles the AST into one absolute-deadline plan before playback, so dispatch
  overhead cannot accumulate into timing drift across a long combo.
- On Windows, requests 1ms multimedia timer resolution for the playback run so
  time.sleep matches short waits better.
- Presses, releases, and replay markers run on the same ordered playback thread.
- Replay markers fire only after the corresponding output press succeeds.
- Group nodes are executed left-to-right (defined order) for macro playback.
- Plain taps emit one 30ms pulse; explicit chain spam is compiled separately.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from pynput import keyboard, mouse

from parser import (
    expanded_ast_from_tokens,
    GroupNode,
    HoldNode,
    HoldWithBodyNode,
    PressNode,
    SequenceNode,
    WaitNode,
)
from profiling.macro_profile_log import MacroProfileLogWriter
from profiling.macro_timing import MacroTimingCollector, MacroTimingProfile

logger = logging.getLogger(__name__)

# Duration (ms) a normal tap is held before its non-blocking release fires.
_TAP_HOLD_MS = 30

# Repeated same-key taps need a real released phase. Without this gap, a 30ms
# pulse at a 30ms spam cadence schedules release and the next press at the same
# deadline, forcing two synchronous OS calls to serialize and making the second
# event late by construction.
_SPAM_RELEASE_GAP_MS = 5

_KB = keyboard.Controller()
_MOUSE = mouse.Controller()

# Map normalized token names → pynput special Key objects.
_SPECIAL: dict[str, keyboard.Key] = {
    "space": keyboard.Key.space,
    "esc": keyboard.Key.esc,
    "escape": keyboard.Key.esc,
    "enter": keyboard.Key.enter,
    "tab": keyboard.Key.tab,
    "shift": keyboard.Key.shift,
    "shift_l": keyboard.Key.shift_l,
    "shift_r": keyboard.Key.shift_r,
    "ctrl": keyboard.Key.ctrl,
    "ctrl_l": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "alt": keyboard.Key.alt,
    "alt_l": keyboard.Key.alt_l,
    "alt_gr": keyboard.Key.alt_gr,
    "up": keyboard.Key.up,
    "down": keyboard.Key.down,
    "left": keyboard.Key.left,
    "right": keyboard.Key.right,
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
    "backspace": keyboard.Key.backspace,
    "delete": keyboard.Key.delete,
    "home": keyboard.Key.home,
    "end": keyboard.Key.end,
    "page_up": keyboard.Key.page_up,
    "page_down": keyboard.Key.page_down,
    "caps_lock": keyboard.Key.caps_lock,
    "insert": keyboard.Key.insert,
    "num_lock": keyboard.Key.num_lock,
    "print_screen": keyboard.Key.print_screen,
    "scroll_lock": keyboard.Key.scroll_lock,
    "pause": keyboard.Key.pause,
    "menu": keyboard.Key.menu,
}

_MOUSE_BTN: dict[str, mouse.Button] = {
    "lmb": mouse.Button.left,
    "rmb": mouse.Button.right,
    "mmb": mouse.Button.middle,
}

_DEFAULT_CHAIN_SPAM_INTERVAL_MS = 100

def _windows_timer_1ms_enter() -> Callable[[], None]:
    """Request 1ms system timer resolution on Windows; return closer to restore."""

    if sys.platform != "win32":
        return lambda: None
    try:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        winmm.timeBeginPeriod.restype = ctypes.c_uint
        winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        winmm.timeEndPeriod.restype = ctypes.c_uint
        if winmm.timeBeginPeriod(1) != 0:
            return lambda: None

        def _end() -> None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                logger.debug("timeEndPeriod failed", exc_info=True)

        return _end
    except Exception:
        logger.debug("timeBeginPeriod unavailable", exc_info=True)
        return lambda: None


class _ReplayState:
    """Tracks per-step timing for karaoke-style visual replay.

    Thread-safe: spam chain schedules fire() on the delay scheduler thread;
    the main playback thread calls it inline for non-chain steps.
    The stop event is checked before each fire so late callbacks
    are silently dropped after the macro has been stopped.

    Delivery: timings are computed on the calling thread (aligned with pynput),
    but the UI callback is enqueued so engine.replay_accept never blocks key I/O.
    """

    def __init__(
        self,
        start_time: float,
        stop: threading.Event,
        deliver_queue: "queue.Queue[tuple[str, float, float, bool]]",
    ):
        self._start_time = start_time
        self._last_time = start_time
        self._stop = stop
        self._lock = threading.Lock()
        self._deliver_queue = deliver_queue

    def fire(self, key: str, pressed: bool, at_time: float | None = None) -> None:
        """Enqueue on_step with wall-clock-accurate step/total timings.

        at_time: absolute perf_counter timestamp when this step logically fires.
        Defaults to now if not supplied (inline, non-timer calls).
        """
        if self._stop.is_set():
            return
        now = at_time if at_time is not None else time.perf_counter()
        with self._lock:
            step_ms = max(0.0, (now - self._last_time) * 1000.0)
            total_ms = max(0.0, (now - self._start_time) * 1000.0)
            self._last_time = now
        try:
            self._deliver_queue.put((key, step_ms, total_ms, pressed))
        except Exception:
            logger.debug("_ReplayState.fire: queue put failed for key %r", key, exc_info=True)


def _resolve_key(name: str) -> keyboard.Key | keyboard.KeyCode | None:
    n = (name or "").strip().lower()
    if n in _MOUSE_BTN:
        return None
    if n in _SPECIAL:
        return _SPECIAL[n]
    if len(n) == 1:
        return keyboard.KeyCode.from_char(n)
    return None


def _extract_press_wait_info(node) -> tuple[str, int] | None:
    """
    Return (key, wait_ms) for SequenceNode([PressNode, WaitNode(mode=soft|hard)]), else None.
    """
    if not isinstance(node, SequenceNode):
        return None
    if len(node.steps) != 2:
        return None
    a, b = node.steps
    if not isinstance(a, PressNode) or not isinstance(b, WaitNode):
        return None
    if b.mode not in ("soft", "hard"):
        return None
    key = (a.key or "").strip().lower()
    wait_ms = int(getattr(b, "duration_ms", 0) or 0)
    if not key or wait_ms <= 0:
        return None
    return key, wait_ms


@dataclass(frozen=True, order=True)
class _PlanEvent:
    """One absolute event in a compiled macro plan."""

    at_s: float
    order: int
    kind: str = field(compare=False)
    key: str = field(compare=False)
    action_id: int = field(compare=False)
    pressed: bool | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _ExecutionPlan:
    events: tuple[_PlanEvent, ...]
    duration_s: float


class _PlanBuilder:
    """Compile combo AST nodes into one absolute-deadline execution plan."""

    def __init__(self, chain_spam_interval_ms: int | None) -> None:
        self.cursor_s = 0.0
        self._spam_interval_ms = chain_spam_interval_ms
        self._events: list[_PlanEvent] = []
        self._order = 0
        self._action_id = 0
        self._key_busy_until: dict[str, float] = defaultdict(float)

    def build(self, nodes: list) -> _ExecutionPlan:
        self._node_list(nodes, self._spam_interval_ms)
        events = tuple(sorted(self._events))
        duration = max((event.at_s for event in events), default=self.cursor_s)
        return _ExecutionPlan(events=events, duration_s=max(duration, self.cursor_s))

    def _validate_key(self, name: str) -> str:
        key = (name or "").strip().lower()
        if key in _MOUSE_BTN or _resolve_key(key) is not None:
            return key
        raise ValueError(f"Unsupported macro input: {name!r}")

    def _event(
        self,
        at_s: float,
        kind: str,
        key: str,
        action_id: int,
        *,
        pressed: bool | None = None,
    ) -> None:
        self._order += 1
        self._events.append(
            _PlanEvent(max(0.0, at_s), self._order, kind, key, action_id, pressed)
        )

    def _next_action(self) -> int:
        self._action_id += 1
        return self._action_id

    def _tap_at(
        self,
        name: str,
        at_s: float,
        *,
        replay: bool = True,
        hold_ms: int = _TAP_HOLD_MS,
    ) -> tuple[int, float]:
        key = self._validate_key(name)
        # A repeated physical key cannot be pressed again before its prior release.
        actual_at = max(at_s, self._key_busy_until[key])
        release_at = actual_at + (max(1, int(hold_ms)) / 1000.0)
        action_id = self._next_action()
        self._event(actual_at, "down", key, action_id)
        if replay:
            self._event(actual_at, "replay", key, action_id, pressed=True)
        self._event(release_at, "up", key, action_id)
        self._key_busy_until[key] = release_at
        return action_id, actual_at

    def _hold(self, name: str, duration_ms: int) -> None:
        key = self._validate_key(name)
        start = max(self.cursor_s, self._key_busy_until[key])
        end = start + max(0, duration_ms) / 1000.0
        action_id = self._next_action()
        self._event(start, "down", key, action_id)
        self._event(start, "replay", key, action_id, pressed=True)
        self._event(end, "up", key, action_id)
        self._event(end, "replay", key, action_id, pressed=False)
        self._key_busy_until[key] = end
        self.cursor_s = end

    def _hold_with_body(self, node: HoldWithBodyNode) -> None:
        key = self._validate_key(node.key)
        start = max(self.cursor_s, self._key_busy_until[key])
        self.cursor_s = start
        action_id = self._next_action()
        self._event(start, "down", key, action_id)
        self._event(start, "replay", key, action_id, pressed=True)
        self._node_list(list(node.body), None)
        end = max(start + max(0, node.duration_ms) / 1000.0, self.cursor_s)
        self._event(end, "up", key, action_id)
        self._event(end, "replay", key, action_id, pressed=False)
        self._key_busy_until[key] = end
        self.cursor_s = end

    def _spam_chain(self, nodes: list, start: int, end: int, key: str, total_ms: int) -> None:
        interval_ms = max(_TAP_HOLD_MS, int(self._spam_interval_ms or _DEFAULT_CHAIN_SPAM_INTERVAL_MS))
        pulse_ms = min(
            _TAP_HOLD_MS,
            max(1, interval_ms - _SPAM_RELEASE_GAP_MS),
        )
        chain_start = self.cursor_s
        physical: list[tuple[float, int]] = []
        for offset_ms in range(0, max(1, total_ms), interval_ms):
            action_id, actual_at = self._tap_at(
                key,
                chain_start + offset_ms / 1000.0,
                replay=False,
                hold_ms=pulse_ms,
            )
            physical.append((actual_at, action_id))

        cumulative_s = 0.0
        for idx in range(start, end):
            info = _extract_press_wait_info(nodes[idx])
            logical_at = chain_start + cumulative_s
            action_id = physical[0][1]
            for physical_at, candidate_id in physical:
                if physical_at <= logical_at + 1e-9:
                    action_id = candidate_id
                else:
                    break
            self._event(logical_at, "replay", key, action_id, pressed=True)
            if info:
                cumulative_s += info[1] / 1000.0
        self.cursor_s = chain_start + total_ms / 1000.0

    def _node_list(self, nodes: list, chain_spam_interval_ms: int | None) -> None:
        previous = self._spam_interval_ms
        self._spam_interval_ms = chain_spam_interval_ms
        try:
            i = 0
            while i < len(nodes):
                info = _extract_press_wait_info(nodes[i]) if chain_spam_interval_ms is not None else None
                if info is not None:
                    key, total_ms = info
                    j = i + 1
                    while j < len(nodes):
                        nxt = _extract_press_wait_info(nodes[j])
                        if nxt is None or nxt[0] != key:
                            break
                        total_ms += nxt[1]
                        j += 1
                    if j - i > 1:
                        self._spam_chain(nodes, i, j, key, total_ms)
                        i = j
                        continue
                self._node(nodes[i], chain_spam_interval_ms)
                i += 1
        finally:
            self._spam_interval_ms = previous

    def _node(self, node, chain_spam_interval_ms: int | None) -> None:
        if isinstance(node, PressNode):
            _action_id, actual_at = self._tap_at(node.key, self.cursor_s)
            self.cursor_s = max(self.cursor_s, actual_at)
        elif isinstance(node, HoldNode):
            self._hold(node.key, node.duration_ms)
        elif isinstance(node, HoldWithBodyNode):
            self._hold_with_body(node)
        elif isinstance(node, WaitNode):
            self.cursor_s += max(0, node.duration_ms) / 1000.0
        elif isinstance(node, SequenceNode):
            self._node_list(list(node.steps), chain_spam_interval_ms)
        elif isinstance(node, GroupNode):
            self._node_list(list(node.items), chain_spam_interval_ms)


class _OutputAdapter(Protocol):
    def press(self, name: str) -> bool: ...
    def release(self, name: str) -> bool: ...


class _SyntheticEventLedger:
    """Matches pynput callbacks to macro-origin events without timing guesses."""

    _TTL_S = 0.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[tuple[str, bool], deque[float]] = defaultdict(deque)

    def expect(self, name: str, pressed: bool) -> None:
        with self._lock:
            self._events[((name or "").strip().lower(), bool(pressed))].append(
                time.perf_counter() + self._TTL_S
            )

    def cancel(self, name: str, pressed: bool) -> None:
        with self._lock:
            q = self._events.get(((name or "").strip().lower(), bool(pressed)))
            if q:
                q.pop()

    def consume(self, name: str, pressed: bool) -> bool:
        key = ((name or "").strip().lower(), bool(pressed))
        now = time.perf_counter()
        with self._lock:
            q = self._events.get(key)
            if not q:
                return False
            while q and q[0] < now:
                q.popleft()
            if not q:
                return False
            q.popleft()
            return True


class _PynputOutput:
    def __init__(self, ledger: _SyntheticEventLedger) -> None:
        self._ledger = ledger

    def _target(self, name: str):
        if name in _MOUSE_BTN:
            return _MOUSE, _MOUSE_BTN[name]
        return _KB, _resolve_key(name)

    def _emit(self, name: str, pressed: bool) -> bool:
        target, value = self._target(name)
        if value is None:
            return False
        self._ledger.expect(name, pressed)
        try:
            (target.press if pressed else target.release)(value)
            return True
        except Exception:
            self._ledger.cancel(name, pressed)
            logger.debug("macro output failed: %s %s", "down" if pressed else "up", name, exc_info=True)
            return False

    def press(self, name: str) -> bool:
        return self._emit(name, True)

    def release(self, name: str) -> bool:
        return self._emit(name, False)


def _wait_until(deadline: float, stop: threading.Event) -> bool:
    """Wait for one absolute deadline. Returns False when interrupted."""
    while not stop.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return True
        if remaining > 0.003:
            stop.wait(remaining - 0.002)
        else:
            # Yield during the short precision tail without adding relative drift.
            time.sleep(0)
    return False


class MacroPlayer:
    """
    Plays back a combo token list by emitting keyboard/mouse events.
    Thread-safe; at most one playback runs at a time.
    """

    def __init__(
        self,
        on_status: Callable[[str, str], None] | None = None,
        on_step: Callable[[str, float, float, bool], None] | None = None,
        *,
        output: _OutputAdapter | None = None,
        profile_log_dir: Path | None = None,
    ):
        """
        on_status(text, color): optional callback on the playback thread for start/complete/stop.
        on_step(key, step_ms, total_ms, pressed): optional karaoke callback, fired once
            per logical combo input phase at its planned time. Must be thread-safe.
        """
        self._on_status = on_status
        self._on_step = on_step
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._chain_spam_interval_ms: int | None = _DEFAULT_CHAIN_SPAM_INTERVAL_MS
        self._synthetic_events = _SyntheticEventLedger()
        self._output: _OutputAdapter = output or _PynputOutput(self._synthetic_events)
        self._last_profile: MacroTimingProfile | None = None
        self._profile_log = (
            MacroProfileLogWriter(profile_log_dir) if profile_log_dir is not None else None
        )

    def set_chain_spam_interval_ms(self, interval_ms: int | None) -> None:
        if interval_ms is None:
            with self._lock:
                self._chain_spam_interval_ms = None
            return
        try:
            iv = int(interval_ms)
        except Exception:
            iv = _DEFAULT_CHAIN_SPAM_INTERVAL_MS
        with self._lock:
            self._chain_spam_interval_ms = iv if iv > 0 else None

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def consume_synthetic_event(self, input_name: str, pressed: bool) -> bool:
        """True when a listener callback matches an event emitted by this macro."""
        return self._synthetic_events.consume(input_name, pressed)

    def last_profile(self, *, include_events: bool = True) -> dict | None:
        """Return a snapshot of the most recently completed playback timing profile."""
        with self._lock:
            profile = self._last_profile
        return profile.to_dict(include_events=include_events) if profile is not None else None

    def start(
        self,
        tokens: list[str],
        *,
        requested_at: float | None = None,
        combo_name: str | None = None,
    ) -> bool:
        """
        Begin macro playback from a list of normalized combo tokens.
        Returns False (and does nothing) if playback is already in progress.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if not tokens:
                return False
            ast = expanded_ast_from_tokens(tokens)
            if not ast:
                return False
            try:
                plan = _PlanBuilder(self._chain_spam_interval_ms).build(list(ast))
            except ValueError as exc:
                logger.warning("Macro validation failed: %s", exc)
                self._notify(f"Macro cannot start: {exc}", "fail")
                return False
            self._stop.clear()
            request_time = time.perf_counter() if requested_at is None else float(requested_at)
            self._last_profile = None
            self._thread = threading.Thread(
                target=self._run,
                args=(plan, request_time, combo_name),
                name="MacroPlayer",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Signal playback to stop; the playback thread releases held inputs."""
        self._stop.set()

    def _run(
        self,
        plan: _ExecutionPlan,
        requested_at: float,
        combo_name: str | None,
    ) -> None:
        stopped_early = False
        failure: str | None = None
        end_period = _windows_timer_1ms_enter()
        held: set[str] = set()
        try:
            start_time = time.perf_counter()
            timing = MacroTimingCollector(
                requested_at=requested_at,
                clock_started_at=start_time,
            )
            replay = None
            replay_q: queue.Queue[tuple[str, float, float, bool] | None] | None = None
            replay_worker_thread: threading.Thread | None = None
            if self._on_step is not None:
                replay_q = queue.Queue()
                on_step = self._on_step
                stop_ev = self._stop

                def _replay_deliver_loop() -> None:
                    while True:
                        item = replay_q.get()
                        if item is None:
                            break
                        if stop_ev.is_set():
                            continue
                        k, sm, tm, pressed = item
                        try:
                            on_step(k, sm, tm, pressed)
                        except Exception:
                            logger.debug(
                                "_replay_deliver_loop: on_step raised for key %r",
                                k,
                                exc_info=True,
                            )

                replay_worker_thread = threading.Thread(
                    target=_replay_deliver_loop, name="MacroReplayDeliver", daemon=True
                )
                replay_worker_thread.start()
                replay = _ReplayState(start_time, self._stop, replay_q)

            action_ok: dict[int, bool] = {}
            try:
                for event in plan.events:
                    if not _wait_until(start_time + event.at_s, self._stop):
                        stopped_early = True
                        break
                    woke_ns = time.perf_counter_ns()
                    if event.kind == "down":
                        dispatch_started_ns = time.perf_counter_ns()
                        ok = self._output.press(event.key)
                        dispatch_completed_ns = time.perf_counter_ns()
                        timing.record(
                            order=event.order,
                            kind=event.kind,
                            key=event.key,
                            planned_offset_s=event.at_s,
                            woke_ns=woke_ns,
                            dispatch_started_ns=dispatch_started_ns,
                            dispatch_completed_ns=dispatch_completed_ns,
                        )
                        action_ok[event.action_id] = ok
                        if not ok:
                            raise RuntimeError(f"could not press {event.key}")
                        held.add(event.key)
                    elif event.kind == "up":
                        if not action_ok.get(event.action_id, False):
                            continue
                        dispatch_started_ns = time.perf_counter_ns()
                        ok = self._output.release(event.key)
                        dispatch_completed_ns = time.perf_counter_ns()
                        timing.record(
                            order=event.order,
                            kind=event.kind,
                            key=event.key,
                            planned_offset_s=event.at_s,
                            woke_ns=woke_ns,
                            dispatch_started_ns=dispatch_started_ns,
                            dispatch_completed_ns=dispatch_completed_ns,
                        )
                        action_ok[event.action_id] = ok
                        if not ok:
                            raise RuntimeError(f"could not release {event.key}")
                        held.discard(event.key)
                    elif event.kind == "replay" and replay is not None:
                        if action_ok.get(event.action_id, False):
                            if event.pressed is None:
                                raise RuntimeError("replay event is missing its input phase")
                            replay.fire(
                                event.key,
                                event.pressed,
                                start_time + event.at_s,
                            )
            finally:
                for key in tuple(held):
                    try:
                        self._output.release(key)
                    except Exception:
                        logger.debug("Emergency macro release failed: %s", key, exc_info=True)
                held.clear()
                if replay_q is not None:
                    replay_q.put(None)
                    if replay_worker_thread is not None:
                        replay_worker_thread.join(timeout=5.0)
            if self._stop.is_set():
                stopped_early = True
        except Exception as exc:
            logger.exception("MacroPlayer: playback failed")
            failure = str(exc)
            stopped_early = True
        finally:
            end_period()
            if "timing" in locals():
                profile = timing.finish()
                with self._lock:
                    self._last_profile = profile
                summary = profile.summary()
                scheduler = summary["scheduler_lateness_ms"]
                collision = summary["same_deadline_lateness_ms"]
                deadline_analysis = summary["deadline_analysis"]
                output = summary["output_duration_ms"]
                interval = summary["interval_error_ms"]
                logger.info(
                    "Macro timing: events=%d start=%.3fms first=%.3fms "
                    "scheduler[p50=%.3f p95=%.3f p99=%.3f max=%.3f]ms "
                    "same_deadline[count=%d p95=%.3f max=%.3f]ms "
                    "output[p95=%.3f max=%.3f]ms interval_error[p95=%.3f max=%.3f]ms",
                    summary["event_count"],
                    summary["request_to_clock_start_ms"],
                    summary["request_to_first_dispatch_ms"] or 0.0,
                    scheduler["p50"],
                    scheduler["p95"],
                    scheduler["p99"],
                    scheduler["max"],
                    deadline_analysis["later_collision_event_count"],
                    collision["p95"],
                    collision["max"],
                    output["p95"],
                    output["max"],
                    interval["p95"],
                    interval["max"],
                )
                logger.debug("Macro timing events: %s", profile.to_dict()["events"])
                outcome = "failed" if failure else "stopped" if stopped_early else "completed"
                if self._profile_log is not None:
                    written_path = self._profile_log.write(
                        profile,
                        outcome=outcome,
                        combo_name=combo_name,
                        plan_duration_ms=plan.duration_s * 1000.0,
                    )
                    if written_path is not None:
                        logger.info("Macro timing profile saved: %s", written_path)
            if failure:
                self._notify(f"Macro failed: {failure}", "fail")
            elif stopped_early:
                self._notify("Macro stopped.", "neutral")
            elif self._on_step is None:
                # Karaoke mode: engine already sent "Combo Complete!" via replay_accept.
                # Plain mode (no on_step): send our own completion notice.
                self._notify("Macro complete.", "success")

    def _notify(self, text: str, color: str) -> None:
        if self._on_status:
            try:
                self._on_status(text, color)
            except Exception:
                pass
