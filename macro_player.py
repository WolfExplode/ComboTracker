"""
MacroPlayer: replays a parsed combo token list by emitting real keyboard/mouse events.
Uses pynput.keyboard.Controller and pynput.mouse.Controller.

Design:
- Runs in a daemon thread; at most one playback at a time.
- Interruptible at any point via a threading.Event (Esc hard-stop).
- Sleeps in coarse slices when far from the deadline, 1ms slices when close, so
  waits stay accurate without adding ~10ms error per step.
- On Windows, requests 1ms multimedia timer resolution for the playback run so
  time.sleep matches short waits better.
- Tap releases and karaoke chain fires use one shared delay thread (heap +
  condition) instead of a threading.Timer per tap.
- Group nodes are executed left-to-right (defined order) for macro playback.
- Plain taps fire twice in a row (release then immediate second tap) to mimic
  light button spam when games drop single synthetic presses; chain spam uses
  single taps per interval so timing stays correct.
"""

from __future__ import annotations

import ctypes
import heapq
import logging
import math
import queue
import sys
import threading
import time
from typing import Callable

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

logger = logging.getLogger(__name__)

# Duration (ms) a "tap" key is held before its non-blocking release fires.
# Must be well below any realistic spam interval to avoid the next press
# arriving before the previous release, which would make it look like a
# key-repeat and get silently ignored by the combo engine.
# Plain taps schedule a second identical tap in the release callback (two pulses
# per combo step); chain spam disables that so interval math stays valid.
_TAP_HOLD_MS = 30

# Extra sleep (ms) inserted after a spam-collapsed chain when more nodes follow.
# Absorbs combo-engine tick-loop drift (~20ms per soft-wait) so the next key
# does not arrive before the engine has finished the last wait in the chain.
_POST_SPAM_PAD_MS = 30

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

# When more than this much time remains, use 10ms sleep slices (low CPU).
# Below it, use 1ms slices so short waits do not systematically overshoot.
_SLEEP_COARSE_THRESHOLD_S = 0.05


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


class _DelayScheduler:
    """Single daemon thread for delayed callbacks (tap release, replay.fire offsets)."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = 0
        self._cv = threading.Condition(threading.Lock())
        self._thread: threading.Thread | None = None

    def _ensure_thread(self) -> None:
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._loop, name="MacroDelaySched", daemon=True
            )
            self._thread.start()

    def schedule(self, delay_s: float, fn: Callable[[], None]) -> None:
        self._ensure_thread()
        fire_at = time.perf_counter() + max(0.0, delay_s)
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, (fire_at, self._seq, fn))
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                now = time.perf_counter()
                fire_at, _seq, fn = self._heap[0]
                if now < fire_at:
                    self._cv.wait(timeout=fire_at - now)
                    continue
                heapq.heappop(self._heap)
            try:
                fn()
            except Exception:
                logger.debug("delay scheduler callback failed", exc_info=True)


_DELAY_SCHEDULER = _DelayScheduler()


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
        deliver_queue: "queue.Queue[tuple[str, float, float]]",
    ):
        self._start_time = start_time
        self._last_time = start_time
        self._stop = stop
        self._lock = threading.Lock()
        self._deliver_queue = deliver_queue

    def fire(self, key: str, at_time: float | None = None) -> None:
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
            self._deliver_queue.put((key, step_ms, total_ms))
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


def _sleep_interruptible(seconds: float, stop: threading.Event) -> None:
    """Sleep until deadline or stop; coarse slices when far, 1ms when near."""
    end = time.perf_counter() + seconds
    while not stop.is_set():
        remaining = end - time.perf_counter()
        if remaining <= 0:
            break
        if remaining > _SLEEP_COARSE_THRESHOLD_S:
            slice_s = min(0.01, remaining)
        else:
            slice_s = min(0.001, remaining)
        time.sleep(slice_s)


def _tap_once(
    name: str,
    stop: threading.Event,
    *,
    on_released: Callable[[], None] | None = None,
) -> None:
    """Single press + scheduled release. Optional callback runs after release (same scheduler thread)."""
    if stop.is_set():
        return
    n = (name or "").strip().lower()
    if n in _MOUSE_BTN:
        btn = _MOUSE_BTN[n]
        try:
            _MOUSE.press(btn)
        except Exception:
            logger.debug("mouse tap press failed: %s", n, exc_info=True)
            return

        def _release_mouse() -> None:
            try:
                _MOUSE.release(btn)
            except Exception:
                pass
            if on_released:
                try:
                    on_released()
                except Exception:
                    pass

        _DELAY_SCHEDULER.schedule(_TAP_HOLD_MS / 1000.0, _release_mouse)
        return
    k = _resolve_key(n)
    if k is None:
        logger.debug("macro_player: unknown key token %r — skipped", n)
        return
    try:
        _KB.press(k)
    except Exception:
        logger.debug("key tap press failed: %s", n, exc_info=True)
        return

    def _release_key() -> None:
        try:
            _KB.release(k)
        except Exception:
            pass
        if on_released:
            try:
                on_released()
            except Exception:
                pass

    _DELAY_SCHEDULER.schedule(_TAP_HOLD_MS / 1000.0, _release_key)


def _tap(name: str, stop: threading.Event, *, double_tap: bool = True) -> None:
    """Press a key (or mouse button) and schedule the release non-blocking.

    The release fires after _TAP_HOLD_MS ms on the delay scheduler so the playback
    timeline is not stalled waiting for the hold to finish.  This keeps the
    combo wait: windows accurate while still sending a realistic press duration
    to the game.

    When double_tap is True (default), a second identical tap runs right after
    the first release — common for games that miss a lone synthetic press.
    Chain spam passes double_tap=False so each interval is still one physical tap.
    """
    if stop.is_set():
        return
    if not double_tap:
        _tap_once(name, stop)
        return

    def _second_tap() -> None:
        if stop.is_set():
            return
        _tap_once(name, stop)

    _tap_once(name, stop, on_released=_second_tap)


def _hold(name: str, duration_ms: int, stop: threading.Event) -> None:
    """Press a key/button and hold for duration_ms ms, then release."""
    if stop.is_set():
        return
    n = (name or "").strip().lower()
    if n in _MOUSE_BTN:
        btn = _MOUSE_BTN[n]
        try:
            _MOUSE.press(btn)
            _sleep_interruptible(duration_ms / 1000.0, stop)
            _MOUSE.release(btn)
        except Exception:
            logger.debug("mouse hold failed: %s", n, exc_info=True)
        return
    k = _resolve_key(n)
    if k is None:
        logger.debug("macro_player: unknown key token %r — skipped", n)
        return
    try:
        _KB.press(k)
        _sleep_interruptible(duration_ms / 1000.0, stop)
        _KB.release(k)
    except Exception:
        logger.debug("key hold failed: %s", n, exc_info=True)


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


def _spam_tap_for_duration(
    name: str,
    total_duration_ms: int,
    interval_ms: int,
    stop: threading.Event,
) -> None:
    """
    Spam taps every interval_ms (inclusive at t=0) until chain duration elapses.
    Example: 1000ms @ 100ms -> floor(1000/100)+1 = 11 taps.
    Visual replay fires are scheduled separately by _execute_node_list.
    """
    if stop.is_set():
        return
    total_ms = max(0, int(total_duration_ms or 0))
    if total_ms <= 0:
        return
    step_ms = max(1, int(interval_ms or _DEFAULT_CHAIN_SPAM_INTERVAL_MS))
    tap_count = math.floor(total_ms / step_ms) + 1
    started = time.perf_counter()
    for i in range(tap_count):
        if stop.is_set():
            break
        _tap(name, stop, double_tap=False)
        if stop.is_set():
            break
        if i >= tap_count - 1:
            # After the last tap, wait out the remainder of the chain window.
            end_at = started + (total_ms / 1000.0)
            remaining = end_at - time.perf_counter()
            if remaining > 0:
                _sleep_interruptible(remaining, stop)
        else:
            next_at = started + ((i + 1) * (step_ms / 1000.0))
            remaining = next_at - time.perf_counter()
            if remaining > 0:
                _sleep_interruptible(remaining, stop)


def _execute_node_list(
    nodes: list,
    stop: threading.Event,
    chain_spam_interval_ms: int | None,
    replay: _ReplayState | None = None,
) -> None:
    """
    Execute nodes in order, collapsing adjacent same-key press+wait nodes into chain spam.
    When replay is set, each logical step fires replay.fire() at the correct wall-clock time
    so the visual timeline advances in sync without depending on combo-engine tick polling.
    """
    i = 0
    n = len(nodes)
    while i < n and not stop.is_set():
        if chain_spam_interval_ms is None:
            _execute(nodes[i], stop, chain_spam_interval_ms, replay)
            i += 1
            continue

        info = _extract_press_wait_info(nodes[i])
        if info is None:
            _execute(nodes[i], stop, chain_spam_interval_ms, replay)
            i += 1
            continue

        key, total_wait_ms = info
        j = i + 1
        while j < n:
            nxt = _extract_press_wait_info(nodes[j])
            if nxt is None or nxt[0] != key:
                break
            total_wait_ms += nxt[1]
            j += 1

        if j - i > 1:
            spam_start = time.perf_counter()

            if replay is not None:
                # Schedule one replay.fire per original node at its cumulative wall-clock offset
                # so the visual timeline advances at the right time independently of the spam.
                cumulative_s = 0.0
                for k in range(i, j):
                    node_info = _extract_press_wait_info(nodes[k])
                    node_key = node_info[0] if node_info else key
                    fire_at = spam_start + cumulative_s
                    if cumulative_s <= 0.0:
                        # First node: fire inline so there is no scheduler delay.
                        replay.fire(node_key, fire_at)
                    else:
                        def _fire(
                            nk: str = node_key, fa: float = fire_at
                        ) -> None:
                            replay.fire(nk, fa)

                        _DELAY_SCHEDULER.schedule(cumulative_s, _fire)
                    if node_info:
                        cumulative_s += node_info[1] / 1000.0

            _spam_tap_for_duration(key, total_wait_ms, chain_spam_interval_ms, stop)
            # After collapsing a chain, give the combo engine's tick loop time to
            # settle the last soft-wait before the next key is dispatched.
            if not stop.is_set() and j < n:
                _sleep_interruptible(_POST_SPAM_PAD_MS / 1000.0, stop)
        else:
            _execute(nodes[i], stop, chain_spam_interval_ms, replay)
        i = j


def _execute(
    node,
    stop: threading.Event,
    chain_spam_interval_ms: int | None,
    replay: _ReplayState | None = None,
) -> None:
    """Recursively execute one AST node, checking stop at each step."""
    if stop.is_set():
        return
    if isinstance(node, PressNode):
        _tap(node.key, stop)
        if replay is not None:
            replay.fire(node.key)
    elif isinstance(node, HoldNode):
        # Karaoke: press + release markers so the UI can run hold_begin/hold_end (same pattern as HoldWithBody).
        if replay is not None:
            replay.fire(node.key)
        _hold(node.key, node.duration_ms, stop)
        if replay is not None:
            replay.fire(node.key)
    elif isinstance(node, HoldWithBodyNode):
        # Press hold key, execute inner body steps (timed relative to hold start), then release.
        # Body waits encode timing from hold start, e.g. {wait:0.15s, q} means press q
        # at 0.15s after hold key down.
        n_key = (node.key or "").strip().lower()
        hold_start = time.perf_counter()
        if n_key in _MOUSE_BTN:
            try:
                _MOUSE.press(_MOUSE_BTN[n_key])
            except Exception:
                logger.debug("hold_with_body: mouse press failed: %s", n_key, exc_info=True)
        else:
            hk = _resolve_key(n_key)
            if hk is not None:
                try:
                    _KB.press(hk)
                except Exception:
                    logger.debug("hold_with_body: key press failed: %s", n_key, exc_info=True)
        if replay is not None:
            # Emit holder key-down as its own replay event so karaoke timing/logging
            # can track the full hold-with-body timeline (start, inner keys, release).
            replay.fire(node.key)
        # Execute inner body (waits + presses) while holder is down
        if not stop.is_set():
            _execute_node_list(list(node.body), stop, None, replay)
        # Sleep any remaining hold time beyond what body steps consumed
        if not stop.is_set():
            elapsed_s = time.perf_counter() - hold_start
            remaining_s = (node.duration_ms / 1000.0) - elapsed_s
            if remaining_s > 0:
                _sleep_interruptible(remaining_s, stop)
        # Release hold key
        if n_key in _MOUSE_BTN:
            try:
                _MOUSE.release(_MOUSE_BTN[n_key])
            except Exception:
                pass
        else:
            hk = _resolve_key(n_key)
            if hk is not None:
                try:
                    _KB.release(hk)
                except Exception:
                    pass
        if replay is not None:
            replay.fire(node.key)
    elif isinstance(node, WaitNode):
        _sleep_interruptible(node.duration_ms / 1000.0, stop)
    elif isinstance(node, SequenceNode):
        _execute_node_list(list(node.steps), stop, chain_spam_interval_ms, replay)
    elif isinstance(node, GroupNode):
        # Execute group items in their defined (left-to-right) order for macro playback.
        _execute_node_list(list(node.items), stop, chain_spam_interval_ms, replay)


class MacroPlayer:
    """
    Plays back a combo token list by emitting keyboard/mouse events.
    Thread-safe; at most one playback runs at a time.
    """

    def __init__(
        self,
        on_status: Callable[[str, str], None] | None = None,
        on_step: Callable[[str, float, float], None] | None = None,
    ):
        """
        on_status(text, color): optional callback on the playback thread for start/complete/stop.
        on_step(key, step_ms, total_ms): optional karaoke callback, fired once per logical combo
            step at the wall-clock moment the key is dispatched.  Must be thread-safe.
        """
        self._on_status = on_status
        self._on_step = on_step
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._chain_spam_interval_ms: int | None = _DEFAULT_CHAIN_SPAM_INTERVAL_MS

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

    def start(self, tokens: list[str]) -> bool:
        """
        Begin macro playback from a list of normalized combo tokens.
        Returns False (and does nothing) if playback is already in progress.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if not tokens:
                return False
            self._stop.clear()
            ast = expanded_ast_from_tokens(tokens)
            if not ast:
                return False
            self._thread = threading.Thread(
                target=self._run, args=(ast,), name="MacroPlayer", daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Signal playback to stop and let any pending tap-release callbacks run."""
        self._stop.set()
        # Do not cancel scheduled releases — let them run so keys are not left pressed.

    def _run(self, ast: list) -> None:
        stopped_early = False
        end_period = _windows_timer_1ms_enter()
        try:
            with self._lock:
                interval_ms = self._chain_spam_interval_ms
            start_time = time.perf_counter()
            replay = None
            replay_q: queue.Queue[tuple[str, float, float]] | None = None
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
                        k, sm, tm = item
                        try:
                            on_step(k, sm, tm)
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

            try:
                _execute_node_list(list(ast), self._stop, interval_ms, replay)
            finally:
                if replay_q is not None:
                    replay_q.put(None)
                    if replay_worker_thread is not None:
                        replay_worker_thread.join(timeout=5.0)
            if self._stop.is_set():
                stopped_early = True
        except Exception:
            logger.exception("MacroPlayer: unhandled error during playback")
            stopped_early = True
        finally:
            end_period()
            if stopped_early:
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
