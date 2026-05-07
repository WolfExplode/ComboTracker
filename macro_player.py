"""
MacroPlayer: replays a parsed combo token list by emitting real keyboard/mouse events.
Uses pynput.keyboard.Controller and pynput.mouse.Controller.

Design:
- Runs in a daemon thread; at most one playback at a time.
- Interruptible at any point via a threading.Event (Esc hard-stop).
- Sleeps in 10ms chunks so the stop signal is acted on quickly.
- Group nodes are executed left-to-right (defined order) for macro playback.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from pynput import keyboard, mouse

from parser import (
    expanded_ast_from_tokens,
    GroupNode,
    HoldNode,
    PressNode,
    SequenceNode,
    WaitNode,
)

logger = logging.getLogger(__name__)

# Duration (ms) a "tap" key is held before releasing.
_TAP_HOLD_MS = 50

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
    """Sleep in 10ms slices so stop signals are honoured quickly."""
    end = time.perf_counter() + seconds
    while not stop.is_set():
        remaining = end - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(0.01, remaining))


def _tap(name: str, stop: threading.Event) -> None:
    """Press and release a key (or mouse button) with a short hold duration."""
    if stop.is_set():
        return
    n = (name or "").strip().lower()
    if n in _MOUSE_BTN:
        btn = _MOUSE_BTN[n]
        try:
            _MOUSE.press(btn)
            _sleep_interruptible(_TAP_HOLD_MS / 1000.0, stop)
            _MOUSE.release(btn)
        except Exception:
            logger.debug("mouse tap failed: %s", n, exc_info=True)
        return
    k = _resolve_key(n)
    if k is None:
        logger.debug("macro_player: unknown key token %r — skipped", n)
        return
    try:
        _KB.press(k)
        _sleep_interruptible(_TAP_HOLD_MS / 1000.0, stop)
        if not stop.is_set():
            _KB.release(k)
    except Exception:
        logger.debug("key tap failed: %s", n, exc_info=True)


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


def _execute(node, stop: threading.Event) -> None:
    """Recursively execute one AST node, checking stop at each step."""
    if stop.is_set():
        return
    if isinstance(node, PressNode):
        _tap(node.key, stop)
    elif isinstance(node, HoldNode):
        _hold(node.key, node.duration_ms, stop)
    elif isinstance(node, WaitNode):
        _sleep_interruptible(node.duration_ms / 1000.0, stop)
    elif isinstance(node, SequenceNode):
        for step in node.steps:
            if stop.is_set():
                break
            _execute(step, stop)
    elif isinstance(node, GroupNode):
        # Execute group items in their defined (left-to-right) order for macro playback.
        for item in node.items:
            if stop.is_set():
                break
            _execute(item, stop)


class MacroPlayer:
    """
    Plays back a combo token list by emitting keyboard/mouse events.
    Thread-safe; at most one playback runs at a time.
    """

    def __init__(self, on_status: Callable[[str, str], None] | None = None):
        """
        on_status(text, color): optional callback invoked on the playback thread
        when playback starts, completes, or is stopped.  Must be thread-safe.
        """
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

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
        """Signal playback to stop. Non-blocking — sets the stop event."""
        self._stop.set()

    def _run(self, ast: list) -> None:
        self._notify("Macro replaying\u2026", "recording")
        stopped_early = False
        try:
            for node in ast:
                if self._stop.is_set():
                    stopped_early = True
                    break
                _execute(node, self._stop)
            if self._stop.is_set():
                stopped_early = True
        except Exception:
            logger.exception("MacroPlayer: unhandled error during playback")
            stopped_early = True
        finally:
            if stopped_early:
                self._notify("Macro stopped.", "neutral")
            else:
                self._notify("Macro complete.", "success")

    def _notify(self, text: str, color: str) -> None:
        if self._on_status:
            try:
                self._on_status(text, color)
            except Exception:
                pass
