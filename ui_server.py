from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer, ThreadingMixIn
from typing import Any, Callable

import websockets
from pynput import keyboard, mouse

from combo_engine import ComboTrackerEngine
from macro_player import MacroPlayer
from transcriber import Transcriber

logger = logging.getLogger(__name__)

TRANSCRIBE_ESC_KEY = "esc"


def _effective_transcribe_min_wait_s() -> float:
    """Seconds: omit wait tokens when gap is shorter than this; 0 keeps all gaps."""
    if not getattr(engine, "transcribe_strip_wait_under_enabled", False):
        return 0.0
    raw = (getattr(engine, "transcribe_strip_wait_under_ms", "") or "").strip()
    if not raw:
        return 0.0
    try:
        ms = int(raw, 10)
    except ValueError:
        return 0.0
    if ms <= 0:
        return 0.0
    return ms / 1000.0


def _sync_transcriber_from_engine() -> None:
    transcriber.set_valid_keys(getattr(engine, "transcribe_valid_keys", "") or "")
    transcriber.set_min_wait_s(_effective_transcribe_min_wait_s())


def _normalized_transcribe_start_key() -> str:
    raw = (getattr(engine, "transcribe_start_key", "") or "").strip().lower()
    return raw if raw else "f"


def _normalized_macro_start_key() -> str:
    raw = (getattr(engine, "macro_start_key", "") or "").strip().lower()
    return raw if raw else "f8"


def _normalized_macro_stop_key() -> str:
    raw = (getattr(engine, "macro_stop_key", "") or "").strip().lower()
    return raw  # empty string = no stop key configured (only Esc stops)

transcribe_mode_enabled = False
macro_mode_enabled = False


HOST_HTTP = "localhost"
PORT_HTTP = 8080
HOST_WS = "localhost"
PORT_WS = 8765


class ThreadedHTTPServer(ThreadingMixIn, TCPServer):
    """Handle each HTTP request in its own thread so multiple tabs can reload without blocking."""


def serve_static() -> None:
    # When packaged with PyInstaller, static files are in sys._MEIPASS
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    static_dir = (base / "static").resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(static_dir), **kwargs)

    with ThreadedHTTPServer((HOST_HTTP, PORT_HTTP), Handler) as httpd:
        print(f"HTTP server running at http://{HOST_HTTP}:{PORT_HTTP}")
        httpd.serve_forever()


connected_clients: set[websockets.WebSocketServerProtocol] = set()


async def broadcast_dict(payload: dict[str, Any]) -> None:
    if not connected_clients:
        return
    msg = json.dumps(payload)
    clients = list(connected_clients)
    results = await asyncio.gather(*(c.send(msg) for c in clients), return_exceptions=True)
    # Drop broken clients
    for c, r in zip(clients, results):
        if isinstance(r, Exception):
            try:
                connected_clients.discard(c)
            except Exception:
                pass


def make_threadsafe_emitter(loop: asyncio.AbstractEventLoop) -> Callable[[dict[str, Any]], None]:
    def emit(payload: dict[str, Any]) -> None:
        try:
            asyncio.run_coroutine_threadsafe(broadcast_dict(payload), loop)
        except Exception:
            # Don't let UI emit plumbing crash the engine, but keep breadcrumbs.
            logger.debug("Emitter failed to schedule broadcast", exc_info=True)

    return emit


def _safe_json_load(s: str) -> dict[str, Any] | None:
    try:
        out: object = json.loads(s)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


async def ws_handler(
    websocket: websockets.WebSocketServerProtocol,
    _path: str | None = None,
) -> None:
    global transcribe_mode_enabled, macro_mode_enabled
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")

    # Send initial state
    await websocket.send(json.dumps(engine.init_payload()))

    try:
        async for message in websocket:
            msg = _safe_json_load(message)
            if not isinstance(msg, dict):
                continue

            mtype = msg.get("type")
            if mtype == "select_combo":
                engine.set_active_combo(str(msg.get("name") or ""))
            elif mtype == "save_combo":
                ok, err = engine.save_or_update_combo(
                    name=str(msg.get("name") or ""),
                    inputs=str(msg.get("inputs") or ""),
                    enders=str(msg.get("enders") or ""),
                    expected_time=str(msg.get("expected_time") or ""),
                    user_difficulty=str(msg.get("user_difficulty") or ""),
                    step_display_mode=str(msg.get("step_display_mode") or ""),
                    key_images=msg.get("key_images"),
                    demo_video=str(msg.get("demo_video") or ""),
                    target_game=str(msg.get("target_game") or ""),
                    ww_team_id=str(msg.get("ww_team_id") or ""),
                )
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "save_team":
                ok, err = engine.save_or_update_ww_team(
                    team_id=str(msg.get("team_id") or ""),
                    team_name=str(msg.get("team_name") or ""),
                    slot1=str(msg.get("slot1") or ""),
                    slot2=str(msg.get("slot2") or ""),
                    slot3=str(msg.get("slot3") or ""),
                )
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "save_character":
                ok, err = engine.save_ww_character(
                    name=str(msg.get("name") or ""),
                    swap_image=str(msg.get("swap_image") or ""),
                    lmb_image=str(msg.get("lmb_image") or ""),
                    ability_images=msg.get("ability_images"),
                )
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "delete_character":
                ok, err = engine.delete_ww_character(str(msg.get("name") or ""))
                if not ok and err:
                    await websocket.send(json.dumps({"type": "alert_notice", "text": err}))
            elif mtype == "update_ww_dash":
                engine.update_ww_dash(str(msg.get("dash_image") or ""))
            elif mtype == "select_team":
                # Stateless team selection: accepts target_game parameter
                target_game = str(msg.get("target_game") or "").strip().lower()
                engine.select_team_stateless(
                    team_id=str(msg.get("team_id") or ""),
                    target_game=target_game
                )
            elif mtype == "update_target_game":
                # Stateless target game update (doesn't persist to JSON)
                target_game = str(msg.get("target_game") or "").strip().lower()
                engine.update_target_game_stateless(target_game)
            elif mtype == "delete_team":
                ok, err = engine.delete_ww_team(str(msg.get("team_id") or ""))
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "delete_combo":
                ok, err = engine.delete_combo(str(msg.get("name") or ""))
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "new_combo":
                engine.new_combo()
            elif mtype == "reorder_timeline_step":
                try:
                    from_idx = int(msg.get("from_step_index"))
                except Exception:
                    from_idx = -1
                raw_before = msg.get("before_step_index")
                before_idx = None
                if raw_before is not None:
                    try:
                        before_idx = int(raw_before)
                    except Exception:
                        before_idx = None
                ok, err = engine.reorder_timeline_step(from_idx, before_idx)
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "delete_timeline_step":
                raw_indices = msg.get("step_indices")
                indices: list[int] = []
                if isinstance(raw_indices, list):
                    for x in raw_indices:
                        try:
                            indices.append(int(x))
                        except Exception:
                            continue
                ok, err = engine.delete_timeline_steps(indices)
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
            elif mtype == "set_transcribe_mode":
                transcribe_mode_enabled = bool(msg.get("enabled", False))
                if transcribe_mode_enabled and macro_mode_enabled:
                    macro_mode_enabled = False
                    if macro_player.is_running():
                        macro_player.stop()
                valid_keys_str = str(msg.get("valid_keys") or "").strip()
                start_raw = str(msg.get("start_key") or "").strip().lower()
                engine.transcribe_start_key = start_raw if start_raw else "f"
                engine.transcribe_valid_keys = valid_keys_str
                engine.transcribe_strip_wait_under_enabled = bool(msg.get("strip_wait_under_enabled", False))
                engine.transcribe_strip_wait_under_ms = str(msg.get("strip_wait_under_ms") or "").strip()
                _sync_transcriber_from_engine()
                engine.save_combos()
            elif mtype == "set_no_fail":
                engine.set_no_fail_mode(bool(msg.get("enabled", False)))
                engine.save_combos()
            elif mtype == "set_macro_mode":
                macro_mode_enabled = bool(msg.get("enabled", False))
                if macro_mode_enabled and transcribe_mode_enabled:
                    transcribe_mode_enabled = False
                    if transcriber.is_recording():
                        transcriber.stop()
                start_raw = str(msg.get("start_key") or "").strip().lower()
                stop_raw = str(msg.get("stop_key") or "").strip().lower()
                spam_raw = str(msg.get("spam_interval_ms") or "").strip()
                engine.macro_start_key = start_raw if start_raw else "f8"
                engine.macro_stop_key = stop_raw
                if spam_raw:
                    try:
                        spam_ms = int(spam_raw)
                    except Exception:
                        spam_ms = int(getattr(engine, "macro_spam_interval_ms", 100) or 100)
                    engine.macro_spam_interval_ms = max(1, spam_ms)
                else:
                    engine.macro_spam_interval_ms = None
                macro_player.set_chain_spam_interval_ms(engine.macro_spam_interval_ms)
                if not macro_mode_enabled and macro_player.is_running():
                    macro_player.stop()
                engine.save_combos()
            elif mtype == "clear_history":
                engine.clear_history_and_stats()
            elif mtype == "clear_history_all":
                engine.clear_all_history_and_stats()
            elif mtype == "load_combos_from_json":
                path = engine.save_path
                if not path.exists():
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "status",
                                "text": f"No combos.json found at {path}. Place the file next to ComboTracker.exe and try again.",
                                "color": "fail",
                            }
                        )
                    )
                else:
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "status",
                                    "text": "combos.json is not valid JSON. Fix the file and try again.",
                                    "color": "fail",
                                }
                            )
                        )
                    except OSError as e:
                        logger.warning("Could not read combos.json: %s", e)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "status",
                                    "text": "Could not read combos.json.",
                                    "color": "fail",
                                }
                            )
                        )
                    else:
                        try:
                            engine.load_combos()
                            _sync_transcriber_from_engine()
                            await broadcast_dict(engine.init_payload())
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "status",
                                        "text": f"Loaded combos from {path.name}.",
                                        "color": "success",
                                    }
                                )
                            )
                        except Exception:
                            logger.exception("load_combos_from_json failed")
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "status",
                                        "text": "Failed to apply combos.json.",
                                        "color": "fail",
                                    }
                                )
                            )
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(connected_clients)}")


def run_ws_server() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Hook engine emitter to this loop (thread-safe)
    engine.set_emitter(make_threadsafe_emitter(loop))

    async def _main():
        async with websockets.serve(ws_handler, HOST_WS, PORT_WS):
            print(f"WebSocket server running at ws://{HOST_WS}:{PORT_WS}")
            await asyncio.Future()  # run forever

    loop.run_until_complete(_main())


# Mouse: merge rapid same-button clicks (down/up/down/up within this window → one logical click)
MOUSE_CLICK_MERGE_S = 0.2


def start_input_listeners() -> tuple[keyboard.Listener, mouse.Listener]:
    # Windows: hold sends repeated key_down and one key_up on release. Coalesce so transcriber
    # sees exactly one key_down and one key_up per physical key (ignore repeat key_downs).
    transcribe_keys_held: set[str] = set()
    # Mouse: merge rapid same-button clicks into one logical click
    transcribe_mouse_last_up: dict[str, float] = {}
    transcribe_mouse_merging: set[str] = set()

    def on_key_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        event_time = time.perf_counter()
        input_name = engine.normalize_key(key)

        # Macro output is matched by origin before hotkey handling. This prevents
        # a synthetic Esc/F9 from stopping its own macro while physical stop keys
        # remain responsive.
        if macro_player.consume_synthetic_event(input_name, True):
            return

        # Esc always hard-stops the macro and immediately resets the visual state.
        if input_name == TRANSCRIBE_ESC_KEY and macro_player.is_running():
            macro_player.stop()
            engine.cancel_attempt()
            return

        if transcribe_mode_enabled:
            if not transcriber.is_recording():
                if input_name == TRANSCRIBE_ESC_KEY:
                    return
                if input_name == _normalized_transcribe_start_key():
                    transcribe_keys_held.clear()
                    transcribe_mouse_last_up.clear()
                    transcribe_mouse_merging.clear()
                    engine.new_combo()
                    transcriber.start()
                    transcriber.key_down(input_name, event_time)
                    transcriber.key_up(input_name, event_time)
                    _push_transcription_preview()
                return
            if input_name == TRANSCRIBE_ESC_KEY:
                transcriber.stop()
                transcribe_keys_held.clear()
                transcribe_mouse_last_up.clear()
                transcribe_mouse_merging.clear()
                return
            if transcriber.is_valid_key(input_name):
                if input_name in transcribe_keys_held:
                    return  # repeat key_down (Windows hold); ignore
                transcribe_keys_held.add(input_name)
                transcriber.key_down(input_name, event_time)
            return
        # Macro mode start/stop hotkeys (only active when not transcribing).
        if macro_mode_enabled:
            start_k = _normalized_macro_start_key()
            stop_k = _normalized_macro_stop_key()
            if input_name == start_k and not macro_player.is_running():
                _start_macro(requested_at=event_time)
                return  # swallow the start key so it doesn't advance the combo
            if stop_k and input_name == stop_k and macro_player.is_running():
                macro_player.stop()
                return  # swallow the stop key

        # While the macro is replaying, its synthetic keystrokes must not feed back
        # into the detection engine — the karaoke replay_accept path handles visuals.
        if macro_player.is_running():
            return

        if input_name == TRANSCRIBE_ESC_KEY:
            engine.cancel_attempt()
            return
        engine.process_press(input_name, event_time=event_time)

    def on_key_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        event_time = time.perf_counter()
        input_name = engine.normalize_key(key)
        if macro_player.consume_synthetic_event(input_name, False):
            return
        if transcribe_mode_enabled:
            if transcriber.is_recording() and input_name != TRANSCRIBE_ESC_KEY and transcriber.is_valid_key(input_name):
                if input_name in transcribe_keys_held:
                    transcribe_keys_held.discard(input_name)
                    transcriber.key_up(input_name, event_time)
                    _push_transcription_preview()
            return
        if input_name == TRANSCRIBE_ESC_KEY:
            return  # Esc cancel already handled on press
        if macro_player.is_running():
            return
        engine.process_release(input_name, event_time=event_time)

    def on_mouse_click(_x: float, _y: float, button: mouse.Button, pressed: bool) -> None:
        event_time = time.perf_counter()
        btn = engine.normalize_mouse(button)
        if macro_player.consume_synthetic_event(btn, pressed):
            return
        if transcribe_mode_enabled:
            if transcriber.is_recording() and transcriber.is_valid_key(btn):
                now = event_time
                if pressed:
                    # Merge rapid same-button clicks: if this down is within MOUSE_CLICK_MERGE_S of last up, skip it (and the next up)
                    if btn in transcribe_mouse_last_up and (now - transcribe_mouse_last_up[btn]) < MOUSE_CLICK_MERGE_S:
                        transcribe_mouse_merging.add(btn)
                        return
                    transcriber.key_down(btn, now)
                else:
                    if btn in transcribe_mouse_merging:
                        transcribe_mouse_merging.discard(btn)
                        return
                    transcribe_mouse_last_up[btn] = now
                    transcriber.key_up(btn, now)
                    _push_transcription_preview()
            return
        if macro_player.is_running():
            return
        if pressed:
            engine.process_press(btn, event_time=event_time)
        else:
            engine.process_release(btn, event_time=event_time)

    kl = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    ml = mouse.Listener(on_click=on_mouse_click)
    kl.start()
    ml.start()
    return kl, ml


engine = ComboTrackerEngine()


def _on_macro_status(text: str, color: str) -> None:
    if color in ("neutral", "fail"):
        # Macro stopped or output failed — reset visual combo state so the timeline doesn't
        # stay mid-progress forever.
        engine.cancel_attempt()
    engine._send({"type": "status", "text": text, "color": color})


def _on_macro_step(key: str, step_ms: float, total_ms: float, pressed: bool) -> None:
    """Karaoke callback: advance the visual combo state for each replayed step."""
    engine.replay_accept(key, step_ms, total_ms, pressed=pressed)


macro_player = MacroPlayer(
    on_status=_on_macro_status,
    on_step=_on_macro_step,
    profile_log_dir=engine.data_dir / "logs" / "macro_profiles",
)
macro_player.set_chain_spam_interval_ms(getattr(engine, "macro_spam_interval_ms", 100))


def _start_macro(*, requested_at: float | None = None) -> None:
    """Begin macro playback using the current active combo tokens."""
    tokens = list(getattr(engine, "active_combo_tokens", []) or [])
    if not tokens:
        engine._send({"type": "status", "text": "No combo loaded — select or save a combo first.", "color": "fail"})
        return
    started = macro_player.start(
        tokens,
        requested_at=requested_at,
        combo_name=engine.active_combo_name,
    )
    if not started:
        pass  # already running — silently ignore per spec


def _on_transcription_done(transcript: str) -> None:
    engine.send_transcription_result(transcript)


def _push_transcription_preview() -> None:
    """Update web UI inputs + timeline from transcriber state (call after each committed key_up)."""
    if not transcribe_mode_enabled or not transcriber.is_recording():
        return
    engine.send_transcription_result(transcriber.current_transcript())


transcriber = Transcriber(on_stop=_on_transcription_done)
# Sync transcriber with persisted transcribe settings (engine.load_combos in ComboTrackerEngine.__init__)
_sync_transcriber_from_engine()


def _warn_if_windows_non_elevated() -> None:
    """When running from source on Windows, elevated games need an elevated listener."""
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return
    try:
        import ctypes

        if not bool(ctypes.windll.shell32.IsUserAnAdmin()):
            print(
                "Warning: ComboTracker is not running as Administrator. "
                "If your game runs as Administrator, keystrokes may not register in-game. "
                "Use an elevated console or the packaged ComboTracker.exe (requests Admin on launch).",
                file=sys.stderr,
            )
    except Exception:
        pass


def setup_logging() -> None:
    """
    Minimal app-level logging.
    Use `COMBOTRACKER_LOG_LEVEL` env var (e.g. DEBUG/INFO/WARNING).
    """
    level_name = str(os.environ.get("COMBOTRACKER_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _warn_if_windows_non_elevated()
    setup_logging()
    # Static UI
    http_thread = threading.Thread(target=serve_static, daemon=True)
    http_thread.start()

    # WebSocket server (owns its asyncio loop)
    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()

    # Input listeners
    kl, ml = start_input_listeners()

    # Tick loop: advance time-based steps (waits / group waits) without requiring another input event.
    # This makes wait tiles complete/turn green automatically when their timer elapses.
    def tick_loop() -> None:
        last_log = 0.0
        while True:
            try:
                engine.tick()
            except Exception:
                # Avoid spamming logs at ~50Hz if something starts throwing.
                now = time.time()
                if now - last_log >= 5.0:
                    last_log = now
                    logger.exception("engine.tick() failed")
            time.sleep(0.02)  # ~50Hz

    tick_thread = threading.Thread(target=tick_loop, daemon=True)
    tick_thread.start()

    print("Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            kl.stop()
        except Exception:
            pass
        try:
            ml.stop()
        except Exception:
            pass
        engine.save_combos()


if __name__ == "__main__":
    main()
