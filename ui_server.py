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
from transcriber import Transcriber

logger = logging.getLogger(__name__)

TRANSCRIBE_ESC_KEY = "esc"


def _normalized_transcribe_start_key() -> str:
    raw = (getattr(engine, "transcribe_start_key", "") or "").strip().lower()
    return raw if raw else "f"

transcribe_mode_enabled = False


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
                    dash_image=str(msg.get("dash_image") or ""),
                    swap_images=msg.get("swap_images"),
                    lmb_images=msg.get("lmb_images"),
                    ability_images=msg.get("ability_images"),
                )
                if not ok and err:
                    await websocket.send(json.dumps({"type": "status", "text": err, "color": "fail"}))
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
            elif mtype == "set_transcribe_mode":
                global transcribe_mode_enabled
                transcribe_mode_enabled = bool(msg.get("enabled", False))
                valid_keys_str = str(msg.get("valid_keys") or "").strip()
                start_raw = str(msg.get("start_key") or "").strip().lower()
                engine.transcribe_start_key = start_raw if start_raw else "f"
                transcriber.set_valid_keys(valid_keys_str)
                engine.transcribe_valid_keys = valid_keys_str
                engine.save_combos()
            elif mtype == "set_no_fail":
                engine.set_no_fail_mode(bool(msg.get("enabled", False)))
                engine.save_combos()
            elif mtype == "clear_history":
                engine.clear_history_and_stats()
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
        input_name = engine.normalize_key(key)
        if transcribe_mode_enabled:
            if not transcriber.is_recording():
                if input_name == TRANSCRIBE_ESC_KEY:
                    return
                if input_name == _normalized_transcribe_start_key():
                    transcribe_keys_held.clear()
                    transcribe_mouse_last_up.clear()
                    transcribe_mouse_merging.clear()
                    now = time.perf_counter()
                    transcriber.start()
                    transcriber.key_down(input_name, now)
                    transcriber.key_up(input_name, now)
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
                transcriber.key_down(input_name, time.perf_counter())
            return
        if input_name == TRANSCRIBE_ESC_KEY:
            engine.cancel_attempt()
            return
        engine.process_press(input_name)

    def on_key_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        input_name = engine.normalize_key(key)
        if transcribe_mode_enabled:
            if transcriber.is_recording() and input_name != TRANSCRIBE_ESC_KEY and transcriber.is_valid_key(input_name):
                if input_name in transcribe_keys_held:
                    transcribe_keys_held.discard(input_name)
                    transcriber.key_up(input_name, time.perf_counter())
            return
        if input_name == TRANSCRIBE_ESC_KEY:
            return  # Esc cancel already handled on press
        engine.process_release(input_name)

    def on_mouse_click(_x: float, _y: float, button: mouse.Button, pressed: bool) -> None:
        btn = engine.normalize_mouse(button)
        if transcribe_mode_enabled:
            if transcriber.is_recording() and transcriber.is_valid_key(btn):
                now = time.perf_counter()
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
            return
        if pressed:
            engine.process_press(btn)
        else:
            engine.process_release(btn)

    kl = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    ml = mouse.Listener(on_click=on_mouse_click)
    kl.start()
    ml.start()
    return kl, ml


engine = ComboTrackerEngine()


def _on_transcription_done(transcript: str) -> None:
    engine.new_combo()
    engine.send_transcription_result(transcript)


transcriber = Transcriber(on_stop=_on_transcription_done)
# Sync transcriber with persisted valid keys (loaded in engine via load_combos)
transcriber.set_valid_keys(getattr(engine, "transcribe_valid_keys", "") or "")


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