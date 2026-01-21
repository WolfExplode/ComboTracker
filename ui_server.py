import asyncio
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

import websockets
from pynput import keyboard, mouse

from combo_engine import ComboTrackerEngine


HOST_HTTP = "localhost"
PORT_HTTP = 8080
HOST_WS = "localhost"
PORT_WS = 8765


def serve_static():
    static_dir = (Path(__file__).resolve().parent / "static").resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

    with TCPServer((HOST_HTTP, PORT_HTTP), Handler) as httpd:
        print(f"HTTP server running at http://{HOST_HTTP}:{PORT_HTTP}")
        httpd.serve_forever()


connected_clients: set[websockets.WebSocketServerProtocol] = set()


async def broadcast_dict(payload: dict):
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


def make_threadsafe_emitter(loop: asyncio.AbstractEventLoop):
    def emit(payload: dict):
        try:
            asyncio.run_coroutine_threadsafe(broadcast_dict(payload), loop)
        except Exception:
            pass

    return emit


def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


async def ws_handler(websocket, _path=None):
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
                engine.set_active_ww_team(str(msg.get("team_id") or ""))
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
            elif mtype == "clear_history":
                engine.clear_history_and_stats()
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(connected_clients)}")


def run_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Hook engine emitter to this loop (thread-safe)
    engine.set_emitter(make_threadsafe_emitter(loop))

    async def _main():
        async with websockets.serve(ws_handler, HOST_WS, PORT_WS):
            print(f"WebSocket server running at ws://{HOST_WS}:{PORT_WS}")
            await asyncio.Future()  # run forever

    loop.run_until_complete(_main())


def start_input_listeners():
    # Input callbacks run off-thread; they call engine directly.
    def on_key_press(key):
        engine.process_press(engine.normalize_key(key))

    def on_key_release(key):
        engine.process_release(engine.normalize_key(key))

    def on_mouse_click(_x, _y, button, pressed):
        btn = engine.normalize_mouse(button)
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


def main():
    # Static UI
    http_thread = threading.Thread(target=serve_static, daemon=True)
    http_thread.start()

    # WebSocket server (owns its asyncio loop)
    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()

    # Input listeners
    kl, ml = start_input_listeners()

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