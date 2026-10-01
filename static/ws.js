// Shared backend connection for every page (tracker, Characters, key overlay).
// connectTracker({ onMessage, onOpen, onClose }) opens ws://localhost:8765 and reconnects with backoff,
// so pages and OBS overlays recover on their own after the backend restarts.
(function () {
    const WS_URL = 'ws://localhost:8765';
    const RECONNECT_START_MS = 500;
    const RECONNECT_MAX_MS = 5000;

    function connectTracker(handlers) {
        const h = handlers || {};
        const conn = {
            socket: null,
            /** Send { type, ...payload }; dropped quietly while disconnected. */
            send(type, payload = {}) {
                const s = conn.socket;
                if (!s || s.readyState !== WebSocket.OPEN) return false;
                s.send(JSON.stringify({ type, ...payload }));
                return true;
            },
            isOpen() {
                return !!conn.socket && conn.socket.readyState === WebSocket.OPEN;
            },
        };
        let delayMs = RECONNECT_START_MS;

        function open() {
            let socket;
            try {
                socket = new WebSocket(WS_URL);
            } catch (_) {
                setTimeout(open, delayMs);
                return;
            }
            conn.socket = socket;
            socket.onopen = () => {
                delayMs = RECONNECT_START_MS;
                if (h.onOpen) h.onOpen();
            };
            socket.onclose = () => {
                if (conn.socket !== socket) return;
                if (h.onClose) h.onClose(delayMs);
                setTimeout(open, delayMs);
                delayMs = Math.min(delayMs * 2, RECONNECT_MAX_MS);
            };
            socket.onmessage = (event) => {
                let msg;
                try { msg = JSON.parse(event.data); } catch (_) { return; }
                if (h.onMessage) h.onMessage(msg);
            };
        }

        open();
        return conn;
    }

    window.connectTracker = connectTracker;
})();
