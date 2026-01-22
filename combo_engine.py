import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from Game_Wuthering_Waves import WutheringWavesGame
import combo_engine_ui as ui
from combo_engine_ui import Status
from persistence import load_engine_state, save_engine_state

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
        self.active_combo_steps: list[dict[str, Any]] = []

        self.current_index = 0
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
    # Normalization helpers
    # -------------------------

    def normalize_key(self, key) -> str:
        """
        Normalize pynput keyboard events to our internal string tokens.

        pynput uses a mix of types:
        - KeyCode: usually has .char (may be None for non-character keys)
        - Key: has .name (e.g. "space", "shift")
        Some edge cases on Windows can produce KeyCode with char=None and no .name.
        This function must never throw (listener callbacks should not crash).
        """
        try:
            ch = getattr(key, "char", None)
            if isinstance(ch, str) and ch:
                return ch.lower()

            name = getattr(key, "name", None)
            if isinstance(name, str) and name:
                return name.lower()

            # Fallback: stringify.
            # Examples:
            # - "Key.space" -> "space"
            # - "'a'" -> "a"
            s = str(key)
            s = s.replace("Key.", "").strip().strip("'").strip('"')
            return s.lower()
        except Exception:
            return ""

    def normalize_mouse(self, button) -> str:
        # Import here to keep module import light in non-mouse contexts
        from pynput import mouse

        if button == mouse.Button.left:
            return "lmb"
        if button == mouse.Button.right:
            return "rmb"
        if button == mouse.Button.middle:
            return "mmb"
        return "mouse_extra"

    # -------------------------
    # Parsing
    # -------------------------

    def split_inputs(self, keys_str: str):
        """
        Split a user-entered Inputs string into top-level comma-separated tokens.

        This parser is intentionally **shallow**: it only understands enough structure to
        avoid splitting commas that are part of nested syntaxes:
        - `hold(...)` parentheses
        - `{...}` hold-duration braces (e.g. `e{350ms}`)
        - `[...]` any-order group brackets

        The returned tokens are *not* validated here; validation happens later in `parse_step()`.

        Examples:

        - Basic:
          `split_inputs("e, 3, r") -> ["e", "3", "r"]`

        - Hold parentheses:
          `split_inputs("lmb, hold(lmb, 0.30), rmb") -> ["lmb", "hold(lmb, 0.30)", "rmb"]`

        - Brace holds:
          `split_inputs("e{350ms}, q") -> ["e{350ms}", "q"]`

        - Any-order groups:
          `split_inputs("[q, e], 2") -> ["[q, e]", "2"]`

        - Mixed:
          `split_inputs("[lmb, wait:0.1s, hold(e,0.30)], r") -> ["[lmb, wait:0.1s, hold(e,0.30)]", "r"]`
        """
        s = keys_str or ""
        out: list[str] = []
        buf: list[str] = []
        paren = 0
        brace = 0
        bracket = 0

        for ch in s:
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)

            if ch == "," and paren == 0 and brace == 0 and bracket == 0:
                token = "".join(buf).strip()
                if token:
                    out.append(token)
                buf = []
                continue
            buf.append(ch)

        token = "".join(buf).strip()
        if token:
            out.append(token)
        return out

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

    def parse_step(self, token: str):
        """
        Parse a single token from `split_inputs()` into a normalized step dict.

        Supported token forms (examples show typical user input; parsing is case-insensitive):

        - Press:
          - `e`, `3`, `r`, `lmb`, `rmb`
          -> `{"input": "<key>", "hold_ms": None, "wait_ms": None}`

        - Soft wait gate (minimum delay; early presses ignored):
          - `wait:0.1` / `wait:0.1s` / `wait:100ms`
          -> `{"wait_ms": 100, "wait_mode": "soft"}`

        - Hard wait gate (early press can drop the combo):
          - `wait_hard:0.2`
          -> `{"wait_ms": 200, "wait_mode": "hard"}`

        - Hold:
          - `hold(e, 0.35)` or `e{350ms}`
          -> `{"input": "e", "hold_ms": 350, "wait_ms": None}`

        - Mandatory animation lock:
          - `wait(r, 1.5)` means: press `r`, then a mandatory wait ≥ 1.5s.
            During this mandatory wait, in-game inputs have no effect and therefore should not drop combos.
          -> `{"composite_steps": [press_step, mandatory_wait_step]}`

        - Any-order group:
          Groups are parsed as a single step that contains multiple required "items" that can be completed in any order.
          Supported group item types:
          - plain press: `q`
          - press+wait pair: `lmb, wait:0.1s` (treated as one atomic item inside the group)
          - hold: `hold(e, 0.30)`
          - mandatory animation lock: `wait(r, 1.5)` (treated as one group item)

          Examples:
          - `[q, e]`
          - `[wait(r, 1.5), q, e]`
          - `[lmb, wait:0.1s, rmb, wait:0.5s, q]`
          - `[hold(e, 0.30), lmb, rmb]`
          - `[lmb, wait:0.1, hold(e, 0.30), lmb, wait:0.5, q]` (same key can appear with different waits)

        Returns:
        - A dict describing one step, OR a dict with `composite_steps` (to be expanded by the caller), OR None.
        """
        t = (token or "").strip()
        if not t:
            return None

        tl = t.lower()

        # Animation-locked (mandatory) wait:
        # Syntax: wait(r, 1.5)  -> press r, then a mandatory minimum wait of 1.5s.
        # Rationale: some abilities have an animation lock where extra inputs have no effect in-game.
        # During this mandatory wait, key presses are ignored by the game and therefore should NOT drop the combo.
        if tl.startswith("wait(") and tl.endswith(")") and len(tl) >= 6:
            inner = tl[len("wait(") : -1].strip()
            parts = [p.strip() for p in self.split_inputs(inner) if p.strip()]
            if len(parts) == 2:
                key = parts[0].strip().lower()
                wait_ms = self._parse_duration(parts[1])
                if key and wait_ms is not None:
                    press_step = {"input": key, "hold_ms": None, "wait_ms": None}
                    wait_step = {
                        "input": None,
                        "hold_ms": None,
                        "wait_ms": int(wait_ms),
                        "wait_mode": "mandatory",  # mandatory wait due to animation lock (see comment above)
                        "wait_for": key,
                    }
                    return {"composite_steps": [press_step, wait_step]}

        # Any-order group step:
        # (See parse_step() docstring for supported syntaxes + examples.)
        # Examples:
        # - [q, e]                 -> press BOTH q and e, in any order
        # - [wait(r, 1.5), q, e]   -> press r/q/e in any order; r triggers a mandatory animation wait before the group can finish
        #
        # Notes:
        # - Inside the group we support: press, press+wait (e.g. lmb, wait:0.1s), hold(e,0.30), and wait(r,t).
        # - During "mandatory" waits (animation locks) and press+wait windows, extra presses are ignored (no fail).
        if tl.startswith("[") and tl.endswith("]") and len(tl) >= 3:
            inner = tl[1:-1].strip()
            parts = [p.strip() for p in self.split_inputs(inner) if p.strip()]
            if len(parts) >= 2:
                # press+wait items inside a group are keyed by signature so the same key can appear with different waits:
                # e.g. [lmb, wait:0.1, ..., lmb, wait:0.5, ...]
                pw_need_counts: dict[str, int] = {}  # sig -> count
                pw_meta: dict[str, dict[str, Any]] = {}  # sig -> {input, wait_ms, wait_mode}
                pw_order_sigs: list[str] = []  # preserve written order for resolving which sig is next
                mandatory_wait: dict[str, Any] | None = None  # {"wait_for": "r", "wait_ms": 1500}
                order: list[dict[str, Any]] = []  # preserves written order of group items (supports duplicates)
                press_need_counts: dict[str, int] = {}  # key -> count (supports duplicates)
                hold_need_counts: dict[str, int] = {}  # sig -> count (supports duplicates)
                hold_meta: dict[str, dict[str, Any]] = {}  # sig -> {input, hold_ms}
                hold_order_sigs: list[str] = []
                ok = True

                j = 0
                while j < len(parts):
                    p = parts[j]
                    s = self.parse_step(p)
                    if not isinstance(s, dict):
                        ok = False
                        break

                    # Composite (wait(r, t)) expands into [press r, mandatory wait] via composite_steps.
                    if s.get("composite_steps") is not None:
                        sub = [x for x in (s.get("composite_steps") or []) if isinstance(x, dict)]
                        if len(sub) != 2:
                            ok = False
                            break
                        press = sub[0]
                        w = sub[1]
                        inp = str(press.get("input") or "").strip().lower()
                        if not inp or press.get("wait_ms") is not None or press.get("hold_ms") is not None:
                            ok = False
                            break
                        w_ms = w.get("wait_ms")
                        w_mode = str(w.get("wait_mode") or "").strip().lower()
                        w_for = str(w.get("wait_for") or "").strip().lower()
                        if w_ms is None or w_mode != "mandatory" or not w_for:
                            ok = False
                            break
                        if mandatory_wait is not None:
                            # Keep it simple for now: only one mandatory wait per group.
                            ok = False
                            break
                        # Treat wait(r, t) as ONE group item (not "r" plus "wait").
                        # We'll still accept the press of r at runtime via group_mandatory_wait.wait_for.
                        mandatory_wait = {"wait_for": w_for, "wait_ms": int(w_ms)}
                        # Preserve written order (de-duped by the single-mandatory-wait rule).
                        order.append({"kind": "anim_wait", "wait_for": w_for, "wait_ms": int(w_ms)})
                        j += 1
                        continue

                    # Hold inside a group is supported as an atomic item.
                    if s.get("hold_ms") is not None:
                        hk = str(s.get("input") or "").strip().lower()
                        hms = int(s.get("hold_ms") or 0)
                        if not hk or hms <= 0:
                            ok = False
                            break
                        sig = f"{hk}:{hms}"
                        hold_need_counts[sig] = int(hold_need_counts.get(sig, 0)) + 1
                        hold_meta[sig] = {"input": hk, "hold_ms": hms}
                        if sig not in hold_order_sigs:
                            hold_order_sigs.append(sig)
                        order.append({"kind": "hold", "sig": sig, "input": hk, "hold_ms": hms})
                        j += 1
                        continue

                    # If this token is a wait gate by itself, it's invalid unless it's paired after a press.
                    if s.get("wait_ms") is not None:
                        ok = False
                        break

                    inp = str(s.get("input") or "").strip().lower()
                    if not inp:
                        ok = False
                        break

                    # Detect "press + wait" pairs inside the group:
                    # [lmb, wait:0.1s, rmb, wait:0.5s] => two atomic items.
                    if j + 1 < len(parts):
                        nxt = self.parse_step(parts[j + 1])
                        if isinstance(nxt, dict) and nxt.get("wait_ms") is not None:
                            mode = str(nxt.get("wait_mode") or "soft").strip().lower()
                            if mode in ("soft", "hard"):
                                wms = int(nxt.get("wait_ms") or 0)
                                if wms > 0:
                                    sig = f"{inp}:{wms}:{mode}"
                                    pw_need_counts[sig] = int(pw_need_counts.get(sig, 0)) + 1
                                    pw_meta[sig] = {"input": inp, "wait_ms": wms, "wait_mode": mode}
                                    if sig not in pw_order_sigs:
                                        pw_order_sigs.append(sig)
                                    order.append({"kind": "press_wait", "sig": sig, "input": inp, "wait_ms": wms, "wait_mode": mode})
                                    j += 2
                                    continue

                    # Plain press item (can repeat; repeats become additional required presses)
                    press_need_counts[inp] = int(press_need_counts.get(inp, 0)) + 1
                    order.append({"kind": "press", "input": inp})
                    j += 1

                uniq_presses = [k for k in press_need_counts.keys()]

                # Disallow listing the same key twice (e.g. [wait(r,1.5), r, q]) to avoid ambiguity.
                if mandatory_wait is not None:
                    mw_for = str(mandatory_wait.get("wait_for") or "").strip().lower()
                    if mw_for and mw_for in uniq_presses:
                        ok = False

                # Group validity: count ALL required items, including duplicates and holds.
                # Examples that must be valid:
                # - [e, e] (2 required presses)
                # - [hold(e,0.30), e] (hold + press)
                required_press_count = sum(int(v or 0) for v in press_need_counts.values())
                required_hold_count = sum(int(v or 0) for v in hold_need_counts.values())
                required_pw_count = sum(int(v or 0) for v in pw_need_counts.values())
                total_items = required_press_count + required_pw_count + required_hold_count + (1 if mandatory_wait is not None else 0)
                if ok and total_items >= 2:
                    return {
                        "input": None,
                        "hold_ms": None,
                        "wait_ms": None,
                        "group_presses": uniq_presses,
                        "group_press_need_counts": press_need_counts,
                        "group_pw_need_counts": pw_need_counts,
                        "group_pw_done_counts": {},
                        "group_pw_meta": pw_meta,
                        "group_pw_order_sigs": pw_order_sigs,
                        "group_done_counts": {},
                        "group_hold_need_counts": hold_need_counts,
                        "group_hold_done_counts": {},
                        "group_hold_meta": hold_meta,
                        "group_hold_order_sigs": hold_order_sigs,
                        "group_hold_active": False,
                        "group_hold_sig": "",
                        "group_hold_for": "",
                        "group_hold_started_at": 0.0,
                        "group_hold_required_ms": 0,
                        "group_mandatory_wait": mandatory_wait,  # or None
                        "group_order": order,
                        "group_wait_active": False,
                        "group_wait_done": False,
                        "group_wait_started_at": 0.0,
                        "group_wait_until": 0.0,
                        "group_pw_active": False,
                        "group_pw_sig": "",
                        "group_pw_until": 0.0,
                    }

        # Wait steps:
        # - wait:0.2        -> soft wait (minimum delay gate; early presses are ignored)
        # - wait_soft:0.2   -> soft wait (alias)
        # - wait_hard:0.2   -> hard wait (early press can drop the combo)
        wait_mode = None
        wait_prefix = None
        if tl.startswith("wait_hard:"):
            wait_mode = "hard"
            wait_prefix = "wait_hard:"
        elif tl.startswith("wait_soft:"):
            wait_mode = "soft"
            wait_prefix = "wait_soft:"
        elif tl.startswith("wait:"):
            wait_mode = "soft"
            wait_prefix = "wait:"

        if wait_prefix is not None:
            dur = tl[len(wait_prefix) :].strip()
            wait_ms = self._parse_duration(dur)
            if wait_ms is not None:
                return {"input": None, "hold_ms": None, "wait_ms": wait_ms, "wait_mode": wait_mode}

        if tl.startswith("hold(") and tl.endswith(")"):
            inner = tl[len("hold(") : -1]
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                hold_ms = self._parse_duration(parts[1])
                if hold_ms is not None:
                    return {"input": parts[0], "hold_ms": hold_ms, "wait_ms": None}

        if "{" in tl and tl.endswith("}"):
            base, rest = tl.split("{", 1)
            ms_str = rest[:-1].replace("ms", "").strip()
            base = base.strip()
            if base:
                hold_ms = self._parse_duration(ms_str)
                if hold_ms is not None:
                    return {"input": base, "hold_ms": hold_ms, "wait_ms": None}

        return {"input": tl, "hold_ms": None, "wait_ms": None}

    def calc_min_combo_time_ms(self, steps):
        total = 0
        for s in steps or []:
            if not isinstance(s, dict):
                continue
            w = s.get("wait_ms")
            h = s.get("hold_ms")
            if w is not None:
                total += int(w)
            if h is not None:
                total += int(h)
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

    def _expected_label_for_step(self, step: dict):
        if not isinstance(step, dict):
            return "—"
        if step.get("group_presses") is not None:
            opts = [str(x or "").strip().lower() for x in (step.get("group_presses") or [])]
            opts = [o for o in opts if o]
            # Include press+wait keys in the group label.
            pw_meta = step.get("group_pw_meta")
            if isinstance(pw_meta, dict) and pw_meta:
                pw_keys = []
                for _sig, meta in pw_meta.items():
                    k = str((meta or {}).get("input") or "").strip().lower()
                    if k:
                        pw_keys.append(k)
                if pw_keys:
                    opts = pw_keys + opts
            # Include holds in the group label (use the hold key name).
            hold_meta = step.get("group_hold_meta")
            hold_need = step.get("group_hold_need_counts")
            if isinstance(hold_meta, dict) and isinstance(hold_need, dict) and hold_need:
                hold_keys = []
                for sig, cnt in hold_need.items():
                    if int(cnt or 0) <= 0:
                        continue
                    meta = hold_meta.get(sig) if isinstance(hold_meta, dict) else None
                    hk = str((meta or {}).get("input") or "").strip().lower()
                    if hk:
                        hold_keys.append(hk)
                if hold_keys:
                    opts = hold_keys + opts
            mw = step.get("group_mandatory_wait")
            if isinstance(mw, dict):
                mw_for = str(mw.get("wait_for") or "").strip().lower()
                if mw_for:
                    opts = [mw_for] + opts
            if opts:
                return f"any-order({ '|'.join(opts) })"
            return "any-order(—)"
        if step.get("wait_ms") is not None:
            w = int(step.get("wait_ms") or 0)
            mode = str(step.get("wait_mode") or "soft").strip().lower()
            if mode == "hard":
                return f"wait-hard(≥{w}ms)"
            if mode == "mandatory":
                k = str(step.get("wait_for") or "").strip().lower()
                if k:
                    return f"anim-wait({k},≥{w}ms)"
                return f"anim-wait(≥{w}ms)"
            return f"wait(≥{w}ms)"
        if step.get("hold_ms") is not None:
            h = int(step.get("hold_ms") or 0)
            inp = str(step.get("input") or "").strip().lower()
            return f"hold({inp},≥{h}ms)" if inp else f"hold(≥{h}ms)"
        inp = str(step.get("input") or "").strip().lower()
        return inp or "—"

    def _find_next_step_index_for_input(self, input_name: str, *, start_index: int) -> int | None:
        """
        Look ahead in the active combo for the next non-wait step that matches input_name.
        Returns the absolute step index, or None if not found.
        """
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return None
        try:
            for j in range(max(0, int(start_index)), len(self.active_combo_steps)):
                s = self.active_combo_steps[j]
                if not isinstance(s, dict):
                    continue
                if s.get("wait_ms") is not None:
                    continue
                if s.get("group_presses") is not None:
                    opts = [str(x or "").strip().lower() for x in (s.get("group_presses") or [])]
                    pw_meta = s.get("group_pw_meta")
                    if isinstance(pw_meta, dict):
                        for _sig, meta in pw_meta.items():
                            k = str((meta or {}).get("input") or "").strip().lower()
                            if k:
                                opts.append(k)
                    hold_meta = s.get("group_hold_meta")
                    hold_need = s.get("group_hold_need_counts")
                    if isinstance(hold_meta, dict) and isinstance(hold_need, dict):
                        for sig, cnt in hold_need.items():
                            if int(cnt or 0) <= 0:
                                continue
                            meta = hold_meta.get(sig) if isinstance(hold_meta, dict) else None
                            hk = str((meta or {}).get("input") or "").strip().lower()
                            if hk:
                                opts.append(hk)
                    mw = s.get("group_mandatory_wait")
                    if isinstance(mw, dict):
                        mw_for = str(mw.get("wait_for") or "").strip().lower()
                        if mw_for:
                            opts = [mw_for] + opts
                    if input_name in opts:
                        return j
                if str(s.get("input") or "").strip().lower() == input_name:
                    return j
        except Exception:
            return None
        return None

    def _find_prev_step_index_for_input(self, input_name: str, *, end_index: int) -> int | None:
        """
        Look backward in the active combo for the most recent non-wait step that matches input_name.
        Searches indices < end_index.
        Returns the absolute step index, or None if not found.
        """
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return None
        try:
            end = max(0, int(end_index))
        except Exception:
            end = 0
        try:
            for j in range(min(end, len(self.active_combo_steps)) - 1, -1, -1):
                s = self.active_combo_steps[j]
                if not isinstance(s, dict):
                    continue
                if s.get("wait_ms") is not None:
                    continue
                if s.get("group_presses") is not None:
                    opts = [str(x or "").strip().lower() for x in (s.get("group_presses") or [])]
                    pw_meta = s.get("group_pw_meta")
                    if isinstance(pw_meta, dict):
                        for _sig, meta in pw_meta.items():
                            k = str((meta or {}).get("input") or "").strip().lower()
                            if k:
                                opts.append(k)
                    hold_meta = s.get("group_hold_meta")
                    hold_need = s.get("group_hold_need_counts")
                    if isinstance(hold_meta, dict) and isinstance(hold_need, dict):
                        for sig, cnt in hold_need.items():
                            if int(cnt or 0) <= 0:
                                continue
                            meta = hold_meta.get(sig) if isinstance(hold_meta, dict) else None
                            hk = str((meta or {}).get("input") or "").strip().lower()
                            if hk:
                                opts.append(hk)
                    mw = s.get("group_mandatory_wait")
                    if isinstance(mw, dict):
                        mw_for = str(mw.get("wait_for") or "").strip().lower()
                        if mw_for:
                            opts = [mw_for] + opts
                    if input_name in opts:
                        return j
                if str(s.get("input") or "").strip().lower() == input_name:
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
            for j in range(max(0, int(start_index)), len(self.active_combo_steps)):
                s = self.active_combo_steps[j]
                if isinstance(s, dict) and s.get("wait_ms") is None:
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
            step = self._active_step()
            if not isinstance(step, dict) or step.get("wait_ms") is None:
                return False
            # If there is any real action after this wait, it is not trailing.
            if self._next_non_wait_step_index(start_index=int(self.current_index) + 1) is not None:
                return False
        except Exception:
            return False

        self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"})
        self.record_combo_success(total_ms)
        self.current_index = 0
        self._reset_hold_state()
        self._reset_wait_state()
        self._reset_group_state()
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
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

    # -------------------------
    # Difficulty (Practical APM + simple timing model)
    # -------------------------

    def _clamp01(self, x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

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

    def _count_combo_actions(self, steps: list[dict[str, Any]] | None) -> tuple[int, int, int]:
        """
        Returns (press_count, hold_count, total_actions).
        - press_count: regular presses (non-wait, non-hold)
        - hold_count: hold(...) steps
        - total_actions: press_count + hold_count
        """
        press = 0
        hold = 0
        for s in steps or []:
            if not isinstance(s, dict):
                continue
            if s.get("wait_ms") is not None:
                continue
            if s.get("hold_ms") is not None:
                hold += 1
                continue
            press += 1
        return press, hold, press + hold

    def practical_apm(self) -> float | None:
        return ui.practical_apm(self)

    def theoretical_max_apm(self) -> float | None:
        return ui.theoretical_max_apm(self)

    def apm_text(self) -> str:
        return ui.apm_text(self)

    def apm_max_text(self) -> str:
        return ui.apm_max_text(self)

    def _wait_triangle_score(self, wait_ms: int) -> float:
        """
        Triangle-shaped wait difficulty:
        - peak at 350ms: roughly "awkward" timing for many players (not instant, not long enough to relax/spam)
        - 0 at 0ms: no minimum delay gate at all
        - 0 at >=600ms: long enough that it’s effectively “free” (you can reposition/spam without losing time)
        """
        try:
            w = float(wait_ms)
        except Exception:
            return 0.0
        if w <= 0.0:
            return 0.0
        # 600ms is treated as "free wait" in this model (see docstring above).
        if w >= 600.0:
            return 0.0
        # Rising edge: 0..350ms ramps up to max difficulty.
        if w < 350.0:
            return self._clamp01(w / 350.0)
        # Falling edge: 350..600ms ramps back down to 0.
        return self._clamp01((600.0 - w) / 250.0)

    def _hold_score(self, hold_ms: int) -> float:
        """
        Hold difficulty: monotonic, saturating. Longer holds are harder (finger commitment).
        """
        try:
            h = float(hold_ms)
        except Exception:
            return 0.0
        if h <= 0.0:
            return 0.0
        # H controls how quickly hold difficulty ramps up.
        # With H=350ms:
        # - ~350ms hold ≈ 63% difficulty
        # - ~700ms hold ≈ 86% difficulty
        # This matches the idea that long holds are meaningfully harder, but it saturates (they don't grow forever).
        H = 350.0
        return self._clamp01(1.0 - (2.718281828 ** (-h / H)))

    def _timing_variation_points(self) -> int:
        """
        Simple rule-based "gotcha timing" score:
        - +1 per distinct non-micro wait duration (ignores >=600ms waits)
        - +1 per distinct hold duration
        - micro waits (<=60ms): +1 if they occur exactly once, else +0
        """
        # micro_thresh: "micro waits" are treated as rhythm that you can standardize by
        # adding extra tiny delays where missing (so they don't necessarily create variation difficulty).
        # If a micro-wait occurs only once, we treat it as a "gotcha" (+1).
        micro_thresh = 60
        waits_micro_count = 0
        non_micro_waits: set[int] = set()
        holds: set[int] = set()

        for s in self.active_combo_steps or []:
            if not isinstance(s, dict):
                continue
            w = s.get("wait_ms")
            h = s.get("hold_ms")
            if w is not None:
                try:
                    w_i = int(w)
                except Exception:
                    continue
                if w_i <= 0:
                    continue
                if w_i >= 600:
                    # Keep consistent with _wait_triangle_score(): >=600ms waits are considered "free"
                    # and don't increase variation difficulty.
                    continue
                if w_i <= micro_thresh:
                    waits_micro_count += 1
                else:
                    non_micro_waits.add(w_i)
            elif h is not None:
                try:
                    h_i = int(h)
                except Exception:
                    continue
                if h_i > 0:
                    holds.add(h_i)

        micro_bonus = 1 if waits_micro_count == 1 else 0
        return int(len(non_micro_waits) + len(holds) + micro_bonus)

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

    def get_editor_payload(self) -> dict[str, Any]:
        return ui.get_editor_payload(self)

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
        # Called from the UI thread; protect against concurrent input/tick threads.
        with self._lock:
            raw = (raw or "").strip()
            if not raw:
                self.combo_enders = {}
                self.save_combos()
                return True, None

            parsed: dict[str, int] = {}
            for token in self.split_inputs(raw):
                t = token.strip()
                if not t:
                    continue

                if ":" in t:
                    k, v = t.split(":", 1)
                    key = k.strip().lower()
                    if not key:
                        continue
                    try:
                        sec = float(v.strip())
                    except ValueError:
                        return False, f"Invalid timing for '{key}'. Use seconds, e.g. {key}:0.2"
                    parsed[key] = max(0, int(sec * 1000))
                else:
                    key = t.strip().lower()
                    if key:
                        parsed[key] = 0

            self.combo_enders = parsed
            self.save_combos()
            return True, None

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
        # Called from the UI thread; protect against concurrent input/tick threads.
        with self._lock:
            name = (name or "").strip()
            keys_str = (inputs or "").strip()
            if not name or not keys_str:
                return False, "Please fill in Name and Inputs."

            ok, err = self.apply_enders_from_text(enders)
            if not ok:
                return False, err

            expected_ms = None
            expected_raw = (expected_time or "").strip()
            if expected_raw:
                expected_ms = self._parse_expected_time_ms(expected_raw)
                if expected_ms is None:
                    return False, "Invalid Expected time. Examples: 1.05s or 1050ms"

            user_diff_val = None
            ud_raw = (user_difficulty or "").strip()
            if ud_raw:
                try:
                    user_diff_val = float(ud_raw)
                except Exception:
                    return False, "Invalid Your difficulty. Use a number from 0 to 10."
                if not (0.0 <= user_diff_val <= 10.0):
                    return False, "Invalid Your difficulty. Use a number from 0 to 10."

            input_list = [k.strip().lower() for k in self.split_inputs(keys_str) if k.strip()]
            if not input_list:
                return False, "Please provide at least one input."

            old_name = self.active_combo_name if self.active_combo_name in self.combos else None
            if old_name and name != old_name:
                if old_name in self.combo_stats and name not in self.combo_stats:
                    self.combo_stats[name] = self.combo_stats.pop(old_name)
                self.combos[name] = input_list
                if old_name != name and old_name in self.combos:
                    del self.combos[old_name]
                if old_name in self.combo_expected_ms:
                    # Drop old key; we'll re-apply below if the UI provided a new value.
                    del self.combo_expected_ms[old_name]
                if old_name in self.combo_user_difficulty:
                    del self.combo_user_difficulty[old_name]
                if old_name in self.combo_step_display_mode and name not in self.combo_step_display_mode:
                    self.combo_step_display_mode[name] = self.combo_step_display_mode.pop(old_name)
                if old_name in self.combo_key_images and name not in self.combo_key_images:
                    self.combo_key_images[name] = self.combo_key_images.pop(old_name)
                # Move any WW/game-specific per-combo metadata.
                self.ww.rename_combo(old_name, name)
            else:
                self.combos[name] = input_list

            # Apply expected execution time (per-combo metadata)
            if expected_ms is not None:
                self.combo_expected_ms[name] = int(expected_ms)
            else:
                self.combo_expected_ms.pop(name, None)

            if user_diff_val is not None:
                self.combo_user_difficulty[name] = float(user_diff_val)
            else:
                self.combo_user_difficulty.pop(name, None)

            # Apply step display mode (per-combo metadata)
            mode_raw = (step_display_mode or "").strip().lower()
            if mode_raw in ("icons", "images"):
                self.combo_step_display_mode[name] = mode_raw
            else:
                self.combo_step_display_mode.pop(name, None)

            # Apply key images mapping (per-combo metadata)
            cleaned_imgs: dict[str, str] = {}
            if isinstance(key_images, dict):
                for k, v in key_images.items():
                    key = str(k).strip().lower()
                    url = str(v).strip()
                    if not key or not url:
                        continue
                    cleaned_imgs[key] = url
            if cleaned_imgs:
                self.combo_key_images[name] = cleaned_imgs
            else:
                # Keep existing images if caller didn't send a dict at all; otherwise allow clearing.
                if isinstance(key_images, dict):
                    self.combo_key_images.pop(name, None)

            # Apply target game + WW team assignment (combo detail)
            g_raw = str(target_game or "").strip().lower()
            self.ww.set_target_game(name, g_raw)
            self.ww.apply_combo_team_assignment(name, target_game=self.ww.get_target_game(name), ww_team_id=ww_team_id)

            self._ensure_combo_stats(name)
            self.set_active_combo(name, emit=False)
            self.save_combos()

            # Broadcast new global+active state
            self._send({"type": "init", **self.init_payload()})
            return True, None

    def delete_combo(self, name: str) -> tuple[bool, str | None]:
        with self._lock:
            name = (name or "").strip()
            if not name or name not in self.combos:
                return False, "Select a combo to delete."

            del self.combos[name]
            if name in self.combo_stats:
                del self.combo_stats[name]
            if name in self.combo_expected_ms:
                del self.combo_expected_ms[name]
            if name in self.combo_user_difficulty:
                del self.combo_user_difficulty[name]
            if name in self.combo_step_display_mode:
                del self.combo_step_display_mode[name]
            if name in self.combo_key_images:
                del self.combo_key_images[name]
            self.ww.delete_combo(name)

            if self.active_combo_name == name:
                self.active_combo_name = None
                self.active_combo_tokens = []
                self.active_combo_steps = []
                self.reset_tracking()

            self.save_combos()
            self._send({"type": "init", **self.init_payload()})
            return True, None

    # -------------------------
    # Wuthering Waves teams (presets)
    # -------------------------

    def set_active_ww_team(self, team_id: str):
        with self._lock:
            tid = str(team_id or "").strip()
            if tid and tid in self.ww_teams:
                self.ww.set_active_ww_team(tid)
                self.save_combos()
                # Refresh editor payload + timeline (so icons update)
                self._send({"type": "combo_data", **self.get_editor_payload()})
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})

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
            name = str(team_name or "").strip()
            if not name:
                return False, "Please provide a Team name."

            ok, err, _tid = self.ww.save_or_update_ww_team(
                team_id=str(team_id or "").strip(),
                team_name=name,
                dash_image=dash_image,
                swap_images=swap_images,
                lmb_images=lmb_images,
                ability_images=ability_images,
            )
            if not ok:
                return False, err

            self.save_combos()
            # Broadcast refresh (updates team dropdown + active team)
            self._send({"type": "init", **self.init_payload()})
            return True, None

    def delete_ww_team(self, team_id: str) -> tuple[bool, str | None]:
        with self._lock:
            ok, err = self.ww.delete_ww_team(team_id)
            if not ok:
                return False, err
            self.save_combos()
            self._send({"type": "init", **self.init_payload()})
            return True, None

    def new_combo(self):
        with self._lock:
            self.active_combo_name = None
            self.active_combo_tokens = []
            self.active_combo_steps = []
            self.reset_tracking()
            self._send({"type": "init", **self.init_payload()})

    def clear_history_and_stats(self):
        with self._lock:
            self.reset_tracking()
            if self.active_combo_name:
                self.combo_stats[self.active_combo_name] = {
                    "success": 0,
                    "fail": 0,
                    "best_ms": None,
                    "total_success_ms": 0,
                    "fail_by_step": {},
                    "fail_by_expected": {},
                    "fail_by_reason": {},
                    "fail_events": [],
                }
                self.save_combos()
            self._send({"type": "clear_results"})
            self._send({"type": "stat_update", "stats": self.stats_text()})
            self._send({"type": "fail_update", "failures": self.failures_by_reason()})
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            st = self.get_status()
            self._send({"type": "status", "text": st.text, "color": st.color})

    def set_active_combo(self, name: str, *, emit: bool = True):
        with self._lock:
            name = (name or "").strip()
            if name not in self.combos:
                self.active_combo_name = None
                self.active_combo_tokens = []
                self.active_combo_steps = []
                self.reset_tracking()
                if emit:
                    self._send({"type": "init", **self.init_payload()})
                return

            self.active_combo_name = name
            self.active_combo_tokens = self.combos[name]
            steps: list[dict[str, Any]] = []
            for t in self.active_combo_tokens:
                s = self.parse_step(t)
                if not s:
                    continue
                # Support composite tokens like wait(r, 1.5) -> [press r, mandatory wait]
                if isinstance(s, dict) and s.get("composite_steps") is not None:
                    try:
                        for sub in (s.get("composite_steps") or []):
                            if isinstance(sub, dict) and sub:
                                steps.append(sub)
                    except Exception:
                        pass
                else:
                    steps.append(s)
            self.active_combo_steps = steps
            self._ensure_combo_stats(name)

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
        if 0 <= self.current_index < len(self.active_combo_steps):
            return self.active_combo_steps[self.current_index]
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
        """
        Clear per-attempt progress for any-order groups ([a, b, c]).
        """
        try:
            for s in self.active_combo_steps or []:
                if isinstance(s, dict) and s.get("group_presses") is not None:
                    # If a group had an internal animation wait running, stop the UI animation.
                    if bool(s.get("group_wait_active")) or bool(s.get("group_pw_active")):
                        self._send({"type": "wait_end"})
                    if bool(s.get("group_hold_active")):
                        self._send({"type": "hold_end"})
                    s["group_done_counts"] = {}
                    s["group_pw_done_counts"] = {}
                    s["group_wait_active"] = False
                    s["group_wait_done"] = False
                    s["group_wait_started_at"] = 0.0
                    s["group_wait_until"] = 0.0
                    s["group_pw_active"] = False
                    s["group_pw_sig"] = ""
                    s["group_pw_until"] = 0.0
                    s["group_hold_done_counts"] = {}
                    s["group_hold_active"] = False
                    s["group_hold_sig"] = ""
                    s["group_hold_for"] = ""
                    s["group_hold_started_at"] = 0.0
                    s["group_hold_required_ms"] = 0
        except Exception:
            pass

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
            step = self._active_step()
            mode = "soft"
            wait_for = ""
            if isinstance(step, dict) and step.get("wait_ms") is not None:
                mode = str(step.get("wait_mode") or "soft").strip().lower() or "soft"
                wait_for = str(step.get("wait_for") or "")
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
        step = self._active_step()
        try:
            if isinstance(step, dict) and step.get("wait_ms") is not None:
                mode = str(step.get("wait_mode") or "soft").strip().lower() or "soft"
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
                expected_label=self._expected_label_for_step(self._active_step()),
                reason="too early",
                elapsed_ms=elapsed_ms,
            )
            self.current_index = 0
            self._reset_hold_state()
            self._reset_wait_state()
            self._reset_group_state()
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return False

        split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0.0
        self.record_hit(label, split_ms, total_ms)
        self.last_input_time = now
        self.current_index += 1
        self._reset_wait_state()
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})

        # If a wait step was (accidentally) the last step, don't get stuck past the end.
        if self.current_index >= len(self.active_combo_steps):
            self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"})
            self.record_combo_success(total_ms)
            self.current_index = 0
            self._reset_hold_state()
            self._reset_wait_state()
            self._reset_group_state()
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
        return True

    def _maybe_start_wait_step(self):
        step = self._active_step()
        if not step:
            return
        wait_ms = step.get("wait_ms")
        if wait_ms is not None and not self.wait_in_progress:
            self._start_wait(int(wait_ms))

    def _complete_hold(self, now: float, *, auto: bool):
        step = self._active_step()
        if not step or step.get("hold_ms") is None:
            return False

        target_input = str(step.get("input") or "")
        target_hold_ms = int(step.get("hold_ms") or 0)

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
                prev = self.active_combo_steps[self.current_index - 1]
                if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                    self._mark_step(self.current_index - 1, "ok")

            self.current_index += 1
            if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                self._reset_hold_state()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                return True
            self._maybe_start_wait_step()

            if self.current_index >= len(self.active_combo_steps):
                self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"})
                self.record_combo_success(total_ms)
                self.current_index = 0
                self._reset_group_state()
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
            self.current_index = 0
            self._reset_group_state()

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
        self._ui_last_success_steps_len = len(self.active_combo_steps or [])
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
        if not self.active_combo_steps:
            return

        while True:
            step = self._active_step()
            if not step:
                return

            group_presses = step.get("group_presses")
            target_input = step.get("input")
            target_hold_ms = step.get("hold_ms")
            target_wait_ms = step.get("wait_ms")
            target_wait_mode = str(step.get("wait_mode") or "soft").strip().lower()

            if target_wait_ms is not None:
                self._maybe_start_wait_step()
                now = time.perf_counter()
                if now < self.wait_until:
                    # Soft-wait training hint:
                    # If you press the *next expected input* during the wait gate, it won't count.
                    # Mark the wait step red immediately so the UI shows the timing mistake.
                    # NOTE: For mandatory waits (animation locks), inputs have no effect in-game.
                    # We intentionally do NOT mark timing mistakes and we do NOT drop the combo during the lock.
                    if target_wait_mode == "soft":
                        next_idx = self._next_non_wait_step_index(start_index=self.current_index + 1)
                        next_expected = None
                        next_expected_set: set[str] | None = None
                        if next_idx is not None:
                            try:
                                nxt = self.active_combo_steps[next_idx]
                                if isinstance(nxt, dict) and nxt.get("group_presses") is not None:
                                    next_expected_set = {
                                        str(x or "").strip().lower() for x in (nxt.get("group_presses") or [])
                                    }
                                else:
                                    next_expected = str(self.active_combo_steps[next_idx].get("input") or "").strip().lower()
                            except Exception:
                                next_expected = None
                                next_expected_set = None
                        if (next_expected and input_name == next_expected) or (
                            next_expected_set and input_name in next_expected_set
                        ):
                            wi = int(self.current_index)
                            self.wait_early_inputs.setdefault(wi, set()).add(input_name)
                            self._mark_step(wi, "early")
                            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                    # Soft wait: ignore any early presses (even enders).
                    # Hard wait: early press can drop the combo (models games where early input consumes/cancels).
                    if target_wait_mode == "hard":
                        # Fail on enders, and also fail if the pressed input matches the next expected non-wait input.
                        next_expected = None
                        next_expected_set: set[str] | None = None
                        try:
                            for j in range(self.current_index + 1, len(self.active_combo_steps)):
                                st = self.active_combo_steps[j]
                                if isinstance(st, dict) and st.get("wait_ms") is None:
                                    if st.get("group_presses") is not None:
                                        next_expected_set = {
                                            str(x or "").strip().lower() for x in (st.get("group_presses") or [])
                                        }
                                    else:
                                        next_expected = str(st.get("input") or "").strip().lower() or None
                                    break
                        except Exception:
                            next_expected = None
                            next_expected_set = None

                        if self._is_combo_ender(input_name) or (next_expected and input_name == next_expected) or (
                            next_expected_set and input_name in next_expected_set
                        ):
                            self._complete_wait(now, fail=True, reason=f"{input_name} too early")
                    return
                self._complete_wait(now, fail=False)
                continue

            if (
                target_hold_ms is not None
                and self.hold_in_progress
                and self.hold_expected_input == target_input
            ):
                if input_name == target_input:
                    return
                now = time.perf_counter()
                held_ms = (now - self.hold_started_at) * 1000
                if held_ms >= float(target_hold_ms):
                    self._complete_hold(now, auto=True)
                    continue
            break

        # Any-order group step: accepts remaining options in any order.
        # Can optionally include a mandatory animation wait (from wait(r, t)).
        if isinstance(step, dict) and step.get("group_presses") is not None:
            opts = [str(x or "").strip().lower() for x in (step.get("group_presses") or [])]
            opts = [o for o in opts if o]
            need_counts = step.get("group_press_need_counts")
            if not isinstance(need_counts, dict) or not need_counts:
                need_counts = {k: 1 for k in opts}
                step["group_press_need_counts"] = need_counts

            done_counts = step.get("group_done_counts")
            if not isinstance(done_counts, dict):
                done_counts = {}
                step["group_done_counts"] = done_counts

            pw_need_counts = step.get("group_pw_need_counts")
            pw_need_counts = pw_need_counts if isinstance(pw_need_counts, dict) else {}
            pw_done_counts = step.get("group_pw_done_counts")
            if not isinstance(pw_done_counts, dict):
                pw_done_counts = {}
                step["group_pw_done_counts"] = pw_done_counts
            pw_meta = step.get("group_pw_meta")
            pw_meta = pw_meta if isinstance(pw_meta, dict) else {}
            pw_order_sigs = step.get("group_pw_order_sigs")
            pw_order_sigs = pw_order_sigs if isinstance(pw_order_sigs, list) else []

            pw_keys = set()
            for _sig, meta in pw_meta.items():
                k = str((meta or {}).get("input") or "").strip().lower()
                if k:
                    pw_keys.add(k)

            hold_need_counts = step.get("group_hold_need_counts")
            hold_need_counts = hold_need_counts if isinstance(hold_need_counts, dict) else {}
            hold_done_counts = step.get("group_hold_done_counts")
            if not isinstance(hold_done_counts, dict):
                hold_done_counts = {}
                step["group_hold_done_counts"] = hold_done_counts
            hold_meta = step.get("group_hold_meta")
            hold_meta = hold_meta if isinstance(hold_meta, dict) else {}
            hold_order_sigs = step.get("group_hold_order_sigs")
            hold_order_sigs = hold_order_sigs if isinstance(hold_order_sigs, list) else []

            mw = step.get("group_mandatory_wait")
            mw_for = ""
            mw_ms = None
            if isinstance(mw, dict):
                mw_for = str(mw.get("wait_for") or "").strip().lower()
                mw_ms = mw.get("wait_ms")
            has_mw = (mw_ms is not None and mw_for)

            # If we're currently inside a hold item inside the group, treat it like a "mini hold step":
            # - other inputs are ignored until the requirement is met (or until it can auto-complete)
            if bool(step.get("group_hold_active")):
                now = time.perf_counter()
                hold_for = str(step.get("group_hold_for") or "").strip().lower()
                sig = str(step.get("group_hold_sig") or "").strip()
                req = int(step.get("group_hold_required_ms") or 0)
                started = float(step.get("group_hold_started_at") or 0.0)

                if input_name == hold_for:
                    return

                held_ms = (now - started) * 1000.0 if started else 0.0
                if req > 0 and held_ms >= float(req):
                    # Auto-complete the hold and then continue processing this press.
                    self._send({"type": "hold_end"})
                    step["group_hold_active"] = False
                    step["group_hold_for"] = ""
                    step["group_hold_sig"] = ""
                    step["group_hold_started_at"] = 0.0
                    step["group_hold_required_ms"] = 0
                    hold_done_counts[sig] = int(hold_done_counts.get(sig, 0)) + 1
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                else:
                    # Not held long enough; ignore other presses while holding.
                    return

            # If we're currently inside the group's animation lock, ignore all inputs until it ends.
            if has_mw and bool(step.get("group_wait_active")):
                now = time.perf_counter()
                until = float(step.get("group_wait_until") or 0.0)
                if now < until:
                    return  # inputs ignored during animation lock (can't fail, can't count)
                # Lock expired; finalize. If the rest of the group is already complete, auto-advance
                # so the current key (often the next step, e.g. '2') is evaluated correctly.
                step["group_wait_active"] = False
                step["group_wait_done"] = True
                self._send({"type": "wait_end"})
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                # If all required presses are already done, the group completes immediately.
                presses_done = all(int(done_counts.get(k, 0)) >= int(need_counts.get(k, 0)) for k in need_counts.keys())
                pw_done_ok = all(int(pw_done_counts.get(sig, 0)) >= int(pw_need_counts.get(sig, 0)) for sig in pw_need_counts.keys())
                if presses_done and pw_done_ok:
                    self._mark_step(int(self.current_index), "ok")
                    self.current_index += 1
                    total_ms = (now - self.start_time) * 1000 if self.start_time else 0
                    if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                        return
                    self._maybe_start_wait_step()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                    # Re-process this same input against the next expected step.
                    return self.process_press(input_name)

            # If we're currently inside a press+wait item, ignore all inputs until it ends.
            if bool(step.get("group_pw_active")):
                now = time.perf_counter()
                until = float(step.get("group_pw_until") or 0.0)
                if now < until:
                    return  # inputs ignored during the post-press animation time
                # The post-press wait finished; mark that press+wait signature complete.
                sig = str(step.get("group_pw_sig") or "").strip()
                if sig:
                    pw_done_counts[sig] = int(pw_done_counts.get(sig, 0)) + 1
                step["group_pw_active"] = False
                step["group_pw_sig"] = ""
                step["group_pw_until"] = 0.0
                self._send({"type": "wait_end"})
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                # If the group is complete after finishing this wait, advance and re-process current input.
                presses_done = all(int(done_counts.get(k, 0)) >= int(need_counts.get(k, 0)) for k in need_counts.keys())
                pw_done_ok = all(int(pw_done_counts.get(sig, 0)) >= int(pw_need_counts.get(sig, 0)) for sig in pw_need_counts.keys())
                mw_done2 = (not has_mw) or bool(step.get("group_wait_done"))
                if presses_done and pw_done_ok and mw_done2:
                    self._mark_step(int(self.current_index), "ok")
                    self.current_index += 1
                    total_ms = (now - self.start_time) * 1000 if self.start_time else 0
                    if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                        return
                    self._maybe_start_wait_step()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return self.process_press(input_name)

            # If this input's requirements are already fully satisfied, treat repeats as enders if applicable.
            # NOTE: keys not present in need_counts should NOT be treated as "already done" (0/0),
            # otherwise hold-only keys would get misclassified as enders.
            already_plain_done = (input_name in need_counts) and (
                int(done_counts.get(input_name, 0)) >= int(need_counts.get(input_name, 0))
            )
            # Consider press+wait requirements for this key satisfied only if all sigs for this key are done.
            if input_name not in pw_keys:
                already_pw_done = True
            else:
                already_pw_done = True
                for sig, meta in (pw_meta or {}).items():
                    k = str((meta or {}).get("input") or "").strip().lower()
                    if k != input_name:
                        continue
                    if int(pw_done_counts.get(sig, 0)) < int(pw_need_counts.get(sig, 0)):
                        already_pw_done = False
                        break
            already_hold_done = True
            if hold_need_counts:
                relevant = [
                    sig
                    for sig in hold_order_sigs
                    if str((hold_meta.get(sig) or {}).get("input") or "").strip().lower() == input_name
                ]
                if relevant:
                    already_hold_done = all(
                        int(hold_done_counts.get(sig, 0)) >= int(hold_need_counts.get(sig, 0)) for sig in relevant
                    )
            if already_plain_done and already_pw_done and already_hold_done:
                # If the key is a combo ender, pressing it again at the wrong time should still drop the combo.
                # (Except when we intentionally ignore immediate re-presses due to grace windows.)
                if self._is_combo_ender(input_name) and not self._should_ignore_ender_miss(input_name):
                    expected = str(self._expected_label_for_step(step) or "").strip().lower()
                    actual = str(input_name or "").strip().lower()
                    self._mark_step(int(self.current_index), "missed")
                    self.record_hit(f"{actual} (Exp: {expected}) [ender]", "FAIL", "FAIL")
                    self._send({"type": "status", "text": "Combo Dropped (Combo Ender)", "color": "fail"})
                    now = time.perf_counter()
                    elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
                    self.record_combo_fail(
                        actual=input_name,
                        expected_step_index=int(self.current_index),
                        expected_label=self._expected_label_for_step(step),
                        reason="wrong input",
                        elapsed_ms=elapsed_ms,
                    )
                    self.current_index = 0
                    self._reset_hold_state()
                    self._reset_wait_state()
                    self._reset_group_state()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return
                return

            # Press+wait group item (e.g. lmb, wait:0.1s) is accepted as one atomic item:
            # press starts the timer; other inputs are ignored until it finishes; only then the item counts as done.
            #
            # IMPORTANT: a key may appear both as a press+wait item *and* as a plain press requirement
            # (e.g. [lmb, wait:0.1s, rmb, wait:0.5s, rmb]). In that case:
            # - the first rmb triggers the press+wait item
            # - after that wait is complete, subsequent rmb presses should count toward the plain press requirement
            if input_name in pw_keys and not bool(step.get("group_pw_active")):
                # Pick the first not-yet-completed press_wait signature for this key (in written order).
                pick_sig = None
                for sig in pw_order_sigs:
                    meta = pw_meta.get(sig) if isinstance(pw_meta, dict) else None
                    k = str((meta or {}).get("input") or "").strip().lower()
                    if k != input_name:
                        continue
                    if int(pw_done_counts.get(sig, 0)) < int(pw_need_counts.get(sig, 0)):
                        pick_sig = str(sig)
                        break
                if not pick_sig:
                    # No remaining press+wait requirement for this key; let it fall through to plain press logic.
                    pick_sig = None
                if pick_sig:
                    meta = pw_meta.get(pick_sig) if isinstance(pw_meta, dict) else None
                    wms = int((meta or {}).get("wait_ms") or 0)
                    if wms <= 0:
                        return

                    now = time.perf_counter()

                    # Start attempt on first accepted press (group can be the first step).
                    if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                        self._insert_attempt_separator()
                        self.start_time = now
                        self.last_input_time = now
                        self._send({"type": "status", "text": "Recording...", "color": "recording"})

                    # Timing row
                    if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                        split_ms = 0
                        total_ms = 0
                    else:
                        split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0
                        total_ms = (now - self.start_time) * 1000 if self.start_time else 0

                    # If this group was gated by a wait right before it, mark that wait green once a valid option counts.
                    if self.current_index > 0:
                        prev = self.active_combo_steps[self.current_index - 1]
                        if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                            self._mark_step(self.current_index - 1, "ok")

                    self.record_hit(input_name, split_ms, total_ms)
                    self.last_input_time = now
                    self.last_success_input = input_name

                    step["group_pw_active"] = True
                    step["group_pw_sig"] = pick_sig
                    step["group_pw_until"] = now + (wms / 1000.0)

                    # Animate the wait fill on the active key+wait tile.
                    self._send({"type": "wait_begin", "required_ms": int(wms), "mode": "soft", "wait_for": input_name})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return

            # Hold group item: pressing the hold key starts the hold timer.
            # We pick the first not-yet-completed hold signature for this key in the order it was written.
            hold_key = input_name
            if hold_need_counts and not bool(step.get("group_hold_active")):
                pick_sig = None
                for sig in hold_order_sigs:
                    meta = hold_meta.get(sig) if isinstance(hold_meta, dict) else None
                    k = str((meta or {}).get("input") or "").strip().lower()
                    if k != hold_key:
                        continue
                    need_n = int(hold_need_counts.get(sig, 0) or 0)
                    done_n = int(hold_done_counts.get(sig, 0) or 0)
                    if done_n < need_n:
                        pick_sig = str(sig)
                        break
                if pick_sig:
                    meta = hold_meta.get(pick_sig) if isinstance(hold_meta, dict) else None
                    req_ms = int((meta or {}).get("hold_ms") or 0)
                    if req_ms > 0:
                        now = time.perf_counter()

                        # Start attempt on first accepted press (group can be the first step).
                        if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                            self._insert_attempt_separator()
                            self.start_time = now
                            self.last_input_time = now
                            self._send({"type": "status", "text": "Recording...", "color": "recording"})

                        # Record the press event itself (consistent with other group items)
                        if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                            split_ms = 0
                            total_ms = 0
                        else:
                            split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0
                            total_ms = (now - self.start_time) * 1000 if self.start_time else 0

                        if self.current_index > 0:
                            prev = self.active_combo_steps[self.current_index - 1]
                            if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                                self._mark_step(self.current_index - 1, "ok")

                        self.record_hit(input_name, split_ms, total_ms)
                        self.last_input_time = now
                        self.last_success_input = input_name

                        step["group_hold_active"] = True
                        step["group_hold_sig"] = pick_sig
                        step["group_hold_for"] = hold_key
                        step["group_hold_started_at"] = now
                        step["group_hold_required_ms"] = req_ms
                        self._send({"type": "hold_begin", "input": str(hold_key or ""), "required_ms": int(req_ms)})
                        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                        return

            # Mandatory-wait key press (e.g. r in wait(r,1.5)) is accepted as part of the group,
            # but it only "counts" once the animation lock finishes.
            if has_mw and input_name == mw_for:
                # If already satisfied, ignore duplicates.
                if bool(step.get("group_wait_done")):
                    return

                now = time.perf_counter()
                # Start attempt on first accepted press (group can be the first step).
                if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                    self._insert_attempt_separator()
                    self.start_time = now
                    self.last_input_time = now
                    self._send({"type": "status", "text": "Recording...", "color": "recording"})

                # Timing row
                if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                    split_ms = 0
                    total_ms = 0
                else:
                    split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0
                    total_ms = (now - self.start_time) * 1000 if self.start_time else 0

                # If this group was gated by a wait right before it, mark that wait green once a valid option counts.
                if self.current_index > 0:
                    prev = self.active_combo_steps[self.current_index - 1]
                    if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                        self._mark_step(self.current_index - 1, "ok")

                self.record_hit(input_name, split_ms, total_ms)
                self.last_input_time = now
                self.last_success_input = input_name

                # Start the internal animation lock now.
                step["group_wait_active"] = True
                step["group_wait_started_at"] = now
                step["group_wait_until"] = now + (int(mw_ms) / 1000.0)
                self._send({"type": "wait_begin", "required_ms": int(mw_ms), "mode": "mandatory", "wait_for": mw_for})
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                return

            if input_name in need_counts:
                now = time.perf_counter()

                # Start of combo (group can be the first step)
                if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                    self._insert_attempt_separator()
                    self.start_time = now
                    self.last_input_time = now
                    self._send({"type": "status", "text": "Recording...", "color": "recording"})

                # Compute timings for hit rows
                if self.current_index == 0 and (sum(int(v) for v in done_counts.values()) + sum(int(v) for v in pw_done_counts.values()) + sum(int(v) for v in hold_done_counts.values())) == 0:
                    split_ms = 0
                    total_ms = 0
                else:
                    split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0
                    total_ms = (now - self.start_time) * 1000 if self.start_time else 0

                # If this group was gated by a wait right before it, mark that wait green once a valid option counts.
                if self.current_index > 0:
                    prev = self.active_combo_steps[self.current_index - 1]
                    if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                        self._mark_step(self.current_index - 1, "ok")

                self.record_hit(input_name, split_ms, total_ms)
                self.last_input_time = now
                self.last_success_input = input_name
                cur_n = int(done_counts.get(input_name, 0))
                need_n = int(need_counts.get(input_name, 0))
                if cur_n < need_n:
                    done_counts[input_name] = cur_n + 1

                # Group completes when all presses are hit, and (if present) the mandatory wait is satisfied.
                presses_done = all(int(done_counts.get(k, 0)) >= int(need_counts.get(k, 0)) for k in need_counts.keys())
                pw_done_ok = all(int(pw_done_counts.get(sig, 0)) >= int(pw_need_counts.get(sig, 0)) for sig in pw_need_counts.keys())
                holds_done = all(int(hold_done_counts.get(sig, 0)) >= int(hold_need_counts.get(sig, 0)) for sig in hold_need_counts.keys())
                mw_done = (not has_mw) or bool(step.get("group_wait_done"))
                if presses_done and pw_done_ok and holds_done and mw_done:
                    self._mark_step(int(self.current_index), "ok")
                    self.current_index += 1
                    if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                        return
                    self._maybe_start_wait_step()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                    if self.current_index >= len(self.active_combo_steps):
                        self._send(
                            {"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"}
                        )
                        self.record_combo_success(total_ms)
                        self.current_index = 0
                        self._reset_hold_state()
                        self._reset_wait_state()
                        self._reset_group_state()
                        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return

                # Partial progress: keep current_index on this group.
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                return

            # If the combo has already started (group is the first step so current_index can still be 0),
            # then combo enders should still be able to drop the combo while you're "inside" the group
            # (except during the group's ignore-wait windows handled above).
            if self.start_time and self.last_input_time:
                if self._is_combo_ender(input_name) and not self._should_ignore_ender_miss(input_name):
                    expected = str(self._expected_label_for_step(step) or "").strip().lower()
                    actual = str(input_name or "").strip().lower()
                    self._mark_step(int(self.current_index), "missed")
                    self.record_hit(f"{actual} (Exp: {expected}) [ender]", "FAIL", "FAIL")
                    self._send({"type": "status", "text": "Combo Dropped (Combo Ender)", "color": "fail"})
                    now = time.perf_counter()
                    elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
                    self.record_combo_fail(
                        actual=input_name,
                        expected_step_index=int(self.current_index),
                        expected_label=self._expected_label_for_step(step),
                        reason="wrong input",
                        elapsed_ms=elapsed_ms,
                    )
                    self.current_index = 0
                    self._reset_hold_state()
                    self._reset_wait_state()
                    self._reset_group_state()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return

        # Start of combo
        if self.current_index == 0:
            if input_name == target_input:
                now = time.perf_counter()
                self._insert_attempt_separator()
                self.start_time = now
                self.last_input_time = now

                if target_hold_ms is None:
                    self.record_hit(input_name, 0, 0)
                    self.current_index += 1
                    if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=0):
                        return
                    self._maybe_start_wait_step()
                    self.last_success_input = input_name
                    self._send({"type": "status", "text": "Recording...", "color": "recording"})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                else:
                    self._start_hold(input_name, int(target_hold_ms), now)
            return

        # During combo
        current_time = time.perf_counter()
        if input_name == target_input:
            if target_hold_ms is None:
                split_ms = (current_time - self.last_input_time) * 1000
                total_ms = (current_time - self.start_time) * 1000
                self.record_hit(input_name, split_ms, total_ms)
                self.last_input_time = current_time
                self.last_success_input = input_name

                # If this step was gated by a wait right before it, mark that wait green (timing satisfied).
                if self.current_index > 0:
                    prev = self.active_combo_steps[self.current_index - 1]
                    if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                        self._mark_step(self.current_index - 1, "ok")

                self.current_index += 1
                if self._maybe_complete_combo_if_trailing_wait(now=current_time, total_ms=total_ms):
                    return
                self._maybe_start_wait_step()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                if self.current_index >= len(self.active_combo_steps):
                    self._send(
                        {"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"}
                    )
                    self.record_combo_success(total_ms)
                    self.current_index = 0
                    self._reset_group_state()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            else:
                self._start_hold(input_name, int(target_hold_ms), current_time)
            return

        # Miss
        if self._is_combo_ender(input_name):
            if self._should_ignore_ender_miss(input_name):
                return

            # More helpful messaging: detect "out of order" presses (likely skipped an expected step).
            expected = str(self._expected_label_for_step(step) or "").strip().lower()
            actual = str(input_name or "").strip().lower()
            skipped_idx = None
            if expected and actual and expected != actual:
                # If the actual input appears later in the combo, you likely hit a later step early.
                skipped_idx = self._find_next_step_index_for_input(actual, start_index=self.current_index + 1)
            passed_idx = None
            if expected and actual and expected != actual:
                # If the actual input appeared earlier than the current expected step,
                # the player is likely repeating a key that the combo has already passed.
                passed_idx = self._find_prev_step_index_for_input(actual, end_index=self.current_index)

            # If the player pressed a later combo input, we can be more specific than "wrong input":
            # - If they pressed the immediate "next action" after the expected one, they likely MISSED the expected input.
            # - If the expected input was pressed during the wait gate right before it, call it "pressed too fast".
            next_action_idx = self._next_non_wait_step_index(start_index=self.current_index + 1)
            is_missed = (skipped_idx is not None) and (next_action_idx is not None) and (skipped_idx == next_action_idx)

            # Detect "pressed too fast": expected was hit during the wait step directly before this expected input.
            pressed_too_fast = False
            if self.current_index > 0:
                prev_idx = self.current_index - 1
                prev = self.active_combo_steps[prev_idx]
                if isinstance(prev, dict) and prev.get("wait_ms") is not None:
                    if input_name and expected:
                        if expected in self.wait_early_inputs.get(prev_idx, set()):
                            pressed_too_fast = True

            # Mark the expected step as missed/wrong for UI feedback.
            self._mark_step(int(self.current_index), "missed")

            # If the actual input exists later in the combo, mark that step red too (shows where you jumped to).
            if skipped_idx is not None:
                self._mark_step(int(skipped_idx), "wrong")

            if pressed_too_fast:
                # Example: expected 'E' after wait, but E was pressed during the wait gate, then player moved on.
                self.record_hit(f"{actual} (Exp: {expected}) [pressed too fast]", "FAIL", "FAIL")
                self._send(
                    {
                        "type": "status",
                        "text": f"Combo Dropped (Pressed Too Fast): '{expected.upper()}' was pressed during the wait, so it didn't count.",
                        "color": "fail",
                    }
                )
                fail_reason = "pressed too fast"
            elif is_missed:
                self.record_hit(f"{actual} (Exp: {expected}) [missed input]", "FAIL", "FAIL")
                self._send(
                    {
                        "type": "status",
                        "text": f"Combo Dropped (Missed Input): expected '{expected.upper()}', but you went to '{actual.upper()}'.",
                        "color": "fail",
                    }
                )
                fail_reason = "missed input"
            elif skipped_idx is not None:
                # Out of order (jumped somewhere later than just the next action).
                self.record_hit(f"{actual} (Exp: {expected}) [out of order]", "FAIL", "FAIL")
                self._send(
                    {
                        "type": "status",
                        "text": f"Combo Dropped (Out of Order): got '{actual.upper()}', expected '{expected.upper()}'.",
                        "color": "fail",
                    }
                )
                fail_reason = "out of order"
            elif passed_idx is not None:
                # The pressed key is part of the combo, but only earlier steps (no longer remaining).
                self.record_hit(f"{actual} (Exp: {expected}) [already passed]", "FAIL", "FAIL")
                self._send(
                    {
                        "type": "status",
                        "text": f"Combo Dropped (Already Passed): '{actual.upper()}' already happened earlier. Expected '{expected.upper()}'.",
                        "color": "fail",
                    }
                )
                fail_reason = "already passed"
            else:
                self.record_hit(f"{actual} (Exp: {expected})", "FAIL", "FAIL")
                self._send({"type": "status", "text": "Combo Dropped (Wrong Input)", "color": "fail"})
                fail_reason = "wrong input"

            elapsed_ms = (current_time - self.start_time) * 1000.0 if self.start_time else None
            self.record_combo_fail(
                actual=input_name,
                expected_step_index=int(self.current_index),
                expected_label=self._expected_label_for_step(self._active_step()),
                reason=fail_reason,
                elapsed_ms=elapsed_ms,
            )
            self.current_index = 0
            self._reset_hold_state()
            self._reset_wait_state()
            self._reset_group_state()
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def process_release(self, input_name: str):
        # Thread-safe wrapper
        with self._lock:
            return self._process_release_unlocked(input_name)

    def _process_release_unlocked(self, input_name: str):
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return

        # Always update pressed-state on release (even if the release is consumed by group-hold logic).
        self.currently_pressed.discard(input_name)

        # Group hold handling (hold items inside []):
        step = self._active_step()
        if isinstance(step, dict) and step.get("group_presses") is not None and bool(step.get("group_hold_active")):
            hold_for = str(step.get("group_hold_for") or "").strip().lower()
            if input_name == hold_for:
                now = time.perf_counter()
                sig = str(step.get("group_hold_sig") or "").strip()
                req = int(step.get("group_hold_required_ms") or 0)
                started = float(step.get("group_hold_started_at") or 0.0)
                held_ms = (now - started) * 1000.0 if started else 0.0

                if req > 0 and held_ms >= float(req):
                    # Complete hold
                    hold_done_counts = step.get("group_hold_done_counts")
                    if not isinstance(hold_done_counts, dict):
                        hold_done_counts = {}
                        step["group_hold_done_counts"] = hold_done_counts
                    hold_done_counts[sig] = int(hold_done_counts.get(sig, 0)) + 1

                    step["group_hold_active"] = False
                    step["group_hold_for"] = ""
                    step["group_hold_sig"] = ""
                    step["group_hold_started_at"] = 0.0
                    step["group_hold_required_ms"] = 0
                    self._send({"type": "hold_end"})
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                else:
                    # Released too early: drop combo (hold too short)
                    self.record_hit(f"{input_name} (hold ≥ {req}ms, {held_ms:.0f}ms)", "FAIL", "FAIL")
                    self._send({"type": "status", "text": "Combo Dropped (Hold Too Short)", "color": "fail"})
                    elapsed_ms = (now - self.start_time) * 1000.0 if self.start_time else None
                    self.record_combo_fail(
                        actual=f"released @ {held_ms:.0f}ms",
                        expected_step_index=int(self.current_index),
                        expected_label=self._expected_label_for_step(step),
                        reason="hold too short",
                        elapsed_ms=elapsed_ms,
                    )
                    self.current_index = 0
                    self._reset_hold_state()
                    self._reset_wait_state()
                    self._reset_group_state()
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                return

        if not self.active_combo_steps:
            return

        step = self._active_step()
        if not step:
            return

        target_input = step.get("input")
        target_hold_ms = step.get("hold_ms")
        if target_hold_ms is None:
            return

        if not self.hold_in_progress or self.hold_expected_input != input_name:
            return
        if input_name != target_input:
            return

        now = time.perf_counter()
        self._complete_hold(now, auto=False)

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
            if not self.active_combo_steps:
                return
            if not self.start_time or not self.last_input_time:
                # Don't auto-advance anything before an attempt has started.
                return

            now = time.perf_counter()

            # 1) Normal wait steps
            step = self._active_step()
            if isinstance(step, dict) and step.get("wait_ms") is not None:
                # Ensure the wait timer has started
                self._maybe_start_wait_step()
                if self.wait_in_progress and now >= float(self.wait_until or 0.0):
                    self._complete_wait(now, fail=False)
                    st = self.get_status()
                    self._send({"type": "status", "text": st.text, "color": st.color})

                    # If we just advanced past the end, finish combo.
                    if self.current_index >= len(self.active_combo_steps):
                        total_ms = (now - self.start_time) * 1000 if self.start_time else 0.0
                        self._send(
                            {"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"}
                        )
                        self.record_combo_success(total_ms)
                        self.current_index = 0
                        self._reset_hold_state()
                        self._reset_wait_state()
                        self._reset_group_state()
                        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                    return

            # 2) Group internal waits (mandatory wait(...) and press+wait items inside [])
            step = self._active_step()
            if not (isinstance(step, dict) and step.get("group_presses") is not None):
                return

            opts = [str(x or "").strip().lower() for x in (step.get("group_presses") or [])]
            opts = [o for o in opts if o]
            need_counts = step.get("group_press_need_counts")
            if not isinstance(need_counts, dict) or not need_counts:
                need_counts = {k: 1 for k in opts}
                step["group_press_need_counts"] = need_counts

            done_counts = step.get("group_done_counts")
            if not isinstance(done_counts, dict):
                done_counts = {}
                step["group_done_counts"] = done_counts

            pw_need_counts = step.get("group_pw_need_counts")
            pw_need_counts = pw_need_counts if isinstance(pw_need_counts, dict) else {}
            pw_done_counts = step.get("group_pw_done_counts")
            if not isinstance(pw_done_counts, dict):
                pw_done_counts = {}
                step["group_pw_done_counts"] = pw_done_counts
            pw_meta = step.get("group_pw_meta")
            pw_meta = pw_meta if isinstance(pw_meta, dict) else {}
            pw_order_sigs = step.get("group_pw_order_sigs")
            pw_order_sigs = pw_order_sigs if isinstance(pw_order_sigs, list) else []

            pw_keys = set()
            for _sig, meta in pw_meta.items():
                k = str((meta or {}).get("input") or "").strip().lower()
                if k:
                    pw_keys.add(k)

            mw = step.get("group_mandatory_wait")
            mw_for = ""
            mw_ms = None
            if isinstance(mw, dict):
                mw_for = str(mw.get("wait_for") or "").strip().lower()
                mw_ms = mw.get("wait_ms")
            has_mw = (mw_ms is not None and mw_for)

            changed = False

            # 2a) Mandatory animation lock inside group
            if has_mw and bool(step.get("group_wait_active")) and now >= float(step.get("group_wait_until") or 0.0):
                step["group_wait_active"] = False
                step["group_wait_done"] = True
                self._send({"type": "wait_end"})
                changed = True

            # 2b) Press+wait item inside group
            if bool(step.get("group_pw_active")) and now >= float(step.get("group_pw_until") or 0.0):
                sig = str(step.get("group_pw_sig") or "").strip()
                if sig:
                    pw_done_counts[sig] = int(pw_done_counts.get(sig, 0)) + 1
                step["group_pw_active"] = False
                step["group_pw_sig"] = ""
                step["group_pw_until"] = 0.0
                self._send({"type": "wait_end"})
                changed = True

            if changed:
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                st = self.get_status()
                self._send({"type": "status", "text": st.text, "color": st.color})

            # If the group is now complete, advance immediately.
            presses_done = all(int(done_counts.get(k, 0)) >= int(need_counts.get(k, 0)) for k in need_counts.keys())
            pw_done_ok = all(int(pw_done_counts.get(sig, 0)) >= int(pw_need_counts.get(sig, 0)) for sig in pw_need_counts.keys())
            mw_done = (not has_mw) or bool(step.get("group_wait_done"))
            if presses_done and pw_done_ok and mw_done:
                self._mark_step(int(self.current_index), "ok")
                self.current_index += 1
                total_ms = (now - self.start_time) * 1000 if self.start_time else 0.0
                if self._maybe_complete_combo_if_trailing_wait(now=now, total_ms=total_ms):
                    return
                self._maybe_start_wait_step()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})
                st = self.get_status()
                self._send({"type": "status", "text": st.text, "color": st.color})
        except Exception:
            return