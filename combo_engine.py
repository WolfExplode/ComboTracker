import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Status:
    text: str
    color: str  # ready|recording|success|fail|wait|neutral


class ComboTrackerEngine:
    """
    Headless combo tracker:
    - Owns combos + stats persistence
    - Owns state machine (press/hold/wait + ender-grace)
    - Emits UI events via a callback (WebSocket, etc.)
    """

    def __init__(self):
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

        # Stats
        self.combo_stats: dict[str, dict[str, Any]] = {}
        # Per-combo metadata (kept minimal on purpose)
        # - expected_ms: user-entered typical execution time (used for Practical APM / difficulty)
        self.combo_expected_ms: dict[str, int] = {}
        # - user_difficulty: user-entered difficulty rating (0..10)
        self.combo_user_difficulty: dict[str, float] = {}

        # Emission
        self._emit: Callable[[dict[str, Any]], None] | None = None

        # Persistence
        self.data_dir = self._get_data_dir()
        self.save_path = self.data_dir / "combos.json"

        # Load persisted state
        self.load_combos()

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
                pass

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
        Split the Inputs field into tokens by commas, but do NOT split commas that are
        inside hold(...) parentheses or inside {...} braces.
        """
        s = keys_str or ""
        out: list[str] = []
        buf: list[str] = []
        paren = 0
        brace = 0

        for ch in s:
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)

            if ch == "," and paren == 0 and brace == 0:
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
        t = (token or "").strip()
        if not t:
            return None

        tl = t.lower()

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
        if step.get("wait_ms") is not None:
            w = int(step.get("wait_ms") or 0)
            mode = str(step.get("wait_mode") or "soft").strip().lower()
            if mode == "hard":
                return f"wait-hard(≥{w}ms)"
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

    # -------------------------
    # Persistence
    # -------------------------

    def _get_data_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def load_combos(self):
        try:
            if not self.save_path.exists():
                return

            data = json.loads(self.save_path.read_text(encoding="utf-8"))

            combos = data.get("combos", {})
            if isinstance(combos, dict):
                sanitized: dict[str, list[str]] = {}
                for name, seq in combos.items():
                    if not isinstance(name, str) or not isinstance(seq, list):
                        continue
                    sanitized[name] = [str(x).strip().lower() for x in seq if str(x).strip()]
                self.combos = sanitized

            enders = data.get("combo_enders", {})
            parsed: dict[str, int] = {}
            if isinstance(enders, dict):
                for k, v in enders.items():
                    key = str(k).strip().lower()
                    if not key:
                        continue
                    try:
                        ms = int(float(v))
                    except Exception:
                        ms = 0
                    parsed[key] = max(0, ms)
            elif isinstance(enders, list):
                for x in enders:
                    key = str(x).strip().lower()
                    if key:
                        parsed[key] = 0
            self.combo_enders = parsed

            stats = data.get("combo_stats", {})
            if isinstance(stats, dict):
                cleaned: dict[str, dict[str, Any]] = {}
                for k, v in stats.items():
                    name = str(k).strip()
                    if not name or not isinstance(v, dict):
                        continue

                    def _as_int(x, default=0):
                        try:
                            return int(x)
                        except Exception:
                            return default

                    s = max(0, _as_int(v.get("success", 0), 0))
                    f = max(0, _as_int(v.get("fail", 0), 0))

                    best_ms = v.get("best_ms", None)
                    try:
                        best_ms = int(best_ms) if best_ms is not None else None
                        if best_ms is not None and best_ms <= 0:
                            best_ms = None
                    except Exception:
                        best_ms = None

                    total_success_ms = max(0, _as_int(v.get("total_success_ms", 0), 0))

                    def _clean_counter_dict(d: Any, key_norm: Callable[[Any], str] | None = None):
                        if not isinstance(d, dict):
                            return {}
                        out: dict[str, int] = {}
                        for kk, vv in d.items():
                            key = key_norm(kk) if key_norm else str(kk)
                            key = str(key).strip()
                            if not key:
                                continue
                            try:
                                cnt = int(vv)
                            except Exception:
                                continue
                            if cnt <= 0:
                                continue
                            out[key] = cnt
                        return out

                    fail_by_step = _clean_counter_dict(
                        v.get("fail_by_step", {}),
                        key_norm=lambda kk: str(int(str(kk))),
                    )
                    fail_by_expected = _clean_counter_dict(
                        v.get("fail_by_expected", {}),
                        key_norm=lambda kk: str(kk).strip().lower(),
                    )
                    fail_by_reason = _clean_counter_dict(
                        v.get("fail_by_reason", {}),
                        key_norm=lambda kk: str(kk).strip().lower(),
                    )

                    fail_events = v.get("fail_events", [])
                    fes: list[dict[str, Any]] = []
                    if isinstance(fail_events, list):
                        for ev in fail_events[-100:]:
                            if not isinstance(ev, dict):
                                continue
                            fes.append(
                                {
                                    "ts": int(ev.get("ts", 0) or 0),
                                    "attempt": max(0, _as_int(ev.get("attempt", 0), 0)),
                                    "step_index": max(0, _as_int(ev.get("step_index", 0), 0)),
                                    "expected": str(ev.get("expected", "") or ""),
                                    "actual": str(ev.get("actual", "") or ""),
                                    "reason": str(ev.get("reason", "") or ""),
                                    "elapsed_ms": (
                                        _as_int(ev.get("elapsed_ms"), 0)
                                        if ev.get("elapsed_ms") is not None
                                        else None
                                    ),
                                }
                            )

                    cleaned[name] = {
                        "success": s,
                        "fail": f,
                        "best_ms": best_ms,
                        "total_success_ms": total_success_ms,
                        "fail_by_step": fail_by_step,
                        "fail_by_expected": fail_by_expected,
                        "fail_by_reason": fail_by_reason,
                        "fail_events": fes,
                    }
                self.combo_stats = cleaned

            # Optional: per-combo expected execution time (ms)
            exp = data.get("combo_expected_ms", {})
            expected_ms: dict[str, int] = {}
            if isinstance(exp, dict):
                for k, v in exp.items():
                    name = str(k).strip()
                    if not name:
                        continue
                    try:
                        ms = int(float(v))
                    except Exception:
                        continue
                    if ms > 0:
                        expected_ms[name] = ms
            self.combo_expected_ms = expected_ms

            # Optional: per-combo user difficulty (0..10)
            ud = data.get("combo_user_difficulty", {})
            user_diff: dict[str, float] = {}
            if isinstance(ud, dict):
                for k, v in ud.items():
                    name = str(k).strip()
                    if not name:
                        continue
                    try:
                        d = float(v)
                    except Exception:
                        continue
                    if 0.0 <= d <= 10.0:
                        user_diff[name] = d
            self.combo_user_difficulty = user_diff

            last_active = data.get("last_active_combo")
            if last_active in self.combos:
                self.set_active_combo(str(last_active), emit=False)
        except Exception:
            self.combos = {}
            self.combo_stats = {}
            self.combo_enders = {}
            self.combo_expected_ms = {}
            self.combo_user_difficulty = {}
            self.active_combo_name = None
            self.active_combo_tokens = []
            self.active_combo_steps = []

    def save_combos(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "last_active_combo": self.active_combo_name,
                "combos": self.combos,
                "combo_enders": dict(self.combo_enders),
                "combo_stats": dict(self.combo_stats),
                "combo_expected_ms": dict(self.combo_expected_ms),
                "combo_user_difficulty": dict(self.combo_user_difficulty),
            }
            self.save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

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
        name = self.active_combo_name
        if not name:
            return "Stats: —"
        self._ensure_combo_stats(name)
        s = int(self.combo_stats[name].get("success", 0))
        f = int(self.combo_stats[name].get("fail", 0))
        pct = self._format_percent(s, f)

        best = self.combo_stats[name].get("best_ms", None)
        avg = self._combo_avg_ms(name)

        # Hardest steps (top 2)
        hardest = ""
        by_step = self.combo_stats[name].get("fail_by_step", {})
        if isinstance(by_step, dict) and by_step:
            pairs: list[tuple[int, int]] = []
            for k, v in by_step.items():
                try:
                    idx = int(k)
                    cnt = int(v)
                except Exception:
                    continue
                if cnt <= 0:
                    continue
                pairs.append((cnt, idx))
            pairs.sort(reverse=True)
            parts: list[str] = []
            for cnt, idx in pairs[:2]:
                label = "—"
                if 0 <= idx < len(self.active_combo_steps):
                    label = self._expected_label_for_step(self.active_combo_steps[idx])
                parts.append(f"#{idx+1}:{label} ({cnt})")
            if parts:
                hardest = " | Hardest: " + ", ".join(parts)

        return (
            f"Stats: {s} success / {f} fail ({pct})"
            f" | Best: {self._format_ms_brief(best)} | Avg: {self._format_ms_brief(avg)}"
            f"{hardest}"
        )

    def failures_by_reason(self) -> dict[str, int]:
        name = self.active_combo_name
        if not name:
            return {}
        self._ensure_combo_stats(name)
        by_reason = self.combo_stats[name].get("fail_by_reason", {})
        if not isinstance(by_reason, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in by_reason.items():
            reason = str(k).strip() or "unknown"
            try:
                cnt = int(v)
            except Exception:
                cnt = 0
            if cnt > 0:
                out[reason] = cnt
        return out

    def min_time_text(self) -> str:
        if not self.active_combo_steps:
            return "Fastest possible: —"
        min_ms = self.calc_min_combo_time_ms(self.active_combo_steps)
        return f"Fastest possible: {self._format_ms(min_ms)}"

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
        """
        Practical APM uses user-entered expected execution time (ms) for the active combo.
        """
        name = self.active_combo_name
        if not name or not self.active_combo_steps:
            return None
        expected_ms = self.combo_expected_ms.get(name)
        if expected_ms is None or expected_ms <= 0:
            return None
        press_count, _hold_count, _actions = self._count_combo_actions(self.active_combo_steps)
        if press_count <= 0:
            return None
        return (60000.0 / float(expected_ms)) * float(press_count)

    def theoretical_max_apm(self) -> float | None:
        """
        Theoretical max APM uses the fastest-possible combo time (sum of waits + holds).
        """
        if not self.active_combo_name or not self.active_combo_steps:
            return None
        min_ms = self.calc_min_combo_time_ms(self.active_combo_steps)
        if min_ms <= 0:
            return None
        press_count, _hold_count, _actions = self._count_combo_actions(self.active_combo_steps)
        if press_count <= 0:
            return None
        return (60000.0 / float(min_ms)) * float(press_count)

    def apm_text(self) -> str:
        apm = self.practical_apm()
        if apm is None:
            return "Practical APM: —"
        return f"Practical APM: {apm:.1f}"

    def apm_max_text(self) -> str:
        apm = self.theoretical_max_apm()
        if apm is None:
            return "Theoretical max APM: —"
        return f"Theoretical max APM: {apm:.1f}"

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
        """
        Returns a 0..10 score (float) or None if there's no active combo.
        """
        if not self.active_combo_steps or not self.active_combo_name:
            return None

        # --- Keys camp (Practical APM + combo length) ---
        apm = self.practical_apm() or 0.0
        press_count, hold_count, actions = self._count_combo_actions(self.active_combo_steps)

        # --- Normalization / scaling constants ---
        # APM normalization: 200 APM maps to 1.0 (anything faster clamps).
        # This is intentionally "tunable": change 200.0 if your game’s APM scale differs.
        apm_norm = self._clamp01(apm / 200.0)

        # Action normalization: 8 actions maps to 1.0.
        # (actions = presses + holds; waits don't count as actions)
        actions_norm = self._clamp01(float(actions) / 8.0)

        # Keys weighting: prioritize speed (APM) over length, but keep length as a reliability tax.
        keys = (0.6 * apm_norm) + (0.4 * actions_norm)

        # --- Timing camp (wait + hold + simple variation points) ---
        wait_scores: list[float] = []
        hold_scores: list[float] = []
        for s in self.active_combo_steps:
            if not isinstance(s, dict):
                continue
            if s.get("wait_ms") is not None:
                try:
                    wait_scores.append(self._wait_triangle_score(int(s.get("wait_ms") or 0)))
                except Exception:
                    continue
            elif s.get("hold_ms") is not None:
                try:
                    hold_scores.append(self._hold_score(int(s.get("hold_ms") or 0)))
                except Exception:
                    continue

        has_wait = 1.0 if wait_scores else 0.0
        has_hold = 1.0 if hold_scores else 0.0
        wait_avg = (sum(wait_scores) / len(wait_scores)) if wait_scores else 0.0
        hold_avg = (sum(hold_scores) / len(hold_scores)) if hold_scores else 0.0

        # Holds weighted higher than waits because they "commit" a finger and restrict spamming.
        # Increase hold_w to make holds matter more relative to waits.
        wait_w = 1.0
        hold_w = 1.5
        denom = (wait_w * has_wait) + (hold_w * has_hold)
        timing_base = 0.0 if denom <= 0 else ((wait_avg * wait_w * has_wait) + (hold_avg * hold_w * has_hold)) / denom

        var_points = self._timing_variation_points()

        # Variation scaling (diminishing returns):
        # We want 1–2 distinct timings to be a big penalty (hard to adapt),
        # but 3–4 to add less (at that point it starts feeling "random" anyway).
        #
        # This uses an exponential saturation curve:
        #   var_norm = 1 - exp(-var_points / K)
        # where K controls how fast it saturates.
        #
        # With K=1.0:
        # - 0 -> 0.000
        # - 1 -> 0.632
        # - 2 -> 0.865
        # - 3 -> 0.950
        # - 4 -> 0.982
        #
        # Increase K to make variation matter less; decrease K to make it spike faster.
        K = 1.0
        var_norm = self._clamp01(1.0 - (2.718281828 ** (-float(var_points) / K)))

        # Timing blend:
        # - timing_base = "how hard are the raw waits/holds"
        # - var_norm    = "how many distinct timing rules do I have to remember"
        # Increase the var_norm weight if you want "one weird timing" to dominate difficulty.
        timing = (0.3 * self._clamp01(timing_base)) + (0.7 * var_norm)

        # Overall blend: slightly emphasize timing over raw key speed/length.
        # Adjust these if you want APM to dominate or timing to dominate.
        combined = (0.45 * keys) + (0.55 * timing)
        return round(10.0 * self._clamp01(combined), 1)

    def difficulty_text(self) -> str:
        d = self.difficulty_score_10()
        if d is None:
            return "Difficulty: —"
        return f"Difficulty: {d:.1f} / 10"

    def user_difficulty_value(self) -> float | None:
        name = self.active_combo_name
        if not name:
            return None
        d = self.combo_user_difficulty.get(name)
        if d is None:
            return None
        try:
            d_f = float(d)
        except Exception:
            return None
        if 0.0 <= d_f <= 10.0:
            return d_f
        return None

    def user_difficulty_text(self) -> str:
        d = self.user_difficulty_value()
        if d is None:
            return "Your difficulty: —"
        return f"Your difficulty: {d:g} / 10"

    # -------------------------
    # UI state snapshots
    # -------------------------

    def get_editor_payload(self) -> dict[str, Any]:
        name = self.active_combo_name or ""
        inputs = ", ".join(self.active_combo_tokens) if self.active_combo_tokens else ""

        enders = ""
        if self.combo_enders:
            parts: list[str] = []
            for k in sorted(self.combo_enders.keys()):
                ms = int(self.combo_enders[k])
                if ms > 0:
                    parts.append(f"{k}:{ms/1000.0:.3g}")
                else:
                    parts.append(k)
            enders = ", ".join(parts)

        expected = ""
        if name:
            ms = self.combo_expected_ms.get(name)
            if ms is not None:
                expected = self._format_ms_brief(ms)
        user_diff = ""
        if name:
            d = self.combo_user_difficulty.get(name)
            if d is not None:
                # Keep it friendly for editing (no trailing .0)
                user_diff = f"{d:g}"
        return {
            "name": name,
            "inputs": inputs,
            "enders": enders,
            "expected_time": expected,
            "user_difficulty": user_diff,
        }

    def get_status(self) -> Status:
        if not self.active_combo_steps:
            return Status("Status: Select a combo to start", "neutral")

        step = self._active_step()
        if not step:
            return Status("Status: Select a combo to start", "neutral")

        if self.current_index == 0:
            start_key = str(step.get("input") or "").upper()
            if step.get("hold_ms") is None:
                return Status(f"Ready! Press '{start_key}' to start.", "ready")
            return Status(
                f"Ready! Hold '{start_key}' for {int(step.get('hold_ms') or 0)}ms to start.",
                "ready",
            )

        if self.wait_in_progress:
            req = self._format_hold_requirement(int(self.wait_required_ms or 0))
            return Status(f"Waiting ≥ {req}...", "wait")
        if self.hold_in_progress:
            req = self._format_hold_requirement(int(self.hold_required_ms or 0))
            inp = str(self.hold_expected_input or "").upper()
            return Status(f"Holding '{inp}' (≥ {req}). Release OR press next input to continue...", "recording")
        return Status("Recording...", "recording")

    def timeline_steps(self) -> list[dict[str, Any]]:
        steps = []
        for idx, s in enumerate(self.active_combo_steps or []):
            mark = self.step_marks.get(idx)
            if s.get("wait_ms") is not None:
                steps.append(
                    {
                        "type": "wait",
                        "input": None,
                        "duration": int(s.get("wait_ms") or 0),
                        "mode": str(s.get("wait_mode") or "soft"),
                        "active": idx == self.current_index,
                        "completed": idx < self.current_index,
                        "mark": mark,
                    }
                )
            elif s.get("hold_ms") is not None:
                steps.append(
                    {
                        "type": "hold",
                        "input": str(s.get("input") or ""),
                        "duration": int(s.get("hold_ms") or 0),
                        "active": idx == self.current_index,
                        "completed": idx < self.current_index,
                        "mark": mark,
                    }
                )
            else:
                steps.append(
                    {
                        "type": "press",
                        "input": str(s.get("input") or ""),
                        "duration": 0,
                        "active": idx == self.current_index,
                        "completed": idx < self.current_index,
                        "mark": mark,
                    }
                )
        return steps

    def init_payload(self) -> dict[str, Any]:
        st = self.get_status()
        return {
            "type": "init",
            "combos": sorted(self.combos.keys()),
            "active_combo": self.active_combo_name,
            "status": {"text": st.text, "color": st.color},
            "stats": self.stats_text(),
            "min_time": self.min_time_text(),
            "difficulty": self.difficulty_text(),
            "difficulty_value": self.difficulty_score_10(),
            "user_difficulty": self.user_difficulty_text(),
            "user_difficulty_value": self.user_difficulty_value(),
            "apm": self.apm_text(),
            "apm_max": self.apm_max_text(),
            "timeline": self.timeline_steps(),
            "failures": self.failures_by_reason(),
            "editor": self.get_editor_payload(),
        }

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
    ) -> tuple[bool, str | None]:
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

        self._ensure_combo_stats(name)
        self.set_active_combo(name, emit=False)
        self.save_combos()

        # Broadcast new global+active state
        self._send({"type": "init", **self.init_payload()})
        return True, None

    def delete_combo(self, name: str) -> tuple[bool, str | None]:
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

        if self.active_combo_name == name:
            self.active_combo_name = None
            self.active_combo_tokens = []
            self.active_combo_steps = []
            self.reset_tracking()

        self.save_combos()
        self._send({"type": "init", **self.init_payload()})
        return True, None

    def new_combo(self):
        self.active_combo_name = None
        self.active_combo_tokens = []
        self.active_combo_steps = []
        self.reset_tracking()
        self._send({"type": "init", **self.init_payload()})

    def clear_history_and_stats(self):
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
        self._send({"type": "stat_update", "stats": self.stats_text()})
        self._send({"type": "fail_update", "failures": self.failures_by_reason()})
        self._send({"type": "timeline_update", "steps": self.timeline_steps()})
        st = self.get_status()
        self._send({"type": "status", "text": st.text, "color": st.color})

    def set_active_combo(self, name: str, *, emit: bool = True):
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
            if s:
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
        self._reset_attempt_marks()
        self._reset_hold_state()
        self._reset_wait_state()

    def _active_step(self):
        if 0 <= self.current_index < len(self.active_combo_steps):
            return self.active_combo_steps[self.current_index]
        return None

    def _insert_attempt_separator(self):
        self.attempt_counter += 1
        name = self.active_combo_name or "Combo"
        # New attempt → clear any per-step failure coloring from the previous attempt.
        self._reset_attempt_marks()
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
        self.wait_in_progress = False
        self.wait_started_at = 0.0
        self.wait_until = 0.0
        self.wait_required_ms = None

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
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            return False

        split_ms = (now - self.last_input_time) * 1000 if self.last_input_time else 0.0
        self.record_hit(label, split_ms, total_ms)
        self.last_input_time = now
        self.current_index += 1
        self._reset_wait_state()
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
            self._maybe_start_wait_step()

            if self.current_index >= len(self.active_combo_steps):
                self._send({"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"})
                self.record_combo_success(total_ms)
                self.current_index = 0
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
                    if target_wait_mode != "hard":
                        next_idx = self._next_non_wait_step_index(start_index=self.current_index + 1)
                        next_expected = None
                        if next_idx is not None:
                            try:
                                next_expected = str(self.active_combo_steps[next_idx].get("input") or "").strip().lower()
                            except Exception:
                                next_expected = None
                        if next_expected and input_name == next_expected:
                            wi = int(self.current_index)
                            self.wait_early_inputs.setdefault(wi, set()).add(input_name)
                            self._mark_step(wi, "early")
                            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                    # Soft wait: ignore any early presses (even enders).
                    # Hard wait: early press can drop the combo (models games where early input consumes/cancels).
                    if target_wait_mode == "hard":
                        # Fail on enders, and also fail if the pressed input matches the next expected non-wait input.
                        next_expected = None
                        try:
                            for j in range(self.current_index + 1, len(self.active_combo_steps)):
                                st = self.active_combo_steps[j]
                                if isinstance(st, dict) and st.get("wait_ms") is None:
                                    next_expected = str(st.get("input") or "").strip().lower() or None
                                    break
                        except Exception:
                            next_expected = None

                        if self._is_combo_ender(input_name) or (next_expected and input_name == next_expected):
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
                self._maybe_start_wait_step()
                self._send({"type": "timeline_update", "steps": self.timeline_steps()})

                if self.current_index >= len(self.active_combo_steps):
                    self._send(
                        {"type": "status", "text": f"Combo '{self.active_combo_name}' Complete!", "color": "success"}
                    )
                    self.record_combo_success(total_ms)
                    self.current_index = 0
                    self._send({"type": "timeline_update", "steps": self.timeline_steps()})
            else:
                self._start_hold(input_name, int(target_hold_ms), current_time)
            return

        # Miss
        if self._is_combo_ender(input_name):
            if self._should_ignore_ender_miss(input_name):
                return

            # More helpful messaging: detect "out of order" presses (likely skipped an expected step).
            expected = str(target_input or "").strip().lower()
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
            self._send({"type": "timeline_update", "steps": self.timeline_steps()})

    def process_release(self, input_name: str):
        input_name = (input_name or "").strip().lower()
        if not input_name:
            return

        self.currently_pressed.discard(input_name)
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

