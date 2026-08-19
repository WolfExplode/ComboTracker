import time
import unittest

from combo_engine import ComboTrackerEngine
from parser import split_inputs, expanded_ast_from_tokens
from state_store import MemoryStateStore
from states import HoldState, SequenceState, WaitState, build_runtime_state
import combo_engine_ui as ui


def _build_engine_for_inputs(inputs: str) -> ComboTrackerEngine:
    e = ComboTrackerEngine(state_store=MemoryStateStore())
    e.active_combo_name = "_test"
    e.active_combo_tokens = split_inputs(inputs)
    ast_steps = expanded_ast_from_tokens(e.active_combo_tokens)
    e.runtime_steps = [build_runtime_state(n) for n in ast_steps]
    e.current_index = 0
    for s in e.runtime_steps:
        s.reset()
    return e


class TimelineViewModelTests(unittest.TestCase):
    def test_sequence_collapses_plain_press_wait(self):
        # inside { ... } we should not emit separate press then wait tiles for plain waits
        combo = "[q, {lmb, wait:0.40s, rmb, wait:0.30s}]"
        e = _build_engine_for_inputs(combo)
        steps = ui.timeline_steps(e, time.perf_counter())
        group = next(s for s in steps if s.get("type") == "group")
        seq = next(it for it in group.get("items", []) if it.get("type") == "sequence")
        items = seq.get("items", [])
        self.assertEqual(items[0]["type"], "press_wait")
        self.assertEqual(items[0]["input"], "lmb")
        self.assertEqual(items[0]["duration"], 400)
        self.assertEqual(items[1]["type"], "press_wait")
        self.assertEqual(items[1]["input"], "rmb")
        self.assertEqual(items[1]["duration"], 300)

    def test_sequence_collapses_anim_wait_composite(self):
        combo = "f, [q, {lmb, wait:0.40s, wait(lmb, 2.9s), lmb, wait:0.30s, lmb}], 2"
        e = _build_engine_for_inputs(combo)
        steps = ui.timeline_steps(e, time.perf_counter())
        group = next(s for s in steps if s.get("type") == "group")
        seq = next(it for it in group.get("items", []) if it.get("type") == "sequence")
        items = seq.get("items", [])
        # expected: press_wait(400), wait(mandatory, 2900), press_wait(300), press
        self.assertEqual([it["type"] for it in items], ["press_wait", "wait", "press_wait", "press"])
        self.assertEqual(items[1]["mode"], "mandatory")
        self.assertEqual(items[1]["wait_for"], "lmb")
        self.assertEqual(items[1]["duration"], 2900)

    def test_buffered_hold_starts_after_mandatory_wait(self):
        # Scenario: during a mandatory wait (animation lock), user holds the next hold key.
        # When the lock ends, the hold should start immediately (buffered input).
        combo = "f, lmb, wait:1s, 2, hold(lmb, 0.3s), wait(1, 1.17s), hold(lmb, 0.60s)"
        e = _build_engine_for_inputs(combo)
        e.reset_tracking()

        # Start attempt: press f, then lmb to reach wait:1s (now one SequenceState: lmb + wait)
        e.process_press("f")
        e.process_press("lmb")

        # Force complete wait:1s without sleeping (wait is inside SequenceState)
        step = e._active_runtime_step()
        if isinstance(step, SequenceState):
            self.assertGreaterEqual(step.current_index, 0)
            wait_step = step.steps[step.current_index]
            self.assertIsInstance(wait_step, WaitState)
        else:
            self.assertIsInstance(step, WaitState)
            wait_step = step
        wait_step.started_at = time.perf_counter() - 2.0
        wait_step.in_progress = True
        e.start_time = e.start_time or time.perf_counter()
        e.last_input_time = e.last_input_time or time.perf_counter()
        e.tick()

        # press 2
        e.process_press("2")
        # start hold(lmb,0.3): press lmb then release after forcing enough held time
        e.process_press("lmb")
        hs = e._active_runtime_step()
        self.assertIsInstance(hs, HoldState)
        hs.started_at = time.perf_counter() - 1.0
        e.hold_started_at = hs.started_at
        e.process_release("lmb")

        # start mandatory wait(1,1.17s) by pressing 1 (engine advances into a WaitState)
        e.process_press("1")
        ws = e._active_runtime_step()
        self.assertIsInstance(ws, WaitState)
        self.assertEqual(ws.mode, "mandatory")
        # during the lock, press+hold lmb (buffer)
        e.process_press("lmb")

        # Force complete the mandatory wait (no sleeping)
        ws.started_at = time.perf_counter() - 5.0
        ws.in_progress = True
        e.tick()

        self.assertTrue(getattr(e, "hold_in_progress", False))
        self.assertEqual((getattr(e, "hold_expected_input", "") or "").lower(), "lmb")

    def test_press_during_hold_can_autocomplete_without_recursion(self):
        # Regression: if a hold has satisfied its duration and you press another key while
        # still holding the hold key, the engine should auto-complete the hold and continue
        # without infinite recursion.
        combo = "f, 2, hold(lmb, 0.3s), wait(1, 1.17s), hold(lmb, 0.60s)"
        e = _build_engine_for_inputs(combo)
        e.reset_tracking()

        e.process_press("f")
        e.process_press("2")
        e.process_press("lmb")  # start hold

        # Force the hold to be long enough
        hs = e._active_runtime_step()
        self.assertIsInstance(hs, HoldState)
        hs.started_at = time.perf_counter() - 1.0
        e.hold_started_at = hs.started_at

        # Press the next key while still holding lmb (this used to recurse forever)
        e.process_press("1")

        # Should have advanced off the hold step (either into wait(1,...) press or beyond)
        self.assertNotIsInstance(e._active_runtime_step(), HoldState)


if __name__ == "__main__":
    unittest.main()

