import threading
import unittest

from combo_engine import ComboTrackerEngine
from parser import expanded_ast_from_tokens, split_inputs
from state_store import MemoryStateStore
from states import HoldState, SequenceState, WaitState, build_runtime_state


class ManualMonotonicClock:
    def __init__(self, initial: float = 1.0) -> None:
        if initial <= 0:
            raise ValueError("Engine clocks must start above its idle sentinel")
        self._value = float(initial)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("A monotonic clock cannot move backwards")
        with self._lock:
            self._value += float(seconds)
            return self._value


def _engine_with_combo(inputs: str, clock: ManualMonotonicClock) -> ComboTrackerEngine:
    engine = ComboTrackerEngine(state_store=MemoryStateStore(), monotonic_now=clock)
    engine.active_combo_name = "_clock_test"
    engine.active_combo_tokens = split_inputs(inputs)
    engine.runtime_steps = [
        build_runtime_state(node)
        for node in expanded_ast_from_tokens(engine.active_combo_tokens)
    ]
    engine.reset_tracking()
    return engine


class ManualMonotonicClockTests(unittest.TestCase):
    def test_clock_rejects_backwards_time(self):
        clock = ManualMonotonicClock(10.0)

        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            clock.advance(-0.001)

        self.assertEqual(clock(), 10.0)

    def test_ender_cooldown_uses_injected_clock(self):
        clock = ManualMonotonicClock(10.0)
        engine = ComboTrackerEngine(state_store=MemoryStateStore(), monotonic_now=clock)
        engine.combo_enders = {"q": 100}
        engine._start_ender_cooldown("q", clock())

        self.assertTrue(engine._ender_on_cooldown("q"))
        clock.advance(0.101)
        self.assertFalse(engine._ender_on_cooldown("q"))

    def test_ender_policy_reuses_the_input_event_timestamp(self):
        clock = ManualMonotonicClock(999.0)
        engine = ComboTrackerEngine(state_store=MemoryStateStore(), monotonic_now=clock)
        engine.combo_enders = {"q": 100}
        engine.last_input_time = 10.0
        engine._ender_cooldown_until["q"] = 10.1

        self.assertFalse(engine._ender_can_drop_now("q", now=10.05))
        self.assertTrue(engine._ender_can_drop_now("q", now=10.101))

    def test_tick_advances_wait_without_sleeping_or_timestamp_mutation(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("f, wait:0.1s, q", clock)

        engine.process_press("f")
        sequence = engine.runtime_steps[0]
        self.assertIsInstance(sequence, SequenceState)

        clock.advance(0.099)
        engine.tick()
        self.assertEqual(engine.current_index, 0)

        clock.advance(0.002)
        engine.tick()
        self.assertEqual(engine.current_index, 1)
        self.assertTrue(sequence.steps[1].completed)

    def test_top_level_wait_uses_one_authoritative_start_timestamp(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("f, wait(r, 0.1s), q", clock)

        engine.process_press("f")
        engine.process_release("f")
        engine.process_press("r")
        wait = engine.runtime_steps[2]
        self.assertIsInstance(wait, WaitState)
        self.assertEqual(wait.started_at, 10.0)
        self.assertEqual(engine.wait_started_at, 10.0)

        clock.advance(0.099)
        engine.tick()
        self.assertEqual(engine.current_index, 2)

        clock.advance(0.001001)
        engine.tick()
        self.assertEqual(engine.current_index, 3)

    def test_hold_auto_completion_uses_injected_clock(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("f, hold(e, 0.1s), q", clock)

        engine.process_press("f")
        engine.process_release("f")
        engine.process_press("e")
        hold = engine.runtime_steps[1]
        self.assertIsInstance(hold, HoldState)

        clock.advance(0.099)
        engine.tick()
        self.assertEqual(engine.current_index, 1)

        clock.advance(0.001001)
        engine.tick()
        self.assertEqual(engine.current_index, 2)

    def test_attempt_rows_reuse_the_transition_timestamp(self):
        clock = ManualMonotonicClock(10.0)
        engine = ComboTrackerEngine(state_store=MemoryStateStore(), monotonic_now=clock)
        engine.start_time = 9.0
        engine._attempt_hit_clock = 9.5

        engine._record_attempt_hit_delta("f", 9.75)

        self.assertEqual(engine._attempt_hit_clock, 9.75)
        self.assertEqual(clock(), 10.0)

    def test_replay_timing_uses_plan_time_not_callback_delivery_time(self):
        clock = ManualMonotonicClock(50.0)
        engine = _engine_with_combo("f, q, r", clock)

        engine.replay_accept("f", 20.0, 20.0)
        self.assertAlmostEqual(engine.start_time, 49.98)
        self.assertAlmostEqual(engine._attempt_hit_clock, 50.0)

        # Simulate a heavily delayed UI callback. Logical replay timing must
        # remain on the compiled plan rather than jumping to wall time.
        clock.advance(5.0)
        engine.replay_accept("q", 30.0, 50.0)
        self.assertAlmostEqual(engine._attempt_hit_clock, 50.03)


if __name__ == "__main__":
    unittest.main()
