import unittest

from combo_engine import ComboTrackerEngine
from parser import expanded_ast_from_tokens, split_inputs
from state_store import MemoryStateStore
from states import build_runtime_state


def _engine_with_combo(inputs: str) -> ComboTrackerEngine:
    engine = ComboTrackerEngine(state_store=MemoryStateStore())
    engine.active_combo_name = "_input_test"
    engine.active_combo_tokens = split_inputs(inputs)
    engine.runtime_steps = [
        build_runtime_state(node)
        for node in expanded_ast_from_tokens(engine.active_combo_tokens)
    ]
    engine.reset_tracking()
    return engine


class InputDetectionTests(unittest.TestCase):
    def test_tick_runtime_errors_are_observable(self):
        engine = ComboTrackerEngine(state_store=MemoryStateStore())
        engine.runtime_steps = [object()]
        engine.start_time = 1.0
        engine.last_input_time = 1.0

        with self.assertRaisesRegex(RuntimeError, "tick transition failed"):
            engine.tick()

    def test_os_repeat_down_cannot_complete_two_identical_steps(self):
        engine = _engine_with_combo("e, e, q")

        engine.process_press("e", event_time=100.0)
        self.assertEqual(engine.current_index, 1)

        engine.process_press("e", event_time=100.1)
        self.assertEqual(engine.current_index, 1)

        engine.process_release("e", event_time=100.2)
        engine.process_press("e", event_time=100.3)
        self.assertEqual(engine.current_index, 2)

    def test_ingress_timestamp_anchors_attempt_before_lock_processing(self):
        engine = _engine_with_combo("f, q")

        engine.process_press("f", event_time=123.456)

        self.assertEqual(engine.start_time, 123.456)
        self.assertEqual(engine.last_input_time, 123.456)


if __name__ == "__main__":
    unittest.main()
