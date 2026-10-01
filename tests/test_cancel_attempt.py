import unittest

from combo_engine import ComboTrackerEngine
from parser import expanded_ast_from_tokens, split_inputs
from state_store import MemoryStateStore
from states import build_runtime_state


def _engine_with_combo(inputs: str) -> tuple[ComboTrackerEngine, list[dict]]:
    sent: list[dict] = []
    engine = ComboTrackerEngine(state_store=MemoryStateStore())
    engine.set_emitter(sent.append)
    engine.active_combo_name = "_cancel_test"
    engine.active_combo_tokens = split_inputs(inputs)
    engine.runtime_steps = [build_runtime_state(n) for n in expanded_ast_from_tokens(engine.active_combo_tokens)]
    engine.reset_tracking()
    return engine, sent


def _types(sent: list[dict]) -> list[str]:
    return [m.get("type") for m in sent]


class CancelAttemptTests(unittest.TestCase):
    """Esc mid-wait must tell the UI the wait ended, or its fill animation latches onto the reset timeline."""

    def test_esc_inside_group_animation_lock_ends_the_wait_animation(self):
        engine, sent = _engine_with_combo("f, [q, e, wait(r, 2.9s)], 2")
        engine.process_press("f")
        engine.process_release("f")
        engine.process_press("r")  # starts the group's 2.9s animation lock
        self.assertIn("wait_begin", _types(sent))

        sent.clear()
        engine.cancel_attempt()

        self.assertIn("wait_end", _types(sent))
        self.assertEqual(engine.current_index, 0)

    def test_esc_with_no_wait_running_sends_no_wait_end(self):
        engine, sent = _engine_with_combo("f, q, e")
        engine.process_press("f")
        engine.process_release("f")
        sent.clear()
        engine.cancel_attempt()
        self.assertNotIn("wait_end", _types(sent))


if __name__ == "__main__":
    unittest.main()
