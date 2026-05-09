"""Regression tests for macro karaoke replay_accept (SequenceState press+wait)."""

import unittest

from combo_engine import ComboTrackerEngine
from parser import split_inputs, expanded_ast_from_tokens
from states import SequenceState, build_runtime_state


def _engine_with_combo(inputs: str) -> ComboTrackerEngine:
    e = ComboTrackerEngine()
    e.active_combo_name = "_test"
    e.active_combo_tokens = split_inputs(inputs)
    ast_steps = expanded_ast_from_tokens(e.active_combo_tokens)
    e.runtime_steps = [build_runtime_state(n) for n in ast_steps]
    e.reset_tracking()
    return e


class ReplayAcceptTests(unittest.TestCase):
    def test_press_soft_wait_sequence_advances_inner_not_top_level(self):
        """`e, wait:Xs` merges to SequenceState; replay must not bump past the whole sequence on `e` alone."""
        e = _engine_with_combo("f, e, wait:0.4s, lmb")
        self.assertEqual(len(e.runtime_steps), 3)
        self.assertIsInstance(e.runtime_steps[1], SequenceState)

        e.replay_accept("f", 10.0, 10.0)
        self.assertEqual(e.current_index, 1)

        seq = e.runtime_steps[1]
        assert isinstance(seq, SequenceState)
        self.assertEqual(seq.current_index, 0)

        e.replay_accept("e", 10.0, 25.0)
        self.assertEqual(e.current_index, 1)
        self.assertEqual(seq.current_index, 1)
        inner_wait = seq.steps[1]
        self.assertTrue(getattr(inner_wait, "in_progress", False))

    def test_cancel_attempt_clears_success_snapshot(self):
        e = _engine_with_combo("f")
        e._ui_last_success_combo = "_test"
        e._ui_last_success_steps_len = len(e.runtime_steps)
        e.cancel_attempt()
        self.assertIsNone(e._ui_last_success_combo)
        self.assertEqual(e._ui_last_success_steps_len, 0)

    def test_clear_blocking_waits_advances_sequence_inner_wait(self):
        """_replay_clear_blocking_waits must force-complete an in-progress inner wait
        inside a SequenceState, regardless of elapsed time."""
        import time as _time
        e = _engine_with_combo("f, e, wait:5s, lmb")
        seq = e.runtime_steps[1]
        self.assertIsInstance(seq, SequenceState)

        # Simulate: "e" was already processed, inner wait started but 5s hasn't passed.
        e.start_time = _time.perf_counter()
        e.last_input_time = e.start_time
        e.current_index = 1
        seq.started = True
        seq.current_index = 1          # pointing at the WaitState
        inner_wait = seq.steps[1]
        inner_wait.in_progress = True
        inner_wait.started_at = _time.perf_counter()  # just started — nowhere near 5s

        # Act: next replay event arrives (lmb). Force-clear should advance past the wait.
        now = _time.perf_counter()
        e._replay_clear_blocking_waits(now)

        # The sequence should be done and current_index advanced to lmb step.
        self.assertEqual(e.current_index, 2)
        self.assertTrue(inner_wait.completed)


if __name__ == "__main__":
    unittest.main()
