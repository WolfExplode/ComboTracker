"""Regression tests for macro karaoke replay_accept (SequenceState press+wait)."""

import time
import unittest

from combo_engine import ComboTrackerEngine
from parser import split_inputs, expanded_ast_from_tokens
from macro_player import MacroPlayer, _PlanBuilder
from states import GroupState, HoldState, SequenceState, build_runtime_state


def _engine_with_combo(inputs: str) -> ComboTrackerEngine:
    e = ComboTrackerEngine()
    e.active_combo_name = "_test"
    e.active_combo_tokens = split_inputs(inputs)
    ast_steps = expanded_ast_from_tokens(e.active_combo_tokens)
    e.runtime_steps = [build_runtime_state(n) for n in ast_steps]
    e.reset_tracking()
    return e


def _replay_plan(e: ComboTrackerEngine, inputs: str, *, through_key: str | None = None) -> None:
    """Feed the engine the logical replay markers produced by the real macro planner."""
    tokens = split_inputs(inputs)
    plan = _PlanBuilder(None).build(list(expanded_ast_from_tokens(tokens)))
    previous_ms = 0.0
    for event in plan.events:
        if event.kind != "replay":
            continue
        total_ms = event.at_s * 1000.0
        e.replay_accept(event.key, total_ms - previous_ms, total_ms)
        previous_ms = total_ms
        if through_key is not None and event.key == through_key:
            return


class _SuccessfulOutput:
    def press(self, _name: str) -> bool:
        return True

    def release(self, _name: str) -> bool:
        return True


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

    def test_group_replay_tracks_each_item_before_advancing(self):
        inputs = "f, [q, e], 2, r"
        e = _engine_with_combo(inputs)

        e.replay_accept("f", 0.0, 0.0)
        e.replay_accept("q", 10.0, 10.0)

        self.assertEqual(e.current_index, 1)
        group = e._active_runtime_step()
        self.assertIsInstance(group, GroupState)
        self.assertEqual([item.completed_count for item in group.items], [1, 0])
        group_tile = e.timeline_steps()[1]
        self.assertEqual(group_tile["progress"], {"done": 1, "total": 2})

        e.replay_accept("e", 10.0, 20.0)
        self.assertEqual(e.current_index, 2)
        self.assertTrue(e.timeline_steps()[1]["completed"])

    def test_group_animation_wait_does_not_consume_next_replay_key(self):
        inputs = "f, [wait(r, 0.2s), q, e], 2, lmb"
        e = _engine_with_combo(inputs)
        _replay_plan(e, inputs, through_key="e")

        self.assertEqual(e.current_index, 2)
        group = e.runtime_steps[1]
        self.assertIsInstance(group, GroupState)
        self.assertEqual([item.completed_count for item in group.items], [1, 1, 1])

    def test_group_sequence_waits_follow_macro_plan(self):
        inputs = "f, [{e, wait:0.08s, e, wait:0.1s}, wait(r, 0.16s)], 2, lmb"
        e = _engine_with_combo(inputs)
        _replay_plan(e, inputs, through_key="2")

        self.assertEqual(e.current_index, 3)
        group = e.runtime_steps[1]
        self.assertIsInstance(group, GroupState)
        self.assertTrue(all(item.completed_count == item.required_count for item in group.items))

    def test_replay_tick_cannot_race_hold_release_marker(self):
        e = _engine_with_combo("f, hold(e, 0.1s), q, r")
        e.replay_accept("f", 0.0, 0.0)
        e.replay_accept("e", 10.0, 10.0)
        hold = e._active_runtime_step()
        self.assertIsInstance(hold, HoldState)
        hold.started_at -= 1.0
        e.hold_started_at = hold.started_at

        e.tick()

        self.assertEqual(e.current_index, 1)
        self.assertTrue(hold.in_progress)

        e.replay_accept("e", 100.0, 110.0)
        self.assertEqual(e.current_index, 2)

    def test_replay_tick_finishes_trailing_press_wait(self):
        emitted: list[dict] = []
        e = _engine_with_combo("f, wait:0.08s")
        e.set_emitter(emitted.append)
        e.record_combo_success = lambda _total=None: None

        e.replay_accept("f", 0.0, 0.0)
        sequence = e._active_runtime_step()
        self.assertIsInstance(sequence, SequenceState)
        wait = sequence.steps[sequence.current_index]
        wait.started_at -= 1.0

        e.tick()

        self.assertEqual(e.current_index, 0)
        self.assertTrue(
            any(
                msg.get("type") == "status" and msg.get("color") == "success"
                for msg in emitted
            )
        )

    def test_macro_plan_completes_every_supported_timeline_shape(self):
        cases = [
            "f, e, wait:0.05s, q",
            "f, wait(r, 0.05s), q",
            "f, hold(e, 0.05s), q",
            "f, hold(lmb, 0.08s, {wait:0.02s, q}), e",
            "f, [q, e], 2",
            "f, [wait(r, 0.05s), q, e], 2",
            "f, [q, hold(e, 0.05s)], 2",
            "f, [{e, wait:0.02s, e, wait:0.03s}, wait(r, 0.04s)], 2",
        ]

        for inputs in cases:
            with self.subTest(inputs=inputs):
                emitted: list[dict] = []
                e = _engine_with_combo(inputs)
                e.set_emitter(emitted.append)
                # Completion should not persist synthetic test statistics.
                e.record_combo_success = lambda _total=None: None

                _replay_plan(e, inputs)

                statuses = [msg for msg in emitted if msg.get("type") == "status"]
                self.assertTrue(any(msg.get("color") == "success" for msg in statuses))
                timeline_updates = [
                    msg for msg in emitted if msg.get("type") == "timeline_update"
                ]
                final_steps = timeline_updates[-1]["steps"]
                self.assertTrue(all(step.get("completed") for step in final_steps))

    def test_macro_player_queue_drives_group_hold_timeline_end_to_end(self):
        inputs = "f, [q, hold(e, 0.05s)], 2"
        emitted: list[dict] = []
        e = _engine_with_combo(inputs)
        e.set_emitter(emitted.append)
        e.record_combo_success = lambda _total=None: None
        player = MacroPlayer(on_step=e.replay_accept, output=_SuccessfulOutput())

        self.assertTrue(player.start(split_inputs(inputs)))
        deadline = time.perf_counter() + 1.0
        while player.is_running() and time.perf_counter() < deadline:
            time.sleep(0.001)

        self.assertFalse(player.is_running())
        self.assertTrue(
            any(
                msg.get("type") == "status" and msg.get("color") == "success"
                for msg in emitted
            )
        )


if __name__ == "__main__":
    unittest.main()
