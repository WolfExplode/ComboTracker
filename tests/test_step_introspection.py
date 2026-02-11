"""Unit tests for step_introspection module."""
import unittest

from states import GroupItemTracker, GroupState, HoldState, PressState, SequenceState, WaitState

import step_introspection


class TestExpectedLabelForStep(unittest.TestCase):
    def test_none_returns_dash(self):
        self.assertEqual(step_introspection.expected_label_for_step(None), "—")

    def test_press_state(self):
        step = PressState(expected="q")
        self.assertEqual(step_introspection.expected_label_for_step(step), "q")

    def test_press_state_optional(self):
        step = PressState(expected="e", optional=True)
        self.assertEqual(step_introspection.expected_label_for_step(step), "e?")

    def test_hold_state(self):
        step = HoldState(expected="lmb", required_ms=300)
        self.assertEqual(step_introspection.expected_label_for_step(step), "hold(lmb,≥300ms)")

    def test_wait_state_hard(self):
        step = WaitState(required_ms=500, mode="hard", wait_for=None)
        self.assertEqual(step_introspection.expected_label_for_step(step), "wait-hard(≥500ms)")

    def test_wait_state_mandatory(self):
        step = WaitState(required_ms=1000, mode="mandatory", wait_for="e")
        self.assertEqual(step_introspection.expected_label_for_step(step), "anim-wait(e,≥1000ms)")

    def test_sequence_state(self):
        step = SequenceState(steps=[PressState(expected="a"), HoldState(expected="b", required_ms=200)])
        self.assertIn("a", step_introspection.expected_label_for_step(step))
        self.assertIn("hold(b,200ms)", step_introspection.expected_label_for_step(step))

    def test_group_state(self):
        items = [
            GroupItemTracker(state=PressState(expected="q"), kind="press"),
            GroupItemTracker(state=HoldState(expected="e", required_ms=100), kind="hold"),
        ]
        step = GroupState(items=items)
        label = step_introspection.expected_label_for_step(step)
        self.assertIn("any-order", label)
        self.assertIn("q", label)
        self.assertIn("e", label)


class TestStartKeysForStep(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(step_introspection.start_keys_for_step(None), set())

    def test_press_state(self):
        step = PressState(expected="q")
        self.assertEqual(step_introspection.start_keys_for_step(step), {"q"})

    def test_hold_state(self):
        step = HoldState(expected="lmb", required_ms=300)
        self.assertEqual(step_introspection.start_keys_for_step(step), {"lmb"})

    def test_wait_state_returns_empty(self):
        step = WaitState(required_ms=500, mode="soft", wait_for=None)
        self.assertEqual(step_introspection.start_keys_for_step(step), set())

    def test_sequence_state_returns_first_step_keys(self):
        step = SequenceState(steps=[PressState(expected="a"), HoldState(expected="b", required_ms=100)])
        self.assertEqual(step_introspection.start_keys_for_step(step), {"a"})

    def test_group_state_collects_all_start_keys(self):
        items = [
            GroupItemTracker(state=PressState(expected="q"), kind="press"),
            GroupItemTracker(state=PressState(expected="e"), kind="press"),
        ]
        step = GroupState(items=items)
        self.assertEqual(step_introspection.start_keys_for_step(step), {"q", "e"})


class TestStepAcceptsInput(unittest.TestCase):
    def test_empty_input_returns_false(self):
        self.assertFalse(step_introspection.step_accepts_input(PressState(expected="q"), ""))
        self.assertFalse(step_introspection.step_accepts_input(PressState(expected="q"), "  "))

    def test_press_state(self):
        step = PressState(expected="q")
        self.assertTrue(step_introspection.step_accepts_input(step, "q"))
        self.assertFalse(step_introspection.step_accepts_input(step, "e"))

    def test_hold_state(self):
        step = HoldState(expected="lmb", required_ms=300)
        self.assertTrue(step_introspection.step_accepts_input(step, "lmb"))
        self.assertFalse(step_introspection.step_accepts_input(step, "rmb"))

    def test_wait_state_returns_false(self):
        step = WaitState(required_ms=500, mode="soft", wait_for=None)
        self.assertFalse(step_introspection.step_accepts_input(step, "q"))

    def test_sequence_state_first_press(self):
        step = SequenceState(steps=[PressState(expected="a"), HoldState(expected="b", required_ms=100)])
        self.assertTrue(step_introspection.step_accepts_input(step, "a"))
        self.assertFalse(step_introspection.step_accepts_input(step, "b"))

    def test_group_state_accepts_any_item_key(self):
        items = [
            GroupItemTracker(state=PressState(expected="q"), kind="press"),
            GroupItemTracker(state=PressState(expected="e"), kind="press"),
        ]
        step = GroupState(items=items)
        self.assertTrue(step_introspection.step_accepts_input(step, "q"))
        self.assertTrue(step_introspection.step_accepts_input(step, "e"))
        self.assertFalse(step_introspection.step_accepts_input(step, "r"))


if __name__ == "__main__":
    unittest.main()
