import unittest

from parser import PressNode, WaitNode, parse_step
from tests.test_runtime_clock import ManualMonotonicClock, _engine_with_combo


class OptionalWaitParseTests(unittest.TestCase):
    def test_dash_wait_is_an_optional_soft_wait(self):
        self.assertEqual(parse_step("-wait:0.15s"), WaitNode(150, "soft", None, optional=True))
        self.assertEqual(parse_step("wait:0.15s"), WaitNode(150, "soft", None))
        self.assertEqual(parse_step("-e"), PressNode("e", optional=True))


class OptionalWaitEngineTests(unittest.TestCase):
    def test_plain_wait_holds_the_next_key_until_it_ends(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("e, wait:1s, x, y", clock)
        engine.process_press("e")
        engine.process_release("e")
        engine.process_press("x")
        self.assertEqual(engine.current_index, 0)

    def test_next_key_cuts_an_optional_wait_after_a_key_short(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("e, -wait:1s, x, y", clock)
        engine.process_press("e")
        engine.process_release("e")
        clock.advance(0.05)
        engine.process_press("x")
        self.assertEqual(engine.current_index, 2)

    def test_optional_wait_can_still_be_waited_out(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("e, -wait:0.1s, x, y", clock)
        engine.process_press("e")
        engine.process_release("e")
        clock.advance(0.2)
        engine.tick()
        self.assertEqual(engine.current_index, 1)
        engine.process_press("x")
        self.assertEqual(engine.current_index, 2)

    def test_standalone_optional_wait(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("hold(e, 0.1s), -wait:1s, x, y", clock)
        engine.process_press("e")
        clock.advance(0.15)
        engine.process_release("e")
        self.assertEqual(engine.current_index, 1)
        engine.process_press("x")
        self.assertEqual(engine.current_index, 3)

    def test_timeline_marks_the_optional_wait(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo("e, -wait:1s, x, -wait:0.5s, y", clock)
        steps = engine.timeline_steps()
        self.assertTrue(steps[0].get("wait_optional"))
        self.assertFalse(steps[0].get("optional"))
        self.assertTrue(steps[1].get("wait_optional"))


if __name__ == "__main__":
    unittest.main()


class MoveNameTests(unittest.TestCase):
    def test_names_ride_along_and_the_engine_ignores_them(self):
        from parser import lower_outside_quotes, runtime_source_token_indices_from_tokens, split_inputs, split_move_name
        tokens = split_inputs('RMB, lmb "Basic: One, Two, Three 1", wait:0.1s, hold(lmb, 0.25s) "Heavy 2: X"')
        self.assertEqual(tokens, ["RMB", 'lmb "Basic: One, Two, Three 1"', "wait:0.1s", 'hold(lmb, 0.25s) "Heavy 2: X"'])
        self.assertEqual(split_move_name(tokens[1]), ("lmb", "Basic: One, Two, Three 1"))
        self.assertEqual(split_move_name("lmb"), ("lmb", None))
        self.assertEqual(parse_step(tokens[1]), PressNode("lmb"))
        self.assertEqual(runtime_source_token_indices_from_tokens(tokens), [[0], [1, 2], [3]])
        self.assertEqual(lower_outside_quotes('LMB "Basic: A 1"'), 'lmb "Basic: A 1"')

    def test_named_combo_plays_like_the_plain_one(self):
        clock = ManualMonotonicClock(10.0)
        engine = _engine_with_combo('rmb, lmb "Basic: Origin Calculus 1", wait:0.1s, x, y', clock)
        engine.process_press("rmb")
        engine.process_release("rmb")
        engine.process_press("lmb")
        engine.process_release("lmb")
        clock.advance(0.2)
        engine.tick()
        engine.process_press("x")
        self.assertEqual(engine.current_index, 3)
