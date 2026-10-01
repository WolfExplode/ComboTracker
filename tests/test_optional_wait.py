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
