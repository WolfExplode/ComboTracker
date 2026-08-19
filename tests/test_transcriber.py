import unittest

from macro_player import _PlanBuilder
from parser import expanded_ast_from_tokens, split_inputs
from transcriber import Transcriber


def _transcriber(*, min_wait_s: float = 0.0, hold_threshold_s: float = 0.2) -> Transcriber:
    transcriber = Transcriber(
        min_wait_s=min_wait_s,
        hold_threshold_s=hold_threshold_s,
    )
    transcriber.set_valid_keys("f, e, q, r, lmb, rmb, 1, 2, 3")
    transcriber.start()
    return transcriber


class TranscriberAccuracyTests(unittest.TestCase):
    def test_preserves_idle_gap_after_hold(self):
        transcriber = _transcriber()
        transcriber.key_down("e", 1.0)
        transcriber.key_up("e", 1.3)
        transcriber.key_down("q", 1.5)
        transcriber.key_up("q", 1.53)

        self.assertEqual(
            transcriber.stop(1.6),
            "hold(e, 0.3s), wait:0.2s, q",
        )

    def test_infers_hold_with_body_from_overlapping_inputs(self):
        transcriber = _transcriber()
        transcriber.key_down("lmb", 1.0)
        transcriber.key_down("q", 1.1)
        transcriber.key_up("q", 1.13)
        transcriber.key_up("lmb", 1.3)

        self.assertEqual(
            transcriber.stop(1.4),
            "hold(lmb, 0.3s, {wait:0.1s, q})",
        )

    def test_orders_cross_device_callbacks_by_capture_timestamp(self):
        transcriber = _transcriber()
        # Simulate mouse and keyboard listener threads arriving out of order.
        transcriber.key_down("q", 1.1)
        transcriber.key_down("lmb", 1.0)
        transcriber.key_up("q", 1.13)
        transcriber.key_up("lmb", 1.3)

        self.assertEqual(
            transcriber.stop(1.4),
            "hold(lmb, 0.3s, {wait:0.1s, q})",
        )

    def test_preserves_rapid_repeated_mouse_clicks(self):
        transcriber = _transcriber()
        transcriber.key_down("lmb", 1.0)
        transcriber.key_up("lmb", 1.025)
        transcriber.key_down("lmb", 1.03)
        transcriber.key_up("lmb", 1.055)

        self.assertEqual(transcriber.stop(1.1), "lmb, wait:0.03s, lmb")

    def test_compacts_three_or_more_same_key_taps_as_spam(self):
        transcriber = _transcriber()
        for started_at in (1.0, 1.15, 1.3, 1.45):
            transcriber.key_down("lmb", started_at)
            transcriber.key_up("lmb", started_at + 0.025)

        self.assertEqual(transcriber.stop(1.6), "spam(lmb, 0.45s)")

    def test_compacts_spam_inside_hold_body(self):
        transcriber = _transcriber()
        transcriber.key_down("lmb", 1.0)
        for started_at in (1.1, 1.25, 1.4):
            transcriber.key_down("r", started_at)
            transcriber.key_up("r", started_at + 0.025)
        transcriber.key_up("lmb", 1.6)

        self.assertEqual(
            transcriber.stop(1.7),
            "hold(lmb, 0.6s, {wait:0.1s, spam(r, 0.3s)})",
        )

    def test_formats_waits_at_millisecond_precision(self):
        transcriber = _transcriber()
        transcriber.key_down("e", 1.0)
        transcriber.key_up("e", 1.03)
        transcriber.key_down("q", 1.123)
        transcriber.key_up("q", 1.153)

        self.assertEqual(transcriber.stop(1.2), "e, wait:0.123s, q")

    def test_stop_closes_still_held_inputs_at_stop_timestamp(self):
        transcriber = _transcriber()
        transcriber.key_down("e", 1.0)

        self.assertEqual(transcriber.stop(1.3), "hold(e, 0.3s)")
        recording = transcriber.last_recording()
        self.assertTrue(recording["events"][-1]["synthetic"])

    def test_ignores_repeat_down_without_losing_physical_release(self):
        transcriber = _transcriber()
        transcriber.key_down("e", 1.0)
        transcriber.key_down("e", 1.1)
        transcriber.key_up("e", 1.3)

        self.assertEqual(transcriber.stop(1.4), "hold(e, 0.3s)")
        recording = transcriber.last_recording()
        self.assertEqual(recording["diagnostics"]["ignored_repeat_downs"], 1)
        self.assertEqual(recording["diagnostics"]["event_count"], 2)

    def test_wait_stripping_is_post_capture_policy(self):
        transcriber = _transcriber(min_wait_s=0.1)
        transcriber.key_down("e", 1.0)
        transcriber.key_up("e", 1.03)
        transcriber.key_down("q", 1.05)
        transcriber.key_up("q", 1.08)

        self.assertEqual(transcriber.stop(1.1), "e, q")
        recording = transcriber.last_recording()
        self.assertEqual(len(recording["events"]), 4)

    def test_transcribed_spam_round_trips_key_down_cadence(self):
        transcriber = _transcriber()
        for started_at in (1.0, 1.03, 1.06):
            transcriber.key_down("lmb", started_at)
            transcriber.key_up("lmb", started_at + 0.025)
        transcript = transcriber.stop(1.1)

        self.assertEqual(transcript, "spam(lmb, 0.06s)")

        plan = _PlanBuilder(30).build(
            list(expanded_ast_from_tokens(split_inputs(transcript)))
        )
        downs = [event.at_s for event in plan.events if event.kind == "down"]

        self.assertEqual(downs, [0.0, 0.03, 0.06])
        self.assertEqual(
            [event["offset_ms"] for event in transcriber.last_recording()["events"]],
            [0.0, 25.0, 30.0, 55.0, 60.0, 85.0],
        )

    def test_transcribed_hold_body_round_trips_inner_press_time(self):
        transcriber = _transcriber()
        transcriber.key_down("lmb", 1.0)
        transcriber.key_down("q", 1.1)
        transcriber.key_up("q", 1.13)
        transcriber.key_up("lmb", 1.3)
        transcript = transcriber.stop(1.4)

        plan = _PlanBuilder(None).build(
            list(expanded_ast_from_tokens(split_inputs(transcript)))
        )
        q_down = next(
            event.at_s
            for event in plan.events
            if event.kind == "down" and event.key == "q"
        )
        holder_up = next(
            event.at_s
            for event in plan.events
            if event.kind == "up" and event.key == "lmb"
        )

        self.assertAlmostEqual(q_down, 0.1)
        self.assertAlmostEqual(holder_up, 0.3)


if __name__ == "__main__":
    unittest.main()
