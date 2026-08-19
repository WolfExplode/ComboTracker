import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from macro_player import MacroPlayer, _PlanBuilder, _SyntheticEventLedger
from profiling.macro_timing import MacroTimingCollector
from parser import expanded_ast_from_tokens, split_inputs


def _plan(inputs: str, spam_interval_ms: int | None = 100):
    ast = expanded_ast_from_tokens(split_inputs(inputs))
    return _PlanBuilder(spam_interval_ms).build(list(ast))


class FakeOutput:
    def __init__(self, *, fail_press: str | None = None):
        self.fail_press = fail_press
        self.events: list[tuple[str, str, float]] = []

    def press(self, name: str) -> bool:
        self.events.append(("down", name, time.perf_counter()))
        return name != self.fail_press

    def release(self, name: str) -> bool:
        self.events.append(("up", name, time.perf_counter()))
        return True


class MacroPlanTests(unittest.TestCase):
    def test_explicit_spam_uses_configured_cadence_and_includes_endpoint(self):
        plan = _plan("spam(lmb, 0.16s)", spam_interval_ms=50)
        downs = [event.at_s for event in plan.events if event.kind == "down"]
        self.assertEqual(downs, [0.0, 0.05, 0.1, 0.16])

    def test_spam_inside_hold_body_keeps_configured_cadence(self):
        plan = _plan(
            "hold(lmb, 0.3s, {wait:0.05s, spam(r, 0.1s)})",
            spam_interval_ms=50,
        )
        r_downs = [event.at_s for event in plan.events if event.kind == "down" and event.key == "r"]
        self.assertEqual(len(r_downs), 3)
        for actual, expected in zip(r_downs, (0.05, 0.1, 0.15)):
            self.assertAlmostEqual(actual, expected)

    def test_plain_press_is_one_pulse_and_next_deadline_is_absolute(self):
        plan = _plan("e, wait:0.1s, q", spam_interval_ms=None)
        downs = [(event.key, event.at_s) for event in plan.events if event.kind == "down"]
        ups = [(event.key, event.at_s) for event in plan.events if event.kind == "up"]

        self.assertEqual(downs, [("e", 0.0), ("q", 0.1)])
        self.assertEqual(ups, [("e", 0.03), ("q", 0.13)])

    def test_chain_spam_excludes_end_deadline_and_has_no_fixed_pad(self):
        plan = _plan("e, wait:0.1s, e, wait:0.1s, q", spam_interval_ms=50)
        e_downs = [event.at_s for event in plan.events if event.kind == "down" and event.key == "e"]
        q_down = next(event.at_s for event in plan.events if event.kind == "down" and event.key == "q")
        replay_e = [event.at_s for event in plan.events if event.kind == "replay" and event.key == "e"]

        self.assertEqual(e_downs, [0.0, 0.05, 0.1, 0.15])
        self.assertAlmostEqual(q_down, 0.2)
        self.assertEqual(replay_e, [0.0, 0.1])

    def test_spam_interval_has_a_released_gap_between_same_key_pulses(self):
        plan = _plan("e, wait:0.05s, e, wait:0.05s, q", spam_interval_ms=5)
        e_downs = [event.at_s for event in plan.events if event.kind == "down" and event.key == "e"]
        e_ups = [event.at_s for event in plan.events if event.kind == "up" and event.key == "e"]
        self.assertEqual(e_downs, [0.0, 0.03, 0.06, 0.09])
        self.assertEqual(len(e_ups), 4)
        for actual, expected in zip(e_ups, [0.025, 0.055, 0.085, 0.115]):
            self.assertAlmostEqual(actual, expected)

    def test_hold_with_body_keeps_holder_down_until_body_and_minimum_finish(self):
        plan = _plan("hold(lmb, 0.2s, {wait:0.05s, q})", spam_interval_ms=None)
        holder = [(event.kind, event.at_s) for event in plan.events if event.key == "lmb"]
        replay_phases = [
            event.pressed
            for event in plan.events
            if event.kind == "replay" and event.key == "lmb"
        ]
        q_down = next(event.at_s for event in plan.events if event.kind == "down" and event.key == "q")

        self.assertEqual(holder, [("down", 0.0), ("replay", 0.0), ("up", 0.2), ("replay", 0.2)])
        self.assertEqual(replay_phases, [True, False])
        self.assertAlmostEqual(q_down, 0.05)


class MacroPlayerTests(unittest.TestCase):
    def test_output_failure_does_not_emit_replay_success(self):
        output = FakeOutput(fail_press="f")
        statuses: list[tuple[str, str]] = []
        replayed: list[str] = []
        player = MacroPlayer(
            on_status=lambda text, color: statuses.append((text, color)),
            on_step=lambda key, _step, _total, _pressed: replayed.append(key),
            output=output,
        )

        self.assertTrue(player.start(["f"]))
        deadline = time.perf_counter() + 1.0
        while player.is_running() and time.perf_counter() < deadline:
            time.sleep(0.001)

        self.assertFalse(player.is_running())
        self.assertEqual(replayed, [])
        self.assertTrue(any(color == "fail" for _text, color in statuses))

    def test_unsupported_input_is_rejected_before_thread_starts(self):
        statuses: list[tuple[str, str]] = []
        player = MacroPlayer(on_status=lambda text, color: statuses.append((text, color)), output=FakeOutput())

        self.assertFalse(player.start(["mouse_extra"]))
        self.assertFalse(player.is_running())
        self.assertTrue(any(color == "fail" for _text, color in statuses))

    def test_completed_run_exposes_dispatch_profile(self):
        player = MacroPlayer(output=FakeOutput())

        self.assertTrue(player.start(["f"]))
        deadline = time.perf_counter() + 1.0
        while player.is_running() and time.perf_counter() < deadline:
            time.sleep(0.001)

        profile = player.last_profile()
        self.assertIsNotNone(profile)
        self.assertEqual(profile["event_count"], 2)
        self.assertEqual([event["kind"] for event in profile["events"]], ["down", "up"])
        self.assertIn("dispatch_start_lateness_ms", profile)

    def test_completed_run_writes_profile_artifacts_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_dir = Path(directory) / "macro_profiles"
            player = MacroPlayer(output=FakeOutput(), profile_log_dir=profile_dir)

            self.assertTrue(player.start(["f"], combo_name="Profile test"))
            deadline = time.perf_counter() + 1.0
            while player.is_running() and time.perf_counter() < deadline:
                time.sleep(0.001)

            latest = profile_dir / "latest.json"
            self.assertTrue(latest.exists())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload["outcome"], "completed")
            self.assertEqual(payload["combo_name"], "Profile test")
            self.assertEqual(payload["profile"]["event_count"], 2)


class MacroTimingCollectorTests(unittest.TestCase):
    def test_summary_reports_lateness_output_cost_and_interval_error(self):
        collector = MacroTimingCollector(requested_at=9.998, clock_started_at=10.0)
        collector.record(
            order=1,
            kind="down",
            key="f",
            planned_offset_s=0.0,
            woke_ns=10_000_500_000,
            dispatch_started_ns=10_001_000_000,
            dispatch_completed_ns=10_001_250_000,
        )
        collector.record(
            order=2,
            kind="up",
            key="f",
            planned_offset_s=0.03,
            woke_ns=10_030_700_000,
            dispatch_started_ns=10_032_000_000,
            dispatch_completed_ns=10_032_500_000,
        )

        summary = collector.finish().summary()

        self.assertEqual(summary["event_count"], 2)
        self.assertAlmostEqual(summary["request_to_clock_start_ms"], 2.0)
        self.assertAlmostEqual(summary["request_to_first_dispatch_ms"], 3.0)
        self.assertAlmostEqual(summary["dispatch_start_lateness_ms"]["max"], 2.0)
        self.assertAlmostEqual(summary["output_duration_ms"]["max"], 0.5)
        self.assertAlmostEqual(summary["interval_error_ms"]["max"], 1.0)
        self.assertEqual(summary["late_event_counts"]["over_1ms"], 1)

    def test_summary_separates_scheduler_and_same_deadline_lateness(self):
        collector = MacroTimingCollector(requested_at=10.0, clock_started_at=10.0)
        collector.record(
            order=1,
            kind="down",
            key="f",
            planned_offset_s=0.0,
            woke_ns=10_000_100_000,
            dispatch_started_ns=10_000_200_000,
            dispatch_completed_ns=10_000_300_000,
        )
        collector.record(
            order=2,
            kind="up",
            key="f",
            planned_offset_s=0.03,
            woke_ns=10_030_100_000,
            dispatch_started_ns=10_030_200_000,
            dispatch_completed_ns=10_030_300_000,
        )
        collector.record(
            order=3,
            kind="down",
            key="q",
            planned_offset_s=0.03,
            woke_ns=10_030_400_000,
            dispatch_started_ns=10_030_500_000,
            dispatch_completed_ns=10_030_600_000,
        )

        summary = collector.finish().summary()

        self.assertEqual(summary["deadline_analysis"]["deadline_count"], 2)
        self.assertEqual(summary["deadline_analysis"]["collision_deadline_count"], 1)
        self.assertEqual(summary["deadline_analysis"]["later_collision_event_count"], 1)
        self.assertAlmostEqual(summary["scheduler_lateness_ms"]["max"], 0.2)
        self.assertAlmostEqual(summary["same_deadline_lateness_ms"]["max"], 0.5)
        self.assertEqual(summary["output_duration_by_input_ms"]["q.down"]["event_count"], 1)


class SyntheticEventLedgerTests(unittest.TestCase):
    def test_event_is_consumed_once_by_origin_and_direction(self):
        ledger = _SyntheticEventLedger()
        ledger.expect("F9", True)

        self.assertFalse(ledger.consume("f9", False))
        self.assertTrue(ledger.consume("f9", True))
        self.assertFalse(ledger.consume("f9", True))


if __name__ == "__main__":
    unittest.main()
