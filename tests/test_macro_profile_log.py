import json
import tempfile
import unittest
from pathlib import Path

from profiling.macro_profile_log import MacroProfileLogWriter
from profiling.macro_timing import MacroTimingCollector


def _profile():
    collector = MacroTimingCollector(requested_at=10.0, clock_started_at=10.001)
    collector.record(
        order=1,
        kind="down",
        key="f",
        planned_offset_s=0.0,
        woke_ns=10_001_100_000,
        dispatch_started_ns=10_001_200_000,
        dispatch_completed_ns=10_001_300_000,
    )
    return collector.finish()


class MacroProfileLogWriterTests(unittest.TestCase):
    def test_writes_timestamped_run_and_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MacroProfileLogWriter(Path(directory) / "macro_profiles")

            run_path = writer.write(
                _profile(),
                outcome="completed",
                combo_name="Example combo",
                plan_duration_ms=123.4567,
            )

            self.assertIsNotNone(run_path)
            self.assertTrue(run_path.exists())
            latest = run_path.parent / "latest.json"
            self.assertTrue(latest.exists())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["outcome"], "completed")
            self.assertEqual(payload["combo_name"], "Example combo")
            self.assertEqual(payload["plan_duration_ms"], 123.457)
            self.assertEqual(payload["profile"]["event_count"], 1)
