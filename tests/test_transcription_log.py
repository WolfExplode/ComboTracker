import json
import tempfile
import unittest
from pathlib import Path

from profiling.transcription_log import TranscriptionLogWriter


class TranscriptionLogWriterTests(unittest.TestCase):
    def test_writes_timestamped_recording_and_latest_snapshot(self):
        recording = {
            "schema_version": 1,
            "transcript": "e, wait:0.123s, q",
            "settings": {"hold_threshold_ms": 200.0, "min_wait_ms": 0.0},
            "diagnostics": {"event_count": 4},
            "events": [
                {"sequence": 1, "key": "e", "phase": "down", "offset_ms": 0.0},
                {"sequence": 2, "key": "e", "phase": "up", "offset_ms": 30.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            writer = TranscriptionLogWriter(Path(directory) / "transcriptions")

            run_path = writer.write(recording)

            self.assertIsNotNone(run_path)
            self.assertTrue(run_path.exists())
            latest = run_path.parent / "latest.json"
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(payload["transcript"], "e, wait:0.123s, q")
            self.assertEqual(payload["diagnostics"]["event_count"], 4)
            self.assertIn("captured_at", payload)


if __name__ == "__main__":
    unittest.main()
