"""Unit tests for format_utils module."""
import unittest

import format_utils


class TestFormatMs(unittest.TestCase):
    def test_whole_seconds(self):
        self.assertEqual(format_utils.format_ms(1000), "1s (1000ms)")
        self.assertEqual(format_utils.format_ms(2000), "2s (2000ms)")

    def test_fractional_seconds(self):
        out = format_utils.format_ms(1500)
        self.assertIn("1500", out)
        self.assertIn("s", out)


class TestFormatMsBrief(unittest.TestCase):
    def test_none_returns_dash(self):
        self.assertEqual(format_utils.format_ms_brief(None), "—")

    def test_under_1000_ms(self):
        self.assertEqual(format_utils.format_ms_brief(500), "500ms")

    def test_over_1000_ms(self):
        out = format_utils.format_ms_brief(1500)
        self.assertIn("1.5", out)
        self.assertIn("s", out)


class TestFormatHoldRequirement(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(format_utils.format_hold_requirement(None), "")

    def test_whole_seconds(self):
        self.assertEqual(format_utils.format_hold_requirement(1000), "1s")
        self.assertEqual(format_utils.format_hold_requirement(3000), "3s")

    def test_fractional(self):
        out = format_utils.format_hold_requirement(350)
        self.assertIn("0.35", out)


class TestParseExpectedTimeMs(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(format_utils.parse_expected_time_ms(""))
        self.assertIsNone(format_utils.parse_expected_time_ms(None))
        self.assertIsNone(format_utils.parse_expected_time_ms("   "))

    def test_seconds_suffix(self):
        self.assertEqual(format_utils.parse_expected_time_ms("1s"), 1000)
        self.assertEqual(format_utils.parse_expected_time_ms("1.5s"), 1500)
        self.assertEqual(format_utils.parse_expected_time_ms("0.3s"), 300)

    def test_ms_suffix(self):
        self.assertEqual(format_utils.parse_expected_time_ms("500ms"), 500)
        self.assertEqual(format_utils.parse_expected_time_ms("1050ms"), 1050)

    def test_invalid_returns_none(self):
        self.assertIsNone(format_utils.parse_expected_time_ms("abc"))
        self.assertIsNone(format_utils.parse_expected_time_ms("1.5.5s"))


if __name__ == "__main__":
    unittest.main()
