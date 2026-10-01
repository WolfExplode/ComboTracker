import unittest

import combo_engine_ui as ui
from combo_engine import ComboTrackerEngine
from state_store import MemoryStateStore


def _engine_with_two_combos():
    engine = ComboTrackerEngine(state_store=MemoryStateStore())
    for name, inputs in (("B", "r, r, r"), ("A", "e, q")):
        engine.new_combo()
        ok, err = engine.save_or_update_combo(
            name=name,
            inputs=inputs,
            enders="",
            expected_time="1.5s" if name == "A" else "",
            user_difficulty="",
            step_display_mode="",
            key_images=None,
            demo_video="",
            ww_team_id="",
        )
        assert ok, err
    return engine


class ComboOverviewTests(unittest.TestCase):
    """The History page and the header combo picker read one row per combo from this."""

    def test_one_row_per_combo_sorted_by_name(self):
        engine = _engine_with_two_combos()
        rows = ui.combo_overview(engine)
        self.assertEqual([r["name"] for r in rows], ["A", "B"])
        self.assertEqual(rows[0]["steps"], 2)
        self.assertEqual(rows[0]["target_ms"], 1500)
        self.assertEqual(rows[1]["target_ms"], None)

    def test_counts_and_average_come_from_saved_stats(self):
        engine = _engine_with_two_combos()
        engine.combo_stats["A"].update(success=2, fail=3, best_ms=900, total_success_ms=2000)
        row = ui.combo_overview(engine)[0]
        self.assertEqual((row["success"], row["fail"], row["best_ms"], row["avg_ms"]), (2, 3, 900, 1000))
        self.assertIsNone(ui.combo_overview(engine)[1]["avg_ms"])

    def test_team_assignment_is_reported(self):
        engine = _engine_with_two_combos()
        engine.ww.combo_ww_team["B"] = "team1"
        rows = {r["name"]: r for r in ui.combo_overview(engine)}
        self.assertEqual(rows["B"]["team_id"], "team1")
        self.assertEqual(rows["A"]["team_id"], "")

    def test_init_and_stat_update_carry_the_overview(self):
        engine = _engine_with_two_combos()
        self.assertEqual(len(engine.init_payload()["overview"]), 2)
        msg = engine.stat_update_payload()
        self.assertEqual(msg["type"], "stat_update")
        self.assertIn("stats", msg)
        self.assertEqual(len(msg["overview"]), 2)


if __name__ == "__main__":
    unittest.main()
