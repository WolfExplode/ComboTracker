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
    """The combo list (grouped by team) reads one row per combo from this."""

    def test_one_row_per_combo_sorted_by_name(self):
        engine = _engine_with_two_combos()
        rows = ui.combo_overview(engine)
        self.assertEqual([r["name"] for r in rows], ["A", "B"])
        self.assertEqual(rows[0]["steps"], 2)
        self.assertEqual(rows[1]["steps"], 3)

    def test_team_assignment_is_reported(self):
        engine = _engine_with_two_combos()
        engine.ww.combo_ww_team["B"] = "team1"
        rows = {r["name"]: r for r in ui.combo_overview(engine)}
        self.assertEqual(rows["B"]["team_id"], "team1")
        self.assertEqual(rows["A"]["team_id"], "")

    def test_init_carries_the_overview(self):
        engine = _engine_with_two_combos()
        self.assertEqual(len(engine.init_payload()["overview"]), 2)


if __name__ == "__main__":
    unittest.main()
