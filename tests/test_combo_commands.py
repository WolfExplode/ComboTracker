import unittest

from _combo_commands import per_combo_maps
from combo_engine import ComboTrackerEngine
from state_store import MemoryStateStore


def _save(engine, name, inputs, **extra):
    return engine.save_or_update_combo(
        name=name,
        inputs=inputs,
        enders=extra.pop("enders", ""),
        expected_time="",
        user_difficulty="",
        step_display_mode="",
        key_images=None,
        demo_video="",
        ww_team_id="",
        **extra,
    )


def _engine_with(*combos):
    engine = ComboTrackerEngine(state_store=MemoryStateStore())
    for name, inputs in combos:
        engine.new_combo()
        _save(engine, name, inputs)
    return engine


class ComboRenameTests(unittest.TestCase):
    def test_rename_onto_existing_combo_is_refused(self):
        engine = _engine_with(("A", "e, q"), ("B", "r, r, r"))
        engine.set_active_combo("A")

        ok, err = _save(engine, "B", "e, q")

        self.assertFalse(ok)
        self.assertIn("already exists", err)
        self.assertEqual(engine.combos["A"], ["e", "q"])
        self.assertEqual(engine.combos["B"], ["r", "r", "r"])

    def test_new_combo_with_taken_name_is_refused(self):
        engine = _engine_with(("B", "r, r, r"))
        engine.new_combo()

        ok, _ = _save(engine, "B", "q")

        self.assertFalse(ok)
        self.assertEqual(engine.combos["B"], ["r", "r", "r"])

    def test_rename_to_new_name_moves_combo(self):
        engine = _engine_with(("A", "e, q"))
        engine.set_active_combo("A")

        ok, _ = _save(engine, "C", "e, q")

        self.assertTrue(ok)
        self.assertNotIn("A", engine.combos)
        self.assertEqual(engine.combos["C"], ["e", "q"])
        self.assertIn("C", engine.combo_stats)

    def test_updating_active_combo_keeps_its_name(self):
        engine = _engine_with(("A", "e, q"))
        engine.set_active_combo("A")

        ok, _ = _save(engine, "A", "e, q, r")

        self.assertTrue(ok)
        self.assertEqual(engine.combos["A"], ["e", "q", "r"])


class PerComboDataTests(unittest.TestCase):
    def test_every_per_combo_dict_is_listed(self):
        engine = _engine_with(("A", "e, q"))
        listed = {id(m) for m in per_combo_maps(engine)}
        # Engine dicts named combo_* are keyed by combo name, except the global ender settings.
        for attr, value in vars(engine).items():
            if attr.startswith("combo_") and isinstance(value, dict) and not attr.startswith("combo_enders"):
                self.assertIn(id(value), listed, attr)

    def test_rename_carries_stats(self):
        engine = _engine_with()
        engine.new_combo()
        _save(engine, "A", "e, q")
        engine.combo_stats["A"]["success"] = 3

        ok, _ = _save(engine, "C", "e, q")

        self.assertTrue(ok)
        self.assertEqual(engine.combo_stats["C"]["success"], 3)
        self.assertNotIn("A", engine.combo_stats)

    def test_delete_clears_every_per_combo_dict(self):
        engine = _engine_with(("A", "e, q"))
        engine.combo_expected_ms["A"] = 1000
        engine.ww.combo_ww_team["A"] = "team1"

        ok, _ = engine.delete_combo("A")

        self.assertTrue(ok)
        for mapping in per_combo_maps(engine):
            self.assertNotIn("A", mapping)


class SaveAsNewTests(unittest.TestCase):
    """The Characters page saves rotations while another combo is active in the tracker."""

    def test_as_new_leaves_active_combo_and_enders_alone(self):
        engine = _engine_with(("A", "e, q"))
        engine.set_active_combo("A")
        _save(engine, "A", "e, q", enders="1:0.5s, r")
        enders_before = dict(engine.combo_enders)
        self.assertEqual(enders_before, {"1": 500, "r": 0})

        ok, _ = _save(engine, "Rotation", "1, e, r", enders=None, as_new=True)

        self.assertTrue(ok)
        self.assertEqual(engine.combos["A"], ["e", "q"])
        self.assertEqual(engine.combos["Rotation"], ["1", "e", "r"])
        self.assertEqual(engine.combo_enders, enders_before)


if __name__ == "__main__":
    unittest.main()
