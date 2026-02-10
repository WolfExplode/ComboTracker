"""Unit tests for Wuthering Waves ender policy (can_ender_drop_combo, on_accepted_key)."""
import unittest

from Game_Wuthering_Waves import WutheringWavesGame


class MockEngine:
    """Minimal engine duck-type for WW policy tests."""
    def __init__(self):
        self.active_combo_name = "test_combo"
        self.last_input_time = None
        self.current_index = 0
        self._combo_enders = set()
        self._ender_cooldown_until = {}

    def _is_combo_ender(self, input_name: str) -> bool:
        return (input_name or "").strip().lower() in self._combo_enders

    def _ender_on_cooldown(self, input_name: str) -> bool:
        key = (input_name or "").strip().lower()
        if not key or key not in self._combo_enders:
            return False
        import time
        until = self._ender_cooldown_until.get(key, 0.0)
        return time.perf_counter() <= until


class TestCanEnderDropCombo(unittest.TestCase):
    def test_empty_input_returns_false(self):
        ww = WutheringWavesGame()
        engine = MockEngine()
        engine._combo_enders = {"e"}
        self.assertFalse(ww.can_ender_drop_combo(engine, ""))
        self.assertFalse(ww.can_ender_drop_combo(engine, "   "))

    def test_non_ender_returns_false(self):
        ww = WutheringWavesGame()
        engine = MockEngine()
        engine._combo_enders = {"e"}
        engine.last_input_time = 1.0
        self.assertFalse(ww.can_ender_drop_combo(engine, "q"))
        self.assertFalse(ww.can_ender_drop_combo(engine, "r"))

    def test_no_input_yet_and_index_zero_returns_false(self):
        ww = WutheringWavesGame()
        engine = MockEngine()
        engine._combo_enders = {"e"}
        engine.last_input_time = None
        engine.current_index = 0
        self.assertFalse(ww.can_ender_drop_combo(engine, "e"))

    def test_ender_after_input_returns_true_for_generic(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "generic"
        engine = MockEngine()
        engine._combo_enders = {"e"}
        engine.last_input_time = 1.0
        engine.current_index = 2
        self.assertTrue(ww.can_ender_drop_combo(engine, "e"))

    def test_ww_current_character_slot_does_not_drop(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "wuthering_waves"
        ww.ww_active_character = "2"
        engine = MockEngine()
        engine._combo_enders = {"1", "2", "3"}
        engine.last_input_time = 1.0
        engine.current_index = 2
        self.assertFalse(ww.can_ender_drop_combo(engine, "2"))
        self.assertTrue(ww.can_ender_drop_combo(engine, "1"))
        self.assertTrue(ww.can_ender_drop_combo(engine, "3"))

    def test_ww_other_character_slot_drops(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "wuthering_waves"
        ww.ww_active_character = "2"
        engine = MockEngine()
        engine._combo_enders = {"1", "2", "3"}
        engine.last_input_time = 1.0
        engine.current_index = 2
        self.assertTrue(ww.can_ender_drop_combo(engine, "1"))
        self.assertTrue(ww.can_ender_drop_combo(engine, "3"))


class TestOnAcceptedKey(unittest.TestCase):
    def test_generic_game_does_nothing(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "generic"
        engine = MockEngine()
        ww.on_accepted_key(engine, "2")
        self.assertIsNone(ww.ww_active_character)

    def test_ww_game_sets_character_slot(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "wuthering_waves"
        engine = MockEngine()
        ww.on_accepted_key(engine, "2")
        self.assertEqual(ww.ww_active_character, "2")
        ww.on_accepted_key(engine, "1")
        self.assertEqual(ww.ww_active_character, "1")

    def test_non_slot_key_does_not_clear_character(self):
        ww = WutheringWavesGame()
        ww.combo_target_game["test_combo"] = "wuthering_waves"
        ww.ww_active_character = "2"
        engine = MockEngine()
        ww.on_accepted_key(engine, "e")
        self.assertEqual(ww.ww_active_character, "2")


if __name__ == "__main__":
    unittest.main()
