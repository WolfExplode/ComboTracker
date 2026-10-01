import json
import tempfile
import unittest
from pathlib import Path

from combo_engine import ComboTrackerEngine
from state_store import JsonStateStore, MemoryStateStore


class MemoryStateStoreTests(unittest.TestCase):
    def test_load_and_save_are_isolated_from_caller_mutation(self):
        original = {"combos": {"one": {"inputs": ["q"]}}}
        store = MemoryStateStore(original)

        original["combos"]["one"]["inputs"].append("e")
        loaded = store.load()
        self.assertEqual(loaded, {"combos": {"one": {"inputs": ["q"]}}})

        loaded["combos"]["one"]["inputs"].append("r")
        self.assertEqual(store.load(), {"combos": {"one": {"inputs": ["q"]}}})

    def test_engine_can_save_without_touching_the_filesystem(self):
        store = MemoryStateStore()
        engine = ComboTrackerEngine(state_store=store)
        engine.combos = {"test": {"inputs": ["q"], "best_ms": None}}

        self.assertTrue(engine.save_combos())
        self.assertIn("test", store.load()["combos"])


class JsonStateStoreTests(unittest.TestCase):
    def test_save_rotates_valid_primary_and_loads_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combos.json"
            store = JsonStateStore(path)

            store.save({"version": 1})
            store.save({"version": 2})

            self.assertEqual(store.load(), {"version": 2})
            self.assertEqual(
                json.loads(store.backup_path.read_text(encoding="utf-8")),
                {"version": 1},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_seed_is_used_only_when_no_save_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combos.json"
            seed = Path(directory) / "combos.example.json"
            seed.write_text(json.dumps({"version": "seed"}), encoding="utf-8")
            store = JsonStateStore(path, seed_path=seed)

            self.assertEqual(store.load(), {"version": "seed"})
            store.save({"version": 1})
            self.assertEqual(store.load(), {"version": 1})
            self.assertEqual(json.loads(seed.read_text(encoding="utf-8")), {"version": "seed"})

    def test_corrupt_primary_falls_back_to_known_good_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combos.json"
            store = JsonStateStore(path)
            store.save({"version": 1})
            store.save({"version": 2})
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(store.load(), {"version": 1})

    def test_corrupt_primary_is_not_rotated_over_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combos.json"
            store = JsonStateStore(path)
            store.save({"version": 1})
            store.save({"version": 2})
            path.write_text("not json", encoding="utf-8")

            store.save({"version": 3})

            self.assertEqual(store.load(), {"version": 3})
            self.assertEqual(
                json.loads(store.backup_path.read_text(encoding="utf-8")),
                {"version": 1},
            )


if __name__ == "__main__":
    unittest.main()


class WutheringWavesDefaultsTests(unittest.TestCase):
    def test_fresh_install_gets_ww_enders_and_transcribe_keys(self):
        engine = ComboTrackerEngine(state_store=MemoryStateStore())
        self.assertEqual(engine.combo_enders.get("r"), 2000)
        self.assertIn("space", engine.combo_enders_soft)
        self.assertIn("lmb", engine.transcribe_valid_keys)

    def test_saved_empty_enders_are_kept(self):
        engine = ComboTrackerEngine(state_store=MemoryStateStore({"combo_enders": {}, "combo_enders_soft": []}))
        self.assertEqual(engine.combo_enders, {})

    def test_legacy_generic_combos_still_load(self):
        data = {"combos": {"old": ["e", "q"]}, "combo_target_game": {"old": "generic"}}
        engine = ComboTrackerEngine(state_store=MemoryStateStore(data))
        self.assertEqual(engine.combos["old"], ["e", "q"])
        self.assertEqual(engine.get_editor_payload()["target_game"], "wuthering_waves")
