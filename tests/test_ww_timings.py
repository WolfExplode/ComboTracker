import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ww_import_timings as imp  # noqa: E402

CHARACTERS = json.loads((ROOT / "static" / "data" / "ww_characters.json").read_text(encoding="utf-8"))["characters"]
SHIPPED = json.loads((ROOT / "static" / "data" / "ww_timings.json").read_text(encoding="utf-8"))


class ImportTimingsTests(unittest.TestCase):
    def test_matches_by_name_and_drops_damage_columns(self):
        scrape = {
            "fetched_at": "2026-10-01 08:51",
            "characters": {
                "roverspectro": {"name": "Rover: Spectro", "abilities": [
                    {"section": "BASIC ATTACK", "name": "Basic 1", "tags": [], "hits": 1, "frames": 23,
                     "cancel": 16, "noswap": 0, "tstop": 0, "mstop": 0, "cd": None, "genre": "BASIC",
                     "mv": 120, "hit_frames": [16]},
                ]},
                "xuanling": {"name": "XuanLing", "abilities": []},
                "nobody": {"name": "Nobody", "abilities": []},
            },
        }
        data, unmatched = imp.build(scrape, CHARACTERS)
        self.assertEqual(unmatched, ["Nobody"])
        rover = data["characters"]["1501"]
        self.assertEqual(rover["name"], "Rover: Spectro")
        self.assertNotIn("mv", rover["abilities"][0])
        self.assertEqual(rover["abilities"][0]["cancel"], 16)
        self.assertIn("1610", data["characters"])  # XuanLing -> Yangyang: Xuanling
        self.assertEqual(json.loads(imp.dump(data)), data)

    def test_shipped_file_is_keyed_by_character_ids_and_timing_only(self):
        ids = {str(c["id"]) for c in CHARACTERS.values()}
        self.assertTrue(SHIPPED["characters"])
        for cid, c in SHIPPED["characters"].items():
            self.assertIn(cid, ids)
            for a in c["abilities"]:
                self.assertLessEqual(set(a), set(imp.KEEP), (c["name"], a["name"]))


if __name__ == "__main__":
    unittest.main()
