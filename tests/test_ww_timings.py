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

    def test_reads_full_scrape_by_header_name(self):
        row = {
            "section": "HEAVY ATTACK3 ABILITIES", "name": "Intro: X", "tags": ["MOTION STOP 6-32F"],
            "values": {"hits": 2, "frames": 100, "cancel": 76, "noswap": 84, "tstop": 0, "mstop": 26,
                       "concerto": 1000, "energy": 0, "cd": "\u2014", "genre": "INTRO", "pri": 11},
            "hits_detail": [{"frame": "48f", "mv": "15.91%"}, {"frame": "52f", "mv": "15.91%"}],
            "timeline": [{"cls": "tl-zone tl-zone--ms", "style": "left: 6%; right: 68%;"}],
        }
        a = imp.ability(imp.flat_row(row))
        self.assertEqual(a["section"], "HEAVY ATTACK")
        self.assertEqual(a["concerto"], 1000)
        self.assertIsNone(a["cd"])
        self.assertEqual(a["hit_frames"], [48, 52])
        self.assertEqual(a["zones"], [{"kind": "ms", "from": 6, "to": 32}])
        self.assertNotIn("mv", a)
        self.assertNotIn("energy", a)
        self.assertNotIn("pri", a)

    def test_drops_cooldowns_from_shifted_columns(self):
        self.assertEqual(imp.cooldown({"cd": 900, "genre": "SKILL"}), 900)
        self.assertIsNone(imp.cooldown({"cd": None, "genre": "BASIC"}))
        self.assertIsNone(imp.cooldown({"cd": 11, "genre": "0:11, 81:2"}))  # priority landed in cd
        self.assertIsNone(imp.cooldown({"cd": "0:2"}))                       # timeline landed in cd

    def test_shipped_file_has_no_shifted_cooldowns(self):
        for c in SHIPPED["characters"].values():
            for a in c["abilities"]:
                self.assertTrue(a["cd"] is None or isinstance(a["cd"], int), (c["name"], a["name"], a["cd"]))
        iuno = next(c for c in SHIPPED["characters"].values() if c["name"] == "Iuno")
        self.assertIsNone(iuno["abilities"][0]["cd"])

    def test_shipped_file_is_keyed_by_character_ids_and_timing_only(self):
        ids = {str(c["id"]) for c in CHARACTERS.values()}
        self.assertTrue(SHIPPED["characters"])
        for cid, c in SHIPPED["characters"].items():
            self.assertIn(cid, ids)
            for a in c["abilities"]:
                self.assertLessEqual(set(a), set(imp.KEEP), (c["name"], a["name"]))


if __name__ == "__main__":
    unittest.main()
