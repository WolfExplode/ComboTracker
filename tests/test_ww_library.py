import json
import tempfile
import unittest
from pathlib import Path

import ww_library
from ww_library import (
    Library,
    LibraryError,
    clean_description,
    handle_api,
    parse_kit,
    parse_roster,
    parse_rotation_sheet,
    parse_transcript,
    transcript_export_url,
)

# Trimmed from the gviz CSV export of AntoCrasher's "2 Minute Calc Compilation" tab.
SHEET_CSV = '''"","AntoCrasher Calc - Zani Team Damage With Different Setups (2 Minutes / 4 Rot + Extension) ","Author","Extra Info","Zani","Slot 2","Slot 3","Tunebreak","Total Damage","DPS","Rotation Time","%","Video Showcase","Rotation Transcript","Calc Sheet","Last Updated","","Personal Thoughts ""Start with the 3NF quickswap rotation.""",""
"","Zani, Phoebe S0R1, Rover S6R1 (Advanced Quickswap)","Cyan","4NF","7,490,611.64","1,179,870.22","1,608,456.23","313,440.43","10,592,378.52","88,269.82","120.00","146.79%","https://www.youtube.com/watch?v=CEQWQrnbQg8","https://docs.google.com/document/d/1Bfs9xX7ZbnYPjXrfLdmBZOVBDJ2uX3q1u6D2v5sHZcQ/edit?tab=t.t9jbtmxsowwk","https://docs.google.com/spreadsheets/d/14_Lj/edit?gid=898931522","","",""
"","Zani, Phoebe S0R1, Shorekeeper S0R1 (123) ","—","2.5NF","5,556,798.55","1,181,070.87","227,631.18","250,752.34","7,216,252.94","60,135.44","120.00","100.00%","https://youtu.be/Vtl3Oh7Qa4c","","","","",""
"","AntoCrasher Calc - Cartethyia Team Damage With Different Setups (2 Minutes / 4 Rot + Extension)","","","","","","","","","","","","","","","","Personal Thoughts","For casual play, learn the optimized 123 rotation.",""
"","","Author","Extra Info","","","","","","","","","Video Showcase","Rotation Transcript","Calc Sheet","Last Updated","","",""
"","Cartethyia Solo","Eugen","DP + 3x BA5","3,984,615.86","0.00","0.00","386,715.28","4,371,331.13","36,427.76","120.00","53.43%","https://youtu.be/YxTsJuCHoXg","","","","",""
'''

TRANSCRIPT = """﻿AntoCrasher Rotations Hub
ba = basic attack
nf = nightfall
lib = liberation

Zani Phoebe Rover ADV QS Cyan 4NF Transcript

Opener

phoebe: eskill > lib > skill > echo > dash > ha > swap
rover: ba23 > ha23 > echo > lib > fskill > swap
zani: skill > ba3 > swap
phoebe: outro
"""

KIT = {
    "Id": 1507,
    "Name": "Zani",
    "ElementName": "Spectro",
    "RoleHeadIcon": "https://example/zani.webp",
    "Skills": [
        {"SkillId": 2, "SkillType": "Resonance Liberation", "SkillName": "Rebellious Instinct",
         "SkillDescribe": "Deal DMG.", "SkillAttributes": []},
        {"SkillId": 1, "SkillType": "Normal Attack", "SkillName": "Routine Negotiation",
         "SkillDescribe": '<span class="font-bold">Basic Attack</span><br>Perform up to 4 attacks &amp; more.',
         "Icon": "https://example/fist.webp",
         "SkillAttributes": [{"attributeName": "Stage 1 DMG", "values": ["29.60%", "32.03%"]}]},
    ],
}


class ParseTests(unittest.TestCase):
    def test_roster_dedupes_rover_and_sorts(self):
        data = {"roleList": [
            {"Id": 1507, "Name": "Zani", "Element": {"Name": "Spectro"}, "WeaponType": {"Name": "Gauntlets"}},
            {"Id": 1501, "Name": "Rover: Spectro", "Element": {"Name": "Spectro"}},
            {"Id": 1502, "Name": "Rover: Spectro", "Element": {"Name": "Spectro"}},
        ]}
        chars = parse_roster(data)["characters"]
        self.assertEqual([c["name"] for c in chars], ["Rover: Spectro", "Zani"])
        self.assertEqual(chars[1]["weapon"], "Gauntlets")

    def test_kit_orders_skills_and_cleans_text(self):
        kit = parse_kit(KIT)
        self.assertEqual([s["type"] for s in kit["skills"]], ["Normal Attack", "Resonance Liberation"])
        na = kit["skills"][0]
        self.assertEqual(na["key"], "LMB")
        self.assertEqual(na["description"], "**Basic Attack**\nPerform up to 4 attacks & more.")
        self.assertEqual(na["multipliers"][0], {"name": "Stage 1 DMG", "values": ["29.60%", "32.03%"]})

    def test_clean_description_collapses_blank_lines(self):
        self.assertEqual(clean_description("a<br><br><br><br>b"), "a\n\nb")

    def test_rotation_sheet_groups(self):
        groups = parse_rotation_sheet(SHEET_CSV)
        self.assertEqual([g["character"] for g in groups], ["Zani", "Cartethyia"])
        zani = groups[0]
        self.assertEqual(zani["thoughts"], "Start with the 3NF quickswap rotation.")
        self.assertEqual(len(zani["teams"]), 2)
        top = zani["teams"][0]
        self.assertEqual(top["members"], ["Zani", "Phoebe", "Rover"])
        self.assertEqual(top["style"], "Advanced Quickswap")
        self.assertEqual(top["author"], "Cyan")
        self.assertAlmostEqual(top["dps"], 88269.82)
        self.assertTrue(top["transcript"].startswith("https://docs.google.com/document/"))
        self.assertEqual(zani["teams"][1]["author"], "")  # "—" means no author
        carte = groups[1]
        self.assertEqual(carte["thoughts"], "For casual play, learn the optimized 123 rotation.")
        self.assertEqual(carte["teams"][0]["members"], ["Cartethyia Solo"])

    def test_transcript(self):
        t = parse_transcript(TRANSCRIPT)
        self.assertEqual(t["glossary"]["nf"], "nightfall")
        self.assertEqual(t["title"], "Zani Phoebe Rover ADV QS Cyan 4NF Transcript")
        self.assertEqual(len(t["sections"]), 1)
        sec = t["sections"][0]
        self.assertEqual(sec["name"], "Opener")
        self.assertEqual([s["character"] for s in sec["steps"]], ["phoebe", "rover", "zani", "phoebe"])
        self.assertEqual(sec["steps"][1]["moves"], ["ba23", "ha23", "echo", "lib", "fskill", "swap"])
        self.assertEqual(sec["steps"][3]["moves"], ["outro"])

    def test_transcript_export_url_keeps_tab(self):
        url = transcript_export_url("https://docs.google.com/document/d/ABC_1/edit?tab=t.xyz")
        self.assertEqual(url, "https://docs.google.com/document/d/ABC_1/export?format=txt&tab=t.xyz")
        self.assertEqual(transcript_export_url("https://example.com/x"), "")


class LibraryCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        self.fail = False

        def fetch(url):
            self.calls.append(url)
            if self.fail:
                raise LibraryError("offline")
            return json.dumps(KIT)

        self.lib = Library(Path(self.tmp.name), fetch=fetch)

    def tearDown(self):
        self.tmp.cleanup()

    def test_caches_and_serves_stale_when_offline(self):
        self.assertEqual(self.lib.kit(1507)["name"], "Zani")
        self.lib.kit(1507)
        self.assertEqual(len(self.calls), 1)  # second read came from cache
        self.fail = True
        stale = self.lib.kit(1507, refresh=True)
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["name"], "Zani")

    def test_api_routes(self):
        status, body = handle_api(self.lib, "/api/ww/kit/1507")
        self.assertEqual((status, body["name"]), (200, "Zani"))
        self.assertEqual(handle_api(self.lib, "/api/ww/nope")[0], 404)
        self.fail = True
        status, body = handle_api(self.lib, "/api/ww/kit/1508")
        self.assertEqual(status, 502)
        self.assertIn("offline", body["error"])


if __name__ == "__main__":
    unittest.main()
