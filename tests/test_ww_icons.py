import asyncio
import unittest
from unittest import mock

from combo_engine import ComboTrackerEngine
from state_store import MemoryStateStore
import ww_sync
from ww_icons import character_slug, parse_character_icons

# Trimmed from a wuthering.gg character page: paths are HTML-escaped and relative.
SAMPLE_PAGE = """
<img src="/_ipx/q_70&amp;s_100x100/images/iconrolehead150/T_IconRoleHead150_53_UI.png">
<img src="/_ipx/q_70&amp;s_32x32/images/iconskill/SP_IconNorKnife.png">
<img src="/_ipx/q_70&amp;s_32x32/images/iconskill/SP_IconAimisiB1.png">
<img src="/_ipx/q_70&amp;s_32x32/images/iconskill/SP_IconAimisiC1.png">
<img src="/_ipx/q_70&amp;s_34x34/images/mstskill/T_MstSkil_34025_UI.png">
"""


class ParseIconsTests(unittest.TestCase):
    def test_parses_full_icon_set(self):
        icons = parse_character_icons(SAMPLE_PAGE)
        self.assertEqual(
            icons["swap_image"],
            "https://wuthering.gg/_ipx/q_70&s_100x100/images/iconrolehead150/T_IconRoleHead150_53_UI.png",
        )
        self.assertTrue(icons["lmb_image"].endswith("SP_IconNorKnife.png"))
        self.assertTrue(icons["ability_images"]["e"].endswith("SP_IconAimisiB1.png"))
        self.assertTrue(icons["ability_images"]["r"].endswith("SP_IconAimisiC1.png"))
        self.assertTrue(icons["ability_images"]["q"].endswith("T_MstSkil_34025_UI.png"))
        self.assertEqual(icons["missing"], [])

    def test_reports_missing_icons(self):
        icons = parse_character_icons("<html>nothing here</html>")
        self.assertEqual(set(icons["missing"]), {"swap_image", "lmb_image", "e", "r", "q"})

    def test_slug(self):
        self.assertEqual(character_slug("  Phrolova "), "phrolova")
        self.assertEqual(character_slug("Jiyan Zhe"), "jiyan-zhe")


class SyncFromWebTests(unittest.TestCase):
    def setUp(self):
        self.engine = ComboTrackerEngine(state_store=MemoryStateStore())
        self.sent = []
        self.engine.set_emitter(self.sent.append)

    def statuses(self):
        return [m["text"] for m in self.sent if m.get("type") == "status"]

    def test_adds_character_from_page(self):
        with mock.patch.object(ww_sync, "fetch_character_icons", return_value=parse_character_icons(SAMPLE_PAGE)):
            asyncio.run(ww_sync.sync_character(self.engine, "aemeath"))
        saved = self.engine.ww.ww_characters["aemeath"]
        self.assertEqual(saved["name"], "Aemeath")
        self.assertTrue(saved["ability_images"]["e"].endswith("SP_IconAimisiB1.png"))
        self.assertIn("Saved Aemeath with icons from wuthering.gg.", self.statuses())

    def test_keeps_existing_icons_the_site_lacks(self):
        self.engine.save_ww_character(name="Zani", swap_image="old.png", lmb_image="lmb.png", ability_images={"q": "echo.png"})
        partial = parse_character_icons(SAMPLE_PAGE.replace("T_MstSkil_34025_UI", "nope"))
        with mock.patch.object(ww_sync, "fetch_character_icons", return_value=partial):
            asyncio.run(ww_sync.sync_character(self.engine, "zani"))
        saved = self.engine.ww.ww_characters["zani"]
        self.assertEqual(saved["ability_images"]["q"], "echo.png")
        self.assertTrue(saved["swap_image"].startswith("https://wuthering.gg/"))

    def test_unknown_character_reports_error(self):
        with mock.patch.object(ww_sync, "fetch_character_icons", side_effect=RuntimeError("no page")):
            asyncio.run(ww_sync.sync_character(self.engine, "nobody"))
        self.assertNotIn("nobody", self.engine.ww.ww_characters)
        self.assertIn("no page", self.statuses())
