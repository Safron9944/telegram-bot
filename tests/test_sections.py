import unittest

from sections import PROTECTED_SECTION_KEYS, UKRAINIAN_LANGUAGE_SECTION_KEY, build_sections, free_section_keys, move_section, reorder_section_group, update_section


class FakeStore:
    def __init__(self):
        self.settings = {}
        self.banks = [{
            "id": 7,
            "slug": "stage-2",
            "title": "Другий етап",
            "source_id": "stage-2.enc",
            "status": "published",
            "display_order": 0,
            "questions_count": 400,
        }]

    async def get_setting(self, key, default=None):
        return self.settings.get(key, default)

    async def set_setting(self, key, value):
        self.settings[key] = value

    async def list_attestation_banks_for_admin(self):
        return list(self.banks)


class SectionCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_banks_and_system_sections_share_one_catalog(self):
        items = await build_sections(FakeStore(), {}, is_admin=False)
        self.assertIn("customs", [item["key"] for item in items])
        dynamic = next(item for item in items if item["key"] == "attestation:7")
        self.assertEqual("Другий етап", dynamic["title"])
        self.assertTrue(dynamic["deletable"])
        self.assertFalse(dynamic["has_access"])
        self.assertEqual(50, dynamic["preview_count"])
        customs = next(item for item in items if item["key"] == "customs")
        self.assertFalse(customs["has_access"])
        self.assertEqual(50, customs["preview_count"])
        language = next(item for item in items if item["key"] == UKRAINIAN_LANGUAGE_SECTION_KEY)
        self.assertTrue(language["manual_grant_only"])
        self.assertFalse(language["has_access"])
        self.assertEqual("ukrainian-language", language["bank_slug"])

    async def test_language_bank_requires_explicit_admin_grant(self):
        store = FakeStore()
        self.assertNotIn(UKRAINIAN_LANGUAGE_SECTION_KEY, await free_section_keys(store))

        full = await build_sections(store, {"sub_infinite": 1, "sub_tier": "full", "section_access": []})
        self.assertFalse(next(item for item in full if item["key"] == UKRAINIAN_LANGUAGE_SECTION_KEY)["has_access"])

        granted = await build_sections(store, {"section_access": [UKRAINIAN_LANGUAGE_SECTION_KEY]})
        self.assertTrue(next(item for item in granted if item["key"] == UKRAINIAN_LANGUAGE_SECTION_KEY)["has_access"])

    async def test_admin_override_has_priority_over_full_subscription(self):
        user = {
            "sub_infinite": True,
            "sub_tier": "full",
            "section_access_overrides": {"customs": False, UKRAINIAN_LANGUAGE_SECTION_KEY: True},
        }
        items = await build_sections(FakeStore(), user, is_admin=False)

        customs = next(item for item in items if item["key"] == "customs")
        language = next(item for item in items if item["key"] == UKRAINIAN_LANGUAGE_SECTION_KEY)
        self.assertFalse(customs["has_access"])
        self.assertTrue(language["has_access"])

    async def test_full_access_unlocks_only_protected_materials_shown_by_admin(self):
        store = FakeStore()
        await update_section(store, "attestation:7", {"price": 0})
        self.assertIn("attestation:7", await free_section_keys(store))
        free = await build_sections(store, {"section_access": []})
        self.assertTrue(next(item for item in free if item["key"] == "attestation:7")["has_access"])
        full = await build_sections(store, {
            "sub_infinite": 1,
            "sub_tier": "full",
            "section_visibility_overrides": {"cases": True},
        })
        protected = [item for item in full if item["key"] in PROTECTED_SECTION_KEYS]
        self.assertTrue(protected)
        self.assertTrue(all(item["has_access"] for item in protected))
        self.assertTrue(next(item for item in protected if item["key"] == "cases")["visible"])
        self.assertTrue(all(not item["visible"] for item in protected if item["key"] != "cases"))
        full_access_keys = {
            item["key"]
            for item in full
            if item["key"] not in PROTECTED_SECTION_KEYS | {UKRAINIAN_LANGUAGE_SECTION_KEY}
        }
        self.assertTrue(all(item["has_access"] for item in full if item["key"] in full_access_keys))

        granted = await build_sections(store, {
            "sub_infinite": 1,
            "sub_tier": "full",
            "section_access": sorted(PROTECTED_SECTION_KEYS),
        })
        self.assertTrue(all(item["has_access"] for item in granted if item["key"] in PROTECTED_SECTION_KEYS))

    async def test_protected_materials_cannot_be_made_free_globally(self):
        store = FakeStore()
        await update_section(store, "cases", {"price": 0})
        self.assertNotIn("cases", await free_section_keys(store))
        items = await build_sections(store, {"section_access": []})
        self.assertFalse(next(item for item in items if item["key"] == "cases")["has_access"])

    async def test_protected_material_is_visible_when_admin_shows_it_but_stays_locked(self):
        items = await build_sections(FakeStore(), {
            "section_access": [],
            "section_visibility_overrides": {"cases": True},
        })
        cases = next(item for item in items if item["key"] == "cases")
        self.assertTrue(cases["visible"])
        self.assertFalse(cases["has_access"])
        self.assertNotIn("manual_grant_only", cases)

    async def test_customs_code_and_support_are_always_free(self):
        store = FakeStore()
        await update_section(store, "customs_code", {"price": 999})
        await update_section(store, "support", {"price": 999})
        self.assertEqual({"customs_code", "support"}, set(await free_section_keys(store)))
        items = await build_sections(store, {})
        for key in ("customs_code", "support"):
            section = next(item for item in items if item["key"] == key)
            self.assertEqual(0, section["price"])
            self.assertTrue(section["has_access"])

    async def test_title_visibility_price_and_order_are_persistent(self):
        store = FakeStore()
        saved = await update_section(store, "customs", {"title": "Навчання", "visible": False, "price": 55})
        self.assertEqual(("Навчання", False, 55), (saved["title"], saved["visible"], saved["price"]))
        self.assertTrue(await move_section(store, "attestation:7", "up"))
        items = await build_sections(store, {}, is_admin=True)
        keys = [item["key"] for item in items]
        self.assertLess(keys.index("attestation:7"), keys.index("support"))

    async def test_default_order_matches_home_groups_and_move_stays_in_group(self):
        store = FakeStore()
        items = await build_sections(store, {}, is_admin=True)
        self.assertEqual(
            ["attestation:7", "customs", "ukrainian_language", "cases", "customs_code", "test_questions", "question_search", "support"],
            [item["key"] for item in items],
        )
        self.assertTrue(await move_section(store, "customs", "up"))
        moved = await build_sections(store, {}, is_admin=True)
        self.assertEqual(["customs", "attestation:7", "ukrainian_language"], [item["key"] for item in moved if item["group"] == "primary"])
        self.assertEqual(["cases", "customs_code", "test_questions", "question_search"], [item["key"] for item in moved if item["group"] == "materials"])

    async def test_drag_order_reorders_exact_group_and_rejects_incomplete_list(self):
        store = FakeStore()
        self.assertTrue(await reorder_section_group(store, "materials", ["question_search", "cases", "customs_code", "test_questions"]))
        items = await build_sections(store, {}, is_admin=True)
        self.assertEqual(
            ["question_search", "cases", "customs_code", "test_questions"],
            [item["key"] for item in items if item["group"] == "materials"],
        )
        self.assertFalse(await reorder_section_group(store, "materials", ["cases"]))


if __name__ == "__main__":
    unittest.main()
