import unittest

from sections import build_sections, free_section_keys, move_section, reorder_section_group, update_section


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

    async def test_zero_price_is_free_and_full_access_unlocks_everything(self):
        store = FakeStore()
        await update_section(store, "attestation:7", {"price": 0})
        self.assertIn("attestation:7", await free_section_keys(store))
        free = await build_sections(store, {"section_access": []})
        self.assertTrue(next(item for item in free if item["key"] == "attestation:7")["has_access"])
        full = await build_sections(store, {"sub_infinite": 1, "sub_tier": "full"})
        self.assertTrue(all(item["has_access"] for item in full))

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
            ["attestation:7", "customs", "cases", "customs_code", "test_questions", "question_search", "support"],
            [item["key"] for item in items],
        )
        self.assertTrue(await move_section(store, "customs", "up"))
        moved = await build_sections(store, {}, is_admin=True)
        self.assertEqual(["customs", "attestation:7"], [item["key"] for item in moved if item["group"] == "primary"])
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
