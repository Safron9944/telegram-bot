from __future__ import annotations

import unittest

from migrate_customs_competencies import (
    BUNDLE_PATH,
    EXPECTED_LAW,
    EXPECTED_OK,
    EXPECTED_OK_MODULES,
    EXPECTED_TOTAL,
    _load_bundle,
    _question_key,
)
from utils import ok_extract_code


class CustomsCompetenciesBundleTests(unittest.TestCase):
    def test_apk_1_9_1_bundle_shape(self):
        items = _load_bundle(BUNDLE_PATH)
        self.assertEqual(EXPECTED_TOTAL, len(items))
        self.assertEqual(EXPECTED_TOTAL, len({int(item["id"]) for item in items}))
        self.assertEqual(EXPECTED_LAW, sum(1 for item in items if not item.get("ok")))
        self.assertEqual(EXPECTED_OK, sum(1 for item in items if item.get("ok")))
        self.assertEqual(
            EXPECTED_OK_MODULES,
            len({ok_extract_code(str(item.get("ok") or "")) for item in items if item.get("ok")}),
        )
        self.assertEqual(len(items), len({_question_key(item) for item in items}))
        competency_keys = {
            (item.get("ok"), item.get("level"), item.get("qnum"))
            for item in items
            if item.get("ok")
        }
        self.assertEqual(EXPECTED_OK, len(competency_keys))

    def test_ok_17_has_three_levels_of_seventy(self):
        items = _load_bundle(BUNDLE_PATH)
        ok17 = [item for item in items if item.get("ok") == "ОК-17"]
        self.assertEqual(210, len(ok17))
        for level in (1, 2, 3):
            self.assertEqual(70, sum(1 for item in ok17 if item.get("level") == level))


if __name__ == "__main__":
    unittest.main()
