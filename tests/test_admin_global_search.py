from types import SimpleNamespace
import unittest

from app import AuthContext, api_admin_global_search
from questions import AttestationBank, Q


class SearchStore:
    async def get_setting(self, key, default=None):
        return default

    async def search_case_questions_all(self, query, limit=15):
        return []

    async def search_test_exam_questions(self, query, limit=15, offset=0):
        return {"items": []}

    async def search_attestation_questions_all(self, query, limit=15):
        return [{
            "id": 81,
            "bank_id": 7,
            "bank_slug": "new-bank",
            "bank_title": "Новий розділ",
            "bank_status": "hidden",
            "qnum": 1,
            "topic": "Підрозділ",
            "question": "Митне питання з APK?",
        }]


class AdminGlobalSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_includes_bundled_and_dynamic_attestation_questions(self):
        bundled = Q(
            id=101,
            section="Атестація",
            topic="Закон",
            ok=None,
            level=None,
            qnum=1,
            question="Митне питання атестації?",
            choices=["Так", "Ні"],
            correct=[1],
            correct_texts=["Так"],
        )
        qb = SimpleNamespace(
            by_id={101: bundled},
            attestation_banks={
                "ukrainian-language": AttestationBank("ukrainian-language", "Державна мова", [101]),
            },
        )
        runtime = SimpleNamespace(store=SearchStore(), qb=qb)
        auth = AuthContext({}, {}, 1, True)

        result = await api_admin_global_search("митне", 15, auth, runtime)

        self.assertEqual([], result["ok"])
        self.assertEqual(2, len(result["attestation"]))
        self.assertEqual({"Державна мова", "Новий розділ"}, {item["bank_title"] for item in result["attestation"]})


if __name__ == "__main__":
    unittest.main()
