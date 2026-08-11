from types import SimpleNamespace
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from admin_apk_import_extension import register_apk_import_routes
from apk_importer.service import ApkImportService


class ManagedStore:
    def __init__(self):
        self.banks = [{
            "id": 7, "slug": "stage-2", "title": "Етап 2",
            "status": "published", "display_order": 0, "questions_count": 400,
        }]
        self.calls = []
        self.settings = {}

    async def get_setting(self, key, default=None):
        return self.settings.get(key, default)

    async def set_setting(self, key, value):
        self.settings[key] = value
        self.calls.append(("setting", key, value))

    async def list_attestation_banks_for_admin(self):
        return list(self.banks)

    async def update_attestation_bank_title(self, bank_id, title):
        self.calls.append(("title", bank_id, title))
        if bank_id != 7:
            return None
        self.banks[0]["title"] = title
        return self.banks[0]

    async def list_attestation_questions_for_admin(self, bank_id, **kwargs):
        self.calls.append(("questions", bank_id, kwargs))
        if bank_id != 7:
            return None
        return {
            "bank": self.banks[0],
            "topics": [{"topic": "Основний", "questions_count": 1}],
            "items": [{
                "id": 81, "qnum": 1, "topic": "Основний", "question": "Питання?",
                "choices": ["Так", "Ні"], "correct": [1], "shuffle_choices": True,
                "managed_manually": False,
            }],
            "total": 1, "offset": kwargs["offset"], "limit": kwargs["limit"],
        }

    async def get_attestation_question_for_admin(self, bank_id, question_id):
        self.calls.append(("question", bank_id, question_id))
        if bank_id != 7 or question_id != 81:
            return None
        return {
            "id": 81, "bank_id": 7, "qnum": 1, "topic": "Основний",
            "question": "Питання?", "choices": ["Так", "Ні"], "correct": [1],
            "shuffle_choices": True, "managed_manually": False,
        }

    async def create_attestation_question(self, bank_id, **question):
        self.calls.append(("create_question", bank_id, question))
        return {"id": 82, "bank_id": bank_id, **question, "managed_manually": True} if bank_id == 7 else None

    async def update_attestation_question(self, bank_id, question_id, **question):
        self.calls.append(("update_question", bank_id, question_id, question))
        return {"id": question_id, "bank_id": bank_id, **question, "managed_manually": True} if (bank_id, question_id) == (7, 81) else None

    async def delete_attestation_question(self, bank_id, question_id):
        self.calls.append(("delete_question", bank_id, question_id))
        return (bank_id, question_id) == (7, 81)

    async def set_attestation_bank_visibility(self, bank_id, *, visible):
        self.calls.append(("visibility", bank_id, visible))
        return self.banks[0] if bank_id == 7 else None

    async def move_attestation_bank(self, bank_id, *, direction):
        self.calls.append(("move", bank_id, direction))
        return bank_id == 7

    async def delete_attestation_bank(self, bank_id):
        self.calls.append(("delete", bank_id))
        return bank_id == 7


class RuntimeBank:
    def __init__(self):
        self.reloads = 0

    async def load_published_attestation_banks(self, store):
        self.reloads += 1


def make_app(*, is_admin=True):
    app = FastAPI()
    store = ManagedStore()
    qb = RuntimeBank()
    app.state.runtime = SimpleNamespace(store=store, qb=qb)

    async def auth():
        return SimpleNamespace(user_id=7, is_admin=is_admin)

    register_apk_import_routes(app, auth, service=ApkImportService(testmsat_passphrase=""))
    return app, store, qb


class AttestationManagementApiTests(unittest.TestCase):
    def test_admin_lists_and_manages_dynamic_banks(self):
        app, store, qb = make_app()
        client = TestClient(app)

        listed = client.get("/api/admin/attestation-banks")
        hidden = client.patch("/api/admin/attestation-banks/7/visibility", json={"visible": False})
        moved = client.post("/api/admin/attestation-banks/7/move", json={"direction": "up"})
        deleted = client.delete("/api/admin/attestation-banks/7")

        self.assertEqual("stage-1", listed.json()["items"][0]["slug"])
        self.assertTrue(listed.json()["items"][0]["system"])
        self.assertEqual([("visibility", 7, False), ("move", 7, "up"), ("delete", 7)], store.calls)
        self.assertEqual(3, qb.reloads)
        self.assertEqual([200, 200, 204], [hidden.status_code, moved.status_code, deleted.status_code])

    def test_admin_can_delete_bundled_stage_1(self):
        app, store, _ = make_app()
        client = TestClient(app)

        deleted = client.delete("/api/admin/attestation-banks/stage-1")
        listed = client.get("/api/admin/attestation-banks")

        self.assertEqual(204, deleted.status_code)
        self.assertEqual(("setting", "attestation_stage_1_deleted", "1"), store.calls[-1])
        self.assertNotIn("stage-1", [item["slug"] for item in listed.json()["items"]])

    def test_admin_can_manage_bank_and_questions(self):
        app, store, qb = make_app()
        client = TestClient(app)
        question = {
            "topic": "Новий підрозділ", "qnum": 5, "question": "Оновлене питання?",
            "choices": ["А", "Б"], "correct": [2], "shuffle_choices": False,
        }

        renamed = client.patch("/api/admin/attestation-banks/7", json={"title": "Нова назва"})
        listed = client.get("/api/admin/attestation-banks/7/questions?q=Питання")
        loaded = client.get("/api/admin/attestation-banks/7/questions/81")
        created = client.post("/api/admin/attestation-banks/7/questions", json=question)
        updated = client.patch("/api/admin/attestation-banks/7/questions/81", json=question)
        deleted = client.delete("/api/admin/attestation-banks/7/questions/81")

        self.assertEqual("Нова назва", renamed.json()["title"])
        self.assertEqual(1, listed.json()["total"])
        self.assertEqual(81, loaded.json()["question"]["id"])
        self.assertTrue(created.json()["question"]["managed_manually"])
        self.assertTrue(updated.json()["question"]["managed_manually"])
        self.assertEqual(204, deleted.status_code)
        self.assertEqual(4, qb.reloads)
        self.assertEqual("Б", store.calls[-2][3]["correct_texts"][0])

    def test_missing_bank_and_non_admin_are_rejected(self):
        app, _, _ = make_app()
        self.assertEqual(404, TestClient(app).delete("/api/admin/attestation-banks/99").status_code)

        user_app, _, _ = make_app(is_admin=False)
        self.assertEqual(403, TestClient(user_app).get("/api/admin/attestation-banks").status_code)


if __name__ == "__main__":
    unittest.main()
