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

    def test_missing_bank_and_non_admin_are_rejected(self):
        app, _, _ = make_app()
        self.assertEqual(404, TestClient(app).delete("/api/admin/attestation-banks/99").status_code)

        user_app, _, _ = make_app(is_admin=False)
        self.assertEqual(403, TestClient(user_app).get("/api/admin/attestation-banks").status_code)


if __name__ == "__main__":
    unittest.main()
