from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from admin_apk_import_extension import register_apk_import_routes
from apk_importer.service import ApkImportService
from apk_importer.sessions import FileSessionStore
from tests.test_apk_import_service import apk_payload


class AdminApkImportApiTests(unittest.TestCase):
    def test_admin_can_complete_preview_and_download_workflow(self):
        with TemporaryDirectory() as directory:
            app = FastAPI()
            service = ApkImportService(
                store=FileSessionStore(Path(directory), clock=lambda: 1_000.0),
                testmsat_passphrase="secret",
            )

            async def admin_auth():
                return SimpleNamespace(user_id=7, is_admin=True)

            register_apk_import_routes(app, admin_auth, service=service)
            client = TestClient(app)
            uploaded = client.post(
                "/api/admin/apk-import/sessions",
                files={"file": ("base.apk", apk_payload("secret"), "application/vnd.android.package-archive")},
            )
            self.assertEqual(200, uploaded.status_code)
            session = uploaded.json()
            bank_id = next(bank["id"] for bank in session["banks"] if bank["supported"])

            parsed = client.post(f"/api/admin/apk-import/sessions/{session['token']}/banks/{bank_id}/parse")
            preview = client.get(f"/api/admin/apk-import/sessions/{session['token']}/preview?limit=1")
            downloaded = client.get(f"/api/admin/apk-import/sessions/{session['token']}/download")
            deleted = client.delete(f"/api/admin/apk-import/sessions/{session['token']}")

            self.assertEqual(2, parsed.json()["summary"]["questions_count"])
            self.assertEqual(1, len(preview.json()["items"]))
            self.assertEqual("application/json", downloaded.headers["content-type"])
            self.assertIn("attachment", downloaded.headers["content-disposition"])
            self.assertEqual(204, deleted.status_code)

    def test_non_admin_is_forbidden(self):
        app = FastAPI()

        async def user_auth():
            return SimpleNamespace(user_id=8, is_admin=False)

        register_apk_import_routes(app, user_auth, service=ApkImportService(testmsat_passphrase=""))
        response = TestClient(app).post(
            "/api/admin/apk-import/sessions",
            files={"file": ("base.apk", b"not-used", "application/octet-stream")},
        )
        self.assertEqual(403, response.status_code)
        self.assertEqual("forbidden", response.json()["detail"]["code"])


if __name__ == "__main__":
    unittest.main()
