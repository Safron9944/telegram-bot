from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apk_importer.models import ArchiveBank
from apk_importer.sessions import (
    FileSessionStore,
    SessionAccessError,
    SessionExpiredError,
    SessionNotFoundError,
)


class FileSessionStoreTests(unittest.TestCase):
    def test_session_is_owner_bound_and_cancel_removes_files(self):
        with TemporaryDirectory() as directory:
            now = [1_000.0]
            store = FileSessionStore(Path(directory), ttl_seconds=1_800, clock=lambda: now[0])
            bank = ArchiveBank("bank", "assets/www/testmsat.enc", "testmsat.enc", 10)

            session = store.create(7, "base.apk", b"archive", (bank,))

            self.assertEqual(2_800.0, session.expires_at)
            self.assertEqual(b"archive", store.read_upload(7, session.token))
            with self.assertRaises(SessionAccessError):
                store.get(8, session.token)
            store.delete(7, session.token)
            self.assertFalse((Path(directory) / session.token).exists())
            with self.assertRaises(SessionNotFoundError):
                store.get(7, session.token)

    def test_expired_session_is_reported_once_and_cleaned(self):
        with TemporaryDirectory() as directory:
            now = [1_000.0]
            store = FileSessionStore(Path(directory), ttl_seconds=10, clock=lambda: now[0])
            session = store.create(7, "base.apk", b"archive", ())
            now[0] = 1_011.0

            with self.assertRaises(SessionExpiredError):
                store.get(7, session.token)

            self.assertFalse((Path(directory) / session.token).exists())
            with self.assertRaises(SessionNotFoundError):
                store.get(7, session.token)

    def test_parsed_json_round_trip_does_not_create_plaintext_file(self):
        with TemporaryDirectory() as directory:
            store = FileSessionStore(Path(directory), clock=lambda: 1_000.0)
            session = store.create(7, "base.apk", b"archive", ())

            store.write_parsed(7, session.token, "bank", {"questions": [{"question": "Україна"}]})

            self.assertEqual(
                {"questions": [{"question": "Україна"}]},
                store.read_parsed(7, session.token),
            )
            names = {path.name for path in (Path(directory) / session.token).iterdir()}
            self.assertEqual({"metadata.json", "upload.bin", "parsed.json"}, names)


if __name__ == "__main__":
    unittest.main()
