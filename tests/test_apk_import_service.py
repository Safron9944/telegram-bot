import base64
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from apk_importer.crypto import evp_bytes_to_key_md5
from apk_importer.service import ApkImportService, UnsupportedBankError
from apk_importer.sessions import FileSessionStore, SessionAccessError


FIXTURE = Path(__file__).parent / "fixtures" / "testms_plaintext_small.txt"


def encrypted_fixture(passphrase: str, header: str = "testmsat 3", *, section: bool = True) -> bytes:
    plaintext_text = FIXTURE.read_text(encoding="utf-8")
    plaintext_text = plaintext_text.replace("testmsat 3", header, 1)
    if not section:
        plaintext_text = plaintext_text.replace("~I. Основи\n\n", "", 1)
    plaintext = plaintext_text.encode("cp1251")
    padding = 16 - len(plaintext) % 16
    padded = plaintext + bytes([padding]) * padding
    salt = b"12345678"
    key, iv = evp_bytes_to_key_md5(passphrase.encode(), salt)
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    encoded = base64.b64encode(b"Salted__" + salt + encryptor.update(padded) + encryptor.finalize())
    return encoded[len(b"U2FsdGVkX1") :]


def apk_payload(passphrase: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("assets/www/testmsat.enc", encrypted_fixture(passphrase))
        archive.writestr("assets/www/testmsca.enc", encrypted_fixture("jYfYZHqt6TuKUaB5", "testms 8"))
        archive.writestr("assets/www/testmsmo.enc", encrypted_fixture("A6KPIz8Rci2ZF3sy", "testmsmo 11"))
        archive.writestr("assets/www/testmsto.enc", encrypted_fixture("ERT2penZTQaDh6mQ", "testmsto 12", section=False))
    return output.getvalue()


class ApkImportServiceTests(unittest.TestCase):
    def test_default_adapter_supports_known_testmsat_bank(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            service = ApkImportService(store=FileSessionStore(Path(directory), clock=lambda: 1_000.0))
            session = service.create_session(7, "base.apk", apk_payload("yYR4XEef3MugI3jb"))
            self.assertTrue(next(bank for bank in session.banks if bank.filename == "testmsat.enc").supported)

    def test_lists_and_parses_all_known_banks_with_friendly_titles(self):
        with TemporaryDirectory() as directory:
            store = FileSessionStore(Path(directory), clock=lambda: 1_000.0)
            service = ApkImportService(store=store, testmsat_passphrase="secret")

            session = service.create_session(7, "base.apk", apk_payload("secret"))

            self.assertEqual(4, len(session.banks))
            supported = [bank for bank in session.banks if bank.supported]
            self.assertEqual(4, len(supported))
            self.assertEqual(
                {"Перший етап", "Митних органів", "Територіальних органів", "Центрального апарату"},
                {bank.title for bank in supported},
            )
            for selected in supported:
                parsed = service.parse_bank(7, session.token, selected.id)
                self.assertEqual(2, parsed.summary.questions_count)
            with self.assertRaises(SessionAccessError):
                service.get_session(8, session.token)

    def test_preview_filters_paginates_and_downloads_utf8_json(self):
        with TemporaryDirectory() as directory:
            service = ApkImportService(
                store=FileSessionStore(Path(directory), clock=lambda: 1_000.0),
                testmsat_passphrase="secret",
            )
            session = service.create_session(7, "base.apk", apk_payload("secret"))
            bank_id = next(bank.id for bank in session.banks if bank.supported)
            service.parse_bank(7, session.token, bank_id)

            page = service.preview(7, session.token, section="Основи", query="таке", offset=0, limit=1)

            self.assertEqual(1, page["total"])
            self.assertEqual("Що таке?", page["items"][0]["question"])
            document = service.download_json(7, session.token)
            self.assertIn("Що таке?", document.decode("utf-8"))
            self.assertEqual(2, json.loads(document)["count"])

    def test_known_bank_is_disabled_without_server_passphrase(self):
        with TemporaryDirectory() as directory:
            service = ApkImportService(
                store=FileSessionStore(Path(directory), clock=lambda: 1_000.0),
                testmsat_passphrase="",
            )

            session = service.create_session(7, "base.apk", apk_payload("secret"))

            selected = next(bank for bank in session.banks if bank.filename == "testmsat.enc")
            self.assertFalse(selected.supported)
            self.assertEqual("missing_passphrase", selected.status)


if __name__ == "__main__":
    unittest.main()
