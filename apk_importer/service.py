from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os

from .archive import inspect_package, read_bank
from .crypto import decrypt_testms_payload
from .models import ParsedBank, ParsedQuestion, ParsedSection
from .sessions import FileSessionStore, ImportSession
from .testms import parse_testms_bank


DEFAULT_TESTMSAT_PASSPHRASE = "yYR4XEef3MugI3jb"


class UnsupportedBankError(ValueError):
    code = "unsupported_bank"


class ApkImportService:
    def __init__(self, *, store: FileSessionStore | None = None, testmsat_passphrase: str | None = None):
        self.store = store or FileSessionStore()
        self.testmsat_passphrase = (
            os.getenv("APK_BANK_TESTMSAT_PASSPHRASE", DEFAULT_TESTMSAT_PASSPHRASE)
            if testmsat_passphrase is None else testmsat_passphrase
        )

    def create_session(self, admin_id: int, filename: str, payload: bytes) -> ImportSession:
        package = inspect_package(payload, filename)
        banks = []
        for bank in package.banks:
            known = bank.filename.casefold() == "testmsat.enc"
            supported = known and bool(self.testmsat_passphrase)
            status = "supported" if supported else "missing_passphrase" if known else "unsupported"
            banks.append(replace(bank, supported=supported, adapter="testms" if known else None, status=status))
        return self.store.create(admin_id, filename, payload, tuple(banks))

    def get_session(self, admin_id: int, token: str) -> ImportSession:
        return self.store.get(admin_id, token)

    def parse_bank(self, admin_id: int, token: str, bank_id: str) -> ParsedBank:
        session = self.store.get(admin_id, token)
        selected = next((bank for bank in session.banks if bank.id == bank_id), None)
        if selected is None or not selected.supported:
            raise UnsupportedBankError("Цей банк поки не підтримується.")
        package = inspect_package(self.store.read_upload(admin_id, token), session.filename)
        payload = read_bank(package, bank_id)
        plaintext = decrypt_testms_payload(payload, self.testmsat_passphrase)
        bank = parse_testms_bank(
            plaintext, source=selected.filename, source_hash=hashlib.sha256(payload).hexdigest()
        )
        self.store.write_parsed(admin_id, token, bank_id, bank.to_dict())
        return bank

    def preview(
        self, admin_id: int, token: str, *, section: str = "", query: str = "",
        offset: int = 0, limit: int = 50,
    ) -> dict:
        payload = self.store.read_parsed(admin_id, token)
        normalized = query.casefold().strip()
        items = [
            item for item in payload["questions"]
            if (not section or item["topic"] == section)
            and (not normalized or normalized in (item["question"] + " " + item["topic"]).casefold())
        ]
        offset = max(0, int(offset)); limit = max(1, min(int(limit), 100))
        return {
            "items": items[offset:offset + limit], "total": len(items),
            "offset": offset, "limit": limit,
            "has_prev": offset > 0, "has_next": offset + limit < len(items),
            "sections": payload["sections"], "count": payload["count"],
        }

    def download_json(self, admin_id: int, token: str) -> bytes:
        return json.dumps(
            self.store.read_parsed(admin_id, token), ensure_ascii=False, indent=2
        ).encode("utf-8")

    def delete_session(self, admin_id: int, token: str) -> None:
        self.store.delete(admin_id, token)
