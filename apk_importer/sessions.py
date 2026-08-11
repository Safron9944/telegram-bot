from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import time

from .models import ArchiveBank


class SessionError(ValueError):
    code = "session_error"


class SessionNotFoundError(SessionError):
    code = "session_not_found"


class SessionExpiredError(SessionError):
    code = "session_expired"


class SessionAccessError(SessionError):
    code = "session_forbidden"


@dataclass(frozen=True)
class ImportSession:
    token: str
    owner_id: int
    filename: str
    expires_at: float
    banks: tuple[ArchiveBank, ...]
    parsed_bank_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "owner_id": self.owner_id,
            "filename": self.filename,
            "expires_at": self.expires_at,
            "banks": [bank.to_dict() for bank in self.banks],
            "parsed_bank_id": self.parsed_bank_id,
        }


class FileSessionStore:
    def __init__(self, root: Path | None = None, *, ttl_seconds: int = 1_800, clock=time.time):
        self.root = Path(root or Path(tempfile.gettempdir()) / "telegram-bot-apk-import")
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, token: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", token or ""):
            raise SessionNotFoundError("Сесію не знайдено.")
        return self.root / token

    @staticmethod
    def _atomic(path: Path, payload: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    def _load(self, token: str) -> ImportSession:
        directory = self._directory(token)
        try:
            data = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            raise SessionNotFoundError("Сесію не знайдено.") from exc
        banks = tuple(ArchiveBank(**bank) for bank in data.get("banks", []))
        return ImportSession(
            token=data["token"], owner_id=int(data["owner_id"]), filename=data["filename"],
            expires_at=float(data["expires_at"]), banks=banks,
            parsed_bank_id=data.get("parsed_bank_id"),
        )

    def cleanup(self, *, exclude: str | None = None) -> None:
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.name == exclude:
                continue
            try:
                session = self._load(directory.name)
            except SessionNotFoundError:
                shutil.rmtree(directory, ignore_errors=True)
                continue
            if session.expires_at <= self.clock():
                shutil.rmtree(directory, ignore_errors=True)

    def create(self, owner_id: int, filename: str, payload: bytes, banks: tuple[ArchiveBank, ...]) -> ImportSession:
        self.cleanup()
        token = secrets.token_urlsafe(32)
        directory = self._directory(token)
        directory.mkdir()
        session = ImportSession(token, int(owner_id), filename, self.clock() + self.ttl_seconds, banks)
        self._atomic(directory / "upload.bin", bytes(payload))
        self._atomic(
            directory / "metadata.json",
            json.dumps(session.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
        return session

    def get(self, owner_id: int, token: str) -> ImportSession:
        session = self._load(token)
        if session.expires_at <= self.clock():
            shutil.rmtree(self._directory(token), ignore_errors=True)
            raise SessionExpiredError("Сесія завершилася.")
        self.cleanup(exclude=token)
        if session.owner_id != int(owner_id):
            raise SessionAccessError("Сесія належить іншому адміністратору.")
        return session

    def read_upload(self, owner_id: int, token: str) -> bytes:
        self.get(owner_id, token)
        return (self._directory(token) / "upload.bin").read_bytes()

    def write_parsed(self, owner_id: int, token: str, bank_id: str, payload: dict) -> None:
        session = self.get(owner_id, token)
        directory = self._directory(token)
        self._atomic(directory / "parsed.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        updated = ImportSession(
            session.token, session.owner_id, session.filename, session.expires_at,
            session.banks, bank_id,
        )
        self._atomic(directory / "metadata.json", json.dumps(updated.to_dict(), ensure_ascii=False).encode("utf-8"))

    def read_parsed(self, owner_id: int, token: str) -> dict:
        self.get(owner_id, token)
        try:
            return json.loads((self._directory(token) / "parsed.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionNotFoundError("Розібраний банк не знайдено.") from exc

    def delete(self, owner_id: int, token: str) -> None:
        self.get(owner_id, token)
        shutil.rmtree(self._directory(token), ignore_errors=True)
