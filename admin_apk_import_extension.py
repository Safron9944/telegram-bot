from __future__ import annotations

import functools
import inspect

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from apk_importer.archive import ArchiveInspectionError, ArchiveLimits
from apk_importer.crypto import BankDecryptError
from apk_importer.service import ApkImportService, UnsupportedBankError
from apk_importer.sessions import (
    SessionAccessError, SessionExpiredError, SessionNotFoundError,
)
from apk_importer.testms import TestMsParseError


_ORIGINAL_FASTAPI_INIT = FastAPI.__init__
_PATCHED = False
_LIMIT = ArchiveLimits().upload_bytes


def _error(status: int, code: str, message: str):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def _translate(exc: Exception):
    if isinstance(exc, SessionAccessError):
        _error(403, exc.code, str(exc))
    if isinstance(exc, SessionExpiredError):
        _error(410, exc.code, str(exc))
    if isinstance(exc, SessionNotFoundError):
        _error(404, exc.code, str(exc))
    if isinstance(exc, ArchiveInspectionError):
        _error(413 if exc.code == "upload_size_limit" else 400, exc.code, str(exc))
    if isinstance(exc, UnsupportedBankError):
        _error(400, exc.code, str(exc))
    if isinstance(exc, (BankDecryptError, TestMsParseError)):
        _error(422, exc.code, str(exc))
    raise exc


async def _read_bounded(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(min(1024 * 1024, _LIMIT + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > _LIMIT:
            _error(413, "upload_size_limit", "APK перевищує 50 MiB.")


def register_apk_import_routes(
    app: FastAPI,
    get_auth_context,
    *,
    service: ApkImportService | None = None,
) -> None:
    if getattr(app.state, "_apk_import_routes_installed", False):
        return
    app.state.apk_import_service = service or ApkImportService()

    def require_admin(auth):
        if not auth.is_admin:
            _error(403, "forbidden", "Потрібні права адміністратора.")
        return app.state.apk_import_service

    @app.post("/api/admin/apk-import/sessions")
    async def upload_apk(file: UploadFile = File(...), auth=Depends(get_auth_context)):
        service = require_admin(auth)
        try:
            session = service.create_session(auth.user_id, file.filename or "upload.apk", await _read_bounded(file))
            return {
                "token": session.token, "expires_at": session.expires_at,
                "filename": session.filename, "banks": [bank.to_dict() for bank in session.banks],
            }
        except Exception as exc:
            _translate(exc)

    @app.post("/api/admin/apk-import/sessions/{token}/banks/{bank_id}/parse")
    async def parse_bank(token: str, bank_id: str, auth=Depends(get_auth_context)):
        service = require_admin(auth)
        try:
            bank = service.parse_bank(auth.user_id, token, bank_id)
            return {"summary": bank.summary.to_dict(), "sections": [item.to_dict() for item in bank.sections]}
        except Exception as exc:
            _translate(exc)

    @app.get("/api/admin/apk-import/sessions/{token}/preview")
    async def preview(token: str, section: str = "", q: str = "", offset: int = 0, limit: int = 50, auth=Depends(get_auth_context)):
        service = require_admin(auth)
        try:
            return service.preview(auth.user_id, token, section=section, query=q, offset=offset, limit=limit)
        except Exception as exc:
            _translate(exc)

    @app.get("/api/admin/apk-import/sessions/{token}/download")
    async def download(token: str, auth=Depends(get_auth_context)):
        service = require_admin(auth)
        try:
            return Response(
                service.download_json(auth.user_id, token), media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="apk_questions.json"',
                    "Cache-Control": "no-store",
                },
            )
        except Exception as exc:
            _translate(exc)

    @app.delete("/api/admin/apk-import/sessions/{token}", status_code=204)
    async def cancel(token: str, auth=Depends(get_auth_context)):
        service = require_admin(auth)
        try:
            service.delete_session(auth.user_id, token)
            return Response(status_code=204)
        except Exception as exc:
            _translate(exc)

    app.state._apk_import_routes_installed = True


@functools.wraps(_ORIGINAL_FASTAPI_INIT)
def _patched_init(self: FastAPI, *args, **kwargs):
    _ORIGINAL_FASTAPI_INIT(self, *args, **kwargs)
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    globals_ = caller.f_globals if caller else {}
    if globals_.get("__name__") == "app" and callable(globals_.get("get_auth_context")):
        register_apk_import_routes(self, globals_["get_auth_context"])


def install() -> None:
    global _PATCHED
    if not _PATCHED:
        FastAPI.__init__ = _patched_init
        _PATCHED = True


install()
