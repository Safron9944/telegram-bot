from __future__ import annotations

import functools
import inspect
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from attestation_publishing import AttestationPublishError, AttestationPublishingService
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


class PublishRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class VisibilityRequest(BaseModel):
    visible: bool


class MoveRequest(BaseModel):
    direction: Literal["up", "down"]


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
    if isinstance(exc, AttestationPublishError):
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
    publisher=None,
) -> None:
    if getattr(app.state, "_apk_import_routes_installed", False):
        return
    app.state.apk_import_service = service or ApkImportService()

    def require_admin(auth):
        if not auth.is_admin:
            _error(403, "forbidden", "Потрібні права адміністратора.")
        return app.state.apk_import_service

    def runtime_for(request: Request):
        return request.app.state.runtime

    async def reload_catalog(runtime) -> None:
        await runtime.qb.load_published_attestation_banks(runtime.store)

    @app.get("/api/admin/attestation-banks")
    async def list_attestation_banks(request: Request, auth=Depends(get_auth_context)):
        require_admin(auth)
        runtime = runtime_for(request)
        dynamic = await runtime.store.list_attestation_banks_for_admin()
        stage_1_count = len(getattr(runtime.qb, "attestation_stage_1", []) or [])
        items = [{
            "id": None,
            "slug": "stage-1",
            "title": "Атестація посадових осіб — 1 етап",
            "status": "published",
            "visible": True,
            "display_order": -1,
            "questions_count": stage_1_count,
            "system": True,
        }]
        items.extend({**row, "visible": row.get("status") == "published", "system": False} for row in dynamic)
        return {"items": items}

    @app.patch("/api/admin/attestation-banks/{bank_id}/visibility")
    async def set_attestation_bank_visibility(
        bank_id: int,
        payload: VisibilityRequest,
        request: Request,
        auth=Depends(get_auth_context),
    ):
        require_admin(auth)
        runtime = runtime_for(request)
        row = await runtime.store.set_attestation_bank_visibility(bank_id, visible=payload.visible)
        if not row:
            _error(404, "attestation_bank_not_found", "Розділ атестації не знайдено.")
        await reload_catalog(runtime)
        return {**row, "visible": payload.visible, "system": False}

    @app.post("/api/admin/attestation-banks/{bank_id}/move")
    async def move_attestation_bank(
        bank_id: int,
        payload: MoveRequest,
        request: Request,
        auth=Depends(get_auth_context),
    ):
        require_admin(auth)
        runtime = runtime_for(request)
        if not await runtime.store.move_attestation_bank(bank_id, direction=payload.direction):
            _error(404, "attestation_bank_not_found", "Розділ атестації не знайдено.")
        await reload_catalog(runtime)
        return {"ok": True}

    @app.delete("/api/admin/attestation-banks/{bank_id}", status_code=204)
    async def delete_attestation_bank(bank_id: int, request: Request, auth=Depends(get_auth_context)):
        require_admin(auth)
        runtime = runtime_for(request)
        if not await runtime.store.delete_attestation_bank(bank_id):
            _error(404, "attestation_bank_not_found", "Розділ атестації не знайдено.")
        await reload_catalog(runtime)
        return Response(status_code=204)

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
            return {
                "summary": bank.summary.to_dict(),
                "sections": [item.to_dict() for item in bank.sections],
                "suggested_title": service.suggested_title(auth.user_id, token),
            }
        except Exception as exc:
            _translate(exc)

    @app.post("/api/admin/apk-import/sessions/{token}/publish")
    async def publish_bank(
        token: str,
        payload: PublishRequest,
        request: Request,
        auth=Depends(get_auth_context),
    ):
        service = require_admin(auth)
        try:
            active_publisher = publisher
            if active_publisher is None:
                runtime = request.app.state.runtime

                async def reload_catalog():
                    await runtime.qb.load_published_attestation_banks(runtime.store)

                active_publisher = AttestationPublishingService(runtime.store, reload_catalog)
            bank = service.get_parsed_bank(auth.user_id, token)
            return await active_publisher.publish(bank, payload.title, changed_by=str(auth.user_id))
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
