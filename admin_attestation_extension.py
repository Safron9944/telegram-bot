"""Add dedicated admin API endpoints for stage-1 attestation navigation.

This module is imported from ``sitecustomize`` before Uvicorn imports
``app``. It patches ``FastAPI.__init__`` once and registers the endpoints on
the application's FastAPI instance without changing the main application
module.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from fastapi import Depends, FastAPI


_ORIGINAL_FASTAPI_INIT = FastAPI.__init__
_PATCHED = False


def _register_routes(app: FastAPI, module_globals: dict[str, Any]) -> None:
    if getattr(app.state, "_admin_attestation_routes_installed", False):
        return

    get_auth_context = module_globals.get("get_auth_context")
    get_runtime = module_globals.get("get_runtime")
    require_http = module_globals.get("require_http")

    if not callable(get_auth_context) or not callable(get_runtime) or not callable(require_http):
        return

    @app.get("/api/admin/attestation-stage-1/sections")
    async def api_admin_attestation_stage_1_sections(
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

        items = runtime.qb.attestation_stage_1_sections()
        return {
            "items": items,
            "total": len(items),
        }

    @app.get("/api/admin/attestation-stage-1/questions")
    async def api_admin_attestation_stage_1_questions(
        section: str,
        offset: int = 0,
        limit: int = 50,
        auth=Depends(get_auth_context),
        runtime=Depends(get_runtime),
    ):
        if not auth.is_admin:
            require_http(403, "forbidden", "Потрібні права адміністратора.")

        section = (section or "").strip()
        if not section:
            require_http(400, "attestation_section_required", "Оберіть розділ атестації.")

        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 100))
        qids = runtime.qb.attestation_stage_1_section_qids(section)
        if not qids:
            require_http(404, "attestation_section_not_found", "Розділ атестації не знайдено.")

        selected = qids[offset : offset + limit]
        items = []
        for qid in selected:
            question = runtime.qb.by_id.get(int(qid))
            if not question:
                continue
            items.append(
                {
                    "id": int(question.id),
                    "qnum": int(question.qnum) if question.qnum is not None else None,
                    "question": question.question or "",
                    "topic": question.topic or section,
                }
            )

        return {
            "section": section,
            "items": items,
            "total": len(qids),
            "offset": offset,
            "limit": limit,
            "has_prev": offset > 0,
            "has_next": offset + limit < len(qids),
        }

    app.state._admin_attestation_routes_installed = True


@functools.wraps(_ORIGINAL_FASTAPI_INIT)
def _fastapi_init_with_admin_attestation(self: FastAPI, *args, **kwargs) -> None:
    _ORIGINAL_FASTAPI_INIT(self, *args, **kwargs)

    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    module_globals = caller.f_globals if caller else {}

    if module_globals.get("__name__") == "app":
        _register_routes(self, module_globals)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    FastAPI.__init__ = _fastapi_init_with_admin_attestation
    _PATCHED = True


install()
