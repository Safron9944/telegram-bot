"""ASGI entry point that installs startup extensions before importing the app."""

# Railway starts Uvicorn with ``python -m uvicorn``. In that mode a project
# level sitecustomize.py is not guaranteed to be imported during interpreter
# startup, so import it explicitly before app.py creates the FastAPI instance.
import sitecustomize as _startup_extensions  # noqa: F401
import app as _app_module

from admin_test_exam_import import register_routes as _register_test_exam_import_routes
from admin_test_exam_import_editor import register_routes as _register_test_exam_editor_routes
from admin_test_exam_crud import register_routes as _register_test_exam_crud_routes

_register_test_exam_import_routes(
    _app_module.app,
    get_auth_context=_app_module.get_auth_context,
    get_runtime=_app_module.get_runtime,
    require_http=_app_module.require_http,
)
_register_test_exam_editor_routes(
    _app_module.app,
    get_auth_context=_app_module.get_auth_context,
    get_runtime=_app_module.get_runtime,
    require_http=_app_module.require_http,
)
_register_test_exam_crud_routes(
    _app_module.app,
    get_auth_context=_app_module.get_auth_context,
    get_runtime=_app_module.get_runtime,
    require_http=_app_module.require_http,
)

app = _app_module.app

__all__ = ["app"]
