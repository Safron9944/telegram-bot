"""ASGI entry point that installs startup extensions before importing the app."""

# Railway starts Uvicorn with ``python -m uvicorn``. In that mode a project
# level sitecustomize.py is not guaranteed to be imported during interpreter
# startup, so import it explicitly before app.py creates the FastAPI instance.
import sitecustomize as _startup_extensions  # noqa: F401

from app import app

__all__ = ["app"]
