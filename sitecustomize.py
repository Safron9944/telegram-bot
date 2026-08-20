"""Register application extensions before Uvicorn imports the FastAPI app."""

import test_exam_verified_extension  # noqa: F401
import admin_apk_import_extension  # noqa: F401
