from __future__ import annotations


HTML_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
VERSIONED_STATIC_CACHE_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
}
UNVERSIONED_STATIC_CACHE_HEADERS = {
    "Cache-Control": "public, max-age=0, must-revalidate",
}
API_NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store",
}


def cache_headers(path: str, *, versioned: bool = False) -> dict[str, str]:
    if path == "/":
        return HTML_NO_CACHE_HEADERS
    if path.startswith("/api/"):
        return API_NO_STORE_HEADERS
    if path.startswith("/static/"):
        return VERSIONED_STATIC_CACHE_HEADERS if versioned else UNVERSIONED_STATIC_CACHE_HEADERS
    return {}
