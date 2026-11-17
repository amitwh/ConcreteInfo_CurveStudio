"""Serves the built web app (``CURVESTUDIO_WEB_DIST``) with a fallback to ``index.html``."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response

from curvestudio_api.errors import ApiError

API_PREFIX = "api"
_IMMUTABLE = "public, max-age=31536000, immutable"
_REVALIDATE = "no-cache"


def mount_spa(app: FastAPI, dist: Path) -> None:
    """Serve static files from ``dist``; any other non-API GET returns ``index.html`` so the
    single-page app can route it. Unknown ``/api`` paths stay JSON 404s."""
    root = dist.resolve()
    index = root / "index.html"
    if not index.is_file():
        raise ValueError(f"CURVESTUDIO_WEB_DIST={dist} has no index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request) -> Response:
        if "\x00" in path:
            raise ApiError(404, "NOT_FOUND", "not found")
        if path == API_PREFIX or path.startswith(f"{API_PREFIX}/"):
            raise ApiError(404, "NOT_FOUND", "no such API endpoint")
        try:
            candidate = (root / path).resolve()
            is_file = path != "" and candidate.is_file() and candidate.is_relative_to(root)
        except OSError, ValueError:  # over-long names and the like: not a file we serve
            is_file = False
        if is_file:
            cache = _IMMUTABLE if "assets" in candidate.relative_to(root).parts else _REVALIDATE
            return FileResponse(candidate, headers={"Cache-Control": cache})
        return FileResponse(index, headers={"Cache-Control": _REVALIDATE})
