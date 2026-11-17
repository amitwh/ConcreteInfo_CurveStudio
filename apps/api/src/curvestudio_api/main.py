"""Process entry point: ``curvestudio-api`` (a single uvicorn process; see the deployment notes)."""

import os

import uvicorn


def run() -> None:
    """Serve the API. Host and port come from ``CURVESTUDIO_HOST`` / ``CURVESTUDIO_PORT``."""
    uvicorn.run(
        "curvestudio_api.app:create_app",
        factory=True,
        host=os.environ.get("CURVESTUDIO_HOST", "127.0.0.1"),
        port=int(os.environ.get("CURVESTUDIO_PORT", "8000")),
        log_config=None,
        proxy_headers=False,  # parsed by the app: CURVESTUDIO_TRUSTED_PROXY_HOPS
    )
