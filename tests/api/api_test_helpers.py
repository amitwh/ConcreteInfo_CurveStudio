"""Shared helpers for the API tests: CSV datasets, sign-up/upload flows, settings.

Top-level (not in ``conftest``) so a spawned interpreter can import them: pytest puts
every collected conftest's directory on ``sys.path``, and in a full session the
last-collected one sits first — so a bare ``import conftest`` there can resolve another
suite's conftest (``tests/unit/tools`` collects last) and die on the missing name. This
module's name is unique across the whole tree, so it resolves from tests/api wherever
the import runs.
"""

import itertools
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from curvestudio_api.settings import Settings

PASSWORD = "correct horse battery staple"
_counter = itertools.count(1)


def make_settings(directory: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "env": "test",
        "database_url": f"sqlite:///{directory}/api.db",
        "data_dir": directory,
        "job_workers": 1,
        "job_grace_s": 0.5,
        "require_email_verification": False,
    }
    values.update(overrides)
    return Settings(**values)


def new_email() -> str:
    return f"user{next(_counter)}@example.org"


def signup(
    client: TestClient, email: str | None = None, password: str = PASSWORD
) -> dict[str, str]:
    """Register and log in a fresh user; returns the Authorization header."""
    email = email or new_email()
    created = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert created.status_code == 201, created.text
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def linear_csv(n: int = 30, header: str = "x,y", slope: float = 2.0) -> bytes:
    lines = [header]
    for i in range(n):
        x = float(i)
        lines.append(f"{x!r},{slope * x + 1.0 + 0.05 * ((i * 7) % 5 - 2)!r}")
    return ("\n".join(lines) + "\n").encode()


def michaelis_csv(n: int = 25) -> bytes:
    vmax, km = 10.0, 2.5
    lines = ["s,v"]
    for i in range(1, n + 1):
        s = 0.4 * i
        v = vmax * s / (km + s) + 0.02 * ((i * 5) % 7 - 3)
        lines.append(f"{s!r},{v!r}")
    return ("\n".join(lines) + "\n").encode()


def upload(
    client: TestClient,
    headers: dict[str, str],
    content: bytes,
    filename: str = "data.csv",
    **form: str,
) -> Any:
    return client.post(
        "/api/v1/datasets", headers=headers, files={"file": (filename, content)}, data=form
    )
