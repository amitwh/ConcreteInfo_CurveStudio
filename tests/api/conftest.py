"""Fixtures for the API tests: settings on a temporary SQLite database, apps and clients."""

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio_api.app import create_app
from curvestudio_api.settings import Settings

PASSWORD = "correct horse battery staple"
_counter = itertools.count(1)


@pytest.fixture(autouse=True)
def starter_models() -> Iterator[None]:
    """Other tests clear the model registry; make sure the starter set is there."""
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


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


@pytest.fixture
def settings_factory(tmp_path: Path) -> Callable[..., Settings]:
    def build(**overrides: Any) -> Settings:
        return make_settings(tmp_path, **overrides)

    return build


@pytest.fixture(scope="module")
def shared_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """One app (and one worker pool) per test module: spawning workers is slow."""
    directory = tmp_path_factory.mktemp("api")
    app = create_app(make_settings(directory))
    app.state.ctx.ip_limiter.limit = 100_000  # tests share one client address
    app.state.ctx.auth_limiter.limit = 100_000
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_factory(
    settings_factory: Callable[..., Settings],
) -> Iterator[Callable[..., TestClient]]:
    """Build apps with custom settings or a custom job worker; closed after the test."""
    opened: list[TestClient] = []

    def build(app_settings: Settings | None = None, **kwargs: Any) -> TestClient:
        app: FastAPI = create_app(app_settings or settings_factory(), **kwargs)
        app.state.ctx.ip_limiter.limit = 100_000
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        return client

    yield build
    for client in reversed(opened):
        client.__exit__(None, None, None)


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
