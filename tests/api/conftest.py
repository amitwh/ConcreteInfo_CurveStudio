"""Fixtures for the API tests: settings on a temporary SQLite database, apps and clients.

The plain helpers (CSV datasets, sign-up/upload flows, settings) live in
``api_test_helpers``: spawned interpreters import test modules by bare name, and a bare
``import conftest`` there can resolve another suite's conftest, while a uniquely named
top-level module always resolves from tests/api.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from api_test_helpers import make_settings
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio_api.app import create_app
from curvestudio_api.routers.system import _catalogue
from curvestudio_api.settings import Settings


@pytest.fixture(autouse=True)
def starter_models() -> Iterator[None]:
    """Other tests clear the model registry; make sure the starter set is there.

    The /models route serves the catalogue from a process cache, so every registry
    rebuild here drops that cache too: a test never sees a catalogue enumerated from
    the previous test's registry.
    """
    _clear_registry()
    register_starter_models()
    _catalogue.cache_clear()
    yield
    _clear_registry()
    register_starter_models()
    _catalogue.cache_clear()


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
    app.state.ctx.install_limiter.limit = 100_000
    app.state.ctx.error_limiter.limit = 100_000
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
        app.state.ctx.install_limiter.limit = 100_000
        app.state.ctx.error_limiter.limit = 100_000
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        return client

    yield build
    for client in reversed(opened):
        client.__exit__(None, None, None)
