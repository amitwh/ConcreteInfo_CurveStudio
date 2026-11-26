"""Shared pytest fixtures.

Installed model plugins are hidden from tests by default. Core tests assert on the exact
contents of the registry after ``_clear_registry()``, so they must not depend on which
catalogue packages (``curvestudio-models``, ``curvestudio-engineering``) happen to be
installed.

A test that needs the real ``curvestudio.models`` entry points carries the marker
``@pytest.mark.real_plugins`` (or ``pytestmark = pytest.mark.real_plugins`` for a module).

The model catalogue's disk cache is pointed at a throwaway directory for the whole
session, so suites neither read nor write the user's real cache (and subprocesses the
tests spawn inherit the redirect).

Spawned worker processes (``multiprocessing`` with ``spawn``, as Auto-fit uses) are fresh
interpreters: they always see the real entry points, whatever this fixture did in the parent.
A test whose workers must find the same models as the parent therefore also needs the marker.
"""

import importlib.metadata
import os
from collections.abc import Iterator
from typing import Any

import pytest

from curvestudio.models.registry import PLUGIN_GROUP


@pytest.fixture(scope="session", autouse=True)
def _catalogue_cache_isolated(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the model catalogue's disk cache out of the user's real cache directory."""
    key = "CURVESTUDIO_CATALOGUE_CACHE_DIR"
    os.environ[key] = str(tmp_path_factory.mktemp("catalogue-cache"))
    yield
    os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def _hide_installed_model_plugins(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Hide the installed ``curvestudio.models`` entry points unless ``real_plugins`` is set."""
    if request.node.get_closest_marker("real_plugins") is not None:
        yield
        return
    real = importlib.metadata.entry_points

    def entry_points(**selection: Any) -> Any:
        if selection.get("group") == PLUGIN_GROUP:
            return []
        return real(**selection)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    yield
