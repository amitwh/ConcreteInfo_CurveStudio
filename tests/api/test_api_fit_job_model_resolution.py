"""The fit-job worker's model resolution against a seeded catalogue cache (regression).

The API's fit jobs resolve their model inside ``run_fit_job`` — the function a warm
worker's forked child runs per job. Under a seeded catalogue cache the registration is
deferred to exactly that resolution, so the worker must be able to load the model
plugins itself even in a process whose once-per-process plugin load was already spent on
an empty scan — the state the full suite leaves behind, where any test's registry call
runs while the root conftest hides the installed entry points. A load latched on that
empty scan made the worker answer ``UNKNOWN_MODEL`` for every plugin-only model.
"""

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.models import ModelSpec, Parameter
from curvestudio.models import registry as registry_module
from curvestudio.models.registry import _clear_registry, register
from curvestudio.models.starter import register_starter_models
from curvestudio.workflow import catalogue_cache

worker = pytest.importorskip("curvestudio_api.worker", reason="the API package is not installed")
storage = pytest.importorskip("curvestudio_api.storage", reason="the API package is not installed")


def _guess(x: object, y: object) -> dict[str, float]:
    return {"a": 1.0}


def _registering() -> Callable[[], None]:
    """A plugin body that registers one simple model when called."""

    def register_one() -> None:
        register(
            ModelSpec(
                id="plugin_linear",
                name="plugin linear",
                category="synthetic",
                expression="a*x",
                parameters=(Parameter("a"),),
                initial_guess=_guess,
            )
        )

    return register_one


class _FakeEntryPoint:
    """The stand-in for ``importlib.metadata.EntryPoint`` the cache tests use too."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.group = registry_module.PLUGIN_GROUP
        self.value = "synthetic_plugin:register"
        self.dist = None

    def load(self) -> Callable[[], None]:
        return _registering()


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, points: Sequence[object]) -> None:
    import importlib.metadata

    real = importlib.metadata.entry_points

    def fake(**selection: object) -> object:
        if selection.get("group") == registry_module.PLUGIN_GROUP:
            return list(points)
        return real(**selection)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)


@pytest.fixture(autouse=True)
def starter_registry_and_fresh_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> Iterator[None]:
    """The starter-only registry and a private cache directory, like a new process."""
    monkeypatch.setenv(catalogue_cache.ENV_CACHE_DIR, str(tmp_path))
    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()
    yield
    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()


def _linear_xy() -> XY:
    x = np.linspace(1.0, 30.0, 30)
    return XY(
        x=x,
        y=2.0 * x,
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="worker-resolution-probe",
    )


def test_the_fit_job_worker_resolves_a_plugin_model_under_a_seeded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fit job, cache pre-seeded: the worker's model resolution loads the plugins.

    The seeding scan is a clean one (starter baseline, registration ahead of it), so the
    record is written and every later listing is served from it. The registry is then
    reset the way the suite leaves it — starter baseline, the once-per-process load spent
    on an empty scan — and the job worker must still resolve the model.
    """
    entry_point = _FakeEntryPoint("synthetic")
    _patch_entry_points(monkeypatch, [entry_point])
    from curvestudio.workflow import list_models as catalogue_list

    catalogue = catalogue_list()
    assert "plugin_linear" in [m.id for m in catalogue]
    assert catalogue_cache.record_path().exists()  # the clean scan wrote the cache

    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()
    _patch_entry_points(monkeypatch, [])  # the entry points disappear (as the conftest does)
    registry_module.load_plugins()  # ...and a registry call in that context scans nothing
    _patch_entry_points(monkeypatch, [entry_point])  # ...and they are back

    result = worker.run_fit_job(
        payload=storage.encode_payload(_linear_xy()),
        x_name="x",
        y_name="y",
        fingerprint="worker-resolution-probe",
        spec={"model": {"kind": "registered", "id": "plugin_linear"}},
    )
    assert result["ok"] is True, result
    assert registry_module.plugins_loaded()  # the resolution loaded the plugins
