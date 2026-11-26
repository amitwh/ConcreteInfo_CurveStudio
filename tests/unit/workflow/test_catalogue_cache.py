"""The model catalogue's disk cache (D1): hits skip registration, misses rebuild.

The built catalogue is decided entirely by the contributing packages and their
``curvestudio.models`` entry points, so it is cached per user under a fingerprint of
exactly those. These tests hold the contract from both sides: a valid cache serves the
identical model list without registering anything, and any mismatch, corruption or
failure rebuilds the ordinary way and repairs the cache. Installed plugins are hidden
by the root conftest; every plugin here is synthetic.
"""

import json
from collections.abc import Callable, Iterator, Sequence

import pytest

from curvestudio.models import ModelSpec, Parameter
from curvestudio.models import registry as registry_module
from curvestudio.models.registry import _clear_registry, register
from curvestudio.models.starter import STARTER_MODELS, register_starter_models
from curvestudio.workflow import catalogue_cache, list_models


def _guess(x: object, y: object) -> dict[str, float]:
    return {"a": 1.0}


def _registering(model_id: str) -> Callable[[], None]:
    """A plugin body that registers one simple model when called."""

    def register_one() -> None:
        register(
            ModelSpec(
                id=model_id,
                name=model_id,
                category="synthetic",
                expression="a*x",
                parameters=(Parameter("a"),),
                initial_guess=_guess,
            )
        )

    return register_one


class _FakeDist:
    """The one thing the fingerprint reads off a distribution: its name and version."""

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version


class _FakeEntryPoint:
    """A stand-in for ``importlib.metadata.EntryPoint``: name, group, value, dist, load."""

    def __init__(
        self,
        name: str,
        function: Callable[[], None],
        dist: _FakeDist | None = None,
        value: str = "synthetic_plugin:register",
    ) -> None:
        self.name = name
        self.group = registry_module.PLUGIN_GROUP
        self.value = value
        self.dist = dist
        self._function = function

    def load(self) -> Callable[[], None]:
        return self._function


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, points: Sequence[_FakeEntryPoint]) -> None:
    import importlib.metadata

    def fake(**selection: object) -> list[_FakeEntryPoint]:
        if selection.get("group") == registry_module.PLUGIN_GROUP:
            return list(points)
        return list(importlib.metadata.entry_points(**selection))  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)


@pytest.fixture(autouse=True)
def starter_registry_and_fresh_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> Iterator[None]:
    """The starter-only registry, a private cache directory, no in-memory memo."""
    monkeypatch.setenv(catalogue_cache.ENV_CACHE_DIR, str(tmp_path))
    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()
    yield
    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()


def _fresh_start() -> None:
    """The registry state a new process starts in: the starter set, nothing loaded."""
    catalogue_cache._reset_for_tests()
    _clear_registry()
    register_starter_models()


def _ids(models: object) -> list[str]:
    return [m.id for m in models]  # type: ignore[attr-defined]


# -- a valid cache serves the catalogue --------------------------------------------------


def test_a_valid_cache_serves_the_same_list_without_registering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    built = list_models()
    assert "synthetic_a" in _ids(built)  # the ordinary build registered the plugin
    assert registry_module.plugins_loaded()
    assert catalogue_cache.record_path().exists()  # a clean scan leaves a cache behind

    _fresh_start()
    assert not registry_module.plugins_loaded()
    assert "synthetic_a" not in registry_module.peek_registered_ids()

    served = list_models()
    assert _ids(served) == _ids(built)  # identical model list, straight from the cache
    assert not registry_module.plugins_loaded()  # registration was skipped
    assert "synthetic_a" not in registry_module.peek_registered_ids()
    # and the skipped registration still happens when a fitted model is asked for
    from curvestudio.workflow import get_model

    assert get_model("synthetic_a").id == "synthetic_a"
    assert registry_module.plugins_loaded()


def test_the_cache_keeps_serving_after_registration_happened_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit registers the plugins mid-session; a later listing is still served, not
    rebuilt (the state a cache may be served under covers both sides of that)."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    built = list_models()
    from curvestudio.models import get_model

    get_model("synthetic_a")  # the lazy registration a fit would trigger
    assert registry_module.plugins_loaded()
    again = list_models()
    assert _ids(again) == _ids(built)


def test_a_changed_entry_point_set_is_not_served_from_the_previous_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same process and registry state under a different entry-point set (tests
    switch it) must rebuild, never serve the previous environment's list. The registry
    itself loads plugins once per process, so the rebuild is observed, not its result."""
    _patch_entry_points(monkeypatch, [])  # an environment with no model plugins at all
    starter_only = list_models()
    assert "synthetic_a" not in _ids(starter_only)
    assert sorted(registry_module.peek_registered_ids()) == sorted(  # the registry did not move
        spec.id for spec in STARTER_MODELS
    )

    from curvestudio.workflow import catalog

    builds: list[int] = []
    real_build = catalog._build_catalogue

    def counting_build() -> list[object]:
        builds.append(1)
        return real_build()

    monkeypatch.setattr(catalog, "_build_catalogue", counting_build)
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    list_models()  # same state at consult, different fingerprint: rebuild, not serve
    assert builds == [1]


def test_a_registry_mutated_beyond_registration_is_never_served_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-process registration (tests, future callers) must reach the listing, so a
    cache built without it is not served: the state check catches it."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    list_models()
    _fresh_start()
    register(
        ModelSpec(
            id="in_process",
            name="in process",
            category="synthetic",
            expression="a*x",
            parameters=(Parameter("a"),),
            initial_guess=_guess,
        )
    )
    served = list_models()
    assert "in_process" in _ids(served)
    assert "synthetic_a" in _ids(served)  # the full scan ran: registration happened
    assert registry_module.plugins_loaded()


# -- damage falls back and repairs -------------------------------------------------------


def _write_record(path: object, **overrides: object) -> None:
    record: dict[str, object] = {
        "format": catalogue_cache.CACHE_FORMAT_VERSION,
        "fingerprint": catalogue_cache.fingerprint(),
        "baseline_ids": sorted(spec.id for spec in STARTER_MODELS),
        "registered_ids": sorted({*[spec.id for spec in STARTER_MODELS], "synthetic_a"}),
        "models": [{"id": "nonsense", "name": "nonsense"}],
    }
    record.update(overrides)
    path.write_text(json.dumps(record), encoding="utf-8")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "damage",
    ["garbage", "empty", "wrong_format", "wrong_fingerprint", "wrong_state", "broken_models"],
)
def test_a_damaged_cache_rebuilds_and_rewrites(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    built = list_models()
    path = catalogue_cache.record_path()
    assert path.exists()
    if damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    elif damage == "empty":
        path.write_text("", encoding="utf-8")
    elif damage == "wrong_format":
        _write_record(path, format=catalogue_cache.CACHE_FORMAT_VERSION + 1)
    elif damage == "wrong_fingerprint":
        _write_record(path, fingerprint="0" * 64)
    elif damage == "wrong_state":
        _write_record(path, baseline_ids=["someone_else"], registered_ids=["someone_else"])
    elif damage == "broken_models":
        _write_record(path, models=[{"nonsense": True}])

    _fresh_start()
    served = list_models()
    assert _ids(served) == _ids(built)  # the ordinary rebuild, not the damaged content
    assert registry_module.plugins_loaded()

    _fresh_start()
    repaired = list_models()
    assert _ids(repaired) == _ids(built)
    assert not registry_module.plugins_loaded()  # the rewritten cache serves again


# -- the fingerprint ---------------------------------------------------------------------


def test_the_fingerprint_tracks_the_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    one = [_FakeEntryPoint("p1", _registering("p1_model"))]
    two = [*one, _FakeEntryPoint("p2", _registering("p2_model"))]
    _patch_entry_points(monkeypatch, one)
    with_one = catalogue_cache.fingerprint()
    _patch_entry_points(monkeypatch, two)
    with_two = catalogue_cache.fingerprint()
    assert with_one != with_two  # a new entry point invalidates the cache
    _patch_entry_points(monkeypatch, one)
    assert catalogue_cache.fingerprint() == with_one  # and is deterministic


def test_the_fingerprint_tracks_a_contributing_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = [_FakeEntryPoint("p1", _registering("p1_model"), dist=_FakeDist("p1-dist", "1.0"))]
    upgraded = [_FakeEntryPoint("p1", _registering("p1_model"), dist=_FakeDist("p1-dist", "1.1"))]
    _patch_entry_points(monkeypatch, plugin)
    before = catalogue_cache.fingerprint()
    _patch_entry_points(monkeypatch, upgraded)
    assert catalogue_cache.fingerprint() != before  # a plugin upgrade invalidates the cache


def test_the_fingerprint_covers_the_contributing_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """An editable install keeps its version while its content changes, so the sources
    are hashed too; without that a dev edit would serve a stale catalogue."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("p1", _registering("p1_model"))])
    first = catalogue_cache.fingerprint()
    family = catalogue_cache._package_roots()[0] / "_cache_probe.py"
    try:
        family.write_text("x = 1\n", encoding="utf-8")
        catalogue_cache._reset_for_tests()
        second = catalogue_cache.fingerprint()
        assert second != first
        family.write_text("x = 2\n", encoding="utf-8")
        catalogue_cache._reset_for_tests()
        assert catalogue_cache.fingerprint() != second
    finally:
        family.unlink(missing_ok=True)
    catalogue_cache._reset_for_tests()
    assert catalogue_cache.fingerprint() == first  # and it is deterministic


# -- what is written ---------------------------------------------------------------------


def test_a_failing_plugin_never_reaches_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalogue built while a plugin failed is not cached: the failure is retried on
    every start-up instead of being frozen into the cache."""

    def boom() -> None:
        raise RuntimeError("plugin exploded")

    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("synthetic", _registering("synthetic_a")),
            _FakeEntryPoint("broken", boom),
        ],
    )
    served = list_models()
    assert "synthetic_a" in _ids(served)
    assert not catalogue_cache.record_path().exists()


def test_a_scan_over_a_mutated_registry_does_not_rewrite_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a clean start-up scan (starter baseline, registration ahead of it) may write
    the cache, so a test's or a caller's in-process state is never persisted."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("synthetic", _registering("synthetic_a"))])
    list_models()
    content = catalogue_cache.record_path().read_text(encoding="utf-8")
    _fresh_start()
    register(
        ModelSpec(
            id="in_process",
            name="in process",
            category="synthetic",
            expression="a*x",
            parameters=(Parameter("a"),),
            initial_guess=_guess,
        )
    )
    assert "in_process" in _ids(list_models())
    assert catalogue_cache.record_path().read_text(encoding="utf-8") == content


# -- where it lives ----------------------------------------------------------------------


def test_the_cache_directory_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(catalogue_cache.ENV_CACHE_DIR, "/tmp/wherever")
    assert catalogue_cache.cache_directory().name == "wherever"


def test_the_default_cache_directory_is_outside_the_repo_and_site_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(catalogue_cache.ENV_CACHE_DIR, raising=False)
    directory = catalogue_cache.cache_directory()
    from pathlib import Path

    assert Path.cwd() not in directory.parents  # never inside the repository
    assert "site-packages" not in str(directory)  # never inside an installation
