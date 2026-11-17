"""Lazy compilation of ModelSpec and the plugin / family registry."""

import importlib.metadata
import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest
import sympy

from curvestudio.models import (
    ModelSpec,
    Parameter,
    PluginError,
    family_errors,
    get_model,
    list_model_ids,
    list_models,
    load_plugins,
    plugin_failures,
    register,
    register_family,
    registry,
)
from curvestudio.models.registry import _clear_registry


def _guess(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"a": 1.0, "b": 1.0}


def _spec(model_id: str, expression: str = "a*exp(b*x)") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=model_id,
        category="exponential",
        expression=expression,
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=_guess,
    )


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture
def lambdify_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = sympy.lambdify

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sympy, "lambdify", counting)
    return calls


# -- lazy compile -----------------------------------------------------------------------


def test_spec_import_does_not_compile(lambdify_calls: list[int]) -> None:
    specs = [_spec(f"m{i}") for i in range(50)]
    assert len(specs) == 50
    assert lambdify_calls == []


def test_first_evaluate_compiles_once(lambdify_calls: list[int]) -> None:
    spec = _spec("m")
    x = np.array([0.0, 1.0, 2.0])
    first = spec.evaluate(x, a=2.0, b=0.5)
    n_after_first = len(lambdify_calls)
    assert n_after_first == 1 + len(spec.parameters)  # value + one derivative per parameter
    again = spec.evaluate(x, a=2.0, b=0.5)
    spec.jacobian(x, a=2.0, b=0.5)
    spec.latex()
    spec.compile()
    assert len(lambdify_calls) == n_after_first
    np.testing.assert_array_equal(first, again)
    np.testing.assert_allclose(first, 2.0 * np.exp(0.5 * x))


def test_compile_is_idempotent_and_compile_alone_compiles(lambdify_calls: list[int]) -> None:
    spec = _spec("m")
    spec.compile()
    n = len(lambdify_calls)
    assert n > 0
    spec.compile()
    assert len(lambdify_calls) == n


@pytest.mark.parametrize(
    "expression",
    ["a*exp(b*x)+", "a*x**b**", "__import__('os')", "a*exp(b*x", "a*foo(x)"],
)
def test_invalid_expression_still_raises_at_construction(expression: str) -> None:
    with pytest.raises(ValueError):
        _spec("bad", expression)


def test_compile_probe_raises_for_unusable_expression() -> None:
    # Passes the grammar (every name and call is whitelisted) but is undefined whatever
    # the data are (x-x is 0 only after SymPy simplifies it), which only the SymPy build sees.
    spec = _spec("probe", "a*b*(1/(x-x))+x")
    with pytest.raises(ValueError, match="not defined for real numbers"):
        spec.compile()
    with pytest.raises(ValueError):
        spec.evaluate(np.array([1.0]), a=1.0, b=1.0)


def test_compile_failure_is_not_cached_as_success() -> None:
    spec = _spec("probe", "a*b*(1/(x-x))+x")
    for _ in range(2):
        with pytest.raises(ValueError):
            spec.compile()


# -- plugins ----------------------------------------------------------------------------


class _EntryPoint:
    def __init__(self, name: str, func: Callable[[], None]) -> None:
        self.name = name
        self._func = func

    def load(self) -> Callable[[], None]:
        return self._func


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, points: Sequence[_EntryPoint], seen: list[str] | None = None
) -> None:
    def fake(*, group: str) -> list[_EntryPoint]:
        if seen is not None:
            seen.append(group)
        return list(points)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)


def test_failing_plugin_never_breaks_lookups_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    def good() -> None:
        calls.append("good")
        register(_spec("from_good"))

    def good_two() -> None:
        calls.append("good_two")
        register(_spec("from_good_two"))

    def boom() -> None:
        calls.append("boom")
        register(_spec("half_registered"))  # registered, then the plugin dies
        raise RuntimeError("plugin exploded")

    groups: list[str] = []
    _patch_entry_points(
        monkeypatch,
        [_EntryPoint("good", good), _EntryPoint("boom", boom), _EntryPoint("good_two", good_two)],
        groups,
    )
    with caplog.at_level(logging.WARNING, logger="curvestudio.models.registry"):
        # The first lookup does not raise: lookups behave the same on every call.
        assert get_model("from_good").id == "from_good"
        assert get_model("from_good_two").id == "from_good_two"
        assert load_plugins() == ["good", "good_two"]
        assert load_plugins() == ["good", "good_two"]
        ids = list_model_ids()
    assert groups == ["curvestudio.models"]
    assert calls == ["good", "boom", "good_two"]  # all tried exactly once
    # The failed plugin is fully absent, not half loaded.
    assert "half_registered" not in ids
    failures = plugin_failures()
    assert [f.name for f in failures] == ["boom"]
    assert isinstance(failures[0].cause, RuntimeError)
    warnings = [r for r in caplog.records if "boom" in r.getMessage()]
    assert len(warnings) == 1  # logged once, not on every lookup
    assert "plugin exploded" in warnings[0].getMessage()


def test_unknown_id_mentions_failed_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("x")

    _patch_entry_points(monkeypatch, [_EntryPoint("boom", boom)])
    with pytest.raises(KeyError, match="plugins that failed to load: 'boom'"):
        get_model("anything")


def test_strict_load_raises_after_loading_every_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    def good() -> None:
        register(_spec("from_good"))

    def boom() -> None:
        raise RuntimeError("plugin exploded")

    def boom_two() -> None:
        raise ValueError("second")

    _patch_entry_points(
        monkeypatch,
        [_EntryPoint("boom", boom), _EntryPoint("good", good), _EntryPoint("boom_two", boom_two)],
    )
    with pytest.raises(PluginError) as info:
        load_plugins(strict=True)
    assert info.value.name == "boom"
    assert isinstance(info.value.cause, RuntimeError)
    assert [f.name for f in info.value.failures] == ["boom", "boom_two"]
    assert "boom_two" in str(info.value)
    assert get_model("from_good").id == "from_good"  # the good plugin loaded regardless
    with pytest.raises(PluginError):  # strict is deterministic on later calls too
        load_plugins(strict=True)
    assert load_plugins() == ["good"]


def test_strict_load_without_failures_returns_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_EntryPoint("ok", lambda: register(_spec("ok_model")))])
    assert load_plugins(strict=True) == ["ok"]
    assert plugin_failures() == ()


@pytest.mark.parametrize("exc", [SystemExit(3), GeneratorExit("g"), MemoryError("m")])
def test_base_exception_plugin_is_a_recorded_failure_not_half_loaded(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    def dies() -> None:
        register(_spec("partial_a"))
        register_family("part_", lambda _e: ["part_1"], lambda i: _spec(i))
        raise exc

    _patch_entry_points(
        monkeypatch,
        [_EntryPoint("dies", dies), _EntryPoint("ok", lambda: register(_spec("ok_model")))],
    )
    assert load_plugins() == ["ok"]
    assert [f.name for f in plugin_failures()] == ["dies"]
    assert plugin_failures()[0].cause is exc
    assert list_model_ids() == ["ok_model"]
    with pytest.raises(KeyError):
        get_model("part_1")
    # The prefix is free again, so the failure left no family behind.
    register_family("part_", lambda _e: [], lambda i: _spec(i))


def test_keyboard_interrupt_aborts_loading_and_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def first() -> None:
        register(_spec("first_model"))

    def interrupted() -> None:
        raise KeyboardInterrupt

    _patch_entry_points(
        monkeypatch, [_EntryPoint("first", first), _EntryPoint("stop", interrupted)]
    )
    with pytest.raises(KeyboardInterrupt):
        load_plugins()
    assert registry._REGISTRY == {}  # nothing registered by the aborted load
    assert plugin_failures() == ()
    # A later call starts over cleanly (the interrupt is not remembered as "done").
    _patch_entry_points(monkeypatch, [_EntryPoint("first", first)])
    assert load_plugins() == ["first"]
    assert get_model("first_model").id == "first_model"


def test_concurrent_first_lookups_both_wait_for_a_slow_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_plugin() -> None:
        time.sleep(0.3)
        register(_spec("slow"))

    _patch_entry_points(monkeypatch, [_EntryPoint("slow_plugin", slow_plugin)])
    barrier = threading.Barrier(2)
    found: list[str] = []
    errors: list[BaseException] = []

    def lookup() -> None:
        try:
            barrier.wait()
            found.append(get_model("slow").id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=lookup) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert found == ["slow", "slow"]
    assert load_plugins() == ["slow_plugin"]


def test_plugin_that_looks_a_model_up_works_and_loaded_flag_is_set_after_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def plugin() -> None:
        register(_spec("first"))
        assert get_model("first").id == "first"  # re-entrant lookup inside the plugin
        seen.append(list_model_ids())
        register(_spec("second"))

    _patch_entry_points(monkeypatch, [_EntryPoint("p", plugin)])
    assert load_plugins() == ["p"]
    assert seen == [["first"]]
    assert list_model_ids() == ["first", "second"]


def test_first_lookup_loads_plugins_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def plugin() -> None:
        calls.append("p")
        register(_spec("plugin_model"))

    _patch_entry_points(monkeypatch, [_EntryPoint("p", plugin)])
    assert [m.id for m in list_models()] == ["plugin_model"]
    assert get_model("plugin_model").id == "plugin_model"
    assert "plugin_model" in list_model_ids()
    assert calls == ["p"]


def test_clear_registry_resets_the_loaded_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def plugin() -> None:
        calls.append("p")
        register(_spec("plugin_model"))

    _patch_entry_points(monkeypatch, [_EntryPoint("p", plugin)])
    assert load_plugins() == ["p"]
    _clear_registry()
    assert load_plugins() == ["p"]
    assert calls == ["p", "p"]
    assert get_model("plugin_model").id == "plugin_model"


# -- families ---------------------------------------------------------------------------


def _fake_family() -> list[str]:
    built: list[str] = []

    def ids(exhaustive: bool) -> list[str]:
        return [f"fake_{i}" for i in range(10 if exhaustive else 3)]

    def build(model_id: str) -> ModelSpec:
        built.append(model_id)
        if not model_id.removeprefix("fake_").isdigit():
            raise KeyError(f"unknown model id {model_id!r}")
        return _spec(model_id)

    register_family("fake_", ids, build)
    return built


def test_families_resolve_ids_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [])
    built = _fake_family()
    assert built == []  # nothing is built until asked for
    spec = get_model("fake_3")
    assert spec.id == "fake_3"
    assert built == ["fake_3"]
    assert get_model("fake_3") is spec  # cached
    assert built == ["fake_3"]
    assert len(list_model_ids(exhaustive=True)) > len(list_model_ids(exhaustive=False))
    assert list_model_ids() == ["fake_0", "fake_1", "fake_2"]
    with pytest.raises(KeyError):
        get_model("fake_x")


def test_registered_ids_come_first_in_listings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [])
    register(_spec("alpha"))
    _fake_family()
    assert list_model_ids()[0] == "alpha"


def test_family_prefix_and_registered_ids_may_not_shadow_each_other() -> None:
    register(_spec("fake_1"))
    with pytest.raises(ValueError, match="collides with registered model ids"):
        _fake_family()
    _clear_registry()
    _fake_family()
    with pytest.raises(ValueError, match="shadow a family member"):
        register(_spec("fake_9"))
    register(_spec("fake"))  # not starting with the prefix "fake_": fine


def test_family_member_that_cannot_be_built_does_not_break_category_listing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_entry_points(monkeypatch, [])

    def build(model_id: str) -> ModelSpec:
        if model_id == "fam_bad":
            raise RuntimeError("cannot build this one")
        return _spec(model_id)

    register_family("fam_", lambda _e: ["fam_a", "fam_bad", "fam_c"], build)
    with caplog.at_level(logging.WARNING, logger="curvestudio.models.registry"):
        assert list_model_ids(category="exponential") == ["fam_a", "fam_c"]
        assert list_model_ids(category="exponential") == ["fam_a", "fam_c"]
    errors = family_errors()
    assert list(errors) == ["fam_bad"]
    assert isinstance(errors["fam_bad"], RuntimeError)
    assert len([r for r in caplog.records if "fam_bad" in r.getMessage()]) == 1
    assert "fam_bad" in list_model_ids()  # still listed without a category filter
    with pytest.raises(RuntimeError):
        get_model("fam_bad")  # a direct request reports the real error


def test_category_filter_matches_first_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [])
    register(_spec("plain"))
    eng = ModelSpec(
        id="eng_model",
        name="Eng",
        category="engineering.strength",
        expression="a*x+b",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=_guess,
    )
    register(eng)
    _fake_family()
    assert list_model_ids(category="engineering") == ["eng_model"]
    assert list_model_ids(category="engineering.strength") == ["eng_model"]
    assert list_model_ids(category="engineering.durability") == []
    assert "plain" in list_model_ids(category="exponential")
    assert "fake_0" in list_model_ids(category="exponential")
    assert [m.id for m in list_models("engineering")] == ["eng_model"]


def test_family_registration_errors() -> None:
    with pytest.raises(ValueError, match="prefix"):
        register_family("", lambda _e: [], _spec)
    _fake_family()
    with pytest.raises(ValueError, match="already registered"):
        _fake_family()


def test_family_building_wrong_id_is_an_error() -> None:
    register_family("w_", lambda _e: ["w_1"], lambda _i: _spec("other"))
    with pytest.raises(ValueError, match="built a model with id"):
        get_model("w_1")


def test_unknown_id_suggests_close_matches() -> None:
    register(_spec("exponential_fit"))
    with pytest.raises(KeyError, match="exponential_fit"):
        get_model("exponential_fi")
