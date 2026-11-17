"""Tests for the starter model set: contract, recovery, guesses, domains, registration."""

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import ModelSpec, get_model, list_models
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import STARTER_MODELS, register_starter_models
from curvestudio.testing.model_contract import check_model_contract

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

EXPECTED_IDS = (
    "exp_growth",
    "exp_decay_offset",
    "exp_offset",
    "log_natural",
    "log_base10",
    "power",
    "michaelis_menten",
    "logistic3",
    "gaussian",
)


@dataclass(frozen=True)
class Case:
    """True parameters and an x grid on which the model is identifiable."""

    truth: dict[str, float]
    x: FloatArray
    # Per-parameter relative tolerance override for the noisy test (default 5 %). exp_offset's
    # asymptote `a` is hidden under the exponential term (standard error ~18 % at 1 % noise).
    noisy_rtol: dict[str, float]


CASES: dict[str, Case] = {
    "exp_growth": Case({"a": 2.0, "b": 0.5}, np.linspace(0.0, 5.0, 25), {}),
    "exp_decay_offset": Case({"a": 5.0, "b": 0.8, "c": 1.0}, np.linspace(0.0, 6.0, 30), {}),
    "exp_offset": Case({"a": 1.0, "b": 2.0, "c": 0.5}, np.linspace(0.0, 4.0, 30), {"a": 0.10}),
    "log_natural": Case({"a": 1.0, "b": 2.0}, np.linspace(0.5, 10.0, 25), {}),
    "log_base10": Case({"a": 1.0, "b": 2.0}, np.linspace(0.5, 10.0, 25), {}),
    "power": Case({"a": 2.0, "b": 1.5}, np.linspace(0.5, 5.0, 25), {}),
    "michaelis_menten": Case({"vmax": 10.0, "km": 2.0}, np.linspace(0.2, 20.0, 25), {}),
    "logistic3": Case({"k": 10.0, "r": 1.2, "x0": 5.0}, np.linspace(0.0, 10.0, 30), {}),
    "gaussian": Case({"a": 4.0, "mu": 3.0, "s": 1.2}, np.linspace(-1.0, 7.0, 41), {}),
}


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Several test modules clear the global registry, so isolate and restore it."""
    _clear_registry()
    yield
    _clear_registry()


def _xy(x: FloatArray, y: FloatArray) -> XY:
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint="starter-test")


def _by_id(model_id: str) -> ModelSpec:
    return next(m for m in STARTER_MODELS if m.id == model_id)


def test_starter_set_has_the_nine_documented_ids_in_order() -> None:
    assert tuple(m.id for m in STARTER_MODELS) == EXPECTED_IDS
    assert set(CASES) == set(EXPECTED_IDS)


def test_expressions_match_the_specification() -> None:
    expected = {
        "exp_growth": "a*exp(b*x)",
        "exp_decay_offset": "a*exp(-b*x)+c",
        "exp_offset": "a+b*exp(c*x)",
        "log_natural": "a+b*log(x)",
        "log_base10": "a+b*log10(x)",
        "power": "a*x**b",
        "michaelis_menten": "vmax*x/(km+x)",
        "logistic3": "k/(1+exp(-r*(x-x0)))",
        "gaussian": "a*exp(-(x-mu)**2/(2*s**2))",
    }
    assert {m.id: m.expression for m in STARTER_MODELS} == expected


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_model_satisfies_contract(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    check_model_contract(spec, case.x, case.truth)


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_noiseless_recovery_from_own_initial_guess(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    y = spec.evaluate(case.x, **case.truth)
    start = spec.initial_guess(case.x, y)
    result = fit(spec, _xy(case.x, y), FitConfig(initial=start))
    assert result.status == FitStatus.CONVERGED, result.warnings
    for estimate in result.parameters:
        truth = case.truth[estimate.name]
        assert estimate.value == pytest.approx(truth, rel=1e-4), estimate.name


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_one_percent_noise_recovery(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    clean = spec.evaluate(case.x, **case.truth)
    rng = np.random.default_rng(20261116)
    y = clean + rng.normal(0.0, 0.01 * float(np.max(np.abs(clean))), size=clean.shape)
    start = spec.initial_guess(case.x, y)
    result = fit(spec, _xy(case.x, y), FitConfig(initial=start))
    assert result.status == FitStatus.CONVERGED, result.warnings
    for estimate in result.parameters:
        truth = case.truth[estimate.name]
        rtol = case.noisy_rtol.get(estimate.name, 0.05)
        assert estimate.value == pytest.approx(truth, rel=rtol), estimate.name


def _assert_in_bounds_and_interior(spec: ModelSpec, guess: dict[str, float]) -> None:
    assert set(guess) == {p.name for p in spec.parameters}
    for p in spec.parameters:
        value = guess[p.name]
        assert np.isfinite(value), f"{spec.id}.{p.name} not finite: {value}"
        assert p.lower < value < p.upper, f"{spec.id}.{p.name}={value} not strictly interior"


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_initial_guess_is_interior_for_constant_y(spec: ModelSpec) -> None:
    x = CASES[spec.id].x
    for level in (3.0, 0.0, -2.0):
        _assert_in_bounds_and_interior(spec, spec.initial_guess(x, np.full_like(x, level)))


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_initial_guess_is_interior_for_three_point_data(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    x = case.x[[0, len(case.x) // 2, -1]]
    y = spec.evaluate(x, **case.truth)
    _assert_in_bounds_and_interior(spec, spec.initial_guess(x, y))


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
@pytest.mark.parametrize("n_points", [0, 1, 2])
def test_initial_guess_survives_tiny_data(spec: ModelSpec, n_points: int) -> None:
    x = CASES[spec.id].x[:n_points]
    y = np.arange(1.0, 1.0 + n_points)
    _assert_in_bounds_and_interior(spec, spec.initial_guess(x, y))


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_initial_guess_survives_nonpositive_and_identical_x(spec: ModelSpec) -> None:
    y = np.array([-1.0, 0.0, 2.0, -3.0, 1.0])
    _assert_in_bounds_and_interior(spec, spec.initial_guess(np.zeros(5), y))
    _assert_in_bounds_and_interior(spec, spec.initial_guess(np.linspace(-2.0, 2.0, 5), y))
    _assert_in_bounds_and_interior(spec, spec.initial_guess(np.linspace(-2.0, 2.0, 5), -np.abs(y)))


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_initial_guess_is_deterministic(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    rng = np.random.default_rng(1)
    y = spec.evaluate(case.x, **case.truth) + rng.normal(0.0, 0.05, size=case.x.shape)
    assert spec.initial_guess(case.x, y) == spec.initial_guess(case.x, y)


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_initial_guess_does_not_mutate_inputs(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    y = spec.evaluate(case.x, **case.truth)
    x_before, y_before = case.x.copy(), y.copy()
    spec.initial_guess(case.x, y)
    np.testing.assert_array_equal(case.x, x_before)
    np.testing.assert_array_equal(y, y_before)


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_metadata_is_complete(spec: ModelSpec) -> None:
    case = CASES[spec.id]
    assert spec.name.strip()
    assert spec.category in {
        "exponential",
        "logarithmic",
        "power",
        "saturation",
        "sigmoidal",
        "peak",
    }
    assert spec.references and all(r.strip() for r in spec.references)
    assert spec.assumptions.strip()
    assert spec.equation_text(case.truth).startswith("y = ")
    assert spec.latex().strip()
    assert all(p.description.strip() for p in spec.parameters)


def test_categories_follow_the_specification() -> None:
    expected = {
        "exp_growth": "exponential",
        "exp_decay_offset": "exponential",
        "exp_offset": "exponential",
        "log_natural": "logarithmic",
        "log_base10": "logarithmic",
        "power": "power",
        "michaelis_menten": "saturation",
        "logistic3": "sigmoidal",
        "gaussian": "peak",
    }
    assert {m.id: m.category for m in STARTER_MODELS} == expected


def test_exp_decay_offset_and_exp_offset_are_distinct_parametrisations() -> None:
    assert _by_id("exp_decay_offset").expression != _by_id("exp_offset").expression
    assert [p.name for p in _by_id("exp_offset").parameters] == ["a", "b", "c"]


@pytest.mark.parametrize("model_id", ["log_natural", "log_base10", "power"])
def test_log_models_reject_nonpositive_x(model_id: str) -> None:
    spec = _by_id(model_id)
    y = np.ones(4)
    reasons = spec.applicability(np.array([-1.0, 0.0, 1.0, 2.0]), y)
    assert reasons
    assert "2 non-positive" in reasons[0]
    assert spec.applicability(np.array([0.5, 1.0, 2.0, 3.0]), y) == []


def test_power_does_not_require_positive_y() -> None:
    assert (
        _by_id("power").applicability(np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0])) == []
    )


def test_michaelis_menten_allows_zero_but_not_negative_x() -> None:
    spec = _by_id("michaelis_menten")
    y = np.ones(3)
    assert spec.applicability(np.array([0.0, 1.0, 2.0]), y) == []
    reasons = spec.applicability(np.array([-0.5, 1.0, 2.0]), y)
    assert reasons
    assert "1 negative" in reasons[0]


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda m: m.id)
def test_every_model_reports_non_finite_values(spec: ModelSpec) -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert spec.applicability(x, np.ones(4))
    assert spec.applicability(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, np.inf, 1.0, 1.0]))


def test_fit_reports_invalid_input_for_log_model_with_nonpositive_x() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = fit(_by_id("log_natural"), _xy(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0])))
    assert result.status == FitStatus.INVALID_INPUT


def test_register_starter_models_registers_all_ids(clean_registry: None) -> None:
    assert list_models() == []
    register_starter_models()
    assert [m.id for m in list_models()] == list(EXPECTED_IDS)
    for model_id in EXPECTED_IDS:
        assert get_model(model_id) is _by_id(model_id)


def test_register_starter_models_is_idempotent(clean_registry: None) -> None:
    register_starter_models()
    register_starter_models()
    assert len(list_models()) == len(EXPECTED_IDS)


def test_importing_models_package_registers_starter_set() -> None:
    """A fresh interpreter (untouched registry) sees all nine starter models after import."""
    code = (
        "from curvestudio.models import list_models; print(','.join(m.id for m in list_models()))"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    # Installed catalogue plugins (curvestudio-models) register more models; the starter set
    # must be there, in its own order.
    ids = done.stdout.strip().split(",")
    assert [i for i in ids if i in EXPECTED_IDS] == list(EXPECTED_IDS)
