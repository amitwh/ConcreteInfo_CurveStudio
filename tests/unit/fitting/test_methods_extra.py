"""Tests for the robust-loss, derivative-free and global fit methods (Task 2.5)."""

import json
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.fitting.methods import check_method_compatibility
from curvestudio.fitting.nonlinear import ROBUST_LOSS_WARNING, _Problem
from curvestudio.models import ModelSpec, Parameter

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@pytest.fixture(autouse=True)
def no_warnings_escape() -> Iterator[None]:
    """Any warning raised while fitting (RuntimeWarning, OptimizeWarning...) fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def make_xy(x: Any, y: Any, weights: Any = None) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="test-fingerprint",
    )


def _exp_decay_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    c = float(np.min(y))
    return {"a": float(y[0] - c), "b": 0.1, "c": c}


EXP_DECAY = ModelSpec(
    id="exp_decay",
    name="Exponential decay with offset",
    category="test",
    expression="a*exp(-b*x) + c",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=_exp_decay_guess,
)

LINE = ModelSpec(
    id="line",
    name="Straight line",
    category="test",
    expression="b0 + b1*x",
    parameters=(Parameter("b0"), Parameter("b1")),
    initial_guess=lambda x, y: {"b0": 0.0, "b1": 0.0},
)

BOUNDS = {"a": (0.0, 20.0), "b": (0.0, 5.0), "c": (-5.0, 10.0)}
TRUTH = {"a": 5.0, "b": 0.4, "c": 2.0}


def prd_data() -> XY:
    """PRD section 20 example: 5*exp(-0.4x) + 2 with N(0, 0.1) noise, default_rng(42)."""
    x = np.linspace(0.0, 10.0, 100)
    y = 5.0 * np.exp(-0.4 * x) + 2.0 + np.random.default_rng(42).normal(0.0, 0.1, x.size)
    return make_xy(x, y)


def values(result: Any) -> dict[str, float]:
    return {p.name: p.value for p in result.parameters}


def assert_near_truth(result: Any, rtol: float) -> None:
    assert result.status is FitStatus.CONVERGED, result.termination_reason
    for name, value in values(result).items():
        assert value == pytest.approx(TRUTH[name], rel=rtol), name


def line_with_outliers() -> XY:
    x = np.linspace(0.0, 10.0, 21)
    y = 2.0 * x + 1.0
    y[[3, 9, 15]] += 50.0
    return make_xy(x, y)


# --- robust losses ---------------------------------------------------------------------


def test_soft_l1_recovers_slope_where_linear_loss_is_pulled_off() -> None:
    data = line_with_outliers()
    robust = fit(LINE, data, FitConfig(loss="soft_l1"))
    plain = fit(LINE, data, FitConfig(loss="linear"))
    assert robust.status is FitStatus.CONVERGED
    assert values(robust)["b1"] == pytest.approx(2.0, rel=0.02)
    assert abs(values(plain)["b1"] - 2.0) > 0.10 * 2.0


def test_robust_loss_adds_the_approximate_covariance_warning() -> None:
    res = fit(LINE, line_with_outliers(), FitConfig(loss="soft_l1"))
    assert ROBUST_LOSS_WARNING in res.warnings
    assert ROBUST_LOSS_WARNING.startswith("ROBUST_LOSS_COVARIANCE_APPROXIMATE: ")


def test_linear_loss_has_no_robust_loss_warning() -> None:
    res = fit(EXP_DECAY, prd_data())
    assert not any("ROBUST_LOSS" in w for w in res.warnings)


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy", "arctan"])
@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_every_robust_loss_runs_with_trf_and_dogbox(method: str, loss: str) -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method=method, loss=loss))
    assert res.status is FitStatus.CONVERGED
    assert (res.method, res.loss) == (method, loss)


def test_lm_with_huber_raises() -> None:
    with pytest.raises(ValueError, match="loss"):
        fit(EXP_DECAY, prd_data(), FitConfig(method="lm", loss="huber"))


@pytest.mark.parametrize(
    "method",
    ["nelder", "powell", "bfgs", "lbfgsb", "differential_evolution", "dual_annealing"],
)
def test_robust_loss_with_a_scalar_method_raises_naming_loss(method: str) -> None:
    bounds = None if method == "bfgs" else BOUNDS  # bfgs would be rejected for bounds first
    with pytest.raises(ValueError, match="loss"):
        fit(EXP_DECAY, prd_data(), FitConfig(method=method, loss="huber", bounds=bounds))


def test_auto_resolves_to_trf_for_robust_loss_and_never_to_a_scalar_method() -> None:
    assert fit(EXP_DECAY, prd_data(), FitConfig(loss="huber")).method == "trf"
    assert fit(EXP_DECAY, prd_data(), FitConfig(bounds=BOUNDS)).method == "trf"
    assert fit(EXP_DECAY, prd_data()).method == "lm"


# --- every method converges on the PRD data ---------------------------------------------


@pytest.mark.parametrize("method", ["dogbox", "nelder", "powell", "lbfgsb"])
def test_bounded_methods_converge_on_the_prd_data(method: str) -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method=method, bounds=BOUNDS))
    assert_near_truth(res, 0.10)
    assert res.method == method


@pytest.mark.parametrize("method", ["nelder", "powell", "bfgs", "lbfgsb"])
def test_unbounded_scalar_methods_converge_on_the_prd_data(method: str) -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method=method))
    assert_near_truth(res, 0.10)


def test_dual_annealing_converges_on_the_prd_data() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method="dual_annealing", bounds=BOUNDS))
    assert_near_truth(res, 0.10)


def test_differential_evolution_recovers_the_prd_parameters_within_5_percent() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method="differential_evolution", bounds=BOUNDS))
    assert_near_truth(res, 0.05)


@pytest.mark.parametrize(
    "method",
    ["nelder", "powell", "bfgs", "lbfgsb", "differential_evolution", "dual_annealing"],
)
def test_scalar_methods_report_the_least_squares_uncertainties(method: str) -> None:
    """Covariance/SEs are s2*pinv(J'J) at the solution, so they match a trf fit's."""
    config = FitConfig(method=method, bounds=BOUNDS if method != "bfgs" else None)
    res = fit(EXP_DECAY, prd_data(), config)
    reference = fit(EXP_DECAY, prd_data(), FitConfig(method="trf", bounds=BOUNDS))
    assert res.covariance is not None and reference.covariance is not None
    assert res.stats is not None and reference.stats is not None
    assert res.stats.df == reference.stats.df == 97
    assert all(p.std_err is not None and p.ci_low is not None for p in res.parameters)
    for got, want in zip(res.parameters, reference.parameters, strict=True):
        assert got.std_err == pytest.approx(want.std_err, rel=0.05)
    assert res.stats.sse == pytest.approx(reference.stats.sse, rel=1e-4)


@pytest.mark.parametrize("scale", [100.0, 1000.0])
def test_nelder_with_default_config_converges_at_large_sse(scale: float) -> None:
    """Regression: nelder's ``fatol`` was the absolute ``ftol`` (1e-14), unreachable at
    SSE ~4e5 / ~4e7, so Nelder-Mead ran out of evaluations instead of converging."""
    x = np.linspace(0.0, 10.0, 60)
    noise = scale * np.random.default_rng(1).normal(0.0, 1.0, x.size)
    data = make_xy(x, scale * (5.0 * np.exp(-0.4 * x) + 2.0) + noise)
    res = fit(EXP_DECAY, data, FitConfig(method="nelder"))
    reference = fit(EXP_DECAY, data, FitConfig(method="trf"))
    assert reference.status is FitStatus.CONVERGED
    assert res.status is FitStatus.CONVERGED, res.termination_reason
    assert res.stats is not None
    assert res.stats.sse > 1e5 * (scale / 100.0) ** 2
    for got, want in zip(res.parameters, reference.parameters, strict=True):
        assert got.value == pytest.approx(want.value, rel=1e-3), got.name


# --- configuration errors ---------------------------------------------------------------


def test_bfgs_with_bounds_raises_naming_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        fit(EXP_DECAY, prd_data(), FitConfig(method="bfgs", bounds=BOUNDS))


@pytest.mark.parametrize("method", ["differential_evolution", "dual_annealing"])
def test_global_methods_without_bounds_raise_naming_finite_bounds(method: str) -> None:
    with pytest.raises(ValueError, match="finite bounds"):
        fit(EXP_DECAY, prd_data(), FitConfig(method=method))


@pytest.mark.parametrize("method", ["differential_evolution", "dual_annealing"])
def test_global_methods_with_only_some_parameters_bounded_raise(method: str) -> None:
    partial = {"a": (0.0, 20.0), "b": (0.0, 5.0)}  # c stays unbounded
    with pytest.raises(ValueError, match="finite bounds"):
        fit(EXP_DECAY, prd_data(), FitConfig(method=method, bounds=partial))
    half_open = {**BOUNDS, "c": (-5.0, float("inf"))}
    with pytest.raises(ValueError, match="finite bounds"):
        fit(EXP_DECAY, prd_data(), FitConfig(method=method, bounds=half_open))


@pytest.mark.parametrize("method", ["differential_evolution", "dual_annealing"])
def test_global_methods_reject_multiple_starts(method: str) -> None:
    with pytest.raises(ValueError, match="n_starts"):
        fit(EXP_DECAY, prd_data(), FitConfig(method=method, bounds=BOUNDS, n_starts=3))


def test_check_method_compatibility_keeps_the_four_argument_form_working() -> None:
    check_method_compatibility("lbfgsb", "linear", True, True)
    with pytest.raises(ValueError, match="finite bounds"):
        check_method_compatibility("dual_annealing", "linear", True, True, all_bounds_finite=False)


# --- determinism and seeds --------------------------------------------------------------


def _json_without_time(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(result.model_dump_json())
    payload["provenance"].pop("created_at")
    return payload


@pytest.mark.parametrize("method", ["differential_evolution", "dual_annealing"])
def test_same_seed_gives_identical_json(method: str) -> None:
    config = FitConfig(method=method, bounds=BOUNDS, seed=7)
    first = fit(EXP_DECAY, prd_data(), config)
    second = fit(EXP_DECAY, prd_data(), config)
    assert _json_without_time(first) == _json_without_time(second)
    assert first.provenance.seed == 7


def test_different_seeds_are_recorded() -> None:
    res = fit(
        EXP_DECAY,
        prd_data(),
        FitConfig(method="differential_evolution", bounds=BOUNDS, seed=11),
    )
    assert res.provenance.seed == 11


# --- gradient, weights, multi-start, failures --------------------------------------------


def test_analytic_gradient_matches_a_central_finite_difference() -> None:
    data = prd_data()
    weights = np.linspace(0.5, 2.0, data.x.size)
    problem = _Problem(
        EXP_DECAY,
        np.asarray(data.x),
        np.asarray(data.y),
        np.sqrt(weights),
        ("a", "b", "c"),
    )
    theta = np.array([4.0, 0.5, 1.5])
    grad = problem.sse_gradient(theta)
    numeric = np.empty(3)
    for j in range(3):
        step = np.zeros(3)
        step[j] = 1e-6
        plus = float(np.sum(problem.residuals(theta + step) ** 2))
        minus = float(np.sum(problem.residuals(theta - step) ** 2))
        numeric[j] = (plus - minus) / 2e-6
    assert grad == pytest.approx(numeric, rel=1e-5)


def test_zero_weight_outlier_is_ignored_by_lbfgsb() -> None:
    x = np.linspace(0.0, 10.0, 21)
    y = 2.0 * x + 1.0
    y[5] += 80.0
    weights = np.ones_like(x)
    weights[5] = 0.0
    res = fit(LINE, make_xy(x, y, weights), FitConfig(method="lbfgsb"))
    assert res.status is FitStatus.CONVERGED
    assert values(res)["b0"] == pytest.approx(1.0, abs=1e-3)
    assert values(res)["b1"] == pytest.approx(2.0, abs=1e-3)
    assert res.residuals[5] == pytest.approx(80.0)  # reported, but influenced nothing


def test_multi_start_applies_to_a_scalar_method() -> None:
    res = fit(
        EXP_DECAY,
        prd_data(),
        FitConfig(method="lbfgsb", bounds=BOUNDS, n_starts=4, seed=3),
    )
    assert_near_truth(res, 0.10)
    assert res.n_starts == 4


def test_scalar_method_at_a_bound_is_flagged() -> None:
    res = fit(
        EXP_DECAY,
        prd_data(),
        FitConfig(method="lbfgsb", bounds={**BOUNDS, "c": (2.5, 10.0)}),
    )
    assert "PARAMETER_AT_BOUND: c" in res.warnings


def test_exhausted_budget_is_not_converged_with_the_scipy_message() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(method="nelder", max_nfev=5))
    assert res.status is FitStatus.NOT_CONVERGED
    assert res.termination_reason
    assert res.covariance is None


def test_too_few_points_is_invalid_input_for_scalar_methods() -> None:
    res = fit(EXP_DECAY, make_xy([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]), FitConfig(method="powell"))
    assert res.status is FitStatus.INVALID_INPUT


def test_non_finite_jacobian_in_the_gradient_fails_the_run_with_a_message() -> None:
    """sqrt(a)*x has an infinite derivative at a=0: lbfgsb needs it, so the fit is FAILED."""
    model = ModelSpec(
        id="sqrt_slope",
        name="sqrt slope",
        category="test",
        expression="sqrt(a)*x + c",
        parameters=(Parameter("a"), Parameter("c")),
        initial_guess=lambda x, y: {"a": 0.0, "c": 0.0},
    )
    x = np.linspace(0.0, 4.0, 20)
    res = fit(model, make_xy(x, 3.0 * x + 1.0), FitConfig(method="lbfgsb"))
    assert res.status is FitStatus.FAILED
    assert "Jacobian is not finite" in res.termination_reason
