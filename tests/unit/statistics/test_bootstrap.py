"""Tests for the seeded residual and case bootstrap."""

import json
import math
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitResult, FitStatus, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import (
    BootstrapResult,
    NonlinearFitter,
    PolynomialFitter,
    StatisticsError,
    bootstrap,
    build_view,
)


def make_xy(x: Any, y: Any, weights: Any = None, fingerprint: str = "fp-bootstrap") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def line_data(n: int = 40, seed: int = 5, slope: float = 0.5, sigma: float = 1.0) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return make_xy(x, 2.0 + slope * x + rng.normal(0.0, sigma, n), fingerprint=f"line-{n}-{seed}")


class RecordingFitter:
    """Wrap a fitter and keep every dataset it is asked to fit."""

    def __init__(self, inner: PolynomialFitter | None = None) -> None:
        self.inner: PolynomialFitter = PolynomialFitter(1) if inner is None else inner
        self.inputs: list[XY] = []

    def fit(self, data: XY) -> FitResult:
        self.inputs.append(data)
        return self.inner.fit(data)

    def predict(self, result: FitResult, x: Any) -> Any:
        return self.inner.predict(result, x)


class FailingFitter:
    """Fail every ``every``-th call, either by raising, by a bad status, or both in turn."""

    def __init__(self, every: int = 3, mode: str = "raise") -> None:
        self.every = every
        self.mode = mode
        self.calls = 0
        self._poly = PolynomialFitter(1)

    def fit(self, data: XY) -> FitResult:
        self.calls += 1
        result = self._poly.fit(data)
        if self.calls % self.every != 0:
            return result
        failing = self.mode == "raise" or (self.mode == "mixed" and self.calls % 6 == 0)
        if failing:
            raise RuntimeError("deliberate bootstrap failure")
        return result.model_copy(
            update={"status": FitStatus.NOT_CONVERGED, "termination_reason": "deliberate failure"}
        )

    def predict(self, result: FitResult, x: Any) -> Any:
        return self._poly.predict(result, x)


class AlwaysFailingFitter:
    """Fail every bootstrap replicate."""

    def fit(self, data: XY) -> FitResult:
        raise RuntimeError("never converges")

    def predict(self, result: FitResult, x: Any) -> Any:
        raise AssertionError("predict must not be called by bootstrap")


def fitted_line(data: XY) -> FitResult:
    return fit_polynomial(data, 1)


def test_same_seed_gives_identical_replicates() -> None:
    data = line_data()
    result = fitted_line(data)
    fitter = PolynomialFitter(1)

    first = bootstrap(data, fitter, result, n_boot=200, seed=42)
    second = bootstrap(data, fitter, result, n_boot=200, seed=42)

    assert first.seed == second.seed == 42
    assert first.n_requested == second.n_requested == 200
    assert first.n_converged == second.n_converged == 200
    assert first.n_failed == second.n_failed == 0
    assert first.status == second.status == "ok"
    assert first.replicates == second.replicates
    assert first.parameters == second.parameters
    assert first.reasons == second.reasons == []

    other = bootstrap(data, fitter, result, n_boot=200, seed=43)
    assert other.replicates != first.replicates


def test_residual_bootstrap_se_matches_the_analytic_se_for_a_line() -> None:
    data = line_data(n=40, seed=5)
    result = fitted_line(data)
    assert result.covariance is not None
    analytic = np.sqrt(np.diag(np.asarray(result.covariance)))

    boot = bootstrap(data, PolynomialFitter(1), result, n_boot=2000, seed=11)

    assert boot.n_converged == 2000 and boot.n_failed == 0
    assert len(boot.parameters) == 2
    for parameter, expected in zip(boot.parameters, analytic, strict=True):
        assert parameter.std_err == pytest.approx(float(expected), rel=0.15)


def test_case_bootstrap_se_is_close_for_a_line() -> None:
    data = line_data(n=40, seed=6)
    result = fitted_line(data)
    assert result.covariance is not None
    analytic = np.sqrt(np.diag(np.asarray(result.covariance)))

    boot = bootstrap(data, PolynomialFitter(1), result, kind="case", n_boot=2000, seed=12)

    assert boot.kind == "case"
    assert boot.n_converged == 2000 and boot.n_failed == 0
    for parameter, expected in zip(boot.parameters, analytic, strict=True):
        assert parameter.std_err == pytest.approx(float(expected), rel=0.25)


def test_failed_replicates_are_counted_and_make_result_unreliable_below_min_success() -> None:
    data = line_data(n=30, seed=8)
    result = fitted_line(data)
    fitter = FailingFitter(every=3, mode="raise")

    boot = bootstrap(data, fitter, result, n_boot=100, seed=3, min_success=0.9)

    assert boot.n_requested == 100
    assert boot.n_failed == 33
    assert boot.n_converged == 67
    assert len(boot.replicates) == 67
    assert all(len(row) == 2 for row in boot.replicates)
    assert boot.status == "unreliable"
    assert any("33 replicate(s)" in reason for reason in boot.reasons)
    assert any("deliberate bootstrap failure" in reason for reason in boot.reasons)
    assert any("below min_success" in reason for reason in boot.reasons)
    # CIs are still reported from the surviving replicates.
    assert all(parameter.ci_low is not None for parameter in boot.parameters)
    assert all(parameter.ci_high is not None for parameter in boot.parameters)


def test_distinct_failure_reasons_are_counted_separately() -> None:
    data = line_data(n=30, seed=8)
    result = fitted_line(data)

    boot = bootstrap(data, FailingFitter(every=3, mode="mixed"), result, n_boot=100, seed=4)

    assert boot.n_failed == 33
    assert boot.n_converged == 67
    failure_lines = [reason for reason in boot.reasons if "replicate(s)" in reason]
    assert len(failure_lines) == 2
    counts = sorted(int(line.split()[0]) for line in failure_lines)
    assert counts == [16, 17]
    assert any("RuntimeError" in line for line in failure_lines)
    assert any("not_converged" in line for line in failure_lines)
    assert sum(counts) == boot.n_failed


class RenamingFitter:
    """Converge, but rename the parameters of every ``every``-th replicate's result."""

    def __init__(self, every: int = 4) -> None:
        self.every = every
        self.calls = 0
        self._poly = PolynomialFitter(1)

    def fit(self, data: XY) -> FitResult:
        self.calls += 1
        result = self._poly.fit(data)
        if self.calls % self.every != 0:
            return result
        renamed = tuple(
            parameter.model_copy(update={"name": f"renamed_{parameter.name}"})
            for parameter in result.parameters
        )
        return result.model_copy(update={"parameters": renamed})

    def predict(self, result: FitResult, x: Any) -> Any:
        return self._poly.predict(result, x)


def test_replicate_with_foreign_parameter_names_is_failed_not_used() -> None:
    # Replicates are summarised by parameter position, so a converged result whose
    # parameter names differ from the original fit's must not enter the pool silently.
    data = line_data(n=30, seed=8)
    result = fitted_line(data)

    boot = bootstrap(data, RenamingFitter(every=4), result, n_boot=100, seed=5, min_success=0.5)

    assert boot.status == "ok"  # 75/100 converged is above min_success=0.5
    assert boot.n_converged == 75
    assert boot.n_failed == 25
    assert len(boot.replicates) == 75
    assert all(len(row) == 2 for row in boot.replicates)
    assert len(boot.reasons) == 1
    assert "fit returned parameters" in boot.reasons[0]
    assert "renamed_b0" in boot.reasons[0] and "renamed_b1" in boot.reasons[0]
    assert "expected ['b0', 'b1']" in boot.reasons[0]
    assert "25 replicate(s)" in boot.reasons[0]


def test_ci_contains_estimate_and_orders_correctly() -> None:
    data = line_data(n=40, seed=5)
    result = fitted_line(data)

    boot = bootstrap(data, PolynomialFitter(1), result, n_boot=400, seed=2)

    assert [parameter.name for parameter in boot.parameters] == ["b0", "b1"]
    for parameter in boot.parameters:
        assert parameter.ci_low is not None and parameter.ci_high is not None
        assert parameter.ci_low < parameter.ci_high
        assert parameter.ci_low <= parameter.mean <= parameter.ci_high
        assert parameter.ci_low <= parameter.estimate <= parameter.ci_high
        assert parameter.std_err > 0.0


EXPONENTIAL_DECAY = ModelSpec(
    id="exp_decay_boot_r1",
    name="Exponential decay",
    category="test",
    expression="a*exp(-b*x)",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": float(np.max(y)), "b": 0.5},
)


def exponential_decay_data(n: int = 25, seed: int = 21) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 5.0, n)
    return make_xy(x, 3.0 * np.exp(-0.4 * x) + rng.normal(0.0, 0.05, n), fingerprint="fp-exp-decay")


def test_bootstrap_with_nonlinear_fitter_on_a_registered_model() -> None:
    _clear_registry()
    register(EXPONENTIAL_DECAY)
    try:
        data = exponential_decay_data()
        fitter = NonlinearFitter(EXPONENTIAL_DECAY, start={"a": 3.0, "b": 0.4})
        result = fitter.fit(data)
        assert result.status is FitStatus.CONVERGED

        boot = bootstrap(data, fitter, result, n_boot=50, seed=13)

        assert boot.status == "ok"
        assert boot.n_requested == 50
        assert boot.n_converged + boot.n_failed == 50
        assert [parameter.name for parameter in boot.parameters] == ["a", "b"]
        for parameter in boot.parameters:
            assert parameter.ci_low is not None and parameter.ci_high is not None
            assert parameter.ci_low <= parameter.ci_high
            assert parameter.std_err > 0.0
    finally:
        _clear_registry()


def test_zero_weight_points_are_not_resampled() -> None:
    rng = np.random.default_rng(9)
    x = np.linspace(0.0, 10.0, 20)
    weights = np.ones_like(x)
    weights[[3, 11]] = 0.0
    data = make_xy(x, 2.0 + 0.5 * x + rng.normal(0.0, 0.5, x.size), weights, "fp-zero")
    active = x[weights > 0.0]
    inactive = set(x[weights == 0.0].tolist())
    result = fit_polynomial(data, 1)

    case_rec = RecordingFitter(PolynomialFitter(1))
    case_boot = bootstrap(data, case_rec, result, kind="case", n_boot=100, seed=5)
    assert case_boot.n_converged == 100
    assert len(case_rec.inputs) == 100
    for sample in case_rec.inputs:
        assert sample.x.size == active.size
        assert set(sample.x.tolist()) <= set(active.tolist())
        assert set(sample.x.tolist()).isdisjoint(inactive)
        assert np.all(sample.weights > 0.0)

    residual_rec = RecordingFitter(PolynomialFitter(1))
    residual_boot = bootstrap(data, residual_rec, result, n_boot=100, seed=5)
    assert residual_boot.n_converged == 100
    for sample in residual_rec.inputs:
        np.testing.assert_array_equal(sample.x, active)
        assert set(sample.x.tolist()).isdisjoint(inactive)
        assert np.all(sample.weights > 0.0)


def test_residual_bootstrap_rebuilds_y_from_inflated_weighted_residuals() -> None:
    data = line_data(n=20, seed=3)
    result = fitted_line(data)
    view = build_view(data, result)
    n = int(view.x.size)
    k = len(view.names)
    adjusted = np.sqrt(view.w) * view.residuals * math.sqrt(n / (n - k))

    rec = RecordingFitter(PolynomialFitter(1))
    bootstrap(data, rec, result, n_boot=50, seed=6)

    for sample in rec.inputs:
        implied = (sample.y - view.fitted) * np.sqrt(view.w)
        for value in implied:
            assert np.any(np.isclose(value, adjusted, rtol=1e-12, atol=1e-12))


def test_case_bootstrap_resamples_whole_active_triples() -> None:
    data = line_data(n=20, seed=10)
    result = fitted_line(data)
    triples = {
        (float(x), float(y), float(w))
        for x, y, w in zip(data.x, data.y, np.ones(data.x.size), strict=True)
    }

    rec = RecordingFitter(PolynomialFitter(1))
    bootstrap(data, rec, result, kind="case", n_boot=50, seed=7)

    for sample in rec.inputs:
        for x, y, w in zip(sample.x, sample.y, sample.weights, strict=True):
            assert (float(x), float(y), float(w)) in triples


def test_result_round_trips_through_json() -> None:
    data = line_data(n=24, seed=13)
    boot = bootstrap(data, PolynomialFitter(1), fitted_line(data), n_boot=60, seed=8)

    text = boot.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    restored = BootstrapResult.model_validate_json(text)
    assert restored == boot
    assert isinstance(restored, BootstrapResult)

    payload = json.loads(text)
    assert set(payload) == {
        "kind",
        "n_requested",
        "n_converged",
        "n_failed",
        "level",
        "seed",
        "status",
        "parameters",
        "replicates",
        "reasons",
    }
    assert len(payload["replicates"]) == boot.n_converged
    assert len(payload["replicates"][0]) == len(boot.parameters)


@pytest.mark.parametrize("n_boot", [0, 1, 49, -5, 50.0, True, "100"])
def test_invalid_n_boot_rejected(n_boot: Any) -> None:
    data = line_data(n=12)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), fitted_line(data), n_boot=n_boot)
    assert info.value.code == "INVALID_N_BOOT"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), True])
def test_invalid_level_rejected(level: Any) -> None:
    data = line_data(n=12)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), fitted_line(data), level=level)
    assert info.value.code == "INVALID_LEVEL"


@pytest.mark.parametrize("min_success", [0.0, 1.0, -0.1, 1.01, float("nan"), True])
def test_invalid_min_success_rejected(min_success: Any) -> None:
    data = line_data(n=12)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), fitted_line(data), min_success=min_success)
    assert info.value.code == "INVALID_MIN_SUCCESS"


@pytest.mark.parametrize("seed", [-1, 1.5, True, "0"])
def test_invalid_seed_rejected(seed: Any) -> None:
    data = line_data(n=12)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), fitted_line(data), seed=seed)
    assert info.value.code == "INVALID_SEED"


def test_invalid_kind_rejected() -> None:
    data = line_data(n=12)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), fitted_line(data), kind="parametric")
    assert info.value.code == "INVALID_KIND"


def test_non_converged_fit_is_refused() -> None:
    data = line_data(n=20)
    result = fitted_line(data).model_copy(update={"status": FitStatus.NOT_CONVERGED})

    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), result, n_boot=50)
    assert info.value.code == "FIT_NOT_CONVERGED"


def test_invalid_input_fit_is_refused() -> None:
    data = line_data(n=20)
    result = fitted_line(data).model_copy(update={"status": FitStatus.INVALID_INPUT})

    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), result, n_boot=50)
    assert info.value.code == "FIT_NOT_CONVERGED"


def test_fit_without_degrees_of_freedom_is_refused() -> None:
    data = line_data(n=20)
    result = fitted_line(data)
    assert result.stats is not None
    no_df = result.model_copy(update={"stats": result.stats.model_copy(update={"df": 0})})
    forged = no_df.model_copy(update={"covariance": None})

    with pytest.raises(StatisticsError) as info:
        bootstrap(data, PolynomialFitter(1), forged, n_boot=50)
    assert info.value.code == "NO_DEGREES_OF_FREEDOM"


def test_all_replicates_failing_raises_with_the_reasons() -> None:
    data = line_data(n=20)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, AlwaysFailingFitter(), fitted_line(data), n_boot=50)
    assert info.value.code == "NO_CONVERGED_REPLICATES"
    assert "only 0 of 50 bootstrap replicates converged" in str(info.value)
    assert "never converges" in str(info.value)


class SucceedsOnceFitter:
    """Converge on the first call only (the full-data refit is not routed through it)."""

    def __init__(self) -> None:
        self.calls = 0
        self._poly = PolynomialFitter(1)

    def fit(self, data: XY) -> FitResult:
        self.calls += 1
        result = self._poly.fit(data)
        if self.calls == 1:
            return result
        return result.model_copy(
            update={"status": FitStatus.NOT_CONVERGED, "termination_reason": "deliberate failure"}
        )

    def predict(self, result: FitResult, x: Any) -> Any:
        return self._poly.predict(result, x)


def test_a_single_converged_replicate_gives_no_standard_error() -> None:
    # One replicate has no spread: a standard error of 0.0 would be a fabricated number,
    # and min_success=0.01 would otherwise let 1/50 pass as "ok".
    data = line_data(n=20)
    with pytest.raises(StatisticsError) as info:
        bootstrap(data, SucceedsOnceFitter(), fitted_line(data), n_boot=50, min_success=0.01)
    assert info.value.code == "NO_CONVERGED_REPLICATES"
    assert "only 1 of 50" in str(info.value)


@pytest.mark.slow
def test_residual_bootstrap_slope_ci_coverage_is_near_nominal() -> None:
    covered = 0
    datasets = 100
    truth = 0.5
    for index in range(datasets):
        rng = np.random.default_rng(5000 + index)
        x = np.linspace(0.0, 10.0, 30)
        y = 2.0 + truth * x + rng.normal(0.0, 1.0, x.size)
        data = make_xy(x, y, fingerprint=f"coverage-{index}")
        result = fit_polynomial(data, 1)

        boot = bootstrap(data, PolynomialFitter(1), result, n_boot=150, seed=9000 + index)

        assert boot.status == "ok"
        slope = boot.parameters[1]
        assert slope.ci_low is not None and slope.ci_high is not None
        covered += int(slope.ci_low <= truth <= slope.ci_high)

    assert 90 <= covered <= 99
