"""Tests for seeded k-fold and leave-one-out cross-validation."""

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, FitStatus
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import (
    CrossValidation,
    Fitter,
    NonlinearFitter,
    PolynomialFitter,
    StatisticsError,
    k_fold_cv,
    loo_cv,
)

SHIFTED_LOG = ModelSpec(
    id="shifted_log_cv",
    name="Shifted logarithm",
    category="test",
    expression="a + b*log(x - c)",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=lambda x, y: {"a": 0.0, "b": 1.0, "c": float(x.min()) + 0.5},
)
"""The default start puts ``c`` inside the data range, where ``log(x - c)`` is not finite."""

WARM_START = {"a": 1.0, "b": 2.0, "c": 1.0}
"""A start near the truth: it makes every fold of the shifted-log problem converge."""


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(SHIFTED_LOG)
    yield
    _clear_registry()


def make_xy(x: Any, y: Any, weights: Any = None, fingerprint: str = "fp-cv") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def noisy_quadratic(n: int = 20, seed: int = 1) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return make_xy(x, 1.0 + 0.5 * x - 0.02 * x**2 + rng.normal(0.0, 0.2, n))


def cubic_data(n: int = 60, seed: int = 11) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    y = 0.3 + 1.2 * x - 0.7 * x**2 + 0.4 * x**3 + rng.normal(0.0, 0.3, n)
    return make_xy(x, y)


def shifted_log_data(n: int = 24, seed: int = 4) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(2.0, 10.0, n)
    y = 1.0 + 2.0 * np.log(x - 1.0) + rng.normal(0.0, 0.02, n)
    return make_xy(x, y)


class RecordingFitter:
    """Wrap a fitter and record the training ``x`` of every fold fit it performs."""

    def __init__(self, inner: Fitter | None = None) -> None:
        self.inner: Fitter = PolynomialFitter(1) if inner is None else inner
        self.train_x: list[np.ndarray[Any, Any]] = []

    def fit(self, data: XY) -> FitResult:
        self.train_x.append(np.array(data.x))
        return self.inner.fit(data)

    def predict(self, result: FitResult, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return self.inner.predict(result, x)


class FailingFitter:
    """Fail exactly one fold, either by raising or by returning ``NOT_CONVERGED``."""

    def __init__(self, fail_on: int = 2, mode: Literal["raise", "status"] = "raise") -> None:
        self.fail_on = fail_on
        self.mode = mode
        self.calls = 0
        self._poly = PolynomialFitter(1)

    def fit(self, data: XY) -> FitResult:
        self.calls += 1
        result = self._poly.fit(data)
        if self.calls != self.fail_on:
            return result
        if self.mode == "raise":
            raise RuntimeError("deliberate fold failure")
        return result.model_copy(
            update={"status": FitStatus.NOT_CONVERGED, "termination_reason": "deliberate failure"}
        )

    def predict(self, result: FitResult, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return self._poly.predict(result, x)


def test_same_seed_same_folds_and_rmse() -> None:
    data = noisy_quadratic(n=24)
    first_rec, second_rec = RecordingFitter(), RecordingFitter()
    first = k_fold_cv(data, first_rec, folds=4, seed=11)
    second = k_fold_cv(data, second_rec, folds=4, seed=11)

    assert first.folds == second.folds == 4
    assert first.seed == second.seed == 11
    assert first.status == second.status == "ok"
    assert first.n_failed == second.n_failed == 0
    assert first.rmse is not None and first.rmse == second.rmse
    assert first.fold_rmse == second.fold_rmse
    assert first.reasons == second.reasons == []
    for a, b in zip(first_rec.train_x, second_rec.train_x, strict=True):
        np.testing.assert_array_equal(a, b)


def test_different_seed_changes_fold_assignment() -> None:
    data = noisy_quadratic(n=24)
    first_rec, second_rec = RecordingFitter(), RecordingFitter()
    k_fold_cv(data, first_rec, folds=4, seed=0)
    k_fold_cv(data, second_rec, folds=4, seed=1)

    differ = [
        not np.array_equal(a, b) for a, b in zip(first_rec.train_x, second_rec.train_x, strict=True)
    ]
    assert any(differ)


def test_every_active_point_is_held_out_exactly_once() -> None:
    data = noisy_quadratic(n=17)
    rec = RecordingFitter()
    cv = k_fold_cv(data, rec, folds=5, seed=3)

    assert cv.status == "ok" and cv.n_failed == 0
    assert len(rec.train_x) == cv.folds == 5
    active = set(np.asarray(data.x).tolist())
    held_sets = [active - set(train.tolist()) for train in rec.train_x]
    held = [value for fold in held_sets for value in fold]
    assert len(held) == len(set(held)) == len(active)
    assert set(held) == active
    held_sizes = [len(fold) for fold in held_sets]
    assert max(held_sizes) - min(held_sizes) <= 1


def test_zero_weight_points_never_appear_in_any_fold() -> None:
    x = np.linspace(0.0, 10.0, 14)
    w = np.ones_like(x)
    w[[2, 9]] = 0.0
    data = make_xy(x, 1.0 + x, w)
    rec = RecordingFitter()
    cv = k_fold_cv(data, rec, folds=3, seed=5)

    assert cv.status == "ok"
    active = set(np.asarray(data.x[w > 0]).tolist())
    inactive = set(np.asarray(data.x[w == 0]).tolist())
    assert len(active) == 12
    total_held = 0
    for train in rec.train_x:
        train_set = set(train.tolist())
        assert train_set <= active
        assert train_set.isdisjoint(inactive)
        held = active - train_set
        assert held.isdisjoint(inactive)
        total_held += len(held)
    assert total_held == len(active)


def test_loo_matches_closed_form_press_for_a_linear_model() -> None:
    rng = np.random.default_rng(7)
    n = 25
    x = np.linspace(-1.0, 1.0, n)
    y = 0.5 - 1.2 * x + 0.8 * x**2 + rng.normal(0.0, 0.1, n)
    data = make_xy(x, y)

    cv = loo_cv(data, PolynomialFitter(2))

    design = np.vander(x, 3, increasing=True)
    leverage = np.diag(design @ np.linalg.inv(design.T @ design) @ design.T)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    press = residuals / (1.0 - leverage)
    expected = float(np.sqrt(np.mean(press**2)))

    assert cv.folds == n and cv.seed == 0
    assert cv.status == "ok" and cv.n_failed == 0
    assert cv.rmse == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(sorted(cv.fold_rmse), sorted(np.abs(press)), rtol=1e-9)


def test_cv_rmse_prefers_true_degree_on_known_truth_data() -> None:
    data = cubic_data()
    line = k_fold_cv(data, PolynomialFitter(1), folds=5, seed=0)
    cubic = k_fold_cv(data, PolynomialFitter(3), folds=5, seed=0)

    assert line.status == cubic.status == "ok"
    assert line.rmse is not None and cubic.rmse is not None
    assert cubic.rmse < line.rmse


def test_weighted_loo_rmse_uses_held_out_weights() -> None:
    x = np.linspace(1.0, 6.0, 9)
    rng = np.random.default_rng(3)
    y = 2.0 + 0.5 * x + rng.normal(0.0, 0.4, x.size)
    w = np.linspace(0.5, 3.0, x.size)
    data = make_xy(x, y, w)

    cv = loo_cv(data, PolynomialFitter(0))

    assert cv.status == "ok" and cv.n_failed == 0
    residuals = np.array(
        [y[i] - np.average(np.delete(y, i), weights=np.delete(w, i)) for i in range(x.size)]
    )
    expected = float(np.sqrt(np.sum(w * residuals**2) / np.sum(w)))
    assert cv.rmse == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(sorted(cv.fold_rmse), sorted(np.abs(residuals)), rtol=1e-9)


@pytest.mark.parametrize("mode", ["raise", "status"])
def test_failed_fold_makes_result_incomplete_and_is_counted(
    mode: Literal["raise", "status"],
) -> None:
    data = noisy_quadratic(n=20)
    fitter = FailingFitter(fail_on=2, mode=mode)
    cv = k_fold_cv(data, fitter, folds=4, seed=0)

    assert cv.status == "incomplete"
    assert cv.n_failed == 1
    assert cv.rmse is None
    assert cv.fold_rmse[1] is None
    assert sum(value is not None for value in cv.fold_rmse) == 3
    assert len(cv.reasons) == 1
    assert "fold 1" in cv.reasons[0]
    if mode == "raise":
        assert "RuntimeError" in cv.reasons[0]
        assert "deliberate fold failure" in cv.reasons[0]
    else:
        assert "not_converged" in cv.reasons[0]
        assert "deliberate failure" in cv.reasons[0]


def test_fold_without_degrees_of_freedom_is_counted_as_failed() -> None:
    data = make_xy([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.9, 3.2])
    cv = loo_cv(data, PolynomialFitter(2))  # 3 parameters, but each fold keeps 3 points

    assert cv.status == "incomplete"
    assert cv.n_failed == 4
    assert cv.fold_rmse == [None, None, None, None]
    assert cv.rmse is None
    assert all("training point(s)" in reason for reason in cv.reasons)


@pytest.mark.parametrize("folds", [1, 0, -2, 13, 2.5, True])
def test_invalid_folds_rejected(folds: Any) -> None:
    data = noisy_quadratic(n=12)
    with pytest.raises(StatisticsError) as info:
        k_fold_cv(data, PolynomialFitter(1), folds=folds)
    assert info.value.code == "INVALID_FOLDS"


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_rejected(seed: Any) -> None:
    data = noisy_quadratic(n=8)
    with pytest.raises(StatisticsError) as info:
        k_fold_cv(data, PolynomialFitter(1), folds=2, seed=seed)
    assert info.value.code == "INVALID_SEED"


def test_loo_rejects_fewer_than_two_active_points() -> None:
    data = make_xy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    with pytest.raises(StatisticsError) as info:
        loo_cv(data, PolynomialFitter(0))
    assert info.value.code == "INVALID_FOLDS"


def test_nonlinear_fitter_warm_start_converges_where_cold_start_would_not() -> None:
    data = shifted_log_data()
    cold = NonlinearFitter(SHIFTED_LOG, config=FitConfig(n_starts=1))
    warm = NonlinearFitter(SHIFTED_LOG, config=FitConfig(n_starts=1), start=WARM_START)

    assert cold.fit(data).status is not FitStatus.CONVERGED
    warm_fit = warm.fit(data)
    assert warm_fit.status is FitStatus.CONVERGED
    assert warm_fit.initial_params == WARM_START

    cold_cv = k_fold_cv(data, cold, folds=4, seed=0)
    warm_cv = k_fold_cv(data, warm, folds=4, seed=0)
    assert cold_cv.status == "incomplete"
    assert cold_cv.n_failed == cold_cv.folds == 4
    assert cold_cv.rmse is None
    assert warm_cv.status == "ok" and warm_cv.n_failed == 0
    assert warm_cv.rmse is not None and warm_cv.rmse < 0.1


def test_start_overrides_config_initial() -> None:
    data = shifted_log_data()
    fitter = NonlinearFitter(
        SHIFTED_LOG,
        config=FitConfig(initial={"a": 9.0, "b": 9.0, "c": 9.0}),
        start=WARM_START,
    )
    result = fitter.fit(data)
    assert result.status is FitStatus.CONVERGED
    assert result.initial_params == WARM_START


def test_result_is_frozen_and_documents_itself() -> None:
    cv = k_fold_cv(noisy_quadratic(n=10), PolynomialFitter(1), folds=2)
    assert isinstance(cv, CrossValidation)
    assert "failed" in CrossValidation.__doc__.lower()
    with pytest.raises(ValueError):
        cv.rmse = 0.0  # type: ignore[misc]
