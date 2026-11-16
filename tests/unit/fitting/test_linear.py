"""Tests for fit_polynomial: weights, origin, invalid input, edge cases, serialisation."""

import json
import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from curvestudio.data import XY
from curvestudio.fitting import FitResult, FitStatus, fit_polynomial


def make_xy(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    weights: list[float] | None = None,
) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="test-fingerprint",
    )


def test_simple_linear_hand_computed() -> None:
    # x = 1..4, y = 2, 4, 5, 8: slope 1.9, intercept 0, sse 0.7 -> hand-checked values.
    res = fit_polynomial(make_xy([1, 2, 3, 4], [2, 4, 5, 8]), 1)
    assert res.status is FitStatus.CONVERGED
    assert res.model_id == "poly1"
    assert res.method == "ols"
    assert res.loss == "linear"
    assert res.n_starts == 1
    assert res.initial_params == {}
    assert res.provenance.dataset_fingerprint == "test-fingerprint"
    assert res.provenance.seed == 0
    assert [p.name for p in res.parameters] == ["b0", "b1"]
    assert res.parameters[0].value == pytest.approx(0.0, abs=1e-12)
    assert res.parameters[1].value == pytest.approx(1.9, abs=1e-12)
    assert res.stats is not None
    assert res.stats.sse == pytest.approx(0.7, abs=1e-12)
    assert res.stats.df == 2
    # Var(b1) = s2 / Sxx with s2 = 0.35, Sxx = 5.
    assert res.parameters[1].std_err == pytest.approx(np.sqrt(0.35 / 5.0), rel=1e-12)
    assert res.fitted == pytest.approx([1.9, 3.8, 5.7, 7.6], abs=1e-12)
    assert res.residuals == pytest.approx([0.1, 0.2, -0.7, 0.4], abs=1e-12)


def test_confidence_interval_uses_student_t_with_df() -> None:
    res = fit_polynomial(make_xy([1, 2, 3, 4], [2, 4, 5, 8]), 1, confidence_level=0.9)
    t = scipy_stats.t.ppf(0.95, 2)
    p = res.parameters[1]
    assert p.std_err is not None and p.ci_low is not None and p.ci_high is not None
    assert p.ci_low == pytest.approx(p.value - t * p.std_err, rel=1e-12)
    assert p.ci_high == pytest.approx(p.value + t * p.std_err, rel=1e-12)


def test_covariance_and_correlation_consistent() -> None:
    res = fit_polynomial(make_xy([1, 2, 3, 4, 5], [1.1, 3.9, 9.2, 15.8, 25.1]), 2)
    assert res.covariance is not None and res.correlation is not None
    cov = np.array(res.covariance)
    corr = np.array(res.correlation)
    se = np.array([p.std_err for p in res.parameters], dtype=float)
    assert np.allclose(np.sqrt(np.diag(cov)), se, rtol=1e-12)
    assert np.allclose(cov, cov.T, rtol=1e-10, atol=0.0)
    assert np.all(np.diag(corr) == 1.0)
    assert np.allclose(corr, cov / np.outer(se, se), rtol=1e-10)
    assert np.all(np.abs(corr) <= 1.0)


def test_weighted_fit_matches_lstsq_on_sqrt_weight_rows() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 10.0]
    w = [1.0, 1.0, 1.0, 100.0]
    res = fit_polynomial(make_xy(x, y, w), 1)
    root_w = np.sqrt(np.array(w))
    design = np.column_stack([np.ones(4), np.array(x)]) * root_w[:, None]
    expected, *_ = np.linalg.lstsq(design, np.array(y) * root_w, rcond=None)
    assert res.method == "wls"
    assert [p.value for p in res.parameters] == pytest.approx(expected.tolist(), rel=1e-12)
    assert res.stats is not None
    resid = np.array(y) - (expected[0] + expected[1] * np.array(x))
    assert res.stats.sse == pytest.approx(float(np.sum(np.array(w) * resid**2)), rel=1e-12)
    # Covariance is s2 * inv(X'WX).
    xtwx = design.T @ design
    s2 = res.stats.sse / res.stats.df
    assert np.allclose(np.array(res.covariance), s2 * np.linalg.inv(xtwx), rtol=1e-10)


def test_zero_weight_points_are_ignored_but_still_reported() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.1, 3.9, 6.2, 8.0, 100.0]  # last point is an outlier given weight 0
    res = fit_polynomial(make_xy(x, y, [1, 1, 1, 1, 0]), 1)
    ref = fit_polynomial(make_xy(x[:4], y[:4]), 1)
    assert [p.value for p in res.parameters] == pytest.approx([p.value for p in ref.parameters])
    assert res.stats is not None and ref.stats is not None
    assert res.stats.n == 4 and res.stats.df == 2
    assert res.stats.residual_se == pytest.approx(ref.stats.residual_se)
    assert np.allclose(np.array(res.covariance), np.array(ref.covariance), rtol=1e-10)
    assert len(res.fitted) == 5 and len(res.residuals) == 5
    assert res.residuals[4] == pytest.approx(100.0 - res.fitted[4])


def test_zero_weights_leaving_too_few_active_points_is_invalid_input() -> None:
    res = fit_polynomial(make_xy([1, 2, 3], [1, 2, 3], [1, 1, 0]), 1)
    assert res.status is FitStatus.INVALID_INPUT


def test_through_origin_recovers_slope() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    res = fit_polynomial(make_xy(x, [2 * v for v in x]), 1, through_origin=True)
    assert res.status is FitStatus.CONVERGED
    assert res.model_id == "poly1_origin"
    assert [p.name for p in res.parameters] == ["b1"]
    assert res.parameters[0].value == pytest.approx(2.0, abs=1e-12)
    assert res.stats is not None
    assert res.stats.r2_definition == "uncentered"
    assert res.stats.k == 1 and res.stats.df == 3
    assert res.equation == "y = b1*x"


def test_through_origin_quadratic_has_no_intercept() -> None:
    x = np.arange(1.0, 8.0)
    res = fit_polynomial(make_xy(x, 3 * x - 0.5 * x**2), 2, through_origin=True)
    assert [p.name for p in res.parameters] == ["b1", "b2"]
    assert [p.value for p in res.parameters] == pytest.approx([3.0, -0.5], abs=1e-10)
    assert res.model_id == "poly2_origin"
    assert res.equation == "y = b1*x + b2*x**2"
    assert res.equation_fitted == "y = 3*x + (-0.5)*x**2"
    assert res.fitted == pytest.approx((3 * x - 0.5 * x**2).tolist(), abs=1e-10)


def test_equations_use_names_and_substituted_values() -> None:
    res = fit_polynomial(make_xy([0, 1, 2, 3], [1, 3, 9, 19]), 2)
    assert res.equation == "y = b0 + b1*x + b2*x**2"
    assert res.equation_fitted is not None
    assert res.equation_fitted.startswith("y = ")
    assert "b0" not in res.equation_fitted
    assert fit_polynomial(make_xy([1, 2, 3], [4, 4, 4]), 0).equation == "y = b0"


def test_degree_zero_is_the_mean() -> None:
    res = fit_polynomial(make_xy([1, 2, 3, 4], [1, 2, 3, 6]), 0)
    assert res.parameters[0].value == pytest.approx(3.0)
    assert res.stats is not None and res.stats.k == 1 and res.stats.df == 3


@pytest.mark.parametrize(("n", "degree"), [(3, 3), (4, 4), (2, 2), (1, 1)])
def test_n_not_greater_than_k_is_invalid_input(n: int, degree: int) -> None:
    x = np.arange(1.0, n + 1)
    res = fit_polynomial(make_xy(x, x**2), degree)
    assert res.status is FitStatus.INVALID_INPUT
    assert "observation" in res.termination_reason
    assert res.parameters == ()
    assert res.covariance is None and res.correlation is None and res.stats is None
    assert res.fitted == [] and res.residuals == []
    assert res.equation == "y = " + " + ".join(
        ["b0", "b1*x", *(f"b{p}*x**{p}" for p in range(2, degree + 1))][: degree + 1]
    )
    assert res.equation_fitted is None
    FitResult.from_json(res.to_json())  # still serialisable


def test_exact_fit_gives_zero_covariance_not_nan() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_polynomial(make_xy([0, 1, 2, 3, 4], [1, 3, 5, 7, 9]), 1)
    assert res.status is FitStatus.CONVERGED
    assert res.stats is not None and res.stats.r2 == 1.0
    assert res.stats.sse < 1e-24
    assert res.covariance is not None
    for p in res.parameters:
        assert p.std_err is not None and np.isfinite(p.std_err)
        assert p.ci_low is not None and p.ci_high is not None
    FitResult.from_json(res.to_json())


def test_perfectly_exact_data_has_zero_se_and_collapsed_ci() -> None:
    # A constant y fitted by its mean is reproduced bit-for-bit, so sse is exactly 0.
    res = fit_polynomial(make_xy([1, 2, 4, 8], [5, 5, 5, 5]), 0)
    assert res.stats is not None and res.stats.sse == 0.0 and res.stats.r2 == 1.0
    assert res.covariance == [[0.0]]
    assert res.correlation is None
    p = res.parameters[0]
    assert p.std_err == 0.0 and p.ci_low == p.value == p.ci_high
    FitResult.from_json(res.to_json())


def test_constant_x_is_rank_deficient_and_failed() -> None:
    res = fit_polynomial(make_xy([2, 2, 2, 2, 2], [1, 2, 3, 4, 5]), 1)
    assert res.status is FitStatus.FAILED
    assert "rank" in res.termination_reason
    assert res.covariance is None and res.correlation is None
    assert all(p.std_err is None and p.ci_low is None for p in res.parameters)
    FitResult.from_json(res.to_json())


def test_too_few_distinct_x_is_failed() -> None:
    res = fit_polynomial(make_xy([1, 1, 2, 2, 2, 1], [1, 2, 3, 4, 5, 6]), 2)
    assert res.status is FitStatus.FAILED
    assert "rank" in res.termination_reason
    assert res.covariance is None


def test_all_zero_x_through_origin_is_failed() -> None:
    res = fit_polynomial(make_xy([0, 0, 0, 0], [1, 2, 3, 4]), 1, through_origin=True)
    assert res.status is FitStatus.FAILED


def test_degree_seven_and_above_warn_high_order() -> None:
    x = np.linspace(0.0, 1.0, 30)
    res = fit_polynomial(make_xy(x, np.sin(3 * x)), 7)
    assert "HIGH_ORDER_POLYNOMIAL: unstable outside the data range" in res.warnings
    low = fit_polynomial(make_xy(x, np.sin(3 * x)), 5)
    assert not any(w.startswith("HIGH_ORDER") for w in low.warnings)
    six = fit_polynomial(make_xy(x, np.sin(3 * x)), 6)
    assert any(w.startswith("HIGH_ORDER") for w in six.warnings)


def test_well_conditioned_fit_has_no_warnings() -> None:
    assert fit_polynomial(make_xy([1, 2, 3, 4], [2, 4, 5, 8]), 1).warnings == ()


def test_wide_scale_x_range_degree_3_is_flagged_or_accurate() -> None:
    x = np.array([1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9])
    true = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.polynomial.polynomial.polyval(x, true) * 1.0
    res = fit_polynomial(make_xy(x, y), 3)
    accurate = res.status is FitStatus.CONVERGED and [p.value for p in res.parameters] == (
        pytest.approx(true.tolist(), rel=1e-6)
    )
    assert "ILL_CONDITIONED" in res.warnings or accurate
    FitResult.from_json(res.to_json())


def test_ill_conditioned_polynomial_on_offset_x_is_warned() -> None:
    x = 1000.0 + np.linspace(0.0, 1.0, 20)
    res = fit_polynomial(make_xy(x, np.sin(x)), 5)
    assert "ILL_CONDITIONED" in res.warnings


def test_overflowing_design_matrix_fails_cleanly() -> None:
    x = np.array([1e200, 2e200, 3e200, 4e200])
    res = fit_polynomial(make_xy(x, [1, 2, 3, 4]), 2)
    assert res.status is FitStatus.FAILED
    assert "overflow" in res.termination_reason


def test_result_round_trips_through_json() -> None:
    res = fit_polynomial(make_xy([1, 2, 3, 4, 5], [1.2, 1.9, 3.2, 3.8, 5.1]), 1)
    again = FitResult.from_json(res.to_json())
    assert again == res
    json.loads(res.to_json())


def test_no_numpy_warnings_escape_on_ordinary_fit() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_polynomial(make_xy([1, 2, 3, 4, 5, 6], [1, 4, 9, 16, 25, 36.5]), 2)


@pytest.mark.parametrize("degree", [-1, 1.5, True, "2"])
def test_invalid_degree_raises(degree: object) -> None:
    with pytest.raises(ValueError, match="degree"):
        fit_polynomial(make_xy([1, 2, 3], [1, 2, 3]), degree)  # type: ignore[arg-type]


def test_degree_zero_through_origin_raises() -> None:
    with pytest.raises(ValueError, match="degree"):
        fit_polynomial(make_xy([1, 2, 3], [1, 2, 3]), 0, through_origin=True)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_confidence_level_raises(level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        fit_polynomial(make_xy([1, 2, 3], [1, 2, 3]), 1, confidence_level=level)
