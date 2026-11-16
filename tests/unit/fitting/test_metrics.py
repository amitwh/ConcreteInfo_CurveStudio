"""Tests for fit_statistics: hand-computed values, edge cases and warning hygiene."""

import math
import warnings

import numpy as np
import pytest

from curvestudio.fitting import FitStatistics, fit_statistics

Y = np.array([1.0, 2.0, 3.0, 4.0])
FITTED = np.array([1.1, 1.9, 3.2, 3.8])


def test_known_case_matches_hand_calculation() -> None:
    s = fit_statistics(Y, FITTED, n_params=2)
    assert isinstance(s, FitStatistics)
    assert s.n == 4
    assert s.k == 2
    assert s.df == 2
    assert s.sse == pytest.approx(0.1, abs=1e-12)
    assert s.mse == pytest.approx(0.025, abs=1e-12)
    assert s.rmse == pytest.approx(math.sqrt(0.025), abs=1e-12)
    # |r| = 0.1, 0.1, 0.2, 0.2 -> 0.6/4. (The task brief's 0.125 is an arithmetic slip.)
    assert s.mae == pytest.approx(0.15, abs=1e-12)
    assert s.r2 == pytest.approx(1 - 0.1 / 5, abs=1e-12)
    assert s.adj_r2 == pytest.approx(1 - 0.02 * 3 / 2, abs=1e-12)
    assert s.residual_se == pytest.approx(math.sqrt(0.1 / 2), abs=1e-12)
    assert s.r2_definition == "centered"


def test_noiseless_is_perfect_and_warning_free() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = fit_statistics(Y, Y.copy(), n_params=2)
    assert s.sse == 0.0
    assert s.r2 == 1.0
    assert s.adj_r2 == 1.0
    assert s.rmse == 0.0
    assert s.residual_se == 0.0


def test_n_equals_k_has_no_degrees_of_freedom() -> None:
    s = fit_statistics(Y[:2], FITTED[:2], n_params=2)
    assert s.df == 0
    assert s.residual_se is None
    assert s.adj_r2 is None
    assert s.r2 is not None


def test_more_parameters_than_points_has_negative_df() -> None:
    s = fit_statistics(Y[:2], FITTED[:2], n_params=3)
    assert s.df == -1
    assert s.residual_se is None
    assert s.adj_r2 is None


def test_weighted_known_case() -> None:
    w = np.array([1.0, 2.0, 3.0, 4.0])
    s = fit_statistics(Y, FITTED, n_params=2, weights=w)
    # sse = 1*0.01 + 2*0.01 + 3*0.04 + 4*0.04 = 0.31
    assert s.sse == pytest.approx(0.31, abs=1e-12)
    # weighted mean = (1+4+9+16)/10 = 3.0; sst = 1*4 + 2*1 + 3*0 + 4*1 = 10
    assert s.r2 == pytest.approx(1 - 0.31 / 10, abs=1e-12)
    # n and df are the unweighted counts
    assert s.mse == pytest.approx(0.31 / 4, abs=1e-12)
    assert s.adj_r2 == pytest.approx(1 - (1 - s.r2) * 3 / 2, abs=1e-12)
    assert s.residual_se == pytest.approx(math.sqrt(0.31 / 2), abs=1e-12)
    # mae mirrors mse: weights multiply |r|, divided by n
    assert s.mae == pytest.approx((0.1 + 2 * 0.1 + 3 * 0.2 + 4 * 0.2) / 4, abs=1e-12)


def test_unit_weights_equal_unweighted() -> None:
    a = fit_statistics(Y, FITTED, n_params=2)
    b = fit_statistics(Y, FITTED, n_params=2, weights=np.ones(4))
    assert a == b


def test_zero_weight_points_contribute_nothing_but_count_in_n() -> None:
    w = np.array([1.0, 1.0, 1.0, 0.0])
    s = fit_statistics(Y, FITTED, n_params=2, weights=w)
    assert s.n == 4
    assert s.df == 2
    assert s.sse == pytest.approx(0.01 + 0.01 + 0.04, abs=1e-12)


def test_constant_y_perfect_fit_is_r2_one_without_warnings() -> None:
    y = np.full(5, 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = fit_statistics(y, y.copy(), n_params=1)
    assert s.r2 == 1.0
    assert s.adj_r2 == 1.0


def test_constant_y_imperfect_fit_has_undefined_r2() -> None:
    y = np.full(5, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = fit_statistics(y, y + 0.5, n_params=1)
    assert s.sse > 0
    assert s.r2 is None
    assert s.adj_r2 is None


def test_through_origin_uses_uncentred_total_sum_of_squares() -> None:
    s = fit_statistics(Y, FITTED, n_params=1, through_origin=True)
    # sst = 1 + 4 + 9 + 16 = 30
    assert s.r2_definition == "uncentered"
    assert s.r2 == pytest.approx(1 - 0.1 / 30, abs=1e-12)
    # uncentred adjusted R2 uses n rather than n-1
    assert s.adj_r2 == pytest.approx(1 - (1 - s.r2) * 4 / 3, abs=1e-12)


def test_through_origin_constant_y_is_not_degenerate() -> None:
    y = np.full(4, 3.0)
    s = fit_statistics(y, y * 0.9, n_params=1, through_origin=True)
    assert s.r2 == pytest.approx(1 - (4 * 0.09) / 36, abs=1e-12)


def test_all_zero_y_through_origin_is_degenerate() -> None:
    y = np.zeros(3)
    perfect = fit_statistics(y, y.copy(), n_params=1, through_origin=True)
    assert perfect.r2 == 1.0
    bad = fit_statistics(y, y + 1.0, n_params=1, through_origin=True)
    assert bad.r2 is None


def test_single_point_zero_params_is_clean() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = fit_statistics(np.array([2.0]), np.array([2.0]), n_params=0)
    assert s.r2 == 1.0
    assert s.df == 1


@pytest.mark.parametrize(
    ("y", "fitted", "n_params", "weights"),
    [
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), 1, None),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf]), 1, None),
        (np.array([1.0, 2.0]), np.array([1.0]), 1, None),
        (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), 1, None),
        (np.array([]), np.array([]), 0, None),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), -1, None),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1, np.array([1.0, np.nan])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1, np.array([1.0, -1.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1, np.array([0.0, 0.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1, np.array([1.0])),
    ],
)
def test_invalid_inputs_raise_value_error(
    y: np.ndarray, fitted: np.ndarray, n_params: int, weights: np.ndarray | None
) -> None:
    with pytest.raises(ValueError):
        fit_statistics(y, fitted, n_params=n_params, weights=weights)


def test_overflow_is_reported_not_warned() -> None:
    y = np.array([1e200, 2e200])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite"):
            fit_statistics(y, -y, n_params=1)
