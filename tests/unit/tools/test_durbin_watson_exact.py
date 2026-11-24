"""Durbin-Watson p-values exact for the fitted model: each part is checked against what it
must reproduce.

Imhof's integral is set against the chi-square (equal weights) and against a ratio of
chi-squares (weights of both signs); the critical values against their own p-values; and the
whole chain (eigenvalues, integral, root finder) against simulated errors for three designs.
"""

import numpy as np
import pytest
from scipy import stats

from curvestudio.tools import ToolError
from curvestudio.tools.durbin_watson import (
    DW_EXACT_MAX_N,
    RANK_RTOL,
    DurbinWatsonCritical,
    DurbinWatsonTest,
    durbin_watson_critical_values,
    durbin_watson_test,
    quadratic_form_cdf,
)

X = np.linspace(0.0, 10.0, 40)
"""Forty equally spaced x values: the order the residuals are taken in."""
DRAWS = 4000
SEED = 20261124
"""An ordinary seed for the simulated errors, taken as it comes."""
QUANTILES = (0.001, 0.1, 0.5, 0.9, 0.999)


def intercept_only() -> np.ndarray:
    return np.ones((40, 1))


def intercept_and_regressor() -> np.ndarray:
    return np.column_stack([np.ones(40), X])


def through_the_origin() -> np.ndarray:
    """A straight line through the origin: one regressor, no intercept."""
    return X[:, None]


def exponential_jacobian(a: float = 2.0, b: float = 0.3) -> np.ndarray:
    """The Jacobian of ``a*exp(b*x)`` at the estimates ``a`` and ``b``."""
    curve = np.exp(b * X)
    return np.column_stack([curve, a * X * curve])


def column_space(design: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the column space of ``design``."""
    return np.linalg.svd(design, full_matrices=False)[0]


def projected_errors(design: np.ndarray, draws: int) -> np.ndarray:
    """Standard-normal error vectors, each projected on the space orthogonal to the design."""
    rng = np.random.default_rng(SEED)
    errors = rng.normal(0.0, 1.0, (draws, design.shape[0]))
    basis = column_space(design)
    return errors - (errors @ basis) @ basis.T


def statistic_rows(residuals: np.ndarray) -> np.ndarray:
    """The Durbin-Watson statistic of every row of ``residuals``."""
    differences = np.diff(residuals, axis=1)
    return (differences**2).sum(axis=1) / (residuals**2).sum(axis=1)


# -- Imhof's integral -----------------------------------------------------------------------------


@pytest.mark.parametrize("count", (1, 2, 3, 5, 10))
def test_the_quadratic_form_cdf_matches_chi_square_for_equal_weights(count: int) -> None:
    points = [float(stats.chi2.ppf(quantile, count)) for quantile in QUANTILES]

    for point in points:
        expected = float(stats.chi2.cdf(point, count))
        found = quadratic_form_cdf(np.ones(count), point)
        assert found == pytest.approx(expected, rel=1e-8), (count, point, found, expected)

    # All weights equal to c: the same sum of squares, times c.
    scale = 3.7
    for point in points:
        expected = float(stats.chi2.cdf(point, count))
        found = quadratic_form_cdf(np.full(count, scale), scale * point)
        assert found == pytest.approx(expected, rel=1e-8), (count, point, found, expected)


def test_the_quadratic_form_cdf_matches_a_ratio_of_chi_squares_at_zero() -> None:
    # z1^2 + z2^2 is Gamma(1, 2) and z3^2 is chi-square(1), so the chance that the first is
    # the smaller is the F(2, 1) distribution at one half.
    ratio = float(stats.f.cdf(0.5, 2, 1))
    weights = np.array([1.0, 1.0, -1.0])
    assert quadratic_form_cdf(weights, 0.0) == pytest.approx(ratio, rel=1e-8)
    # The negated weights give the complementary chance, read from its own integral.
    assert quadratic_form_cdf(-weights, 0.0) == pytest.approx(1.0 - ratio, rel=1e-8)
    # A zero weight takes no part; with only zero weights the form is identically zero.
    assert quadratic_form_cdf(np.array([1.0, 0.0, -1.0]), 0.0) == pytest.approx(0.5, abs=1e-12)
    assert quadratic_form_cdf(np.zeros(3), 0.0) == 1.0
    assert quadratic_form_cdf(np.zeros(3), -0.5) == 0.0


# -- the test -------------------------------------------------------------------------------------


def test_the_p_values_are_the_two_tails_of_the_statistic() -> None:
    design = intercept_and_regressor()
    test = durbin_watson_test(1.7, design)
    assert isinstance(test, DurbinWatsonTest)
    assert test.df == 38
    assert test.statistic == 1.7
    assert 0.0 < test.positive < 0.5 < test.negative < 1.0
    # The two tails share the whole chance.
    assert test.positive + test.negative == pytest.approx(1.0, abs=1e-9)
    # Same input, same output.
    assert test == durbin_watson_test(1.7, design)

    far_below = durbin_watson_test(0.0, design)
    far_above = durbin_watson_test(4.0, design)
    assert far_below is not None and far_above is not None
    assert far_below.positive == 0.0 and far_below.negative == 1.0
    assert far_above.positive == 1.0 and far_above.negative == 0.0


def test_the_critical_values_bracket_their_own_p_values() -> None:
    design = intercept_and_regressor()
    critical = durbin_watson_critical_values(design, alpha=0.05)
    assert isinstance(critical, DurbinWatsonCritical)
    assert critical.df == 38 and critical.alpha == 0.05
    assert critical.lower < critical.upper

    at_lower = durbin_watson_test(critical.lower, design)
    at_upper = durbin_watson_test(critical.upper, design)
    assert at_lower is not None and at_upper is not None
    assert at_lower.positive == pytest.approx(0.05, abs=0.005)
    assert at_upper.negative == pytest.approx(0.05, abs=0.005)


def test_the_critical_values_follow_the_design() -> None:
    # With an intercept the residuals leave out the trend, a direction in which the statistic
    # is large; through the origin they may still carry a constant, a direction in which it is
    # small. So both critical values are the lower for the line through the origin.
    with_intercept = durbin_watson_critical_values(intercept_and_regressor())
    without = durbin_watson_critical_values(through_the_origin())
    assert with_intercept is not None and without is not None
    assert with_intercept.lower > without.lower
    assert with_intercept.upper > without.upper


@pytest.mark.parametrize(
    "design",
    (intercept_only(), intercept_and_regressor(), exponential_jacobian()),
    ids=("intercept", "intercept-and-regressor", "exponential"),
)
def test_the_simulated_rejection_rate_is_the_nominal_one(design: np.ndarray) -> None:
    errors = projected_errors(design, DRAWS)
    assert np.allclose(errors @ column_space(design), 0.0, atol=1e-10)
    statistic = statistic_rows(errors)

    critical = durbin_watson_critical_values(design, alpha=0.05)
    assert critical is not None
    # A p-value is below 0.05 exactly where the statistic has crossed its critical value.
    lower_rate = float(np.mean(statistic < critical.lower))
    upper_rate = float(np.mean(statistic > critical.upper))
    assert lower_rate == pytest.approx(0.05, abs=0.01), lower_rate
    assert upper_rate == pytest.approx(0.05, abs=0.01), upper_rate

    # The p-values themselves say the same, on a share of the same draws taken one by one.
    df = design.shape[0] - design.shape[1]
    for value in statistic[:120]:
        test = durbin_watson_test(float(value), design)
        assert test is not None and test.df == df
        assert (test.positive < 0.05) == (value < critical.lower), value
        assert (test.negative < 0.05) == (value > critical.upper), value
        assert test.positive + test.negative == pytest.approx(1.0, abs=1e-9)


# -- what is refused ------------------------------------------------------------------------------


def test_a_dependent_design_and_a_large_one() -> None:
    duplicated = np.column_stack([np.ones(40), X, X])
    with pytest.raises(ToolError) as dependent:
        durbin_watson_test(1.5, duplicated)
    assert dependent.value.code == "RANK_DEFICIENT_DESIGN"
    with pytest.raises(ToolError) as dependent_values:
        durbin_watson_critical_values(duplicated)
    assert dependent_values.value.code == "RANK_DEFICIENT_DESIGN"

    # The line is RANK_RTOL: a third column whose singular value is that fraction of the
    # largest makes the design dependent; twice that fraction, and it stands on its own.
    trend = intercept_and_regressor()
    curved = X**2 - trend @ np.linalg.lstsq(trend, X**2, rcond=None)[0]
    largest = float(np.linalg.svd(trend, compute_uv=False)[0])
    for factor, refused in ((0.5, True), (2.0, False)):
        small = factor * RANK_RTOL * largest * curved / float(np.linalg.norm(curved))
        jacobian = np.column_stack([trend, small])
        if refused:
            with pytest.raises(ToolError) as at_the_line:
                durbin_watson_test(1.5, jacobian)
            assert at_the_line.value.code == "RANK_DEFICIENT_DESIGN"
        else:
            assert isinstance(durbin_watson_test(1.5, jacobian), DurbinWatsonTest)

    too_many = np.column_stack([np.ones(DW_EXACT_MAX_N + 1), np.linspace(0.0, 1.0, 1001)])
    assert durbin_watson_test(1.5, too_many) is None
    assert durbin_watson_critical_values(too_many) is None
    at_the_limit = too_many[:DW_EXACT_MAX_N]
    assert isinstance(durbin_watson_test(1.5, at_the_limit), DurbinWatsonTest)

    with pytest.raises(ToolError) as empty:
        durbin_watson_test(1.5, np.empty((0, 2)))
    assert empty.value.code == "NO_DATA"

    for alpha in (0.6, 0.0, -0.05, 0.5):
        with pytest.raises(ValueError):
            durbin_watson_critical_values(intercept_and_regressor(), alpha=alpha)
