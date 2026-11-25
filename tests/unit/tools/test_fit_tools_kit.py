"""The fit-tool kit on real fits: a target, a grid, a band's half-width and the curve.

Every fit here actually runs through the workflow. The prediction half-width is checked
against the delta-method formula written out in the test, not against the function it
wraps, so a change in the shared band code cannot make the kit's number pass alone.
"""

import time
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sps

from curvestudio.data import XY
from curvestudio.statistics import build_view
from curvestudio.tools._fit_tools import (
    CancelledFitTool,
    FitTarget,
    cancel_checker,
    curve_of,
    evaluation_grid,
    prediction_half_width,
    resolve_target,
)
from curvestudio.workflow import FitSpec, PolynomialRef, RegisteredRef, fit_dataset

X = np.linspace(0.0, 10.0, 30)
NOISE = np.random.default_rng(0).normal(0.0, 0.1, X.size)
"""The plain seed-0 noise: the generator's first draw, taken as it comes."""
LINE = FitSpec(model=PolynomialRef(degree=1))
QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
GAUSSIAN = FitSpec(model=RegisteredRef(id="gaussian"))


def xy_of(x: Sequence[float], y: Sequence[float]) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None,
        x_name="x",
        y_name="y",
        fingerprint="fp-fit-tools",
    )


def line_xy() -> XY:
    return xy_of(X, 2.0 + 0.5 * X + NOISE)


def quadratic_xy() -> XY:
    return xy_of(X, 1.0 + 0.5 * X - 0.03 * X**2 + NOISE)


def gaussian_xy() -> XY:
    x = np.linspace(-2.0, 6.0, 30)
    return xy_of(x, 5.0 * np.exp(-((x - 2.0) ** 2) / (2.0 * 1.2**2)) + NOISE)


def values_of(target: FitTarget) -> list[float]:
    return [parameter.value for parameter in target.result.parameters]


def test_a_bundle_a_bare_result_and_a_refit_agree() -> None:
    xy = line_xy()
    bundle = fit_dataset(xy, LINE)
    given = resolve_target(xy, LINE, bundle=bundle)
    bare = resolve_target(xy, LINE, bundle.result)
    refit = resolve_target(xy, LINE)

    assert given.refitted is False
    assert bare.refitted is False
    assert refit.refitted is True
    assert given.bundle is bundle
    assert bare.bundle is None
    assert refit.bundle is not None
    assert given.note is None
    assert bare.note is None
    assert isinstance(refit.note, str)

    grid = evaluation_grid(xy)
    for target in (bare, refit):
        assert values_of(target) == pytest.approx(values_of(given), abs=1e-12)
        assert_allclose(curve_of(target, grid), curve_of(given, grid), rtol=0.0, atol=1e-12)


def test_the_target_keeps_the_bounds_the_fit_ran_with() -> None:
    gaussian = resolve_target(gaussian_xy(), GAUSSIAN)
    assert gaussian.bounds == {"a": (None, None), "mu": (None, None), "s": (0.0, None)}

    narrowed = GAUSSIAN.model_copy(update={"bounds": {"mu": (0.0, 4.0)}})
    assert resolve_target(gaussian_xy(), narrowed).bounds["mu"] == (0.0, 4.0)

    # A polynomial declares no bound of its own.
    assert resolve_target(line_xy(), LINE).bounds == {}


def test_the_grid_spans_the_data_range_inclusively() -> None:
    xy = line_xy()
    grid = evaluation_grid(xy, points=51)

    assert grid.size == 51
    assert grid[0] == pytest.approx(float(xy.x.min()))
    assert grid[-1] == pytest.approx(float(xy.x.max()))
    assert np.all(np.isfinite(grid))
    assert np.all(np.diff(grid) > 0.0)


def test_the_grid_takes_the_range_it_is_given() -> None:
    grid = evaluation_grid(line_xy(), points=41, low=-2.0, high=12.0)

    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(12.0)
    assert grid.size == 41


def test_the_grid_is_deduplicated_when_the_range_is_too_narrow() -> None:
    # At 1e16 the spacing of the doubles is 2, so 200 linspace values over 4 units repeat.
    xy = xy_of([1e16, 1e16 + 2.0, 1e16 + 4.0], [0.0, 1.0, 2.0])
    grid = evaluation_grid(xy, points=200)

    assert grid.size < 200
    assert grid.size >= 2
    assert np.all(np.diff(grid) > 0.0)


@pytest.mark.parametrize(
    ("x", "message"),
    [([2.0, 2.0, 2.0], "low < high"), ([3.0], "at least 2 usable x values")],
)
def test_a_grid_needs_two_usable_x_values(x: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation_grid(xy_of(x, [0.0] * len(x)))


def test_a_grid_refuses_a_range_that_is_not_strictly_increasing() -> None:
    xy = line_xy()

    with pytest.raises(ValueError, match="low < high"):
        evaluation_grid(xy, low=5.0, high=5.0)
    with pytest.raises(ValueError, match="low < high"):
        evaluation_grid(xy, low=6.0, high=5.0)


def test_a_grid_refuses_fewer_than_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        evaluation_grid(line_xy(), points=1)


@pytest.mark.parametrize("level", [0.90, 0.95])
def test_the_prediction_half_width_is_the_delta_method_formula(level: float) -> None:
    xy = line_xy()
    view = build_view(xy, fit_dataset(xy, LINE).result)
    grid = evaluation_grid(xy, points=25)

    covariance = view.cov
    assert covariance is not None
    assert view.s2 is not None
    design = np.column_stack([np.ones_like(grid), grid])
    mean_variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    quantile = float(sps.t.ppf(1.0 - (1.0 - level) / 2.0, view.df))
    expected = quantile * np.sqrt(mean_variance + view.s2)

    assert_allclose(prediction_half_width(view, grid, level), expected, rtol=0.0, atol=1e-12)


def test_without_a_token_the_checker_does_nothing() -> None:
    check = cancel_checker(None)

    check()
    check()


def test_a_false_token_never_stops_the_tool() -> None:
    check = cancel_checker(lambda: False)

    for _ in range(3):
        check()


def test_a_true_token_stops_the_tool_at_the_next_check() -> None:
    allowed = False
    check = cancel_checker(lambda: allowed)

    check()
    allowed = True
    with pytest.raises(CancelledFitTool):
        check()


def test_the_checker_reads_the_token_each_time_it_is_called() -> None:
    states = iter([False, True])
    check = cancel_checker(lambda: next(states))

    check()
    with pytest.raises(CancelledFitTool):
        check()


def test_curve_of_a_polynomial_evaluates_its_fitted_coefficients() -> None:
    xy = quadratic_xy()
    target = resolve_target(xy, QUADRATIC)
    coefficients = np.array(values_of(target))
    grid = evaluation_grid(xy)

    assert target.model is None
    assert_allclose(
        curve_of(target, grid),
        np.polynomial.polynomial.polyval(grid, coefficients),
        rtol=0.0,
        atol=1e-12,
    )


def test_curve_of_a_library_model_is_its_evaluate_at_the_fitted_values() -> None:
    xy = gaussian_xy()
    target = resolve_target(xy, GAUSSIAN)
    model = target.model
    assert model is not None
    fitted = {parameter.name: parameter.value for parameter in target.result.parameters}
    grid = evaluation_grid(xy)

    assert_allclose(curve_of(target, grid), model.evaluate(grid, **fitted), rtol=0.0, atol=1e-12)


def test_a_small_fit_resolves_well_under_a_hundred_milliseconds() -> None:
    cases: list[tuple[str, XY, FitSpec]] = [
        ("a line", line_xy(), LINE),
        ("a quadratic", quadratic_xy(), QUADRATIC),
        ("a library peak", gaussian_xy(), GAUSSIAN),
    ]
    for name, xy, spec in cases:
        resolve_target(xy, spec)  # compile the model untimed; the fit below is the subject
        started = time.perf_counter()
        target = resolve_target(xy, spec)
        elapsed = time.perf_counter() - started
        assert target.refitted is True, name
        assert elapsed < 0.1, f"{name} took {elapsed * 1000:.1f} ms"
