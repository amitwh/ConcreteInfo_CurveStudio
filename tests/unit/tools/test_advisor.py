"""The next-measurement advisor on real fits: the points, the narrowing, and the refusals.

Every fit actually runs through the workflow. The straight-line cases check the design a
textbook would give — the two ends of the range — and the quadratic case is sparse enough
that its centre point is a real choice: with the ends in place, the chosen interior point is
independently checked to be the one that most narrows the curvature parameter's standard
error. The standard errors after every step are checked against the fit's own, so a design
that claims a narrowing it does not buy cannot pass.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitResult
from curvestudio.statistics import build_view
from curvestudio.tools import ToolError, Verdict
from curvestudio.tools._fit_tools import evaluation_grid
from curvestudio.tools.advisor import AdvisorRequest, advise
from curvestudio.tools.stats import Estimate, MultiLinePlot, Table, ToolResult
from curvestudio.workflow import FitSpec, PolynomialRef, fit_dataset

X = np.linspace(0.0, 10.0, 30)
LINE_NOISE = np.random.default_rng(0).normal(0.0, 0.1, X.size)
LINE = FitSpec(model=PolynomialRef(degree=1))
QUADRATIC = FitSpec(model=PolynomialRef(degree=2))

QUADRATIC_X = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
"""A sparse quadratic sample, so that the interior of the range still has information to give."""

QUADRATIC_NOISE = np.random.default_rng(1).normal(0.0, 0.05, QUADRATIC_X.size)

CANDIDATES = 400
"""The default candidate count the grid-step assertions are written for."""

LOW = float(X.min())
HIGH = float(X.max())
STEP = (HIGH - LOW) / (CANDIDATES - 1)
"""One candidate grid step over the data's range."""


def _xy(x: Sequence[float], y: Sequence[float], fingerprint: str) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None,
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def line_xy() -> XY:
    return _xy(X, 2.0 + 0.5 * X + LINE_NOISE, "fp-advisor-line")


def quadratic_xy() -> XY:
    y = 1.0 + 0.5 * QUADRATIC_X - 0.03 * QUADRATIC_X**2 + QUADRATIC_NOISE
    return _xy(QUADRATIC_X, y, "fp-advisor-quadratic")


def _table(result: ToolResult, title: str) -> Table:
    return next(found for found in result.tables if found.title == title)


def _column(result: ToolResult, title: str, key: str) -> list[object]:
    found = _table(result, title)
    at = [column.key for column in found.columns].index(key)
    return [row[at] for row in found.rows]


def _points(result: ToolResult) -> list[float]:
    return [float(value) for value in _column(result, "Recommended points", "x")]


def _estimates(result: ToolResult) -> dict[str, float]:
    return {estimate.label: estimate.value for estimate in result.estimates}


@pytest.fixture(scope="module")
def line_fit() -> FitResult:
    return fit_dataset(line_xy(), LINE).result


@pytest.fixture(scope="module")
def quadratic_fit() -> FitResult:
    return fit_dataset(quadratic_xy(), QUADRATIC).result


@pytest.fixture(scope="module")
def line_two(line_fit: FitResult) -> ToolResult:
    return advise(
        line_xy(), LINE, line_fit, request=AdvisorRequest(points=2, candidates=CANDIDATES)
    )


@pytest.fixture(scope="module")
def line_three(line_fit: FitResult) -> ToolResult:
    return advise(
        line_xy(), LINE, line_fit, request=AdvisorRequest(points=3, candidates=CANDIDATES)
    )


@pytest.fixture(scope="module")
def quadratic_three(quadratic_fit: FitResult) -> ToolResult:
    return advise(quadratic_xy(), QUADRATIC, quadratic_fit, request=AdvisorRequest(points=3))


@pytest.fixture(scope="module")
def quadratic_parameter(quadratic_fit: FitResult) -> ToolResult:
    return advise(
        quadratic_xy(),
        QUADRATIC,
        quadratic_fit,
        request=AdvisorRequest(points=3, objective="parameter", parameter="b1"),
    )


# ------------------------------------------------------------------ the recommended points


def test_two_points_are_the_two_ends_of_a_straight_line(line_two: ToolResult) -> None:
    points = _points(line_two)

    assert len(points) == 2
    assert min(points) <= LOW + STEP, f"the low end is not chosen: {points}"
    assert max(points) >= HIGH - STEP, f"the high end is not chosen: {points}"


def test_three_points_keep_both_ends(line_three: ToolResult) -> None:
    points = _points(line_three)

    assert len(points) == 3
    assert min(points) <= LOW + STEP, f"the low end was dropped: {points}"
    assert max(points) >= HIGH - STEP, f"the high end was dropped: {points}"


def test_a_quadratic_is_measured_at_both_ends_and_near_the_centre(
    quadratic_three: ToolResult,
) -> None:
    points = _points(quadratic_three)
    interior = [point for point in points if LOW + STEP < point < HIGH - STEP]

    assert len(points) == 3
    assert min(points) <= LOW + STEP, f"the low end was dropped: {points}"
    assert max(points) >= HIGH - STEP, f"the high end was dropped: {points}"
    assert len(interior) == 1, f"exactly one point should be interior: {points}"
    assert abs(interior[0] - 5.0) <= STEP + 1e-9, f"the interior point is not central: {points}"


def test_the_centre_most_narrows_the_curvature_parameter(
    quadratic_three: ToolResult, quadratic_fit: FitResult
) -> None:
    """With the two ends in place, the chosen interior point is the best one for ``b2``."""
    points = _points(quadratic_three)
    ends = [min(points), max(points)]
    interior = next(point for point in points if LOW + STEP < point < HIGH - STEP)

    candidate_x = evaluation_grid(quadratic_xy(), points=CANDIDATES)
    view = build_view(quadratic_xy(), quadratic_fit)
    information = view.jacobian_at(view.x).T @ view.jacobian_at(view.x)
    for end in ends:
        row = view.jacobian_at(np.array([end]))[0]
        information = information + np.outer(row, row)
    assert view.s2 is not None

    def curvature_se(x: float) -> float:
        row = view.jacobian_at(np.array([x]))[0]
        covariance = view.s2 * np.linalg.pinv(information + np.outer(row, row), hermitian=True)
        return float(np.sqrt(covariance[2, 2]))

    inside = (candidate_x > LOW + STEP) & (candidate_x < HIGH - STEP)
    best = min(curvature_se(float(x)) for x in candidate_x[inside])
    chosen = curvature_se(interior)
    assert chosen <= best + 1e-12 * best, f"{chosen} is not the best interior SE {best}"


# ------------------------------------------------------------------------- the narrowing


def test_no_standard_error_rises_when_a_point_is_added(
    line_three: ToolResult, quadratic_three: ToolResult
) -> None:
    for result in (line_three, quadratic_three):
        before = [float(value) for value in _column(result, "Narrowing", "before")]
        rows = _table(result, "Recommended points").rows
        assert len(rows) == 3
        previous = before
        for row in rows:
            step_se = [float(value) for value in row[3:]]
            for value, earlier in zip(step_se, previous, strict=True):
                assert value <= earlier + 1e-9 * max(1.0, abs(earlier)), (
                    f"a standard error rose: {value} after {earlier}"
                )
            previous = step_se


def test_the_parameter_objective_narrows_its_parameter_at_least_as_much(
    quadratic_parameter: ToolResult, quadratic_three: ToolResult
) -> None:
    targeted = _estimates(quadratic_parameter)["SE of b1"]
    general = _estimates(quadratic_three)["SE of b1"]

    assert targeted <= general + 1e-15 * general, (
        f"the parameter objective gives {targeted}, the D-optimal one {general}"
    )


def test_the_parameter_objective_says_so_in_its_methods(
    quadratic_parameter: ToolResult,
) -> None:
    assert "minimising the variance of 'b1'" in quadratic_parameter.methods_fragment
    assert "greedy exchange" in quadratic_parameter.methods_fragment


# --------------------------------------------------------------------- the result's blocks


def test_a_result_has_the_expected_blocks(quadratic_three: ToolResult) -> None:
    assert type(quadratic_three) is ToolResult
    assert quadratic_three.tool == "advisor"
    assert [found.title for found in quadratic_three.tables] == [
        "Recommended points",
        "Narrowing",
    ]
    assert [column.label for column in _table(quadratic_three, "Recommended points").columns] == [
        "Step",
        "x",
        "Replicates",
        "SE of b0",
        "SE of b1",
        "SE of b2",
    ]
    assert [column.label for column in _table(quadratic_three, "Narrowing").columns] == [
        "Parameter",
        "SE before",
        "SE after",
        "Change %",
    ]
    assert all(isinstance(estimate, Estimate) for estimate in quadratic_three.estimates)
    labels = [estimate.label for estimate in quadratic_three.estimates]
    assert labels == [
        "SE of b0",
        "SE of b1",
        "SE of b2",
        "Prediction band half-width at the grid, mean (before)",
        "Prediction band half-width at the grid, mean (after)",
    ]
    assert [check.id for check in quadratic_three.assumptions] == [
        "covariance_as_truth",
        "model_as_given",
    ]
    assert [check.verdict for check in quadratic_three.assumptions] == [
        Verdict.PASS,
        Verdict.NOT_APPLICABLE,
    ]
    plot = quadratic_three.plots[0]
    assert isinstance(plot, MultiLinePlot)
    assert plot.title == "Prediction band narrowing"
    assert [series.label for series in plot.series] == ["Before", "After"]
    assert len(plot.series[0].x) == len(plot.series[0].y) == len(plot.series[1].y)
    assert "D-optimality" in quadratic_three.methods_fragment
    assert "information matrix" in quadratic_three.methods_fragment
    assert "sigma^2" in quadratic_three.methods_fragment
    assert "SSE/df" in quadratic_three.methods_fragment
    assert "Measure next at" in quadratic_three.interpretation


# ---------------------------------------------------------------- determinism, JSON, cancel


def test_the_same_request_gives_the_same_json(line_fit: FitResult) -> None:
    request = AdvisorRequest(points=3, candidates=50)
    first = advise(line_xy(), LINE, line_fit, request=request).model_dump_json()
    second = advise(line_xy(), LINE, line_fit, request=request).model_dump_json()
    assert first == second


def test_the_result_survives_a_json_round_trip_without_nan(
    quadratic_three: ToolResult,
) -> None:
    text = quadratic_three.model_dump_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert ToolResult.model_validate_json(text) == quadratic_three


def test_a_cancelled_run_returns_the_partial_result_and_the_warning(
    line_fit: FitResult,
) -> None:
    limit = AdvisorRequest(points=3, candidates=10).candidates + 2
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > limit

    result = advise(
        line_xy(),
        LINE,
        line_fit,
        request=AdvisorRequest(points=3, candidates=10),
        should_cancel=should_cancel,
    )
    assert type(result) is ToolResult
    assert len(_table(result, "Recommended points").rows) == 1
    assert result.warnings[-1] == "cancelled after 1 of 3 points"
    assert "cancelled after 1 of 3 points" in result.interpretation
    assert "NaN" not in result.model_dump_json()


# --------------------------------------------------------------------- the refusals


def test_a_parameter_name_the_model_does_not_have_is_refused(line_fit: FitResult) -> None:
    with pytest.raises(ToolError) as refused:
        advise(
            line_xy(),
            LINE,
            line_fit,
            request=AdvisorRequest(objective="parameter", parameter="slope"),
        )
    assert refused.value.code == "UNKNOWN_COLUMN"
    assert "slope" in refused.value.message


@pytest.mark.parametrize(
    "overrides",
    [
        {"points": 0},
        {"points": 21},
        {"candidates": 9},
        {"candidates": 1_001},
        {"replicates": 0},
        {"objective": "parameter", "parameter": None},
    ],
)
def test_an_out_of_range_request_is_refused(
    line_fit: FitResult, overrides: dict[str, object]
) -> None:
    with pytest.raises(ToolError) as refused:
        advise(
            line_xy(),
            LINE,
            line_fit,
            request=AdvisorRequest(**overrides),  # type: ignore[arg-type]
        )
    assert refused.value.code == "INVALID_OPTION"


def test_a_range_that_does_not_rise_is_refused(line_fit: FitResult) -> None:
    with pytest.raises(ToolError) as refused:
        advise(line_xy(), LINE, line_fit, request=AdvisorRequest(low=8.0, high=2.0))
    assert refused.value.code == "INVALID_OPTION"


def test_the_parameter_field_is_ignored_under_d_optimal(line_fit: FitResult) -> None:
    plain = advise(line_xy(), LINE, line_fit, request=AdvisorRequest(points=2))
    ignored = advise(
        line_xy(), LINE, line_fit, request=AdvisorRequest(points=2, parameter="not a parameter")
    )
    assert plain.model_dump_json() == ignored.model_dump_json()


# -------------------------------------------------------------- replicates, range, weights


def test_replicates_are_added_at_each_location(line_fit: FitResult) -> None:
    result = advise(line_xy(), LINE, line_fit, request=AdvisorRequest(points=2, replicates=2))

    assert _column(result, "Recommended points", "replicates") == [2, 2]
    one_row = advise(line_xy(), LINE, line_fit, request=AdvisorRequest(points=2, replicates=1))
    assert _estimates(result)["SE of b1"] <= _estimates(one_row)["SE of b1"]


def test_a_range_beyond_the_data_is_flagged(line_fit: FitResult) -> None:
    result = advise(
        line_xy(), LINE, line_fit, request=AdvisorRequest(points=3, low=-2.0, high=12.0)
    )

    assert any("extends beyond" in warning for warning in result.warnings)
    caution = next(check for check in result.assumptions if check.id == "extrapolation")
    assert caution.verdict is Verdict.CAUTION
    plot = result.plots[0]
    assert isinstance(plot, MultiLinePlot)
    assert plot.series[0].x[0] == pytest.approx(-2.0)
    assert plot.series[0].x[-1] == pytest.approx(12.0)


def test_a_few_point_fit_says_the_ends_are_all_it_can_offer() -> None:
    x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    y = 2.0 + 0.5 * x + np.random.default_rng(1).normal(0.0, 0.05, x.size)
    xy = _xy(x, y, "fp-advisor-few")
    result = advise(xy, LINE, fit_dataset(xy, LINE).result, request=AdvisorRequest(points=2))

    assert any("Every recommended point lies at an end" in warning for warning in result.warnings)


def test_a_zero_weight_row_is_listed_as_excluded() -> None:
    weights = np.ones(X.size)
    weights[7] = 0.0
    xy = XY(
        X,
        2.0 + 0.5 * X + LINE_NOISE,
        weights,
        x_name="x",
        y_name="y",
        fingerprint="fp-advisor-weighted",
    )
    spec = FitSpec(model=PolynomialRef(degree=1), use_weights=True)
    result = advise(xy, spec, fit_dataset(xy, spec).result, request=AdvisorRequest(points=2))

    assert result.counts.n_offered == X.size
    assert result.counts.n_used == X.size - 1
    assert result.counts.n_excluded == 1
    assert [(row.index, row.reason) for row in result.excluded] == [(7, "weight is zero")]


def test_the_expected_blocks_are_the_only_ones(line_two: ToolResult) -> None:
    """A result is built from its blocks: no extra table, plot or test sneaks in."""
    assert line_two.tests == ()
    assert len(line_two.tables) == 2
    assert len(line_two.plots) == 1
    assert isinstance(line_two.warnings, tuple)
