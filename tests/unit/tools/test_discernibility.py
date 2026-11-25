"""The model discernibility report on real fits and on rankings built to pin one number.

The report refits every row it reads, so the curve comparisons are real fits of the real
data. Rankings built by hand carry the AICc values that place a model in the leading group;
the curves behind them still come from the workflow's own fitter, never from a formula
written in a test.
"""

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CandidateMetrics,
    RankedFit,
    autofit,
)
from curvestudio.models.custom import CustomModelDefinition
from curvestudio.tools._errors import ToolError
from curvestudio.tools._fit_tools import curve_of, evaluation_grid, resolve_target
from curvestudio.tools.discernibility import DiscernibilityRequest, discernibility
from curvestudio.tools.stats import ToolResult
from curvestudio.tools.stats._display import MultiLinePlot
from curvestudio.workflow import CustomRef, FitSpec, PolynomialRef

X = np.linspace(0.0, 10.0, 40)
"""The x values of the constructed rankings."""


def xy_of(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="fp-discernibility",
    )


def row(
    model_id: str,
    name: str,
    parameters: int,
    aicc: float,
    *,
    delta_aicc: float | None = None,
    rank: int = 1,
    n: int = 40,
) -> RankedFit:
    """One ranked row: its metrics carry the AICc the leading group is read from."""
    return RankedFit(
        rank=rank,
        model_id=model_id,
        name=name,
        metrics=CandidateMetrics(k=parameters, n=n, aicc=aicc),
        delta_aicc=delta_aicc,
        status="converged",
    )


def ranking_of(xy: XY, rows: list[RankedFit], config: AutoFitConfig | None = None) -> AutoFitResult:
    """A ranking as the report reads it: the rows, the data's fingerprint and the settings."""
    return AutoFitResult(
        config=AutoFitConfig(workers=1, n_starts=1) if config is None else config,
        dataset_fingerprint=xy.fingerprint,
        n_active=int(np.count_nonzero(np.ones(xy.x.size))),
        ranked=tuple(rows),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def records(result: ToolResult, title: str) -> list[dict[str, object]]:
    """A table's rows as dicts keyed by the column keys."""
    found = next(table for table in result.tables if table.title == title)
    keys = [column.key for column in found.columns]
    return [dict(zip(keys, cells, strict=True)) for cells in found.rows]


def quadratic_xy(noise: float = 0.3) -> XY:
    y = 2.0 + X + 0.15 * X**2 + np.random.default_rng(7).normal(0.0, noise, X.size)
    return xy_of(X, y)


def bent_quadratic_xy() -> XY:
    """A quadratic whose noise lets a slightly wrong shape sit just outside the interval."""
    y = 2.0 + X + 0.15 * X**2 + np.random.default_rng(9).normal(0.0, 0.5, X.size)
    return xy_of(X, y)


COSINE = CustomModelDefinition(
    id="discern_cos", name="Quadratic with a cosine bend", expression="a + b*x + c*cos(0.5*x)"
)
SINE = CustomModelDefinition(
    id="discern_sin", name="Quadratic with a sine bend", expression="a + b*x + c*sin(x)"
)
BENT_CONFIG = AutoFitConfig(workers=1, n_starts=1, custom_models=(COSINE, SINE))


def curve_difference(xy: XY, equation: str) -> float:
    """The root-mean-square difference between the quadratic and a custom curve on the grid."""
    grid = evaluation_grid(xy, points=200)
    quadratic = resolve_target(xy, FitSpec(model=PolynomialRef(degree=2)))
    custom = resolve_target(
        xy, FitSpec(model=CustomRef(equation=equation, parameters=["a", "b", "c"]))
    )
    return float(np.sqrt(np.mean((curve_of(quadratic, grid) - curve_of(custom, grid)) ** 2)))


def test_a_model_the_data_were_generated_from_is_in_the_leading_group() -> None:
    x = np.linspace(0.5, 6.0, 25)
    y = 10.0 * x / (2.0 + x) + np.random.default_rng(3).normal(0.0, 0.05, x.size)
    xy = xy_of(x, y)
    config = AutoFitConfig(
        families=("saturation",),
        include_polynomials=False,
        include_generated=False,
        n_starts=1,
        workers=1,
        cv_folds=None,
        staged_starts=False,
    )
    result = autofit(xy, config)

    report = discernibility(xy, result)

    models = records(report, "Models")
    assert any("michaelis_menten" in str(model["model"]) for model in models)
    assert models[0]["rank"] == 1
    assert "michaelis_menten" in str(models[0]["model"])


def test_two_reparametrisations_of_one_curve_are_one_predictive_group() -> None:
    x = np.linspace(0.0, 10.0, 40)
    y = 5.0 * np.exp(-0.8 * x) + 1.0 + np.random.default_rng(5).normal(0.0, 0.02, x.size)
    xy = xy_of(x, y)
    rows = [
        row("exp_decay_offset", "Exponential decay with offset", 3, 10.0, delta_aicc=0.0),
        row("exp_offset", "Exponential approach to a limit", 3, 11.0, delta_aicc=1.0, rank=2),
    ]

    report = discernibility(xy, ranking_of(xy, rows), request=DiscernibilityRequest(profile=False))

    groups = records(report, "Predictive groups")
    assert len(groups) == 1
    assert "exp_decay_offset" in str(groups[0]["models"])
    assert "exp_offset" in str(groups[0]["models"])
    statuses = [str(model["status"]) for model in records(report, "Models")]
    assert all(status.startswith("Assessed") for status in statuses)


def test_a_simpler_model_is_in_the_leading_group_only_within_two_aicc() -> None:
    xy = quadratic_xy()
    quadratic = row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0)
    line = row("poly1", "Straight line", 2, 101.9, delta_aicc=1.9, rank=2)

    inside = discernibility(
        xy, ranking_of(xy, [quadratic, line]), request=DiscernibilityRequest(profile=False)
    )
    assert len(records(inside, "Models")) == 2
    assert records(inside, "Models")[0]["rank"] == 1
    assert "poly2" in str(records(inside, "Models")[0]["model"])
    simplest = records(inside, "The simplest model that can be defended")
    assert len(simplest) == 1
    assert "poly1" in str(simplest[0]["model"])

    outside = discernibility(
        xy,
        ranking_of(
            xy,
            [
                quadratic,
                row("poly1", "Straight line", 2, 102.1, delta_aicc=2.1, rank=2),
            ],
        ),
        request=DiscernibilityRequest(profile=False),
    )
    assert len(records(outside, "Models")) == 1
    assert "poly1" not in str(records(outside, "Models")[0]["model"])


def test_a_generated_quadratic_leads_and_the_line_is_separated_with_a_stated_range() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row("poly1", "Straight line", 2, 101.0, delta_aicc=1.0, rank=2),
    ]

    report = discernibility(xy, ranking_of(xy, rows), request=DiscernibilityRequest(profile=False))

    models = records(report, "Models")
    assert "poly2" in str(models[0]["model"])
    groups = records(report, "Predictive groups")
    if len(groups) == 1:
        assert "poly1" in str(groups[0]["models"])
        assert "poly2" in str(groups[0]["models"])
    else:
        separating = records(report, "What would separate the groups")
        assert separating
        assert separating[0]["x_range"] != "none"
        assert isinstance(separating[0]["extra_points"], int)
        assert separating[0]["extra_points"] >= 1


def test_the_extra_points_figure_falls_as_the_curve_difference_grows() -> None:
    xy = bent_quadratic_xy()
    quadratic = row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0)
    cosine = row("discern_cos", "Quadratic with a cosine bend", 3, 101.0, delta_aicc=1.0, rank=2)
    sine = row("discern_sin", "Quadratic with a sine bend", 3, 101.0, delta_aicc=1.0, rank=2)

    near = discernibility(
        xy,
        ranking_of(xy, [quadratic, cosine], BENT_CONFIG),
        request=DiscernibilityRequest(profile=False),
    )
    far = discernibility(
        xy,
        ranking_of(xy, [quadratic, sine], BENT_CONFIG),
        request=DiscernibilityRequest(profile=False),
    )

    near_points = records(near, "What would separate the groups")[0]["extra_points"]
    far_points = records(far, "What would separate the groups")[0]["extra_points"]
    assert isinstance(near_points, int)
    assert isinstance(far_points, int)
    near_difference = curve_difference(xy, COSINE.expression)
    far_difference = curve_difference(xy, SINE.expression)
    assert near_difference < far_difference
    assert far_points < near_points
    assert f"about {near_points} more points" in near.interpretation


def test_a_cancelled_run_returns_the_partial_report_with_a_warning() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row("poly1", "Straight line", 2, 101.0, delta_aicc=1.0, rank=2),
        row("poly3", "Cubic", 4, 101.5, delta_aicc=1.5, rank=3),
    ]
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    report = discernibility(
        xy,
        ranking_of(xy, rows),
        request=DiscernibilityRequest(profile=False),
        should_cancel=should_cancel,
    )

    assert type(report) is ToolResult
    assert any("cancelled" in warning.lower() for warning in report.warnings)
    statuses = [str(model["status"]) for model in records(report, "Models")]
    assert statuses[0].startswith("Assessed")
    assert all(status == "Not assessed: the report was cancelled" for status in statuses[1:])


def test_a_polynomial_through_the_origin_is_refitted_with_its_own_form() -> None:
    x = np.linspace(0.0, 10.0, 40)
    y = 2.0 * x + 0.1 * x**2 + np.random.default_rng(4).normal(0.0, 0.1, x.size)
    xy = xy_of(x, y)
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row(
            "poly2_origin",
            "Quadratic through the origin",
            2,
            101.0,
            delta_aicc=1.0,
            rank=2,
        ),
    ]

    report = discernibility(xy, ranking_of(xy, rows), request=DiscernibilityRequest(profile=False))

    statuses = [str(model["status"]) for model in records(report, "Models")]
    assert len(statuses) == 2
    assert all(status.startswith("Assessed") for status in statuses)
    assert records(report, "Predictive groups")


def test_the_leading_group_is_computed_from_aicc_when_the_delta_is_missing() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0),
        row("poly1", "Straight line", 2, 101.5, rank=2),
        row("poly3", "Cubic", 4, 104.0, rank=3),
    ]

    report = discernibility(xy, ranking_of(xy, rows), request=DiscernibilityRequest(profile=False))

    models = records(report, "Models")
    assert len(models) == 2
    assert models[0]["delta_aicc"] == pytest.approx(0.0)
    assert models[1]["delta_aicc"] == pytest.approx(1.5)


def test_the_model_cap_is_stated_and_the_rows_left_out_are_listed() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row("poly1", "Straight line", 2, 101.0, delta_aicc=1.0, rank=2),
        row("poly3", "Cubic", 4, 101.5, delta_aicc=1.5, rank=3),
    ]

    report = discernibility(
        xy,
        ranking_of(xy, rows),
        request=DiscernibilityRequest(profile=False, max_models=2),
    )

    models = records(report, "Models")
    assert len(models) == 3
    assert str(models[2]["status"]).startswith("Not assessed")
    assert any("at most 2" in warning for warning in report.warnings)


def test_the_report_is_the_same_on_a_second_run_and_survives_a_json_round_trip() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row("poly1", "Straight line", 2, 101.0, delta_aicc=1.0, rank=2),
    ]
    ranking = ranking_of(xy, rows)

    first = discernibility(xy, ranking, request=DiscernibilityRequest(profile=False))
    second = discernibility(xy, ranking, request=DiscernibilityRequest(profile=False))

    assert type(first) is ToolResult
    assert first.model_dump_json() == second.model_dump_json()
    text = first.model_dump_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert ToolResult.model_validate_json(text) == first
    assert len(first.counts.model_dump()) == 3
    assert first.counts.n_offered == xy.x.size
    assert first.counts.n_excluded == 0


def test_the_plot_draws_the_leading_group_curves_on_the_grid() -> None:
    xy = quadratic_xy()
    rows = [
        row("poly2", "Quadratic", 3, 100.0, delta_aicc=0.0),
        row("poly1", "Straight line", 2, 101.0, delta_aicc=1.0, rank=2),
    ]

    report = discernibility(
        xy,
        ranking_of(xy, rows),
        request=DiscernibilityRequest(profile=False, grid_points=51),
    )

    plot = next(item for item in report.plots if isinstance(item, MultiLinePlot))
    assert plot.title == "Models in the leading group"
    assert {series.label for series in plot.series} == {"Quadratic", "Straight line"}
    for series in plot.series:
        assert len(series.x) == len(series.y) == 51
        assert series.x[0] == pytest.approx(float(xy.x.min()))
        assert series.x[-1] == pytest.approx(float(xy.x.max()))


def test_a_ranking_without_a_ranked_model_is_refused() -> None:
    xy = quadratic_xy()
    with pytest.raises(ToolError, match="no converged model"):
        discernibility(xy, ranking_of(xy, []))
