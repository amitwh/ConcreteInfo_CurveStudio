"""The experiment feasibility simulator: coverage, power, and the ways it refuses to run.

The straight-line case is measured against the closed-form reference for the same design:
a t test on 50 observations, which is what the slope's test is when the design's two x
levels are at -1 and 1 (so the slope's standard error is the mean's). Its coverage is held
to 2 points of the nominal 95 % over 2,000 simulated datasets, and its power to 0.05 of
statsmodels' ``TTestPower``. A deliberately confounded design must report absent intervals
and a warning rather than a NaN.
"""

import math

import pytest
from statsmodels.stats.power import TTestPower

from curvestudio.tools import ToolError, Verdict
from curvestudio.tools.feasibility import (
    DesignSpec,
    NoiseSpec,
    SimulationRequest,
    simulate,
)
from curvestudio.tools.stats import Estimate, Table, ToolResult
from curvestudio.workflow import CustomRef, FitSpec, PolynomialRef, RegisteredRef

STRAIGHT = FitSpec(model=PolynomialRef(degree=1))
"""The model every case fits: a straight line, with parameters b0 and b1."""

TRUTH = {"b0": 1.0, "b1": 2.0}
"""The true values the simulated data are drawn from."""

SD = 0.5
"""The noise's standard deviation in the straight-line case."""

DIFFERENCE = {"b1": 0.2}
"""The slope difference whose power the straight-line case measures."""

REPLICATES = 25
"""Twice this many observations: the design puts 25 points at -1 and 25 at 1."""

OBSERVATIONS = 2 * REPLICATES
"""The design's total observations, the n of the comparable one-sample t test."""

SEED = 20261125
"""One fixed seed, so the measurements below are reproducible."""

STRAIGHT_LINE_DESIGN = DesignSpec(low=-1.0, high=1.0, count=2, replicates=REPLICATES)
"""Two x levels at -1 and 1, repeated: the slope's SE then equals the mean's."""


def _request(**overrides: object) -> SimulationRequest:
    """A small valid request, with the named fields replaced."""
    options: dict[str, object] = {
        "truth": dict(TRUTH),
        "design": DesignSpec(low=-1.0, high=1.0, count=2, replicates=5),
        "noise": NoiseSpec(kind="sd", value=SD),
        "runs": 5,
        "seed": 3,
    }
    options.update(overrides)
    return SimulationRequest(**options)  # type: ignore[arg-type]


def _table(result: ToolResult, title: str) -> Table:
    return next(found for found in result.tables if found.title == title)


def _row(result: ToolResult, title: str, parameter: str) -> dict[str, object]:
    """The named parameter's row of a table, by column key."""
    found = _table(result, title)
    keys = [column.key for column in found.columns]
    row = next(entry for entry in found.rows if entry[keys.index("parameter")] == parameter)
    return dict(zip(keys, row, strict=True))


def _estimate(result: ToolResult, label: str) -> float:
    return next(estimate.value for estimate in result.estimates if estimate.label == label)


@pytest.fixture(scope="module")
def straight_line() -> ToolResult:
    """2,000 simulated straight-line datasets, fitted and measured once for the module."""
    return simulate(
        STRAIGHT,
        SimulationRequest(
            truth=TRUTH,
            design=STRAIGHT_LINE_DESIGN,
            noise=NoiseSpec(kind="sd", value=SD),
            runs=2_000,
            seed=SEED,
            difference=DIFFERENCE,
        ),
    )


# ------------------------------------------------------------------- the straight line


def test_straight_line_coverage_is_within_two_points_of_95(straight_line: ToolResult) -> None:
    for parameter in TRUTH:
        coverage = _row(straight_line, "Parameter recovery", parameter)["coverage"]
        assert isinstance(coverage, float)
        assert abs(coverage - 95.0) <= 2.0, f"{parameter} coverage is {coverage} %"


def test_straight_line_power_matches_the_closed_form(straight_line: ToolResult) -> None:
    power = _row(straight_line, "Power", "b1")["power"]
    reference = float(TTestPower().power(DIFFERENCE["b1"] / SD, OBSERVATIONS, 0.05))
    assert isinstance(power, float)
    assert abs(power - reference) <= 0.05, f"simulated {power}, closed form {reference}"


def test_straight_line_bias_is_near_zero(straight_line: ToolResult) -> None:
    for parameter in TRUTH:
        bias = _row(straight_line, "Parameter recovery", parameter)["bias"]
        assert isinstance(bias, float)
        assert abs(bias) < 0.01, f"{parameter} bias is {bias}"


def test_a_straight_line_result_has_the_expected_blocks(straight_line: ToolResult) -> None:
    assert type(straight_line) is ToolResult
    assert straight_line.tool == "feasibility"
    assert _estimate(straight_line, "Runs") == 2_000.0
    assert _estimate(straight_line, "Simulations that converged %") == 100.0
    assert [found.title for found in straight_line.tables] == [
        "Parameter recovery",
        "Design",
        "Power",
    ]
    assert [column.label for column in _table(straight_line, "Parameter recovery").columns] == [
        "Parameter",
        "True value",
        "Mean estimate",
        "Bias",
        "Empirical SE",
        "Mean reported SE",
        "Coverage %",
        "Converged %",
    ]
    assert [(column.label, column.kind) for column in _table(straight_line, "Design").columns] == [
        ("Point", "number"),
        ("Replicates", "int"),
    ]
    design_rows = _table(straight_line, "Design").rows
    assert design_rows == ((-1.0, REPLICATES), (1.0, REPLICATES))
    assert all(isinstance(estimate, Estimate) for estimate in straight_line.estimates)
    assert [check.id for check in straight_line.assumptions] == [
        "noise_model",
        "fit_settings",
    ]
    assert all(check.verdict is Verdict.PASS for check in straight_line.assumptions)
    plot = straight_line.plots[0]
    assert plot.kind == "box"
    assert [box.label for box in plot.boxes] == ["b0", "b1"]


# --------------------------------------------------------------------- a bad design


def test_a_confounded_design_is_reported_not_hidden() -> None:
    request = _request(
        design=DesignSpec(locations=(2.0,), replicates=4),
        noise=NoiseSpec(kind="sd", value=0.2),
        runs=10,
    )
    result = simulate(STRAIGHT, request)
    assert type(result) is ToolResult
    assert _estimate(result, "Simulations that converged %") == 0.0
    for parameter in TRUTH:
        row = _row(result, "Parameter recovery", parameter)
        assert row["mean_estimate"] is None
        assert row["empirical_se"] is None
        assert row["coverage"] is None
        assert row["converged"] == 0.0
    assert any("did not converge" in warning for warning in result.warnings)
    assert "NaN" not in result.model_dump_json()
    assert math.isfinite(_estimate(result, "Runs"))


# -------------------------------------------------------------------- refusal cases


def test_five_runs_are_enough() -> None:
    result = simulate(STRAIGHT, _request(runs=5))
    assert _estimate(result, "Runs") == 5.0


def test_more_than_five_thousand_runs_is_too_many() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(runs=5_001))
    assert refused.value.code == "TOO_MANY_ROWS"


def test_a_truth_missing_a_parameter_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(truth={"b0": 1.0}))
    assert refused.value.code == "INVALID_OPTION"
    assert "b1" in refused.value.message


def test_a_difference_naming_an_unknown_parameter_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(difference={"b2": 0.1}))
    assert refused.value.code == "INVALID_OPTION"
    assert "b2" in refused.value.message


def test_a_design_with_locations_and_a_range_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(
            STRAIGHT,
            _request(design=DesignSpec(locations=(0.0, 1.0), low=0.0, high=2.0, count=3)),
        )
    assert refused.value.code == "INVALID_OPTION"


def test_a_design_with_no_replicates_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(design=DesignSpec(low=0.0, high=10.0, count=5, replicates=0)))
    assert refused.value.code == "INVALID_OPTION"


def test_a_range_of_more_than_two_hundred_points_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(design=DesignSpec(low=0.0, high=10.0, count=201)))
    assert refused.value.code == "INVALID_OPTION"


def test_a_level_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, _request(level=1.5))
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_that_is_not_the_tools_own_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        simulate(STRAIGHT, {"truth": TRUTH})  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------- cancellation and JSON


def test_a_cancelled_run_returns_what_it_finished() -> None:
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 3

    result = simulate(
        STRAIGHT,
        _request(runs=200, design=DesignSpec(low=0.0, high=10.0, count=5)),
        should_cancel=should_cancel,
    )
    assert _estimate(result, "Runs") == 3.0
    assert result.warnings[-1] == "cancelled after 3 of 200 runs"
    assert result.counts.n_offered == 200
    assert result.counts.n_used == 3
    assert result.counts.n_excluded == 197


def test_the_same_request_and_seed_give_the_same_json() -> None:
    request = _request(runs=20, design=STRAIGHT_LINE_DESIGN)
    first = simulate(STRAIGHT, request).model_dump_json()
    second = simulate(STRAIGHT, request).model_dump_json()
    assert first == second


def test_the_result_survives_a_json_round_trip_without_nan(straight_line: ToolResult) -> None:
    text = straight_line.model_dump_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert ToolResult.model_validate_json(text) == straight_line


# ----------------------------------------------------------- the other model kinds


def test_a_coefficient_of_variation_design_is_fitted_with_matching_weights() -> None:
    result = simulate(
        STRAIGHT,
        SimulationRequest(
            truth=TRUTH,
            design=DesignSpec(low=1.0, high=5.0, count=9),
            noise=NoiseSpec(kind="cv", value=0.1),
            runs=40,
            seed=17,
        ),
    )
    assert _estimate(result, "Simulations that converged %") == 100.0
    bias = _row(result, "Parameter recovery", "b1")["bias"]
    assert isinstance(bias, float)
    assert abs(bias) < 0.5
    assert "inverse variances" in result.methods_fragment


def test_a_design_outside_the_models_example_range_says_it_extrapolates() -> None:
    result = simulate(
        FitSpec(model=RegisteredRef(id="exp_growth")),
        SimulationRequest(
            truth={"a": 2.0, "b": 0.5},
            design=DesignSpec(locations=(-1.0, 1.0, 6.0)),
            noise=NoiseSpec(kind="sd", value=0.1),
            runs=5,
            seed=23,
        ),
    )
    note = _table(result, "Design").note
    assert note is not None
    assert "extrapolated" in note
    assert "-1" in note


def test_a_custom_equation_is_evaluated_and_fitted() -> None:
    result = simulate(
        FitSpec(model=CustomRef(equation="a*x + b", parameters=["a", "b"])),
        SimulationRequest(
            truth={"a": 2.0, "b": 1.0},
            design=DesignSpec(low=0.0, high=10.0, count=6),
            noise=NoiseSpec(kind="sd", value=0.2),
            runs=10,
            seed=29,
        ),
    )
    assert _estimate(result, "Simulations that converged %") == 100.0
    assert {row[0] for row in _table(result, "Parameter recovery").rows} == {"a", "b"}
