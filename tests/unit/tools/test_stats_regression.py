"""The linear regression tool: coefficients, intervals, R squared, F test and predictions.

Every number is checked against ``statsmodels.api.OLS(...).fit()``: the coefficients, their
standard errors, t statistics, p-values and intervals, R squared, adjusted R squared, the
residual standard error, the F test, the standardised coefficients (by refitting standardised
variables) and the variance inflation factors (against
``statsmodels.stats.outliers_influence.variance_inflation_factor``). Prediction intervals are
checked against ``get_prediction``'s mean and observation intervals.

The worked example is the Longley (1967) dataset, whose ordinary-least-squares summary is
tabulated in the statsmodels documentation. The NIST/SEMATECH e-Handbook was searched on
2026-11-24 (the Process Modeling chapter's detailed contents and every quantitative-techniques
page): its numeric regression examples are simple regressions (pressure on temperature) and
the multiple-regression pages are conceptual, with no tabulated coefficients, so no multiple
regression worked example exists there to reproduce. The example used instead is the summary
table in the statsmodels documentation, "Ordinary Least Squares" notebook, section
"Multicollinearity" (``statsmodels.org/stable/examples/notebooks/generated/ols.html``), read on
2026-11-24, on the Longley dataset the package ships. Its printed values are reproduced here
to the precision the table prints.

The size of the t test of a null coefficient and of the overall F test is measured on
independent normal draws at a fixed seed. The properties every least-squares fit has are
checked too: scaling a predictor scales its coefficient by the inverse and leaves its
statistic unchanged, and adding a constant to the outcome moves only the intercept.
"""

from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError
from statsmodels.api import OLS, add_constant
from statsmodels.datasets.longley import load_pandas
from statsmodels.stats.outliers_influence import variance_inflation_factor

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof, num
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    QQPlot,
    ScatterPlot,
    ToolResult,
    Verdict,
    assert_size,
    build_request,
    p_text,
    run_tool,
    simulated_size,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue
from curvestudio.tools.stats.regression import (
    RegressionRequest,
    _durbin_watson_statistic,
    _fit,
    run_regression,
)

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""

INTERCEPT = "(Intercept)"


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_regression(RegressionRequest(**options))  # type: ignore[arg-type]


def _table(result: ToolResult, title: str) -> list[tuple[object, ...]]:
    found = next(table for table in result.tables if table.title == title)
    return list(found.rows)


def _coefficients(result: ToolResult) -> dict[object, dict[str, object]]:
    """The Coefficients table as a dictionary per term, by column key."""
    table = next(found for found in result.tables if found.title == "Coefficients")
    keys = [column.key for column in table.columns]
    return {row[0]: dict(zip(keys, row, strict=True)) for row in table.rows}


def _variance_inflation(result: ToolResult) -> dict[object, float]:
    table = next(found for found in result.tables if found.title == "Variance inflation")
    return {row[0]: float(row[1]) for row in table.rows}  # type: ignore[arg-type]


def _sample(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two predictors drawn independently, and an outcome linear in them with normal noise."""
    rng = np.random.default_rng(seed)
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    outcome = 1.0 + 2.0 * first - 0.5 * second + rng.normal(size=n)
    return outcome, first, second


def _design(*columns: np.ndarray) -> np.ndarray:
    return add_constant(np.column_stack(columns))


# --------------------------------------------------------------------------- the references


def test_every_number_of_a_two_predictor_fit_is_statsmodels() -> None:
    outcome, first, second = _sample()
    design = _design(first, second)
    reference = OLS(outcome, design).fit()

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x1", first), _column("x2", second)),
        level=0.95,
    )
    cells = _coefficients(result)
    interval = np.asarray(reference.conf_int())
    for index, term in enumerate((INTERCEPT, "x1", "x2")):
        assert cells[term]["estimate"] == pytest.approx(
            float(np.asarray(reference.params)[index]), rel=STAT_RTOL
        )
        assert cells[term]["se"] == pytest.approx(
            float(np.asarray(reference.bse)[index]), rel=STAT_RTOL
        )
        assert cells[term]["statistic"] == pytest.approx(
            float(np.asarray(reference.tvalues)[index]), rel=STAT_RTOL
        )
        assert cells[term]["p"] == pytest.approx(
            float(np.asarray(reference.pvalues)[index]), abs=P_ATOL
        )
        assert cells[term]["lower"] == pytest.approx(interval[index, 0], rel=STAT_RTOL)
        assert cells[term]["upper"] == pytest.approx(interval[index, 1], rel=STAT_RTOL)

    assert result.estimates[0].value == pytest.approx(reference.rsquared, rel=STAT_RTOL)
    assert result.estimates[1].value == pytest.approx(reference.rsquared_adj, rel=STAT_RTOL)
    assert result.estimates[2].value == pytest.approx(
        float(np.sqrt(reference.mse_resid)), rel=STAT_RTOL
    )
    test = result.tests[0]
    assert test.statistic == pytest.approx(float(reference.fvalue), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.f_pvalue), abs=P_ATOL)
    assert test.df == 2.0
    assert test.df_denominator == 37.0


def test_the_standardised_coefficients_are_the_refit_of_standardised_variables() -> None:
    outcome, first, second = _sample(seed=1)

    def standardise(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std(ddof=1)

    design = _design(standardise(first), standardise(second))
    reference = OLS(standardise(outcome), design).fit()
    expected = np.asarray(reference.params)

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x1", first), _column("x2", second)),
    )
    cells = _coefficients(result)

    assert cells[INTERCEPT]["standardised"] is None
    assert cells["x1"]["standardised"] == pytest.approx(float(expected[1]), rel=STAT_RTOL)
    assert cells["x2"]["standardised"] == pytest.approx(float(expected[2]), rel=STAT_RTOL)


def test_the_variance_inflation_factors_are_statsmodels_own() -> None:
    outcome, first, second = _sample(seed=2)
    design = _design(first, second)

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x1", first), _column("x2", second)),
    )
    found = _variance_inflation(result)

    assert found["x1"] == pytest.approx(float(variance_inflation_factor(design, 1)), rel=STAT_RTOL)
    assert found["x2"] == pytest.approx(float(variance_inflation_factor(design, 2)), rel=STAT_RTOL)


def test_the_prediction_intervals_are_statsmodels_mean_and_observation_intervals() -> None:
    outcome, first, second = _sample(n=30, seed=3)
    points = ((0.0, 0.0), (1.0, -1.0), (-0.5, 2.0))
    design = _design(first, second)
    reference = OLS(outcome, design).fit()
    frame = reference.get_prediction(add_constant(np.asarray(points))).summary_frame(alpha=0.05)

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x1", first), _column("x2", second)),
        predictions=points,
        level=0.95,
    )
    rows = _table(result, "Predictions")

    for index, row in enumerate(rows):
        assert row[0] == index + 1
        assert row[1] == pytest.approx(float(frame["mean"].iloc[index]), rel=STAT_RTOL)
        assert row[2] == pytest.approx(float(frame["mean_ci_lower"].iloc[index]), rel=STAT_RTOL)
        assert row[3] == pytest.approx(float(frame["mean_ci_upper"].iloc[index]), rel=STAT_RTOL)
        assert row[4] == pytest.approx(float(frame["obs_ci_lower"].iloc[index]), rel=STAT_RTOL)
        assert row[5] == pytest.approx(float(frame["obs_ci_upper"].iloc[index]), rel=STAT_RTOL)


def test_the_longley_multiple_regression_reproduces_the_tabulated_summary() -> None:
    """The Longley (1967) OLS summary, as tabulated by the statsmodels documentation.

    The source and the date are named in this module's docstring. The printed table rounds its
    numbers, so each is held to that rounding: the coefficients to three parts in a thousand,
    the p-values to half a thousandth, R squared and adjusted R squared to a thousandth, and
    the F statistic to a thousandth of itself.
    """
    dataset = load_pandas()
    frame = dataset.exog
    outcome = np.asarray(dataset.endog, dtype=np.float64)
    names = list(frame.columns)
    result = _run(
        outcome=_column("TOTEMP", outcome),
        predictors=tuple(_column(name, frame[name].to_numpy()) for name in names),
        names=tuple(names),
    )
    cells = _coefficients(result)

    printed = {
        "const": -3.482e06,
        "GNPDEFL": 15.0619,
        "GNP": -0.0358,
        "UNEMP": -2.0202,
        "ARMED": -1.0332,
        "POP": -0.0511,
        "YEAR": 1829.1515,
    }
    terms = {name: name for name in names} | {"const": INTERCEPT}
    for term, tabulated in printed.items():
        assert cells[terms[term]]["estimate"] == pytest.approx(tabulated, rel=2e-3), term

    printed_p = {
        "const": 0.004,
        "GNPDEFL": 0.863,
        "GNP": 0.313,
        "UNEMP": 0.003,
        "ARMED": 0.001,
        "POP": 0.826,
        "YEAR": 0.003,
    }
    for term, tabulated in printed_p.items():
        assert cells[terms[term]]["p"] == pytest.approx(tabulated, abs=6e-4), term

    assert result.estimates[0].value == pytest.approx(0.995, abs=1e-3)
    assert result.estimates[1].value == pytest.approx(0.992, abs=1e-3)
    test = result.tests[0]
    assert test.statistic == pytest.approx(330.3, rel=1e-3)
    assert test.p_value == pytest.approx(4.98e-10, rel=1e-2)


# --------------------------------------------------------------------------- size


def _null_draw(size: int = 50) -> Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]:
    """A design of an intercept and two independent predictors, and an outcome independent of
    both: every null hypothesis of the fit is true."""

    def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        predictors = rng.standard_normal((size, 2))
        return np.column_stack([np.ones(size), predictors]), rng.standard_normal(size)

    return draw


def _coefficient_p_value(sample: tuple[np.ndarray, np.ndarray]) -> float | None:
    design, outcome = sample
    model = _fit(design, outcome, 0.95)
    return None if model.p_value is None else float(model.p_value[1])


def _overall_p_value(sample: tuple[np.ndarray, np.ndarray]) -> float | None:
    design, outcome = sample
    return _fit(design, outcome, 0.95).f_p_value


def test_the_t_test_of_a_null_coefficient_rejects_at_its_level() -> None:
    share = simulated_size(_null_draw(), _coefficient_p_value, draws=4_000, seed=0)

    assert_size(share)


def test_the_overall_f_test_rejects_at_its_level_under_the_null() -> None:
    share = simulated_size(_null_draw(), _overall_p_value, draws=4_000, seed=1)

    assert_size(share)


# --------------------------------------------------------------------------- the properties


def test_scaling_a_predictor_scales_its_coefficient_and_leaves_its_statistic() -> None:
    outcome, first, second = _sample(seed=4)
    plain = _coefficients(
        _run(
            outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
        )
    )
    scaled = _coefficients(
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x1", 3.0 * first), _column("x2", second)),
        )
    )

    assert scaled["x1"]["estimate"] == pytest.approx(float(plain["x1"]["estimate"]) / 3.0, rel=1e-8)
    assert scaled["x1"]["se"] == pytest.approx(float(plain["x1"]["se"]) / 3.0, rel=1e-8)
    assert scaled["x1"]["statistic"] == pytest.approx(float(plain["x1"]["statistic"]), rel=1e-8)
    assert scaled["x1"]["p"] == pytest.approx(float(plain["x1"]["p"]), rel=1e-8)
    assert scaled["x2"]["estimate"] == pytest.approx(float(plain["x2"]["estimate"]), rel=1e-8)
    assert scaled[INTERCEPT]["estimate"] == pytest.approx(
        float(plain[INTERCEPT]["estimate"]), rel=1e-8
    )


def test_adding_a_constant_to_the_outcome_moves_only_the_intercept() -> None:
    outcome, first, second = _sample(seed=5)
    plain_result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )
    shifted_result = _run(
        outcome=_column("y", outcome + 5.0),
        predictors=(_column("x1", first), _column("x2", second)),
    )
    plain = _coefficients(plain_result)
    shifted = _coefficients(shifted_result)

    assert shifted[INTERCEPT]["estimate"] == pytest.approx(
        float(plain[INTERCEPT]["estimate"]) + 5.0, rel=1e-8
    )
    assert shifted["x1"]["estimate"] == pytest.approx(float(plain["x1"]["estimate"]), rel=1e-8)
    assert shifted["x2"]["estimate"] == pytest.approx(float(plain["x2"]["estimate"]), rel=1e-8)
    assert shifted_result.estimates[0].value == pytest.approx(
        plain_result.estimates[0].value, rel=1e-8
    )
    assert shifted_result.estimates[2].value == pytest.approx(
        plain_result.estimates[2].value, rel=1e-8
    )
    assert shifted_result.tests[0].statistic == pytest.approx(
        plain_result.tests[0].statistic, rel=1e-8
    )


# --------------------------------------------------------------------------- the blocks


def test_the_result_is_the_contract_with_the_coefficients_and_the_f_test() -> None:
    outcome, first, second = _sample(seed=6)

    result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )

    assert type(result) is ToolResult
    assert result.tool == "regression"
    assert result.title == "Linear regression of y on x1 and x2"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=40, n_used=40, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert [estimate.label for estimate in result.estimates] == [
        "R squared",
        "Adjusted R squared",
        "Residual standard error",
    ]
    assert len(result.tests) == 1
    test = result.tests[0]
    assert test.name == "F test of the regression"
    assert test.method == "Ordinary least squares regression, F test"
    assert test.effect is not None
    assert test.effect.value == result.estimates[0].value
    assert test.p_value_text == p_text(test.p_value)
    assert [table.title for table in result.tables] == ["Coefficients", "Variance inflation"]
    assert [column.label for column in result.tables[0].columns] == [
        "Term",
        "Estimate",
        "SE",
        "Statistic",
        "p",
        "Lower",
        "Upper",
        "Standardised",
    ]
    assert len(result.tables[0].rows) == 3
    assert result.tables[0].rows[0][0] == INTERCEPT
    assert result.tables[0].rows[0][7] is None


def test_one_predictor_draws_the_data_with_a_fitted_line_and_a_band() -> None:
    rng = np.random.default_rng(7)
    predictor = rng.normal(size=25)
    outcome = 3.0 + 2.0 * predictor + rng.normal(size=25)

    result = _run(outcome=_column("y", outcome), predictors=(_column("x", predictor),))

    assert [type(plot) for plot in result.plots] == [ScatterPlot, ScatterPlot, QQPlot]
    line = result.plots[0]
    assert isinstance(line, ScatterPlot)
    assert line.x_label == "x" and line.y_label == "y"
    assert len(line.lines) == 1 and line.lines[0].label == "Fitted line"
    assert line.band is not None
    assert len(line.band.x) == len(line.band.lower) == len(line.band.upper)
    assert all(low <= high for low, high in zip(line.band.lower, line.band.upper, strict=True))
    assert "Variance inflation" not in [table.title for table in result.tables]


def test_two_predictors_draw_the_residuals_and_a_qq_plot_only() -> None:
    outcome, first, second = _sample(seed=8)

    result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )

    assert [type(plot) for plot in result.plots] == [ScatterPlot, QQPlot]
    residuals = result.plots[0]
    assert isinstance(residuals, ScatterPlot)
    assert residuals.title == "Residuals against fitted values"
    assert len(residuals.lines) == 1
    assert residuals.lines[0].y0 == residuals.lines[0].y1 == 0.0
    qq = result.plots[1]
    assert isinstance(qq, QQPlot)
    assert len(qq.theoretical) == len(qq.ordered)


def test_the_predictions_table_appears_only_when_asked_for() -> None:
    outcome, first, second = _sample(n=20, seed=9)
    request = {
        "outcome": _column("y", outcome),
        "predictors": (_column("x1", first), _column("x2", second)),
    }

    without = _run(**request)
    with_predictions = _run(**request, predictions=((0.0, 0.0), (1.0, 1.0)))

    assert "Predictions" not in [table.title for table in without.tables]
    assert [table.title for table in with_predictions.tables] == [
        "Coefficients",
        "Variance inflation",
        "Predictions",
    ]
    assert len(_table(with_predictions, "Predictions")) == 2


def test_the_independence_check_uses_the_exact_durbin_watson_statistic() -> None:
    outcome, first, second = _sample(seed=10)
    result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )

    independence = next(check for check in result.assumptions if check.id == "independence")
    coefficients = _coefficients(result)
    fitted = (
        float(coefficients[INTERCEPT]["estimate"])
        + float(coefficients["x1"]["estimate"]) * first
        + float(coefficients["x2"]["estimate"]) * second
    )
    residual = np.asarray(outcome) - fitted
    expected = _durbin_watson_statistic(residual)

    assert independence.statistic == f"D = {num(expected)}"
    assert independence.verdict is not Verdict.NOT_APPLICABLE


def test_the_durbin_watson_statistic_is_the_ratio_of_the_differenced_squares() -> None:
    assert _durbin_watson_statistic(np.asarray([1.0, -1.0, 1.0, -1.0])) == pytest.approx(3.0)


def test_above_a_thousand_rows_the_independence_check_gives_the_statistic_alone() -> None:
    size = 1_005
    predictor = np.arange(size, dtype=np.float64)
    outcome = 1.0 + 0.5 * predictor + np.random.default_rng(11).normal(size=size)

    result = _run(outcome=_column("y", outcome), predictors=(_column("x", predictor),))
    independence = next(check for check in result.assumptions if check.id == "independence")

    assert independence.verdict is Verdict.NOT_APPLICABLE
    assert independence.statistic is not None and independence.statistic.startswith("D = ")
    assert "above 1,000 rows" in independence.evidence


def test_the_assumptions_check_the_four_things_a_least_squares_fit_assumes() -> None:
    outcome, first, second = _sample(seed=12)

    result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )

    assert [check.id for check in result.assumptions] == [
        "normality",
        "constant_variance",
        "independence",
        "linearity",
    ]
    linearity = result.assumptions[-1]
    assert linearity.verdict is Verdict.NOT_APPLICABLE
    assert linearity.next_step == "Look at the residuals against each predictor."


# --------------------------------------------------------------------------- missing data


def test_a_row_missing_in_a_predictor_is_left_out_and_listed() -> None:
    outcome = _column("y", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    predictor = _column("x", [1.0, 2.0, float("nan"), 4.0, 5.0, 7.0])

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.counts == Counts(n_offered=6, n_used=5, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in x"),)


def test_a_row_missing_in_the_outcome_is_left_out_and_listed() -> None:
    outcome = _column("y", [1.0, 2.0, float("nan"), 4.0, 5.0])
    predictor = _column("x", [1.0, 2.0, 3.0, 4.0, 6.0])

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in y"),)


def test_a_row_with_an_infinite_value_is_listed_as_not_finite() -> None:
    outcome = _column("y", [1.0, 2.0, 3.0, 4.0])
    predictor = _column("x", [1.0, 2.0, float("inf"), 4.0])

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.excluded == (ExcludedRow(index=2, reason="not a finite number in x"),)


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    outcome = _column("y", [float("nan")] * 205 + list(np.arange(1.0, 11.0)))
    predictor = _column("x", np.arange(1.0, 216.0))

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.counts.n_excluded == 205
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


# --------------------------------------------------------------------------- the refusals


def test_no_predictor_or_more_than_twenty_is_refused() -> None:
    outcome = _column("y", np.arange(1.0, 25.0))
    with pytest.raises(ToolError) as refused:
        _run(outcome=outcome, predictors=())
    assert refused.value.code == "INVALID_OPTION"

    many = tuple(_column(f"x{index}", np.arange(1.0, 25.0)) for index in range(21))
    with pytest.raises(ToolError) as refused:
        _run(outcome=outcome, predictors=many)
    assert refused.value.code == "INVALID_OPTION"
    assert "between 1 and 20" in refused.value.message


def test_names_of_the_wrong_length_are_refused() -> None:
    outcome, first, _second = _sample(n=12, seed=13)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_column("y", outcome), predictors=(_column("x", first),), names=("a", "b"))
    assert refused.value.code == "INVALID_OPTION"
    assert "one name per predictor" in refused.value.message


def test_more_than_a_thousand_prediction_rows_are_refused() -> None:
    outcome, first, _second = _sample(n=12, seed=14)
    too_many = tuple((0.0,) for _ in range(1_001))
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x", first),),
            predictions=too_many,
        )
    assert refused.value.code == "INVALID_OPTION"


def test_a_prediction_row_of_the_wrong_length_is_refused() -> None:
    outcome, first, second = _sample(n=12, seed=15)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x1", first), _column("x2", second)),
            predictions=((0.0,),),
        )
    assert refused.value.code == "INVALID_OPTION"
    assert "prediction row 1" in refused.value.message


def test_a_non_finite_prediction_is_refused() -> None:
    outcome, first, _second = _sample(n=12, seed=16)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x", first),),
            predictions=((float("nan"),),),
        )
    assert refused.value.code == "INVALID_OPTION"


def test_a_constant_predictor_is_refused() -> None:
    outcome, _first, _second = _sample(n=12, seed=17)
    constant = _column("c", [2.0] * 12)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_column("y", outcome), predictors=(constant,))
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "c" in refused.value.message


def test_a_constant_outcome_is_refused() -> None:
    predictor = _column("x", np.arange(12.0))
    with pytest.raises(ToolError) as refused:
        _run(outcome=_column("y", [4.0] * 12), predictors=(predictor,))
    assert refused.value.code == "CONSTANT_COLUMN"


def test_an_exactly_collinear_predictor_is_refused() -> None:
    rng = np.random.default_rng(18)
    first = rng.normal(size=15)
    second = rng.normal(size=15)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", rng.normal(size=15)),
            predictors=(
                _column("a", first),
                _column("b", second),
                _column("c", first + second),
            ),
        )
    assert refused.value.code == "COLLINEAR"
    assert "a, b and c" in refused.value.message


def test_fewer_rows_than_predictors_plus_two_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", [1.0, 2.0, 3.0]),
            predictors=(_column("a", [1.0, 2.0, 3.0]), _column("b", [2.0, 3.0, 5.0])),
        )
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 4 complete rows for 2 predictors" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    outcome, first, _second = _sample(n=12, seed=19)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_column("y", outcome), predictors=(_column("x", first),), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_regression(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    outcome, first, _second = _sample(n=12, seed=20)
    with pytest.raises(ValidationError):
        RegressionRequest(  # type: ignore[call-arg]
            outcome=_column("y", outcome), predictors=(_column("x", first),), method="ols"
        )


# ---------------------------------------------------------------------- the words and the registry


def test_an_exact_fit_reports_no_test_and_no_assumption_it_cannot_check() -> None:
    predictor = np.arange(10.0)
    outcome = 2.0 + 3.0 * predictor

    result = _run(outcome=_column("y", outcome), predictors=(_column("x", predictor),))

    assert result.estimates[0].value == 1.0
    assert result.estimates[2].value == 0.0
    test = result.tests[0]
    assert test.statistic is None and test.p_value is None
    assert test.reason is not None and "exact" in test.reason
    assert all(check.verdict is Verdict.NOT_APPLICABLE for check in result.assumptions)
    assert "exact" in result.interpretation


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    outcome, first, second = _sample(seed=21)

    result = _run(
        outcome=_column("y", outcome), predictors=(_column("x1", first), _column("x2", second))
    )

    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
    for text in texts:
        assert not claims_proof(text), text


def test_every_request_field_is_described_and_the_names_field_is_labelled() -> None:
    found = next(info for info in catalogue() if info.name == "regression")

    assert found.family == "relationships"
    assert [field.name for field in found.fields] == [
        "outcome",
        "predictors",
        "names",
        "level",
        "predictions",
    ]
    assert all(field.description for field in found.fields)
    assert found.fields[0].kind == "column"
    assert found.fields[1].kind == "columns"
    assert found.fields[2].label == "Predictor names"


def test_the_registry_builds_and_runs_the_tool_from_column_names() -> None:
    outcome, first, second = _sample(n=15, seed=22)
    columns = {"y": _column("y", outcome), "x1": _column("x1", first), "x2": _column("x2", second)}

    def group(name: str) -> object:
        raise AssertionError("regression takes no group column")

    request = build_request(
        "regression",
        {
            "outcome": "y",
            "predictors": ["x1", "x2"],
            "names": ["first", "second"],
            "predictions": [[0.0, 0.0]],
            "level": 0.9,
        },
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("regression", request)

    assert isinstance(request, RegressionRequest)
    assert result.tool == "regression"
    assert result.level == 0.9
    assert result.tables[0].rows[1][0] == "first"
    assert "curvestudio.tools.stats.regression" in TOOL_MODULES
