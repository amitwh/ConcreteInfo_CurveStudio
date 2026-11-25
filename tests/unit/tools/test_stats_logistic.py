"""The logistic regression tool: coefficients, intervals, pseudo R squared, AUC and calibration.

Every number is checked against ``statsmodels.api.Logit(...).fit()``: the coefficients, their
standard errors, z statistics, p-values and Wald intervals, the AIC and the log-likelihood.
McFadden's and Nagelkerke's pseudo R squared are checked against their formulas written out in
this file, and the area under the ROC curve against the Mann-Whitney probability with ties
counted as one half.

The worked example is the UCLA Statistical Consulting Group's logistic-regression example,
"Logistic Regression | Stata Data Analysis Examples",
``https://stats.oarc.ucla.edu/stata/dae/logistic-regression/``, fetched and read on 2026-11-25.
The page's "Logistic regression" output for ``logit admit gre gpa i.rank`` on 400 observations
tabulates the coefficients and their 95 % intervals, and its ``logit, or`` output tabulates the
odds ratios with their intervals; the page prints LR chi2(5) = 41.46, log likelihood =
-229.25875 and pseudo R2 = 0.0829. The 400 rows of the page's dataset,
``https://stats.oarc.ucla.edu/stat/data/binary.csv``, are embedded below so the test runs
offline. The three rank indicators are built from the published ``rank`` column, so the model
reproduces the page's categorical-rank fit with numeric columns.

The properties every logistic fit has are checked too: adding a constant to a predictor moves
only the intercept, scaling a predictor scales its coefficient by the inverse and leaves its
statistic unchanged, and modelling the other outcome level negates every coefficient.
"""

import io

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats
from scipy.special import expit
from statsmodels.api import Logit, add_constant

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof, num
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    ScatterPlot,
    ToolResult,
    Verdict,
    build_request,
    p_text,
    run_tool,
)
from curvestudio.tools.stats.logistic import (
    LogisticRequest,
    _fit,
    run_logistic,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""

INTERCEPT = "(Intercept)"

BINARY_CSV = """\
admit,gre,gpa,rank
0,380,3.61,3
1,660,3.67,3
1,800,4,1
1,640,3.19,4
0,520,2.93,4
1,760,3,2
1,560,2.98,1
0,400,3.08,2
1,540,3.39,3
0,700,3.92,2
0,800,4,4
0,440,3.22,1
1,760,4,1
0,700,3.08,2
1,700,4,1
0,480,3.44,3
0,780,3.87,4
0,360,2.56,3
0,800,3.75,2
1,540,3.81,1
0,500,3.17,3
1,660,3.63,2
0,600,2.82,4
0,680,3.19,4
1,760,3.35,2
1,800,3.66,1
1,620,3.61,1
1,520,3.74,4
1,780,3.22,2
0,520,3.29,1
0,540,3.78,4
0,760,3.35,3
0,600,3.4,3
1,800,4,3
0,360,3.14,1
0,400,3.05,2
0,580,3.25,1
0,520,2.9,3
1,500,3.13,2
1,520,2.68,3
0,560,2.42,2
1,580,3.32,2
1,600,3.15,2
0,500,3.31,3
0,700,2.94,2
1,460,3.45,3
1,580,3.46,2
0,500,2.97,4
0,440,2.48,4
0,400,3.35,3
0,640,3.86,3
0,440,3.13,4
0,740,3.37,4
1,680,3.27,2
0,660,3.34,3
1,740,4,3
0,560,3.19,3
0,380,2.94,3
0,400,3.65,2
0,600,2.82,4
1,620,3.18,2
0,560,3.32,4
0,640,3.67,3
1,680,3.85,3
0,580,4,3
0,600,3.59,2
0,740,3.62,4
0,620,3.3,1
0,580,3.69,1
0,800,3.73,1
0,640,4,3
0,300,2.92,4
0,480,3.39,4
0,580,4,2
0,720,3.45,4
0,720,4,3
0,560,3.36,3
1,800,4,3
0,540,3.12,1
1,620,4,1
0,700,2.9,4
0,620,3.07,2
0,500,2.71,2
0,380,2.91,4
1,500,3.6,3
0,520,2.98,2
0,600,3.32,2
0,600,3.48,2
0,700,3.28,1
1,660,4,2
0,700,3.83,2
1,720,3.64,1
0,800,3.9,2
0,580,2.93,2
1,660,3.44,2
0,660,3.33,2
0,640,3.52,4
0,480,3.57,2
0,700,2.88,2
0,400,3.31,3
0,340,3.15,3
0,580,3.57,3
0,380,3.33,4
0,540,3.94,3
1,660,3.95,2
1,740,2.97,2
1,700,3.56,1
0,480,3.13,2
0,400,2.93,3
0,480,3.45,2
0,680,3.08,4
0,420,3.41,4
0,360,3,3
0,600,3.22,1
0,720,3.84,3
0,620,3.99,3
1,440,3.45,2
0,700,3.72,2
1,800,3.7,1
0,340,2.92,3
1,520,3.74,2
1,480,2.67,2
0,520,2.85,3
0,500,2.98,3
0,720,3.88,3
0,540,3.38,4
1,600,3.54,1
0,740,3.74,4
0,540,3.19,2
0,460,3.15,4
1,620,3.17,2
0,640,2.79,2
0,580,3.4,2
0,500,3.08,3
0,560,2.95,2
0,500,3.57,3
0,560,3.33,4
0,700,4,3
0,620,3.4,2
1,600,3.58,1
0,640,3.93,2
1,700,3.52,4
0,620,3.94,4
0,580,3.4,3
0,580,3.4,4
0,380,3.43,3
0,480,3.4,2
0,560,2.71,3
1,480,2.91,1
0,740,3.31,1
1,800,3.74,1
0,400,3.38,2
1,640,3.94,2
0,580,3.46,3
0,620,3.69,3
1,580,2.86,4
0,560,2.52,2
1,480,3.58,1
0,660,3.49,2
0,700,3.82,3
0,600,3.13,2
0,640,3.5,2
1,700,3.56,2
0,520,2.73,2
0,580,3.3,2
0,700,4,1
0,440,3.24,4
0,720,3.77,3
0,500,4,3
0,600,3.62,3
0,400,3.51,3
0,540,2.81,3
0,680,3.48,3
1,800,3.43,2
0,500,3.53,4
1,620,3.37,2
0,520,2.62,2
1,620,3.23,3
0,620,3.33,3
0,300,3.01,3
0,620,3.78,3
0,500,3.88,4
0,700,4,2
1,540,3.84,2
0,500,2.79,4
0,800,3.6,2
0,560,3.61,3
0,580,2.88,2
0,560,3.07,2
0,500,3.35,2
1,640,2.94,2
0,800,3.54,3
0,640,3.76,3
0,380,3.59,4
1,600,3.47,2
0,560,3.59,2
0,660,3.07,3
1,400,3.23,4
0,600,3.63,3
0,580,3.77,4
0,800,3.31,3
1,580,3.2,2
1,700,4,1
0,420,3.92,4
1,600,3.89,1
1,780,3.8,3
0,740,3.54,1
1,640,3.63,1
0,540,3.16,3
0,580,3.5,2
0,740,3.34,4
0,580,3.02,2
0,460,2.87,2
0,640,3.38,3
1,600,3.56,2
1,660,2.91,3
0,340,2.9,1
1,460,3.64,1
0,460,2.98,1
1,560,3.59,2
0,540,3.28,3
0,680,3.99,3
1,480,3.02,1
0,800,3.47,3
0,800,2.9,2
1,720,3.5,3
0,620,3.58,2
0,540,3.02,4
0,480,3.43,2
1,720,3.42,2
0,580,3.29,4
0,600,3.28,3
0,380,3.38,2
0,420,2.67,3
1,800,3.53,1
0,620,3.05,2
1,660,3.49,2
0,480,4,2
0,500,2.86,4
0,700,3.45,3
0,440,2.76,2
1,520,3.81,1
1,680,2.96,3
0,620,3.22,2
0,540,3.04,1
0,800,3.91,3
0,680,3.34,2
0,440,3.17,2
0,680,3.64,3
0,640,3.73,3
0,660,3.31,4
0,620,3.21,4
1,520,4,2
1,540,3.55,4
1,740,3.52,4
0,640,3.35,3
1,520,3.3,2
1,620,3.95,3
0,520,3.51,2
0,640,3.81,2
0,680,3.11,2
0,440,3.15,2
1,520,3.19,3
1,620,3.95,3
1,520,3.9,3
0,380,3.34,3
0,560,3.24,4
1,600,3.64,3
1,680,3.46,2
0,500,2.81,3
1,640,3.95,2
0,540,3.33,3
1,680,3.67,2
0,660,3.32,1
0,520,3.12,2
1,600,2.98,2
0,460,3.77,3
1,580,3.58,1
1,680,3,4
1,660,3.14,2
0,660,3.94,2
0,360,3.27,3
0,660,3.45,4
0,520,3.1,4
1,440,3.39,2
0,600,3.31,4
1,800,3.22,1
1,660,3.7,4
0,800,3.15,4
0,420,2.26,4
1,620,3.45,2
0,800,2.78,2
0,680,3.7,2
0,800,3.97,1
0,480,2.55,1
0,520,3.25,3
0,560,3.16,1
0,460,3.07,2
0,540,3.5,2
0,720,3.4,3
0,640,3.3,2
1,660,3.6,3
1,400,3.15,2
1,680,3.98,2
0,220,2.83,3
0,580,3.46,4
1,540,3.17,1
0,580,3.51,2
0,540,3.13,2
0,440,2.98,3
0,560,4,3
0,660,3.67,2
0,660,3.77,3
1,520,3.65,4
0,540,3.46,4
1,300,2.84,2
1,340,3,2
1,780,3.63,4
1,480,3.71,4
0,540,3.28,1
0,460,3.14,3
0,460,3.58,2
0,500,3.01,4
0,420,2.69,2
0,520,2.7,3
0,680,3.9,1
0,680,3.31,2
1,560,3.48,2
0,580,3.34,2
0,500,2.93,4
0,740,4,3
0,660,3.59,3
0,420,2.96,1
0,560,3.43,3
1,460,3.64,3
1,620,3.71,1
0,520,3.15,3
0,620,3.09,4
0,540,3.2,1
1,660,3.47,3
0,500,3.23,4
1,560,2.65,3
0,500,3.95,4
0,580,3.06,2
0,520,3.35,3
0,500,3.03,3
0,600,3.35,2
0,580,3.8,2
0,400,3.36,2
0,620,2.85,2
1,780,4,2
0,620,3.43,3
1,580,3.12,3
0,700,3.52,2
1,540,3.78,2
1,760,2.81,1
0,700,3.27,2
0,720,3.31,1
1,560,3.69,3
0,720,3.94,3
1,520,4,1
1,540,3.49,1
0,680,3.14,2
0,460,3.44,2
1,560,3.36,1
0,480,2.78,3
0,460,2.93,3
0,620,3.63,3
0,580,4,1
0,800,3.89,2
1,540,3.77,2
1,680,3.76,3
1,680,2.42,1
1,620,3.37,1
0,560,3.78,2
0,560,3.49,4
0,620,3.63,2
1,800,4,2
0,640,3.12,3
0,540,2.7,2
0,700,3.65,2
1,540,3.49,2
0,540,3.51,2
0,660,4,1
1,480,2.62,2
0,420,3.02,1
1,740,3.86,2
0,580,3.36,2
0,640,3.17,2
0,640,3.51,2
1,800,3.05,2
1,660,3.88,2
1,600,3.38,3
1,620,3.75,2
1,460,3.99,3
0,620,4,2
0,560,3.04,3
0,460,2.63,2
0,700,3.65,2
0,600,3.89,3"""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _group(name: str, labels: object, rows: object | None = None) -> GroupColumn:
    """A group column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(labels, dtype=object)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_logistic(LogisticRequest(**options))  # type: ignore[arg-type]


def _coefficients(result: ToolResult) -> dict[object, dict[str, object]]:
    """The Coefficients table as a dictionary per term, by column key."""
    table = next(found for found in result.tables if found.title == "Coefficients")
    keys = [column.key for column in table.columns]
    return {row[0]: dict(zip(keys, row, strict=True)) for row in table.rows}


def _table(result: ToolResult, title: str) -> list[tuple[object, ...]]:
    found = next(table for table in result.tables if table.title == title)
    return list(found.rows)


def _sample(
    size: int = 150, seed: int = 20261125
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three independent normal predictors, and a two-level outcome that depends on them.

    The rows are ordered so the first label is ``no`` and the second, the modelled event, is
    ``yes``; the outcome keeps both classes.
    """
    rng = np.random.default_rng(seed)
    first = rng.normal(size=size)
    second = rng.normal(size=size)
    third = rng.normal(size=size)
    event = (rng.random(size) < expit(-0.4 + 0.8 * first - 0.5 * second + 0.3 * third)).astype(
        np.float64
    )
    order = np.argsort(event, kind="stable")
    event, first, second, third = event[order], first[order], second[order], third[order]
    labels = np.where(event == 1.0, "yes", "no")
    return labels, event, first, second, third


def _binary_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The UCLA binary dataset: admit, gre, gpa and rank."""
    data = np.loadtxt(io.StringIO(BINARY_CSV), delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


# --------------------------------------------------------------------------- the references


def test_every_number_of_a_three_predictor_fit_is_statsmodels() -> None:
    labels, event, first, second, third = _sample()
    design = add_constant(np.column_stack([first, second, third]))
    reference = Logit(event, design).fit(disp=0)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )
    cells = _coefficients(result)
    interval = np.asarray(reference.conf_int())
    for index, term in enumerate((INTERCEPT, "x1", "x2", "x3")):
        assert cells[term]["estimate"] == pytest.approx(
            float(np.asarray(reference.params)[index]), rel=STAT_RTOL
        )
        assert cells[term]["se"] == pytest.approx(
            float(np.asarray(reference.bse)[index]), rel=STAT_RTOL
        )
        assert cells[term]["z"] == pytest.approx(
            float(np.asarray(reference.tvalues)[index]), rel=STAT_RTOL
        )
        assert cells[term]["p"] == pytest.approx(
            float(np.asarray(reference.pvalues)[index]), abs=P_ATOL
        )
        assert cells[term]["lower"] == pytest.approx(interval[index, 0], rel=STAT_RTOL)
        assert cells[term]["upper"] == pytest.approx(interval[index, 1], rel=STAT_RTOL)
        assert cells[term]["odds_ratio"] == pytest.approx(
            float(np.exp(np.asarray(reference.params)[index])), rel=STAT_RTOL
        )
        assert cells[term]["or_lower"] == pytest.approx(
            float(np.exp(interval[index, 0])), rel=STAT_RTOL
        )
        assert cells[term]["or_upper"] == pytest.approx(
            float(np.exp(interval[index, 1])), rel=STAT_RTOL
        )

    assert result.estimates[2].value == pytest.approx(reference.aic, rel=STAT_RTOL)
    model = _fit(design, event, 0.95)
    assert model.log_likelihood == pytest.approx(reference.llf, rel=STAT_RTOL)
    null = Logit(event, np.ones((event.size, 1))).fit(disp=0)
    assert model.null_log_likelihood == pytest.approx(null.llf, rel=STAT_RTOL)


def test_the_pseudo_r_squared_values_are_their_formulas() -> None:
    labels, event, first, second, third = _sample()
    design = add_constant(np.column_stack([first, second, third]))
    reference = Logit(event, design).fit(disp=0)
    null = Logit(event, np.ones((event.size, 1))).fit(disp=0)
    mcfadden = 1.0 - reference.llf / null.llf
    cox_snell = 1.0 - np.exp(2.0 / event.size * (null.llf - reference.llf))
    maximum = 1.0 - np.exp(2.0 / event.size * null.llf)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    assert result.estimates[0].value == pytest.approx(mcfadden, rel=STAT_RTOL)
    assert result.estimates[1].value == pytest.approx(cox_snell / maximum, rel=STAT_RTOL)


def test_the_area_under_the_curve_is_the_mann_whitney_probability() -> None:
    labels, event, first, second, third = _sample()
    design = add_constant(np.column_stack([first, second, third]))
    model = _fit(design, event, 0.95)
    positive = model.fitted[event == 1.0]
    negative = model.fitted[event == 0.0]
    expected = float(
        np.mean(
            [(above > below) + 0.5 * (above == below) for above in positive for below in negative]
        )
    )

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    assert result.estimates[3].value == pytest.approx(expected, rel=1e-10)


def test_the_rates_at_the_half_cut_off_are_their_counts() -> None:
    labels, event, first, second, third = _sample()
    design = add_constant(np.column_stack([first, second, third]))
    model = _fit(design, event, 0.95)
    predicted = model.fitted >= 0.5
    true_positive = int(np.count_nonzero(predicted & (event == 1.0)))
    false_negative = int(np.count_nonzero(~predicted & (event == 1.0)))
    true_negative = int(np.count_nonzero(~predicted & (event == 0.0)))
    false_positive = int(np.count_nonzero(predicted & (event == 0.0)))

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    assert result.estimates[4].value == pytest.approx(
        true_positive / (true_positive + false_negative)
    )
    assert result.estimates[5].value == pytest.approx(
        true_negative / (true_negative + false_positive)
    )
    assert result.estimates[6].value == pytest.approx((true_positive + true_negative) / event.size)
    rows = _table(result, "Classification at 0.5")
    assert rows[0] == ("yes", true_positive, false_negative)
    assert rows[1] == ("no", false_positive, true_negative)
    assert rows[2] == ("Total", true_positive + false_positive, false_negative + true_negative)


def test_the_ucla_binary_example_reproduces_the_tabulated_coefficients_and_odds_ratios() -> None:
    """The UCLA example's ``logit admit gre gpa i.rank`` table, read on 2026-11-25.

    The source and the date are named in this module's docstring. The printed table rounds its
    numbers, so each is held to that rounding: the coefficients and the odds ratios to five
    parts in a hundred thousand, the odds ratio intervals to a part in a thousand, the log
    likelihood to a part in a million, the likelihood ratio statistic to a part in a thousand,
    and the pseudo R squared to a ten-thousandth.
    """
    admit, gre, gpa, rank = _binary_example()
    labels = np.where(admit == 1.0, "admit", "no admit")
    predictors = (
        _column("gre", gre),
        _column("gpa", gpa),
        _column("rank 2", (rank == 2.0).astype(np.float64)),
        _column("rank 3", (rank == 3.0).astype(np.float64)),
        _column("rank 4", (rank == 4.0).astype(np.float64)),
    )

    result = _run(outcome=_group("admit", labels), predictors=predictors)
    cells = _coefficients(result)

    printed = {
        INTERCEPT: -3.989979,
        "gre": 0.0022644,
        "gpa": 0.8040377,
        "rank 2": -0.6754429,
        "rank 3": -1.340204,
        "rank 4": -1.551464,
    }
    for term, tabulated in printed.items():
        assert cells[term]["estimate"] == pytest.approx(tabulated, rel=5e-5), term

    printed_odds = {
        "gre": (1.002267, 1.00012, 1.004418),
        "gpa": (2.234545, 1.166122, 4.281877),
        "rank 2": (0.5089309, 0.2736922, 0.9463578),
        "rank 3": (0.2617923, 0.1330551, 0.5150889),
        "rank 4": (0.2119375, 0.0934435, 0.4806919),
    }
    for term, (odds, lower, upper) in printed_odds.items():
        assert cells[term]["odds_ratio"] == pytest.approx(odds, rel=5e-5), term
        assert cells[term]["or_lower"] == pytest.approx(lower, rel=1e-3), term
        assert cells[term]["or_upper"] == pytest.approx(upper, rel=1e-3), term

    test = result.tests[0]
    assert test.statistic == pytest.approx(41.46, rel=1e-3)
    assert test.df == 5.0
    assert test.p_value is not None and test.p_value < 1e-6
    assert result.estimates[0].value == pytest.approx(0.0829, abs=1e-4)
    design = add_constant(
        np.column_stack(
            [
                gre,
                gpa,
                (rank == 2.0).astype(np.float64),
                (rank == 3.0).astype(np.float64),
                (rank == 4.0).astype(np.float64),
            ]
        )
    )
    assert _fit(design, admit, 0.95).log_likelihood == pytest.approx(-229.25875, rel=1e-6)
    assert result.counts == Counts(n_offered=400, n_used=400, n_excluded=0)


# --------------------------------------------------------------------------- the properties


def test_adding_a_constant_to_a_predictor_moves_only_the_intercept() -> None:
    labels, _event, first, second, third = _sample(seed=1)
    plain = _coefficients(
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
        )
    )
    shifted = _coefficients(
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", first + 5.0), _column("x2", second), _column("x3", third)),
        )
    )

    assert shifted[INTERCEPT]["estimate"] == pytest.approx(
        float(plain[INTERCEPT]["estimate"]) - 5.0 * float(plain["x1"]["estimate"]), rel=1e-8
    )
    for term in ("x1", "x2", "x3"):
        assert shifted[term]["estimate"] == pytest.approx(
            float(plain[term]["estimate"]), rel=1e-8
        ), term


def test_scaling_a_predictor_scales_its_coefficient_and_leaves_its_statistic() -> None:
    labels, _event, first, second, third = _sample(seed=2)
    plain = _coefficients(
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
        )
    )
    scaled = _coefficients(
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", 3.0 * first), _column("x2", second), _column("x3", third)),
        )
    )

    assert scaled["x1"]["estimate"] == pytest.approx(float(plain["x1"]["estimate"]) / 3.0, rel=1e-8)
    assert scaled["x1"]["se"] == pytest.approx(float(plain["x1"]["se"]) / 3.0, rel=1e-8)
    assert scaled["x1"]["z"] == pytest.approx(float(plain["x1"]["z"]), rel=1e-8)
    assert scaled["x1"]["p"] == pytest.approx(float(plain["x1"]["p"]), rel=1e-8)
    for term in ("x2", "x3", INTERCEPT):
        assert scaled[term]["estimate"] == pytest.approx(
            float(plain[term]["estimate"]), rel=1e-8
        ), term


def test_modelling_the_other_level_negates_every_non_intercept_coefficient() -> None:
    """The event is the level second in order of first appearance, so presenting the rows in
    the opposite order makes the other level the event; the model must then negate."""
    labels, _event, first, second, third = _sample(seed=3)
    plain = _coefficients(
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
        )
    )
    reversed_rows = _coefficients(
        _run(
            outcome=_group("outcome", labels[::-1]),
            predictors=(
                _column("x1", first[::-1]),
                _column("x2", second[::-1]),
                _column("x3", third[::-1]),
            ),
        )
    )

    for term in ("x1", "x2", "x3", INTERCEPT):
        assert reversed_rows[term]["estimate"] == pytest.approx(
            -float(plain[term]["estimate"]), abs=1e-8
        ), term


# --------------------------------------------------------------------------- the blocks


def test_the_result_is_the_contract_with_its_estimates_tests_tables_and_plot() -> None:
    labels, _event, first, second, third = _sample(seed=4)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    assert type(result) is ToolResult
    assert result.tool == "logistic"
    assert result.title == "Logistic regression"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=150, n_used=150, n_excluded=0)
    assert result.excluded == () and result.excluded_unlisted == 0
    assert [estimate.label for estimate in result.estimates] == [
        "McFadden's pseudo R squared",
        "Nagelkerke's pseudo R squared",
        "AIC",
        "Area under the ROC curve",
        "Sensitivity at 0.5",
        "Specificity at 0.5",
        "Accuracy at 0.5",
    ]
    assert len(result.tests) == 5
    assert result.tests[0].name == "Likelihood ratio test of the model"
    assert result.tests[0].df == 3.0
    assert result.tests[0].effect is not None
    assert result.tests[0].effect.value == result.estimates[0].value
    assert [test.name for test in result.tests[1:]] == [
        "(Intercept) Wald test, two-sided",
        "x1 Wald test, two-sided",
        "x2 Wald test, two-sided",
        "x3 Wald test, two-sided",
    ]
    for test in result.tests[1:]:
        assert test.method == "Wald z test, two-sided"
        assert test.effect is not None and test.effect.value is not None
    assert [table.title for table in result.tables] == [
        "Coefficients",
        "Classification at 0.5",
    ]
    assert [column.label for column in result.tables[0].columns] == [
        "Term",
        "Estimate",
        "SE",
        "z",
        "p",
        "Lower",
        "Upper",
        "Odds ratio",
        "OR lower",
        "OR upper",
    ]
    assert [check.id for check in result.assumptions] == [
        "hosmer_lemeshow",
        "linearity",
        "independence",
    ]
    assert result.assumptions[1].verdict is Verdict.NOT_APPLICABLE
    assert (
        result.assumptions[1].next_step == "Check the fitted probabilities against each predictor."
    )
    assert result.assumptions[2].verdict is Verdict.NOT_APPLICABLE
    assert [type(plot) for plot in result.plots] == [ScatterPlot]
    plot = result.plots[0]
    assert isinstance(plot, ScatterPlot)
    assert plot.title == "Calibration"
    assert len(plot.x) == len(plot.y) == 10
    assert plot.lines[0].label == "Perfect calibration"
    assert plot.lines[0].x0 == plot.lines[0].y0 == 0.0
    assert plot.lines[0].x1 == plot.lines[0].y1 == 1.0
    assert result.tests[0].p_value_text == p_text(result.tests[0].p_value)


def test_the_calibration_plot_and_the_hosmer_lemeshow_test_use_the_same_deciles() -> None:
    labels, event, first, second, third = _sample(seed=5)
    design = add_constant(np.column_stack([first, second, third]))
    model = _fit(design, event, 0.95)
    order = np.argsort(model.fitted, kind="stable")
    groups = np.array_split(order, 10)
    statistic = 0.0
    for group in groups:
        size = float(group.size)
        observed = float(event[group].sum())
        expected = float(model.fitted[group].sum())
        statistic += (observed - expected) ** 2 / (expected * (1.0 - expected / size))

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    check = result.assumptions[0]
    assert check.statistic == f"chi-square = {num(statistic)}"
    assert "Hosmer-Lemeshow" in check.evidence
    assert check.verdict is not Verdict.NOT_APPLICABLE
    plot = result.plots[0]
    assert isinstance(plot, ScatterPlot)
    assert plot.x == tuple(float(np.mean(model.fitted[group])) for group in groups)
    assert plot.y == tuple(float(np.mean(event[group])) for group in groups)


def test_the_predictions_table_is_statsmodels_prediction_with_a_wald_interval() -> None:
    labels, event, first, second, third = _sample(size=80, seed=6)
    design = add_constant(np.column_stack([first, second, third]))
    reference = Logit(event, design).fit(disp=0)
    points = ((0.0, 0.0, 0.0), (1.0, -1.0, 0.5))
    centred = add_constant(np.asarray(points))
    predicted = np.asarray(reference.predict(centred))
    covariance = np.asarray(reference.cov_params())
    critical = float(scipy_stats.norm.ppf(0.975))

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
        predictions=points,
    )
    rows = _table(result, "Predictions")

    assert len(rows) == 2
    for index, row in enumerate(rows):
        eta = float(centred[index] @ np.asarray(reference.params))
        error = float(np.sqrt(centred[index] @ covariance @ centred[index]))
        assert row[0] == index + 1
        assert row[1] == pytest.approx(float(predicted[index]), rel=STAT_RTOL)
        assert row[2] == pytest.approx(expit(eta - critical * error), rel=STAT_RTOL)
        assert row[3] == pytest.approx(expit(eta + critical * error), rel=STAT_RTOL)


def test_the_predictions_table_appears_only_when_asked_for() -> None:
    labels, _event, first, second, third = _sample(size=60, seed=7)
    request = {
        "outcome": _group("outcome", labels),
        "predictors": (_column("x1", first), _column("x2", second), _column("x3", third)),
    }

    without = _run(**request)
    with_predictions = _run(**request, predictions=((0.0, 0.0, 0.0),))

    assert "Predictions" not in [table.title for table in without.tables]
    assert [table.title for table in with_predictions.tables] == [
        "Coefficients",
        "Classification at 0.5",
        "Predictions",
    ]


def test_the_default_names_are_the_predictors_own_names() -> None:
    labels, _event, first, second, third = _sample(size=60, seed=8)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
        names=("first", "second", "third"),
    )

    assert [row[0] for row in result.tables[0].rows] == [
        INTERCEPT,
        "first",
        "second",
        "third",
    ]


def test_the_result_survives_a_json_round_trip_without_a_nan() -> None:
    labels, _event, first, second, third = _sample(size=60, seed=9)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )
    payload = result.model_dump_json()

    assert "NaN" not in payload
    assert ToolResult.model_validate_json(payload) == result


# --------------------------------------------------------------------------- missing data


def test_a_row_missing_in_a_predictor_is_left_out_and_listed() -> None:
    labels = _group("outcome", ["no", "no", "yes", "yes", "no", "yes"])
    predictor = _column("x", [1.0, 2.0, float("nan"), 4.0, 5.0, 7.0])

    result = _run(outcome=labels, predictors=(predictor,))

    assert result.counts == Counts(n_offered=6, n_used=5, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in x"),)


def test_a_row_without_an_outcome_label_is_left_out_and_listed() -> None:
    outcome = _group("outcome", ["no", "no", "yes", "yes"], rows=(0, 1, 3, 4))
    predictor = _column("x", [1.0, 2.0, 3.0, 4.0, 5.0])

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="no group label in outcome"),)


def test_a_row_with_an_infinite_value_is_listed_as_not_finite() -> None:
    labels = _group("outcome", ["no", "no", "yes", "yes"])
    predictor = _column("x", [1.0, 2.0, float("inf"), 4.0])

    result = _run(outcome=labels, predictors=(predictor,))

    assert result.excluded == (ExcludedRow(index=2, reason="not a finite number in x"),)


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    labels = np.where(np.arange(11) % 2 == 0, "no", "yes")
    outcome = _group("outcome", labels, rows=range(205, 216))
    predictor = _column("x", np.arange(216.0))

    result = _run(outcome=outcome, predictors=(predictor,))

    assert result.counts == Counts(n_offered=216, n_used=11, n_excluded=205)
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


# --------------------------------------------------------------------------- the refusals


def test_no_predictor_or_more_than_twenty_is_refused() -> None:
    labels = _group("outcome", ["no", "no", "yes", "yes"])
    with pytest.raises(ToolError) as refused:
        _run(outcome=labels, predictors=())
    assert refused.value.code == "INVALID_OPTION"

    many = tuple(_column(f"x{index}", np.arange(25.0)) for index in range(21))
    with pytest.raises(ToolError) as refused:
        _run(outcome=labels, predictors=many)
    assert refused.value.code == "INVALID_OPTION"
    assert "between 1 and 20" in refused.value.message


def test_names_of_the_wrong_length_are_refused() -> None:
    labels, _event, first, _second, _third = _sample(size=12, seed=10)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_group("outcome", labels), predictors=(_column("x", first),), names=("a", "b"))
    assert refused.value.code == "INVALID_OPTION"
    assert "one name per predictor" in refused.value.message


def test_more_than_a_thousand_prediction_rows_are_refused() -> None:
    labels, _event, first, _second, _third = _sample(size=12, seed=11)
    too_many = tuple((0.0,) for _ in range(1_001))
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x", first),),
            predictions=too_many,
        )
    assert refused.value.code == "INVALID_OPTION"


def test_a_prediction_row_of_the_wrong_length_is_refused() -> None:
    labels, _event, first, second, _third = _sample(size=12, seed=12)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x1", first), _column("x2", second)),
            predictions=((0.0,),),
        )
    assert refused.value.code == "INVALID_OPTION"
    assert "prediction row 1" in refused.value.message


def test_a_non_finite_prediction_is_refused() -> None:
    labels, _event, first, _second, _third = _sample(size=12, seed=13)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_group("outcome", labels),
            predictors=(_column("x", first),),
            predictions=((float("nan"),),),
        )
    assert refused.value.code == "INVALID_OPTION"


def test_an_outcome_without_exactly_two_levels_is_refused() -> None:
    predictor = _column("x", np.arange(12.0))
    with pytest.raises(ToolError) as refused:
        _run(outcome=_group("outcome", ["yes"] * 12), predictors=(predictor,))
    assert refused.value.code == "LEVELS_REQUIRED"

    with pytest.raises(ToolError) as refused:
        _run(outcome=_group("outcome", ["a", "b", "c"] * 4), predictors=(predictor,))
    assert refused.value.code == "LEVELS_REQUIRED"


def test_a_constant_predictor_is_refused() -> None:
    labels, _event, _first, _second, _third = _sample(size=12, seed=14)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_group("outcome", labels), predictors=(_column("c", [2.0] * 12),))
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "c" in refused.value.message


def test_an_exactly_collinear_predictor_is_refused() -> None:
    rng = np.random.default_rng(15)
    first = rng.normal(size=15)
    second = rng.normal(size=15)
    labels = np.where(np.arange(15) % 2 == 0, "no", "yes")
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_group("outcome", labels),
            predictors=(
                _column("a", first),
                _column("b", second),
                _column("c", first + second),
            ),
        )
    assert refused.value.code == "COLLINEAR"
    assert "a, b and c" in refused.value.message


def test_fewer_rows_than_predictors_plus_two_is_refused() -> None:
    labels = _group("outcome", ["no", "yes"] * 5 + ["no"])
    predictors = tuple(_column(f"x{index}", np.arange(11.0) * (index + 1)) for index in range(10))
    with pytest.raises(ToolError) as refused:
        _run(outcome=labels, predictors=predictors)
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 12 complete rows for 10 predictors" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    labels, _event, first, _second, _third = _sample(size=12, seed=16)
    with pytest.raises(ToolError) as refused:
        _run(outcome=_group("outcome", labels), predictors=(_column("x", first),), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_logistic(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    labels, _event, first, _second, _third = _sample(size=12, seed=17)
    with pytest.raises(ValidationError):
        LogisticRequest(  # type: ignore[call-arg]
            outcome=_group("outcome", labels), predictors=(_column("x", first),), method="logit"
        )


# --------------------------------------------------------------------------- separation and words


def test_perfect_separation_is_reported_and_the_coefficients_are_shown() -> None:
    predictor = np.arange(8.0)
    labels = np.where(predictor >= 4.0, "yes", "no")

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x", predictor),),
    )

    assert result.warnings
    assert any("separation" in warning and "x" in warning for warning in result.warnings)
    rows = result.tables[0].rows
    assert len(rows) == 2
    assert rows[1][0] == "x"
    assert rows[1][1] is not None
    assert result.estimates[3].value == pytest.approx(1.0)
    assert "NaN" not in result.model_dump_json()


def test_fewer_than_ten_rows_warns_and_a_small_group_warns() -> None:
    predictor = np.arange(8.0)
    labels = np.where(predictor >= 4.0, "yes", "no")

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x", predictor),),
    )

    assert any("fewer than 10" in warning for warning in result.warnings)
    assert any("Hosmer-Lemeshow group" in warning for warning in result.warnings)


def test_the_words_name_the_event_and_claim_no_proof() -> None:
    labels, _event, first, second, third = _sample(seed=18)

    result = _run(
        outcome=_group("outcome", labels),
        predictors=(_column("x1", first), _column("x2", second), _column("x3", third)),
    )

    assert "yes" in result.interpretation and "no" in result.interpretation
    assert "association" in result.interpretation
    assert "causes" in result.interpretation
    assert "iteratively reweighted" in result.methods_fragment
    assert "Wald" in result.methods_fragment
    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
    for text in texts:
        assert not claims_proof(text), text


# ---------------------------------------------------------------------- the words and the registry


def test_every_request_field_is_described_and_the_names_field_is_labelled() -> None:
    found = next(info for info in catalogue() if info.name == "logistic")

    assert found.family == "relationships"
    assert found.title == "Logistic regression"
    assert [field.name for field in found.fields] == [
        "outcome",
        "predictors",
        "names",
        "predictions",
        "level",
    ]
    assert all(field.description for field in found.fields)
    assert found.fields[0].kind == "group"
    assert found.fields[1].kind == "columns"
    assert found.fields[2].label == "Predictor names"


def test_the_registry_builds_and_runs_the_tool_from_column_names() -> None:
    labels, _event, first, second, third = _sample(size=60, seed=19)
    columns = {
        "x1": _column("x1", first),
        "x2": _column("x2", second),
        "x3": _column("x3", third),
    }
    groups = {"outcome": _group("outcome", labels)}

    request = build_request(
        "logistic",
        {
            "outcome": "outcome",
            "predictors": ["x1", "x2", "x3"],
            "names": ["first", "second", "third"],
            "predictions": [[0.0, 0.0, 0.0]],
            "level": 0.9,
        },
        column=columns.__getitem__,
        group=groups.__getitem__,
    )
    result = run_tool("logistic", request)

    assert isinstance(request, LogisticRequest)
    assert result.tool == "logistic"
    assert result.level == 0.9
    assert result.tables[0].rows[1][0] == "first"
    assert "curvestudio.tools.stats.logistic" in TOOL_MODULES
