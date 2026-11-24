"""Deming and Passing-Bablok regression between two methods.

Every number is checked against an independent implementation written here from the formulas.
Deming's slope is the closed-form solution of the Deming equations, ``Sxy b**2 +
(error_ratio Sxx - Syy) b - error_ratio Sxy = 0``, with the root whose sign follows the
covariance; its intercept follows from the slope; and its standard errors are the delete-one
jackknife over the pairs, ``sqrt((n - 1) / n)`` times the spread of the leave-one-out
estimates. Passing-Bablok is checked against a brute-force implementation that forms the
angle of every pair of points and applies the shifted-median and rank rules of Passing and
Bablok (1983) directly.

The worked example is the ``X.td15`` / ``Y.td15`` data set and the ``calanas`` reference values
in the ``mcr`` R package's unit test ``inst/unitTests/runit.mcPaBa.R``, function
``test.mc.paba.call``, fetched from github.com/cran/mcr on 2026-11-24; the package's help page
``man/mcreg.Rd`` documents the same exact algorithm (the radian slope measure, the shifted
median, and the rank-based intervals). The values there are the ones the ``calanas`` program
print for 50 pairs, reproduced here to the tolerance stated below. Passing and Bablok (1983),
"A new biometrical procedure for testing the equality of measurements from two different
analytical methods", J Clin Chem Clin Biochem 21:709-720, is not freely available, so the mcr
package is the fetched public source and the algorithm is written out independently in this
file.

The properties each estimator has are checked too: Deming's fit is symmetric when the error
ratio is one, so swapping the two methods turns the slope into its reciprocal; and every
surface states evidence, never proof.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import scipy.stats as scipy_stats
from pydantic import ValidationError

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    ScatterPlot,
    ToolResult,
    Verdict,
    build_request,
    run_tool,
)
from curvestudio.tools.stats.comparison_lines import (
    DemingRequest,
    PassingBablokRequest,
    _deming_fit,
    run_deming,
    run_passingbablok,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue

Runner = Callable[..., ToolResult]
"""A tool's run function, called with the options of its request as keywords."""

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""

TIE_EPS = 1e-12
"""The relative difference below which two readings count as equal."""

CALANAS_TD15_SLOPE = 1.02058852149448
CALANAS_TD15_SLOPE_LCI = 0.95848878631593
CALANAS_TD15_SLOPE_UCI = 1.09232009780764
CALANAS_TD15_INTERCEPT = -0.639752984809869
CALANAS_TD15_INTERCEPT_LCI = -2.3330808182894
CALANAS_TD15_INTERCEPT_UCI = 0.840973102642057


class _PabaExpected(NamedTuple):
    """A Passing-Bablok fit as the brute-force reference computes it."""

    slope: float
    intercept: float
    slope_lower: float | None
    slope_upper: float | None
    intercept_lower: float | None
    intercept_upper: float | None


def _column(name: str, values: object) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    return Column(name=name, values=array, rows=np.arange(array.size, dtype=np.intp))


def _deming(x: np.ndarray, y: np.ndarray, error_ratio: float) -> tuple[float, float]:
    """The Deming slope and intercept, written out from the quadratic in the slope."""
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centred_x = x - mean_x
    centred_y = y - mean_y
    sxx = float(centred_x @ centred_x)
    syy = float(centred_y @ centred_y)
    sxy = float(centred_x @ centred_y)
    difference = syy - error_ratio * sxx
    slope = (difference + math.sqrt(difference**2 + 4.0 * error_ratio * sxy**2)) / (2.0 * sxy)
    return slope, mean_y - slope * mean_x


def _jackknife(values: list[float]) -> float:
    """The delete-one jackknife standard error of the estimates."""
    size = len(values)
    mean = sum(values) / size
    return math.sqrt((size - 1) / size * sum((value - mean) ** 2 for value in values))


def _deming_expected(
    x: np.ndarray, y: np.ndarray, error_ratio: float = 1.0
) -> tuple[float, float, float, float]:
    """The Deming slope, intercept and both jackknife standard errors."""
    slope, intercept = _deming(x, y, error_ratio)
    slopes: list[float] = []
    intercepts: list[float] = []
    for leave_out in range(x.size):
        keep = np.ones(x.size, dtype=bool)
        keep[leave_out] = False
        left, right = _deming(x[keep], y[keep], error_ratio)
        slopes.append(left)
        intercepts.append(right)
    return slope, intercept, _jackknife(slopes), _jackknife(intercepts)


def _paba_reference(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> _PabaExpected:
    """Passing-Bablok by brute force: the angle of every pair, the shifted median, the ranks."""
    size = x.size
    positive = float(scipy_stats.kendalltau(x, y).statistic) >= 0.0
    angles: list[float] = []
    for left in range(size):
        for right in range(left + 1, size):
            dx = float(x[right] - x[left])
            dy = float(y[right] - y[left])
            same_x = dx == 0.0 or abs(dx) < TIE_EPS * (abs(x[left]) + abs(x[right])) / 2.0
            same_y = dy == 0.0 or abs(dy) < TIE_EPS * (abs(y[left]) + abs(y[right])) / 2.0
            if same_x:
                if not same_y:
                    angles.append(math.pi / 2.0 if positive else -math.pi / 2.0)
            else:
                angles.append(math.atan(dy / dx))
    ordered = sorted(angles)
    total = len(ordered)
    below = sum(1 for angle in ordered if angle <= -math.pi / 4.0)
    below_strict = sum(1 for angle in ordered if angle < -math.pi / 4.0)
    above = sum(1 for angle in ordered if angle >= math.pi / 4.0)
    above_strict = sum(1 for angle in ordered if angle > math.pi / 4.0)
    offset = below + below_strict if positive else -(above + above_strict)

    ranked = total + offset
    half = (ranked + 1) // 2
    if ranked % 2 == 0:
        slope = math.tan((ordered[half - 1] + ordered[half]) / 2.0)
    else:
        slope = math.tan(ordered[half - 1])

    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    span = math.floor(quantile * math.sqrt(size * (size - 1.0) * (2.0 * size + 5.0) / 18.0) + 0.5)
    lowest = 2 * (below - below_strict) + 1 if positive else 2 * (above - above_strict) + 1

    def slope_bound(rank: int, *, upper: bool) -> float | None:
        index = (rank + 1) // 2
        if (index < lowest) if not upper else (index > total):
            return None
        if rank % 2 == 0:
            if index >= total:
                return None
            return math.tan((ordered[index - 1] + ordered[index]) / 2.0)
        return math.tan(ordered[index - 1])

    slope_lower = slope_bound(total - span + offset, upper=False)
    slope_upper = slope_bound(total + span + offset, upper=True)
    intercept = float(np.median(y - slope * x))
    intercept_lower = None if slope_upper is None else float(np.median(y - slope_upper * x))
    intercept_upper = None if slope_lower is None else float(np.median(y - slope_lower * x))
    return _PabaExpected(
        slope=slope,
        intercept=intercept,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
        intercept_lower=intercept_lower,
        intercept_upper=intercept_upper,
    )


def _sample(n: int = 40, seed: int = 0, noise: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Two methods that measure a common reading, the second with independent noise."""
    rng = np.random.default_rng(seed)
    first = rng.normal(50.0, 10.0, n)
    return first, first + rng.normal(0.0, noise, n)


def _run_deming(**options: object) -> ToolResult:
    return run_deming(DemingRequest(**options))  # type: ignore[arg-type]


def _run_paba(**options: object) -> ToolResult:
    return run_passingbablok(PassingBablokRequest(**options))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- Deming


def test_deming_matches_the_closed_form_and_its_jackknife_standard_errors() -> None:
    first, second = _sample(seed=0)
    slope, intercept, se_slope, se_intercept = _deming_expected(first, second)

    result = _run_deming(method_a=_column("a", first), method_b=_column("b", second))

    found_slope, found_intercept = result.estimates
    assert found_slope.label == "Slope"
    assert found_intercept.label == "Intercept"
    assert found_slope.value == pytest.approx(slope, rel=STAT_RTOL)
    assert found_slope.se == pytest.approx(se_slope, rel=STAT_RTOL)
    assert found_intercept.value == pytest.approx(intercept, rel=STAT_RTOL)
    assert found_intercept.se == pytest.approx(se_intercept, rel=STAT_RTOL)

    critical = float(scipy_stats.t.ppf(0.975, first.size - 1))
    assert found_slope.lower == pytest.approx(slope - critical * se_slope, rel=STAT_RTOL)
    assert found_slope.upper == pytest.approx(slope + critical * se_slope, rel=STAT_RTOL)
    assert found_intercept.lower == pytest.approx(
        intercept - critical * se_intercept, rel=STAT_RTOL
    )
    assert found_intercept.upper == pytest.approx(
        intercept + critical * se_intercept, rel=STAT_RTOL
    )


def test_deming_reads_the_error_ratio_as_the_second_method_over_the_first() -> None:
    first, second = _sample(seed=1)
    centred_x = first - first.mean()
    centred_y = second - second.mean()
    ordinary = float(centred_x @ centred_y) / float(centred_x @ centred_x)
    reverse = float(centred_y @ centred_y) / float(centred_x @ centred_y)

    loud = _run_deming(
        method_a=_column("a", first), method_b=_column("b", second), error_ratio=1.0e9
    )
    quiet = _run_deming(
        method_a=_column("a", first), method_b=_column("b", second), error_ratio=1.0e-9
    )

    assert loud.estimates[0].value == pytest.approx(ordinary, rel=1e-6)
    assert quiet.estimates[0].value == pytest.approx(reverse, rel=1e-6)


def test_swapping_the_methods_at_equal_error_turns_the_deming_slope_into_its_reciprocal() -> None:
    first, second = _sample(seed=2)

    forwards = _run_deming(method_a=_column("a", first), method_b=_column("b", second))
    backwards = _run_deming(method_a=_column("b", second), method_b=_column("a", first))

    assert forwards.estimates[0].value * backwards.estimates[0].value == pytest.approx(
        1.0, rel=STAT_RTOL
    )


def test_the_deming_level_moves_only_the_intervals() -> None:
    first, second = _sample(n=30, seed=3)

    ninety = _run_deming(method_a=_column("a", first), method_b=_column("b", second), level=0.90)
    default = _run_deming(method_a=_column("a", first), method_b=_column("b", second))

    assert ninety.estimates[0].value == pytest.approx(default.estimates[0].value, rel=STAT_RTOL)
    assert ninety.estimates[0].se == pytest.approx(default.estimates[0].se or 0.0, rel=STAT_RTOL)
    assert ninety.estimates[0].lower is not None and default.estimates[0].lower is not None
    assert ninety.estimates[0].lower > default.estimates[0].lower


# --------------------------------------------------------------------------- Passing-Bablok


def test_passing_bablok_matches_the_brute_force_reference() -> None:
    first, second = _sample(n=40, seed=4)
    expected = _paba_reference(first, second)

    result = _run_paba(method_a=_column("a", first), method_b=_column("b", second))

    slope, intercept = result.estimates
    assert slope.label == "Slope"
    assert intercept.label == "Intercept"
    assert slope.value == pytest.approx(expected.slope, rel=STAT_RTOL)
    assert slope.lower == pytest.approx(expected.slope_lower, rel=STAT_RTOL)
    assert slope.upper == pytest.approx(expected.slope_upper, rel=STAT_RTOL)
    assert slope.se is None
    assert intercept.value == pytest.approx(expected.intercept, rel=STAT_RTOL)
    assert intercept.lower == pytest.approx(expected.intercept_lower, rel=STAT_RTOL)
    assert intercept.upper == pytest.approx(expected.intercept_upper, rel=STAT_RTOL)
    assert intercept.se is None


def test_passing_bablok_handles_a_falling_relation() -> None:
    rng = np.random.default_rng(5)
    first = rng.normal(30.0, 5.0, 35)
    second = 90.0 - 2.0 * first + rng.normal(0.0, 3.0, 35)
    expected = _paba_reference(first, second)

    result = _run_paba(method_a=_column("a", first), method_b=_column("b", second))

    assert result.estimates[0].value == pytest.approx(expected.slope, rel=STAT_RTOL)
    assert result.estimates[1].value == pytest.approx(expected.intercept, rel=STAT_RTOL)
    assert result.estimates[0].lower == pytest.approx(expected.slope_lower, rel=STAT_RTOL)
    assert result.estimates[0].upper == pytest.approx(expected.slope_upper, rel=STAT_RTOL)


def test_passing_bablok_reproduces_the_mcr_td15_worked_example() -> None:
    """The X.td15/Y.td15 data and calanas values from mcr's runit.mcPaBa.R, fetched 2026-11-24.

    The reference values are the calanas columns in ``test.mc.paba.call`` for TD15: slope
    1.02058852149448 with interval 0.95848878631593 to 1.09232009780764, intercept
    -0.639752984809869 with interval -2.3330808182894 to 0.840973102642057.
    """
    first = np.asarray(
        [
            18.654599109456,
            17.0610823229594,
            37.8076135618173,
            51.1413961380522,
            103.985667589592,
            34.8605133072495,
            35.1146708951621,
            16.005996343635,
            68.5890100912778,
            62.1699013684068,
            18.1641040703209,
            94.2563842860796,
            31.67278631354,
            18.8059773112482,
            67.9347403613216,
            18.260239156489,
            87.1550332248675,
            98.5843421029197,
            20.3747255514258,
            4.20199417418543,
            27.7662087802102,
            16.6797331091928,
            52.3335230459024,
            1.25970439645986,
            2.55285163008995,
            71.1422300479687,
            99.3428902581639,
            40.6747911609727,
            82.5924169862313,
            64.7062991090824,
            1.46968233573658,
            83.5389530283147,
            19.8368624090347,
            44.339918470983,
            19.2879672940529,
            86.5484515948081,
            79.8461255550168,
            66.1164680506251,
            32.8785828616434,
            85.634465208222,
            15.772867025856,
            102.866851788539,
            16.4442973704484,
            23.0916021823352,
            83.3011761592876,
            9.47357732139885,
            15.4574423263928,
            67.0394769874472,
            32.7239965276836,
            71.7361088013058,
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [
            15.9116325401596,
            17.9423397346645,
            58.1755698848695,
            49.0779434653576,
            113.758268975683,
            35.5525894416901,
            39.4088027819502,
            15.0167183931276,
            65.8450151148211,
            62.6667874119841,
            15.8020862675435,
            83.9418317622942,
            33.5524291118479,
            18.5775089262923,
            73.872592767028,
            21.600509295785,
            93.4269960011703,
            89.8278248717022,
            20.130360709819,
            4.94910345310486,
            23.1566064984898,
            17.336339530418,
            56.2280624424048,
            1.70969794990576,
            3.40128933880261,
            87.5197563754811,
            67.3466761237754,
            35.6882873995953,
            90.4041198524321,
            62.2603160684644,
            1.52718831259026,
            84.5412183739216,
            19.0003036715583,
            51.0624306952549,
            12.0291575952956,
            55.3970074896586,
            96.7628055558332,
            69.6940969840453,
            29.1380990869214,
            85.4597993067675,
            18.1508502210932,
            92.1385811560549,
            17.012175857013,
            23.0242126162252,
            95.7510267835212,
            8.70125548126123,
            13.4955271740672,
            45.8284262464427,
            31.1175741976669,
            69.5490311988868,
        ],
        dtype=np.float64,
    )

    result = _run_paba(method_a=_column("a", first), method_b=_column("b", second))
    slope, intercept = result.estimates

    assert slope.value == pytest.approx(CALANAS_TD15_SLOPE, rel=STAT_RTOL)
    assert slope.lower == pytest.approx(CALANAS_TD15_SLOPE_LCI, rel=STAT_RTOL)
    assert slope.upper == pytest.approx(CALANAS_TD15_SLOPE_UCI, rel=STAT_RTOL)
    assert intercept.value == pytest.approx(CALANAS_TD15_INTERCEPT, rel=STAT_RTOL)
    assert intercept.lower == pytest.approx(CALANAS_TD15_INTERCEPT_LCI, rel=STAT_RTOL)
    assert intercept.upper == pytest.approx(CALANAS_TD15_INTERCEPT_UCI, rel=STAT_RTOL)


def test_the_passing_bablok_level_moves_the_intervals() -> None:
    first, second = _sample(n=30, seed=6)
    expected = _paba_reference(first, second, level=0.90)

    result = _run_paba(method_a=_column("a", first), method_b=_column("b", second), level=0.90)

    assert result.estimates[0].lower == pytest.approx(expected.slope_lower, rel=STAT_RTOL)
    assert result.estimates[0].upper == pytest.approx(expected.slope_upper, rel=STAT_RTOL)
    assert result.estimates[0].value == pytest.approx(
        _paba_reference(first, second).slope, rel=STAT_RTOL
    )


def test_three_pairs_give_a_slope_and_an_intercept_but_no_interval() -> None:
    result = _run_paba(
        method_a=_column("a", [1.0, 2.0, 3.0]),
        method_b=_column("b", [1.1, 2.0, 3.2]),
    )

    assert result.estimates[0].value == pytest.approx(1.05, rel=STAT_RTOL)
    assert result.estimates[0].lower is None and result.estimates[0].upper is None
    assert result.estimates[1].lower is None and result.estimates[1].upper is None
    assert result.tables == ()


# --------------------------------------------------------------------------- the blocks


@pytest.mark.parametrize(
    ("tool", "run"),
    [("deming", _run_deming), ("passingbablok", _run_paba)],
)
def test_the_result_is_the_contract(tool: str, run: Runner) -> None:
    first, second = _sample(n=25, seed=7)
    result = run(method_a=_column("a", first), method_b=_column("b", second))

    assert type(result) is ToolResult
    assert result.tool == tool
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=25, n_used=25, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert [estimate.label for estimate in result.estimates] == ["Slope", "Intercept"]
    assert result.tests == ()
    assert len(result.assumptions) == 1
    assert result.assumptions[0].verdict is Verdict.NOT_APPLICABLE
    assert [table.title for table in result.tables] == ["Parameters"]


def test_the_deming_parameters_table_names_the_jackknife_and_holds_the_standard_errors() -> None:
    first, second = _sample(n=20, seed=8)

    result = _run_deming(method_a=_column("a", first), method_b=_column("b", second))

    parameters = result.tables[0]
    assert [column.label for column in parameters.columns] == [
        "Parameter",
        "Estimate",
        "SE",
        "Lower",
        "Upper",
    ]
    assert parameters.note is not None
    assert parameters.note.startswith("Deming regression; jackknife standard errors")
    assert [row[0] for row in parameters.rows] == ["Slope", "Intercept"]
    assert parameters.rows[0][2] == pytest.approx(result.estimates[0].se or 0.0)


def test_the_passing_bablok_parameters_table_names_the_method_and_leaves_the_se_empty() -> None:
    first, second = _sample(n=25, seed=9)

    result = _run_paba(method_a=_column("a", first), method_b=_column("b", second))

    parameters = result.tables[0]
    assert parameters.note is not None
    assert parameters.note.startswith("Passing\u2013Bablok regression")
    assert [row[2] for row in parameters.rows] == [None, None]


@pytest.mark.parametrize(
    ("tool", "run"),
    [("deming", _run_deming), ("passingbablok", _run_paba)],
)
def test_the_plot_draws_method_b_against_a_with_the_fitted_and_identity_lines(
    tool: str, run: Runner
) -> None:
    first, second = _sample(n=25, seed=10)
    result = run(method_a=_column("a", first), method_b=_column("b", second))

    assert len(result.plots) == 1
    plot = result.plots[0]
    assert isinstance(plot, ScatterPlot)
    assert plot.x_label == "a" and plot.y_label == "b"
    assert np.allclose(np.asarray(plot.x), first)
    assert np.allclose(np.asarray(plot.y), second)
    assert [line.label for line in plot.lines] == ["Fitted line", "Identity"]
    assert plot.lines[1].y0 == plot.lines[1].x0
    assert plot.lines[1].y1 == plot.lines[1].x1
    slope = result.estimates[0].value
    intercept = result.estimates[1].value
    assert plot.lines[0].y0 == pytest.approx(intercept + slope * plot.lines[0].x0)
    assert plot.lines[0].y1 == pytest.approx(intercept + slope * plot.lines[0].x1)


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    first, second = _sample(seed=11)
    demands = _run_deming(method_a=_column("a", first), method_b=_column("b", second))
    paba = _run_paba(method_a=_column("a", first), method_b=_column("b", second))

    for result in (demands, paba):
        texts = [result.title, result.interpretation, result.methods_fragment]
        for check in result.assumptions:
            texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
        for table_block in result.tables:
            texts.extend([table_block.title, table_block.note or ""])
        for plot in result.plots:
            texts.extend([plot.title, plot.note or ""])
        for text in texts:
            assert not claims_proof(text), text


# --------------------------------------------------------------------------- missing data


@pytest.mark.parametrize(("tool", "run"), [("deming", _run_deming), ("passingbablok", _run_paba)])
def test_a_row_missing_in_one_method_is_left_out_of_both_and_listed(tool: str, run: Runner) -> None:
    first = _column("a", [1.0, 2.0, float("nan"), 4.0, 5.0])
    second = _column("b", [1.0, 2.0, 3.0, 4.0, 5.0])

    result = run(method_a=first, method_b=second)

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in a"),)


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    first = _column("a", [float("nan")] * 205 + list(np.arange(1.0, 11.0)))
    second = _column("b", np.arange(1.0, 216.0))

    result = _run_paba(method_a=first, method_b=second)

    assert result.counts.n_excluded == 205
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


# --------------------------------------------------------------------------- the refusals


@pytest.mark.parametrize(("tool", "run"), [("deming", _run_deming), ("passingbablok", _run_paba)])
def test_a_constant_method_is_refused(tool: str, run: Runner) -> None:
    with pytest.raises(ToolError) as refused:
        run(
            method_a=_column("a", [3.0, 3.0, 3.0, 3.0]), method_b=_column("b", [1.0, 2.0, 4.0, 8.0])
        )
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "a" in refused.value.message


@pytest.mark.parametrize(("tool", "run"), [("deming", _run_deming), ("passingbablok", _run_paba)])
def test_fewer_than_three_pairs_is_refused(tool: str, run: Runner) -> None:
    with pytest.raises(ToolError) as refused:
        run(method_a=_column("a", [1.0, 2.0]), method_b=_column("b", [1.0, 3.0]))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 3 complete pairs" in refused.value.message


def test_deming_refuses_more_than_ten_thousand_pairs() -> None:
    """Its jackknife refits the line once per pair, so the cost is quadratic: the tool says
    so rather than tying a request up for over a minute."""
    values = np.arange(1.0, 10_002.0)

    with pytest.raises(ToolError) as refused:
        _run_deming(method_a=_column("a", values), method_b=_column("b", values * 1.01))

    assert refused.value.code == "TOO_MANY_ROWS"
    assert "10,000" in refused.value.message
    assert "jackknife" in refused.value.message


def test_passing_bablok_refuses_more_than_a_thousand_pairs() -> None:
    values = np.arange(1.0, 1002.0)

    with pytest.raises(ToolError) as refused:
        _run_paba(method_a=_column("a", values), method_b=_column("b", values * 1.01))

    assert refused.value.code == "TOO_MANY_ROWS"
    assert "1,000" in refused.value.message


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_a_deming_error_ratio_outside_zero_and_infinity_is_refused(ratio: float) -> None:
    first, second = _sample(n=10, seed=12)
    with pytest.raises(ToolError) as refused:
        _run_deming(method_a=_column("a", first), method_b=_column("b", second), error_ratio=ratio)
    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize(("tool", "run"), [("deming", _run_deming), ("passingbablok", _run_paba)])
@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(
    tool: str, run: Runner, level: float
) -> None:
    first, second = _sample(n=10, seed=13)
    with pytest.raises(ToolError) as refused:
        run(method_a=_column("a", first), method_b=_column("b", second), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_deming(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as refused:
        run_passingbablok(
            DemingRequest(
                method_a=_column("a", [1.0, 2.0, 3.0]), method_b=_column("b", [1.0, 2.0, 3.0])
            )
        )
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    first, second = _sample(n=10, seed=14)
    with pytest.raises(ValidationError):
        PassingBablokRequest(  # type: ignore[call-arg]
            method_a=_column("a", first),
            method_b=_column("b", second),
            error_ratio=2.0,
        )


# ---------------------------------------------------------------------- the registry


@pytest.mark.parametrize(
    ("name", "title", "fields", "labels", "kinds"),
    [
        (
            "deming",
            "Deming regression",
            ["method_a", "method_b", "error_ratio", "level"],
            ["Method A", "Method B", "Error ratio", "Level"],
            ["column", "column", "option", "option"],
        ),
        (
            "passingbablok",
            "Passing\u2013Bablok regression",
            ["method_a", "method_b", "level"],
            ["Method A", "Method B", "Level"],
            ["column", "column", "option"],
        ),
    ],
)
def test_every_request_field_is_described_and_the_tool_is_in_its_family(
    name: str,
    title: str,
    fields: list[str],
    labels: list[str],
    kinds: list[str],
) -> None:
    found = next(info for info in catalogue() if info.name == name)

    assert found.family == "comparison"
    assert found.title == title
    assert found.needs_data
    assert [field.name for field in found.fields] == fields
    assert [field.label for field in found.fields] == labels
    assert [field.kind for field in found.fields] == kinds
    assert all(field.description for field in found.fields)
    assert "curvestudio.tools.stats.comparison_lines" in TOOL_MODULES


def test_the_registry_builds_and_runs_both_tools_from_column_names() -> None:
    first, second = _sample(n=15, seed=15)
    columns = {"a": _column("a", first), "b": _column("b", second)}

    def group(name: str) -> object:
        raise AssertionError("neither tool takes a group column")

    deming = build_request(
        "deming",
        {"method_a": "a", "method_b": "b", "error_ratio": 2.0, "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    assert isinstance(deming, DemingRequest) and deming.error_ratio == 2.0
    assert run_tool("deming", deming).title == "Deming regression of b on a"

    paba = build_request(
        "passingbablok",
        {"method_a": "a", "method_b": "b", "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    assert isinstance(paba, PassingBablokRequest)
    assert run_tool("passingbablok", paba).level == 0.9


def test_deming_handles_a_covariance_of_zero_with_a_horizontal_line() -> None:
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    second = np.asarray([2.0, -1.0, -1.0, 2.0])

    slope, intercept = _deming_fit(first, second, 1.0)

    assert slope == 0.0
    assert intercept == pytest.approx(float(np.mean(second)), rel=STAT_RTOL)
