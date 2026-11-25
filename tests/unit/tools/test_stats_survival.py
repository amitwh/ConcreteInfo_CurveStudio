"""Survival analysis: the Kaplan-Meier curve, the log-rank test and the Cox model.

The reference numbers come from statsmodels: ``SurvfuncRight`` for the Kaplan-Meier curve and
its Greenwood standard error, ``survdiff`` for the log-rank statistic, and ``PHReg`` with
Breslow ties for the Cox coefficients and their standard errors.

Two examples are published, both fetched and read on 2026-11-25:

- The survival package's own vignette, "A Package for Survival Analysis in R" (Terry Therneau),
  https://cran.r-project.org/web/packages/survival/vignettes/survival.pdf. Section 2.1 prints
  ``survfit(Surv(futime, fustat) ~ resid.ds, ovarian)``: for resid.ds = 1, 11 patients and 3
  events with the median not reached; for resid.ds = 2, 15 patients and 9 events with a median
  of 464 days. Its summary at 0.5, 1.0 and 1.5 years tabulates the survival 0.800, 0.600 and
  0.375 with standard errors 0.103, 0.126 and 0.130, and 0.909 with 0.0867 for resid.ds = 1.
  Section 3.6 prints ``coxph(Surv(futime, fustat) ~ rx + age + resid.ds, ovarian)``: the
  coefficients -0.8489, 0.1285 and 0.6964, the hazard ratios 0.4279, 1.1372 and 2.0065, the z
  values -1.328, 2.718 and 0.918, the p-values 0.18416, 0.00657 and 0.35858, and a
  likelihood-ratio test of 16.77 on 3 degrees of freedom with p = 0.0007889. Section 2.6 prints
  ``survdiff(Surv(time, status) ~ x, aml)``: 11 maintained patients with 7 events and 12
  nonmaintained with 11, a chi-square of 3.4 on 1 degree of freedom and p = 0.07.
- The data are the Rdatasets copies of ``survival::ovarian`` and ``survival::aml``, fetched on
  2026-11-25 from
  https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/ovarian.csv
  and https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/aml.csv.

The restricted mean survival is checked against a numerical integral of the same curve, and
against the variance written out in the test. Harrell's concordance index is checked against
its definition, a loop over the comparable pairs. The method the other numbers would need,
a full score process for proportional hazards, is not what the tool computes: the check is the
correlation approximation, and the tests only hold its table and its verdict to their shape.
"""

import json
import math

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import ToolResult, Verdict, p_text
from curvestudio.tools.stats.registry import FAMILIES, TOOL_MODULES, TOOLS, catalogue
from curvestudio.tools.stats.survival import (
    ALL_GROUP,
    MIN_EVENTS,
    SCHOENFELD_MIN_EVENTS,
    SurvivalRequest,
    _concordance,
    cox_fit,
    kaplan_meier,
    logrank,
    run_survival,
)

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""
COX_ATOL = 1e-6
"""The tolerance on a Cox coefficient or standard error against statsmodels."""

OVARIAN = (
    (59, 1, 72.3315, 2, 1, 1),
    (115, 1, 74.4932, 2, 1, 1),
    (156, 1, 66.4658, 2, 1, 2),
    (421, 0, 53.3644, 2, 2, 1),
    (431, 1, 50.3397, 2, 1, 1),
    (448, 0, 56.4301, 1, 1, 2),
    (464, 1, 56.937, 2, 2, 2),
    (475, 1, 59.8548, 2, 2, 2),
    (477, 0, 64.1753, 2, 1, 1),
    (563, 1, 55.1781, 1, 2, 2),
    (638, 1, 56.7562, 1, 1, 2),
    (744, 0, 50.1096, 1, 2, 1),
    (769, 0, 59.6301, 2, 2, 2),
    (770, 0, 57.0521, 2, 2, 1),
    (803, 0, 39.2712, 1, 1, 1),
    (855, 0, 43.1233, 1, 1, 2),
    (1040, 0, 38.8932, 2, 1, 2),
    (1106, 0, 44.6, 1, 1, 1),
    (1129, 0, 53.9068, 1, 2, 1),
    (1206, 0, 44.2055, 2, 2, 1),
    (1227, 0, 59.589, 1, 2, 2),
    (268, 1, 74.5041, 2, 1, 2),
    (329, 1, 43.137, 2, 1, 1),
    (353, 1, 63.2192, 1, 2, 2),
    (365, 1, 64.4247, 2, 2, 1),
    (377, 0, 58.3096, 1, 2, 1),
)
"""The ovarian data: futime, fustat, age, resid.ds, rx, ecog.ps."""

AML_TIME = (
    9,
    13,
    13,
    18,
    23,
    28,
    31,
    34,
    45,
    48,
    161,
    5,
    5,
    8,
    8,
    12,
    16,
    23,
    27,
    30,
    33,
    43,
    45,
)
AML_STATUS = (1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1)
AML_GROUP = ("Maintained",) * 11 + ("Nonmaintained",) * 12


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _group(name: str, labels: object, rows: object | None = None) -> GroupColumn:
    """A group column whose labels are the data layer's object array of strings."""
    array = np.asarray(labels, dtype=object)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=array, rows=np.asarray(positions, dtype=np.intp))


def _status(labels: object) -> NDArray[np.bool_]:
    """A boolean event flag from the label column the helper builds."""
    return np.asarray(np.asarray(labels, dtype=object) == "event", dtype=bool)


def _request(
    time: object,
    status: object,
    groups: object | None = None,
    predictors: object = (),
    *,
    event_level: str | None = "event",
    **options: object,
) -> SurvivalRequest:
    """A request over the labels ``event`` and ``censored``, built from positions 0..n-1."""
    event = _group(
        "status",
        ["event" if flag else "censored" for flag in np.asarray(status, dtype=bool)],
    )
    group = None if groups is None else _group("arm", groups)
    columns = tuple(
        _column(f"x{index}", values)
        for index, values in enumerate(predictors)  # type: ignore[arg-type]
    )
    return SurvivalRequest(
        time=_column("time", time),
        event=event,
        groups=group,
        predictors=columns,
        event_level=event_level,
        **options,  # type: ignore[arg-type]
    )


def _run(
    time: object,
    status: object,
    groups: object | None = None,
    predictors: object = (),
    *,
    event_level: str | None = "event",
    **options: object,
) -> ToolResult:
    return run_survival(
        _request(time, status, groups, predictors, event_level=event_level, **options)
    )


def _three(seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[str]]:
    """Three groups of unequal size, with a little censoring and some ties."""
    rng = np.random.default_rng(seed)
    sizes = (25, 30, 20)
    scales = (12.0, 8.0, 15.0)
    time = np.concatenate(
        [
            np.round(rng.exponential(scale, size), 1) + 0.5
            for scale, size in zip(scales, sizes, strict=True)
        ]
    )
    status = rng.random(int(sum(sizes))) < 0.72
    groups = [
        label for label, size in zip(("a", "b", "c"), sizes, strict=True) for _ in range(size)
    ]
    return time, status, groups


def _survival_at(curve: object, when: float) -> tuple[float, float] | None:
    """The curve's value and standard error at the last event time at or before ``when``."""
    values = [
        (float(value), float(error))
        for time, value, error in zip(
            curve.times,
            curve.survival,
            curve.se,
            strict=True,  # type: ignore[attr-defined]
        )
        if float(time) <= when
    ]
    return values[-1] if values else None


# --------------------------------------------------------------------------- reference numbers


def test_the_kaplan_meier_curve_and_greenwood_error_match_statsmodels() -> None:
    time, status, groups = _three(1)
    result = _run(time, status, groups)
    for label in ("a", "b", "c"):
        mask = np.asarray(groups) == label
        reference = SurvfuncRight(time[mask], status[mask].astype(int))
        curve = kaplan_meier(label, time[mask], status[mask], 0.95)
        assert np.allclose(curve.times, reference.surv_times, rtol=0.0, atol=0.0)
        assert np.allclose(curve.survival, reference.surv_prob, rtol=1e-10, atol=0.0)
        keep = np.isfinite(curve.se)
        assert np.allclose(curve.se[keep], reference.surv_prob_se[keep], rtol=STAT_RTOL)
        assert curve.median == pytest.approx(float(reference.quantile(0.5)), rel=0.0, abs=1e-12)
    labels = [test.name for test in result.tests]
    assert labels[0] == "Log-rank test"


def test_the_restricted_mean_survival_follows_the_integral_of_its_curve() -> None:
    time, status, groups = _three(2)
    for label in ("a", "b", "c"):
        mask = np.asarray(groups) == label
        reference = SurvfuncRight(time[mask], status[mask].astype(int))
        tau = float(np.max(time[mask]))
        grid = np.linspace(0.0, tau, 200_001)
        index = np.searchsorted(reference.surv_times, grid, side="right")
        curve = np.where(
            index == 0,
            1.0,
            reference.surv_prob[np.clip(index - 1, 0, reference.surv_prob.size - 1)],
        )
        integrated = float(np.trapezoid(curve, grid))
        fitted = kaplan_meier(label, time[mask], status[mask], 0.95)
        assert fitted.rmst == pytest.approx(integrated, abs=1e-4)

        variance = 0.0
        for when in fitted.times:
            risk = int(np.count_nonzero(time[mask] >= when))
            dead = int(np.count_nonzero((time[mask] == when) & status[mask]))
            after = np.searchsorted(reference.surv_times, when, side="right")
            area = 0.0
            previous = float(when)
            for step in range(after, reference.surv_times.size):
                following = float(reference.surv_times[step])
                area += float(reference.surv_prob[step - 1]) * (following - previous)
                previous = following
            area += float(reference.surv_prob[-1]) * (tau - previous)
            if area <= 0.0:
                continue
            if risk == dead:
                # The curve reaches zero before the largest observed time: Greenwood's sum
                # divides by zero, so the tool reports no standard error.
                assert fitted.rmst_se is None
                break
            variance += area * area * dead / (risk * (risk - dead))
        else:
            assert fitted.rmst_se == pytest.approx(math.sqrt(variance), rel=1e-4)


def test_the_log_rank_statistic_matches_statsmodels_with_ties() -> None:
    time, status, groups = _three(3)
    codes = np.array([{"a": 0, "b": 1, "c": 2}[label] for label in groups], dtype=np.intp)
    found = logrank(time, status, codes, 3)
    reference_statistic, reference_p = survdiff(time, status.astype(int), codes)
    assert found.statistic == pytest.approx(float(reference_statistic), rel=STAT_RTOL)
    assert found.p_value == pytest.approx(float(reference_p), abs=P_ATOL)
    assert found.df == 2.0

    pair = codes < 2
    again = logrank(time[pair], status[pair], codes[pair], 2)
    reference_statistic, reference_p = survdiff(time[pair], status[pair].astype(int), codes[pair])
    assert again.statistic == pytest.approx(float(reference_statistic), rel=STAT_RTOL)
    assert again.p_value == pytest.approx(float(reference_p), abs=P_ATOL)
    assert again.df == 1.0


def test_the_pairwise_log_rank_comparisons_carry_holms_adjustment() -> None:
    time, status, groups = _three(4)
    result = _run(time, status, groups)
    assert [test.name for test in result.tests] == [
        "Log-rank test",
        "Log-rank test, a against b",
        "Log-rank test, a against c",
        "Log-rank test, b against c",
    ]
    comparisons = result.tests[1:]
    raw = sorted(test.p_value for test in comparisons if test.p_value is not None)
    family = len(raw)
    running = 0.0
    expected: list[float] = []
    for step, value in enumerate(raw, start=1):
        running = max(running, (family - step + 1) * value)
        expected.append(min(1.0, running))
    adjusted = sorted(test.adjusted_p for test in comparisons if test.adjusted_p is not None)
    assert adjusted == pytest.approx(expected)
    for comparison in comparisons:
        assert comparison.adjustment == "Holm"
        assert comparison.df == 1.0
        raw_pair = [test.p_value for test in comparisons if test.p_value is not None]
        assert raw_pair  # the raw p-values are kept beside the adjusted ones

    two = _run(time, status, ["a" if group == "a" else "b" for group in groups])
    assert [test.name for test in two.tests] == ["Log-rank test"]


def test_the_cox_model_matches_statsmodels_with_breslow_ties() -> None:
    rng = np.random.default_rng(11)
    size = 90
    time = np.sort(np.round(rng.exponential(10.0, size)) + 1.0)
    status = rng.random(size) < 0.75
    status[0] = True
    first = rng.normal(size=size)
    second = rng.integers(0, 2, size).astype(np.float64)
    design = np.column_stack([first, second])
    reference = PHReg(time, design, status=status.astype(int)).fit()

    fitted = cox_fit(time, status, design, ("first", "second"), 0.95)
    assert fitted.converged
    assert fitted.beta == pytest.approx(reference.params, abs=COX_ATOL)
    assert fitted.se == pytest.approx(reference.bse, abs=COX_ATOL)
    assert fitted.log_likelihood == pytest.approx(float(reference.llf), rel=1e-9)
    assert fitted.p_value == pytest.approx(reference.pvalues, abs=1e-8)
    assert fitted.hazard_ratio == pytest.approx(np.exp(reference.params), rel=1e-6)
    # The intercept-only partial likelihood is written out here: one term per event time.
    null_likelihood = 0.0
    for when in np.unique(time[status]):
        dead = float(np.count_nonzero((time == when) & status))
        risk = float(np.count_nonzero(time >= when))
        null_likelihood -= dead * math.log(risk)
    assert fitted.log_likelihood_null == pytest.approx(null_likelihood, rel=1e-9)
    assert fitted.lr_statistic == pytest.approx(
        2.0 * (float(reference.llf) - null_likelihood), abs=1e-6
    )

    result = _run(time, status, predictors=(first, second))
    test = result.tests[-1]
    assert test.name == "Likelihood-ratio test of the Cox model"
    assert test.statistic == pytest.approx(fitted.lr_statistic, rel=1e-10)
    assert test.p_value == pytest.approx(fitted.lr_p_value, rel=1e-10)
    assert test.df == 2.0
    rows = {row[0]: row for row in result.tables[1].rows}
    for index, name in enumerate(("x0", "x1")):
        assert rows[name][1] == pytest.approx(float(np.exp(reference.params[index])), rel=1e-6)
        assert rows[name][4] == pytest.approx(float(reference.params[index] / reference.bse[index]))
        assert rows[name][5] == pytest.approx(float(reference.pvalues[index]), abs=1e-8)


def test_the_concordance_index_follows_its_definition() -> None:
    time = np.array([2.0, 4.0, 4.0, 6.0, 8.0, 9.0])
    event = np.array([True, True, False, True, True, False])
    risk = np.array([0.9, 0.2, 0.5, 0.7, 0.1, 0.8])

    concordant = 0.0
    comparable = 0
    for left in range(time.size):
        if not event[left]:
            continue
        for right in range(time.size):
            if time[right] <= time[left]:
                continue
            comparable += 1
            if risk[left] > risk[right]:
                concordant += 1.0
            elif risk[left] == risk[right]:
                concordant += 0.5
    assert comparable == 11
    assert concordant == 7.0
    assert _concordance(risk, time, event) == pytest.approx(concordant / comparable)

    # A risk that reverses every comparable pair scores what the first did not.
    assert _concordance(-risk, time, event) == pytest.approx(4.0 / 11.0)
    assert _concordance(np.zeros_like(risk), time, event) == pytest.approx(0.5)


def test_the_concordance_index_agrees_with_the_fitted_coefficients() -> None:
    time, status, groups = _three(5)
    design = np.column_stack(
        [
            np.asarray([1.0 if group == "a" else 0.0 for group in groups]),
            np.linspace(0, 1, time.size),
        ]
    )
    fitted = cox_fit(time, status, design, ("a", "x"), 0.95)
    result = _run(
        time,
        status,
        predictors=(
            np.asarray([1.0 if group == "a" else 0.0 for group in groups]),
            np.linspace(0, 1, time.size),
        ),
    )
    estimate = next(e for e in result.estimates if e.label == "Harrell's concordance index")
    assert estimate.value == pytest.approx(
        _concordance(design @ fitted.beta, time, status)  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- published examples


def test_the_published_ovarian_example_reproduces_its_median_and_curves() -> None:
    futime = np.array([row[0] for row in OVARIAN], dtype=np.float64)
    fustat = np.array([row[1] for row in OVARIAN], dtype=bool)
    resid = np.array([row[3] for row in OVARIAN], dtype=np.float64)

    result = run_survival(
        SurvivalRequest(
            time=_column("futime", futime),
            event=_group("fustat", ["dead" if flag else "alive" for flag in fustat]),
            groups=_group("resid.ds", [f"{value:g}" for value in resid]),
            event_level="dead",
        )
    )
    rows = {row[0]: row for row in result.tables[0].rows}
    assert rows["1"][1:3] == (11, 3)
    assert rows["1"][3] is None
    assert rows["2"][1:3] == (15, 9)
    assert rows["2"][3] == pytest.approx(464.0)

    one = kaplan_meier("1", futime[resid == 1], fustat[resid == 1], 0.95)
    two = kaplan_meier("2", futime[resid == 2], fustat[resid == 2], 0.95)
    assert _survival_at(one, 365.0) == pytest.approx((0.909, 0.0867), abs=5e-4)
    for when, survival, error in (
        (182.5, 0.800, 0.103),
        (365.0, 0.600, 0.126),
        (547.5, 0.375, 0.130),
    ):
        assert _survival_at(two, when) == pytest.approx((survival, error), abs=5e-4)

    # The median interval in the vignette uses the log transform; the tool's is the
    # complementary log-log band the request asks for, so only the point estimates are pinned.
    assert two.median == pytest.approx(464.0)


def test_the_published_ovarian_example_reproduces_its_cox_model() -> None:
    futime = np.array([row[0] for row in OVARIAN], dtype=np.float64)
    fustat = np.array([row[1] for row in OVARIAN], dtype=bool)
    age = np.array([row[2] for row in OVARIAN], dtype=np.float64)
    resid = np.array([row[3] for row in OVARIAN], dtype=np.float64)
    rx = np.array([row[4] for row in OVARIAN], dtype=np.float64)

    result = run_survival(
        SurvivalRequest(
            time=_column("futime", futime),
            event=_group("fustat", ["dead" if flag else "alive" for flag in fustat]),
            event_level="dead",
            predictors=(_column("rx", rx), _column("age", age), _column("resid.ds", resid)),
        )
    )
    rows = {row[0]: row for row in result.tables[1].rows}
    expected = {
        "rx": (-0.8489, 0.4279, -1.328, 0.18416),
        "age": (0.1285, 1.1372, 2.718, 0.00657),
        "resid.ds": (0.6964, 2.0065, 0.918, 0.35858),
    }
    for term, (coefficient, hazard_ratio, z, p_value) in expected.items():
        assert math.log(rows[term][1]) == pytest.approx(coefficient, abs=5e-4)
        assert rows[term][1] == pytest.approx(hazard_ratio, rel=5e-4)
        assert rows[term][4] == pytest.approx(z, abs=5e-4)
        assert rows[term][5] == pytest.approx(p_value, abs=5e-5)

    lr = result.tests[-1]
    assert lr.name == "Likelihood-ratio test of the Cox model"
    assert lr.statistic == pytest.approx(16.77, abs=5e-3)
    assert lr.df == 3.0
    assert lr.p_value == pytest.approx(0.0007889, abs=1e-6)

    # A hazard ratio above 1 for age is a higher hazard, which is what the vignette prints.
    age_estimate = next(e for e in result.estimates if e.label == "Hazard ratio, age")
    assert age_estimate.lower < age_estimate.value < age_estimate.upper
    assert age_estimate.lower == pytest.approx(1.0365, abs=5e-4)
    assert age_estimate.upper == pytest.approx(1.2476, abs=5e-4)


def test_the_published_aml_example_reproduces_the_log_rank() -> None:
    result = run_survival(
        SurvivalRequest(
            time=_column("time", AML_TIME),
            event=_group("status", ["event" if flag else "censored" for flag in AML_STATUS]),
            groups=_group("x", AML_GROUP),
            event_level="event",
        )
    )
    rows = {row[0]: row for row in result.tables[0].rows}
    assert (rows["Maintained"][1], rows["Maintained"][2]) == (11, 7)
    assert (rows["Nonmaintained"][1], rows["Nonmaintained"][2]) == (12, 11)
    test = result.tests[0]
    assert test.name == "Log-rank test"
    assert test.statistic == pytest.approx(3.4, abs=0.05)
    assert test.df == 1.0
    assert test.p_value == pytest.approx(0.07, abs=5e-3)
    assert test.p_value_text == p_text(test.p_value)


# --------------------------------------------------------------------------- properties


def test_scaling_time_scales_the_median_and_leaves_the_tests_unchanged() -> None:
    time, status, groups = _three(6)
    predictor = rng_value = np.random.default_rng(7).normal(size=time.size)
    base = _run(time, status, groups, predictors=(predictor,))
    scaled = _run(time * 3.0, status, groups, predictors=(predictor,))

    base_medians = [e.value for e in base.estimates if e.label.startswith("Median")]
    scaled_medians = [e.value for e in scaled.estimates if e.label.startswith("Median")]
    assert scaled_medians == pytest.approx([value * 3.0 for value in base_medians], rel=1e-10)
    base_rmst = [e.value for e in base.estimates if "Restricted mean" in e.label]
    scaled_rmst = [e.value for e in scaled.estimates if "Restricted mean" in e.label]
    assert scaled_rmst == pytest.approx([value * 3.0 for value in base_rmst], rel=1e-10)

    assert scaled.tests[0].statistic == pytest.approx(base.tests[0].statistic, rel=1e-10)
    assert scaled.tests[0].p_value == pytest.approx(base.tests[0].p_value, abs=1e-12)
    base_ratios = [e.value for e in base.estimates if e.label.startswith("Hazard ratio")]
    scaled_ratios = [e.value for e in scaled.estimates if e.label.startswith("Hazard ratio")]
    assert scaled_ratios == pytest.approx(base_ratios, rel=1e-10)
    assert rng_value.size == time.size


def test_a_group_with_no_events_has_a_curve_that_never_falls() -> None:
    time = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], float)
    censored = np.zeros(time.size, dtype=bool)
    alive_time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], float)
    alive_event = np.array(
        [True, True, True, True, True, True, True, True, True, True, True, False]
    )
    result = _run(
        np.concatenate([time, alive_time]),
        np.concatenate([censored, alive_event]),
        ["censored"] * time.size + ["events"] * alive_time.size,
    )
    curve = kaplan_meier("censored", time, censored, 0.95)
    assert curve.events == 0
    assert np.all(curve.survival == 1.0) or curve.survival.size == 0
    assert curve.median is None
    assert curve.rmst == pytest.approx(float(time.max()))

    plot = next(plot for plot in result.plots if plot.title == "Kaplan–Meier")  # noqa: RUF001
    series = {one.label: one for one in plot.series}
    assert set(series["censored"].y) == {1.0}
    assert any("has no events" in warning for warning in result.warnings)


def test_an_exponential_samples_median_is_near_log_two_over_lambda() -> None:
    rate = 0.2
    rng = np.random.default_rng(8)
    time = rng.exponential(1.0 / rate, 4_000)
    event = np.ones(time.size, dtype=bool)
    curve = kaplan_meier("all", time, event, 0.95)
    assert curve.median is not None
    assert curve.median == pytest.approx(math.log(2.0) / rate, rel=0.05)


# --------------------------------------------------------------------------- blocks and words


def test_the_result_carries_the_summary_the_curves_and_the_words() -> None:
    time, status, groups = _three(9)
    predictor = np.random.default_rng(10).normal(size=time.size)
    result = _run(time, status, groups, predictors=(predictor,))

    assert type(result) is ToolResult
    assert result.tool == "survival"
    assert result.title == "Survival analysis of time by arm"
    assert result.level == 0.95
    assert [table.title for table in result.tables] == [
        "Survival summary",
        "Cox model",
        "Schoenfeld check",
    ]
    assert [plot.title for plot in result.plots] == ["Kaplan–Meier", "Schoenfeld residuals"]  # noqa: RUF001
    assert [plot.kind for plot in result.plots] == ["lines", "scatter"]
    assert [estimate.label.split(",")[0] for estimate in result.estimates][:2] == [
        "Median survival",
        "Restricted mean survival",
    ]
    assert any(
        label.startswith("Hazard ratio, x0") for label in (e.label for e in result.estimates)
    )
    assert any(e.label == "Harrell's concordance index" for e in result.estimates)
    assert [check.id for check in result.assumptions] == [
        "proportional_hazards",
        "independent_censoring",
    ]
    assert [test.name for test in result.tests][-1] == "Likelihood-ratio test of the Cox model"
    assert result.counts.n_offered == time.size
    assert result.counts.n_used == time.size
    assert result.counts.n_excluded == 0
    assert not claims_proof(result.interpretation)
    assert not claims_proof(result.methods_fragment)
    assert "Kaplan-Meier" in result.methods_fragment
    assert "Breslow" in result.methods_fragment
    assert "concordance" in result.methods_fragment
    assert "complementary log-log" in result.methods_fragment

    restored = ToolResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert "NaN" not in result.model_dump_json()
    assert json.loads(result.model_dump_json())["tool"] == "survival"


def test_the_event_coding_is_the_second_level_unless_named() -> None:
    time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], float)
    flags = np.array([False, True, False, True, False, True, False, True, False, True, False, True])
    labels = ["censored" if not flag else "event" for flag in flags]
    assert labels[0] == "censored" and labels[1] == "event"

    default = run_survival(
        SurvivalRequest(
            time=_column("time", time),
            event=_group("status", labels),
            groups=_group("arm", ["a"] * 6 + ["b"] * 6),
        )
    )
    assert default.interpretation.startswith(
        "With 'event' read as the event and 'censored' as censoring"
    )
    assert "The event is 'event'" in default.tables[0].note

    named = run_survival(
        SurvivalRequest(
            time=_column("time", time),
            event=_group("status", labels),
            groups=_group("arm", ["a"] * 6 + ["b"] * 6),
            event_level="censored",
        )
    )
    assert named.interpretation.startswith(
        "With 'censored' read as the event and 'event' as censoring"
    )


def test_the_summary_reports_the_event_coding_and_the_zero_time_rule() -> None:
    time = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], float)
    status = np.array([True, True, False, True, True, False, True, True, True, False, True, True])
    result = _run(time, status, ["a"] * 6 + ["b"] * 6)
    note = result.tables[0].note
    assert note is not None
    assert "A survival time of zero is allowed." in note
    assert "'event'" in note and "'censored'" in note
    plot = result.plots[0]
    assert plot.series[0].x[0] == 0.0


def test_the_schoenfeld_check_has_a_row_per_term_and_a_plot_of_the_first() -> None:
    time, status, _groups = _three(12)
    first = np.random.default_rng(13).normal(size=time.size)
    second = np.random.default_rng(14).normal(size=time.size)
    result = _run(time, status, predictors=(first, second))
    table = next(table for table in result.tables if table.title == "Schoenfeld check")
    assert [row[0] for row in table.rows] == ["x0", "x1"]
    for row in table.rows:
        assert row[1] is not None and -1.0 <= float(row[1]) <= 1.0
        assert row[2] is not None and 0.0 <= float(row[2]) <= 1.0
    check = next(c for c in result.assumptions if c.id == "proportional_hazards")
    assert check.verdict in (Verdict.PASS, Verdict.CAUTION)
    plot = next(plot for plot in result.plots if plot.title == "Schoenfeld residuals")
    assert "x0" in plot.y_label
    assert len(plot.x) == len(plot.y)


def test_fewer_than_ten_events_warns_and_skips_the_proportional_hazards_check() -> None:
    time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], float)
    status = np.zeros(time.size, dtype=bool)
    status[[1, 3, 5, 7, 9, 11]] = True
    predictor = np.random.default_rng(15).normal(size=time.size)
    result = _run(time, status, predictors=(predictor,))
    assert any("fewer than the 10" in warning for warning in result.warnings)
    check = next(c for c in result.assumptions if c.id == "proportional_hazards")
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert all(table.title != "Schoenfeld check" for table in result.tables)
    assert all(plot.title != "Schoenfeld residuals" for plot in result.plots)


def test_every_row_left_out_is_counted_and_listed() -> None:
    values = np.arange(1, 13, dtype=np.float64)
    values[2] = np.nan
    time = _column("time", values)
    labels = _group("status", ["event"] * 8 + ["censored"] * 3, rows=np.arange(11, dtype=np.intp))
    result = run_survival(
        SurvivalRequest(
            time=time,
            event=labels,
            event_level="event",
            groups=_group("arm", ["a"] * 6 + ["b"] * 6),
        )
    )
    assert result.counts.n_offered == 12
    assert result.counts.n_used == 10
    assert result.counts.n_excluded == 2
    reasons = {row.index: row.reason for row in result.excluded}
    assert reasons == {2: "missing in time", 11: "no group label in status"}
    assert result.counts.n_offered == result.counts.n_used + result.counts.n_excluded


# --------------------------------------------------------------------------- edge cases


def test_four_events_are_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.zeros(time.size, dtype=bool)
    status[[2, 5, 8, 12]] = True
    with pytest.raises(ToolError) as refused:
        _run(time, status, ["a"] * 10 + ["b"] * 10)
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "4 events" in refused.value.message
    assert MIN_EVENTS == 5


def test_an_event_column_that_never_recorded_an_event_is_refused() -> None:
    # All-censored data hold one level: a column that never recorded the event cannot say
    # which level is the event, so the two-level column the tool takes is missing one level.
    time = np.arange(1, 21, dtype=np.float64)
    status = np.zeros(time.size, dtype=bool)
    with pytest.raises(ToolError) as refused:
        _run(time, status, ["a"] * 10 + ["b"] * 10)
    assert refused.value.code == "SINGLE_GROUP"


def test_a_negative_time_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    time[0] = -0.5
    status = np.array([True] * 10 + [False] * 10)
    with pytest.raises(ToolError) as refused:
        _run(time, status, ["a"] * 10 + ["b"] * 10)
    assert refused.value.code == "INVALID_OPTION"
    assert "below zero" in refused.value.message


def test_a_single_group_level_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.ones(time.size, dtype=bool)
    with pytest.raises(ToolError) as refused:
        _run(time, status, ["only"] * time.size)
    assert refused.value.code == "SINGLE_GROUP"


def test_a_predictor_collinear_with_the_group_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.array([True] * 10 + [False] * 10)
    indicator = np.array([1.0] * 10 + [0.0] * 10)
    with pytest.raises(ToolError) as refused:
        _run(time, status, predictors=(indicator, indicator * 2.0))
    assert refused.value.code == "COLLINEAR"


def test_a_constant_predictor_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.array([True] * 10 + [False] * 10)
    with pytest.raises(ToolError) as refused:
        _run(time, status, predictors=(np.ones(time.size),))
    assert refused.value.code == "COLLINEAR"


def test_neither_groups_nor_predictors_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.ones(time.size, dtype=bool)
    with pytest.raises(ToolError) as refused:
        _run(time, status)
    assert refused.value.code == "INVALID_OPTION"
    assert "groups" in refused.value.message and "predictors" in refused.value.message


def test_names_must_name_one_predictor_each() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.ones(time.size, dtype=bool)
    predictor = np.random.default_rng(16).normal(size=time.size)
    with pytest.raises(ToolError) as refused:
        _run(time, status, predictors=(predictor, predictor + 1.0), names=("only one",))
    assert refused.value.code == "INVALID_OPTION"
    assert "one name per predictor" in refused.value.message


def test_an_event_column_of_three_levels_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    labels = ["event"] * 8 + ["censored"] * 8 + ["lost"] * 4
    with pytest.raises(ToolError) as refused:
        run_survival(
            SurvivalRequest(
                time=_column("time", time),
                event=_group("status", labels),
                groups=_group("arm", ["a"] * 10 + ["b"] * 10),
            )
        )
    assert refused.value.code == "LEVELS_REQUIRED"


def test_an_event_level_not_in_the_column_is_refused() -> None:
    time = np.arange(1, 21, dtype=np.float64)
    with pytest.raises(ToolError) as refused:
        run_survival(
            SurvivalRequest(
                time=_column("time", time),
                event=_group("status", ["event"] * 10 + ["censored"] * 10),
                groups=_group("arm", ["a"] * 10 + ["b"] * 10),
                event_level="missing",
            )
        )
    assert refused.value.code == "LEVELS_REQUIRED"
    assert "missing" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 2.0])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    time = np.arange(1, 21, dtype=np.float64)
    status = np.ones(time.size, dtype=bool)
    with pytest.raises(ToolError) as refused:
        _run(time, status, ["a"] * 10 + ["b"] * 10, level=level)
    assert refused.value.code == "INVALID_OPTION"


class _Other(BaseModel):
    """A request of no tool at all, to hold the type check to its word."""

    value: int = 0


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_survival(_Other())
    assert refused.value.code == "INVALID_OPTION"
    assert "SurvivalRequest" in refused.value.message


def test_the_advanced_family_and_the_registry_hold_the_tool() -> None:
    assert "curvestudio.tools.stats.survival" in TOOL_MODULES
    assert ("advanced", "Advanced models") in FAMILIES

    info = TOOLS.info("survival")
    assert (info.name, info.family, info.title, info.needs_data, info.max_rows) == (
        "survival",
        "advanced",
        "Survival analysis",
        True,
        None,
    )
    assert [(field.name, field.kind, field.required) for field in info.fields] == [
        ("time", "column", True),
        ("event", "group", True),
        ("groups", "group", False),
        ("predictors", "columns", False),
        ("names", "option", False),
        ("event_level", "option", False),
        ("level", "option", False),
    ]
    assert all(field.description for field in info.fields)
    assert "survival" in [entry.name for entry in catalogue()]

    time, status, groups = _three(17)
    columns = {
        "time": _column("time", time),
        "status": _group("status", ["event" if flag else "censored" for flag in status]),
        "arm": _group("arm", groups),
    }
    request = TOOLS.build_request(
        "survival",
        {"time": "time", "event": "status", "groups": "arm", "event_level": "event"},
        column=lambda name: columns[name],
        group=lambda name: columns[name],
    )
    assert isinstance(request, SurvivalRequest)
    assert request.groups is not None and request.groups.name == "arm"
    assert TOOLS.run("survival", request).tool == "survival"
    assert ALL_GROUP == "All"
    assert SCHOENFELD_MIN_EVENTS == 10
