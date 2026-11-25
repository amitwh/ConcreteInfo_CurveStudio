"""Survival analysis: Kaplan-Meier curves, the log-rank test and a Cox model.

The time-to-event tools read one numeric column of follow-up times and one two-level column
whose labels say which rows observed the event and which were censored. Rows are complete in
every column the tool is given; a row left out is counted and listed, never dropped without a
word.

The Kaplan-Meier estimator is the product-limit estimate of the survival function. Its
variance is Greenwood's, its interval the complementary log-log (log(-log S)) transform, its
median the first time the curve reaches 0.5 with the Brookmeyer-Crowley interval obtained by
inverting that band, and its restricted mean the area under the curve to the largest observed
time with the Greenwood-based variance (Klein J.P. and Moeschberger M.L., "Survival Analysis:
Techniques for Censored and Truncated Data", 2nd edition, Springer, 2003, sections 4.2-4.5;
Brookmeyer R. and Crowley J., "A confidence interval for the median survival time", Biometrics
38:29-41, 1982).

The log-rank (Mantel-Haenszel) test compares the groups' observed event counts with the counts
the pooled hazard implies at every event time, its statistic chi-square on k - 1 degrees of
freedom (Mantel N. and Haenszel W., Journal of the National Cancer Institute 22:719-748, 1959;
Peto R. and Peto J., British Journal of Cancer 26:257-265, 1972). The pairwise comparisons
carry Holm's step-down adjustment.

The Cox model is fitted by Newton-Raphson on the partial likelihood with Breslow's handling of
tied event times (Cox D.R., "Regression models and life-tables", Journal of the Royal
Statistical Society B 34:187-220, 1972; Breslow N.E., Biometrics 30:89-99, 1974), to a
tolerance of 1e-9 or 50 iterations; a fit that does not converge is reported with the last
change, never returned as if it had. The coefficients carry Wald intervals, the model a
likelihood-ratio test, and the ordering of the risks Harrell's concordance index (Harrell
F.E. et al., Journal of the American Medical Association 247:2543-2546, 1982). Proportional
hazards is checked approximately by the correlation of the scaled Schoenfeld residuals with
the event time, on at least ten events (Grambsch P.M. and Therneau T.M., Biometrika
81:515-526, 1994); it is an approximation, not the full score process.

A survival time must not be below zero; a time of zero is allowed and named. At least five
events and two distinct times are needed. Censoring is assumed independent of the event time,
and that is an assumption the data cannot show.

A p-value is a statement about evidence, never a statement of proof.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.data.dataset import IntArray, StrArray
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import joined, num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    MAX_PLOT_POINTS,
    Cell,
    ColumnKind,
    FloatArray,
    MultiLinePlot,
    ScatterPlot,
    XYSeries,
)
from curvestudio.tools.stats._tables import holm, table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_EVENTS = 5
"""The fewest events a survival analysis needs: a curve of fewer falls too little to read."""

MIN_DISTINCT_TIMES = 2
"""The fewest distinct times the curve needs to have more than one step."""

SCHOENFELD_MIN_EVENTS = 10
"""The fewest events the proportional-hazards check is read on."""

NEWTON_TOLERANCE = 1e-9
"""The largest change in a coefficient the Cox fit accepts as convergence."""

NEWTON_MAX_ITERATIONS = 50
"""The most Newton-Raphson iterations the Cox fit takes."""

COLLINEAR_RTOL = 1e-10
"""A Cox information matrix counts as singular when its smallest singular value is this small
a fraction of its largest."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

ALL_GROUP = "All"
"""The label of the single curve when no group column is given."""

_NAMES_LABEL = "Predictor names"

_CAUTION_THRESHOLD = f"p >= {num(CAUTION_P)} pass; p < {num(CAUTION_P)} caution"
"""The rule on the proportional-hazards check: it has no failing verdict."""

_SURVIVAL_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("n", "n", "int"),
    ("events", "Events", "int"),
    ("median", "Median", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
    ("rmst", "Restricted mean", "number"),
    ("se", "SE", "number"),
)

_COX_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("hazard_ratio", "Hazard ratio", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
    ("z", "z", "number"),
    ("p", "p", "p_value"),
)

_SCHOENFELD_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("correlation", "Correlation with time", "number"),
    ("p", "p", "p_value"),
)


class SurvivalRequest(BaseModel):
    """Follow-up times, the event column, and the groups or predictors the analysis takes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: Column = Field(
        description=(
            "The numeric follow-up time of each row: the time to the event, or to censoring. "
            "A time must not be below zero; a time of zero is allowed."
        )
    )
    event: GroupColumn = Field(
        description=(
            "The two-level column that says which rows observed the event and which were "
            "censored. The level second in order of first appearance is the event, unless a "
            "level is named."
        )
    )
    groups: GroupColumn | None = Field(
        default=None,
        description=(
            "A group column whose levels are compared by the Kaplan-Meier curves and the "
            "log-rank test. Leave empty for a Cox model alone."
        ),
    )
    predictors: tuple[Column, ...] = Field(
        default=(),
        description=(
            "Numeric columns for the Cox model, each contributing its own hazard ratio. Leave "
            "empty for the Kaplan-Meier curves and the log-rank test alone."
        ),
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "What the predictors are called in the tables and the words, one per predictor. "
            "Left empty, each predictor's own column name is used."
        ),
        json_schema_extra={"label": _NAMES_LABEL},
    )
    event_level: str | None = Field(
        default=None,
        description=(
            "The level of the event column that is the event; the other level is censoring. "
            "Left empty, the level second in order of first appearance is the event."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Coding(NamedTuple):
    """Which level of the event column is the event, and which is censoring."""

    event: str
    censored: str


class _Complete(NamedTuple):
    """The rows complete in every column given, and the positions left out."""

    positions: IntArray
    times: FloatArray
    event: NDArray[np.bool_]
    groups: StrArray | None
    predictors: tuple[FloatArray, ...]
    excluded: tuple[ExcludedRow, ...]
    n_offered: int


class KaplanMeier(NamedTuple):
    """One group's Kaplan-Meier curve and the numbers read from it."""

    label: str
    n: int
    events: int
    times: FloatArray
    """The distinct event times, ascending: where the curve falls."""
    survival: FloatArray
    """The curve after each event time."""
    se: FloatArray
    """Greenwood's standard error at each event time; NaN where it is undefined."""
    lower: FloatArray
    """The complementary log-log lower limit at each event time; NaN where undefined."""
    upper: FloatArray
    """The complementary log-log upper limit at each event time; NaN where undefined."""
    median: float | None
    median_lower: float | None
    median_upper: float | None
    rmst: float
    rmst_se: float | None


class LogRank(NamedTuple):
    """The log-rank statistic, its degrees of freedom and its p-value."""

    statistic: float
    df: float
    p_value: float


class Cox(NamedTuple):
    """A fitted Cox model and everything read from it."""

    beta: FloatArray
    se: FloatArray
    z: FloatArray
    p_value: FloatArray
    hazard_ratio: FloatArray
    lower: FloatArray
    upper: FloatArray
    log_likelihood: float
    log_likelihood_null: float
    lr_statistic: float
    lr_p_value: float
    concordance: float | None
    converged: bool
    last_change: float
    iterations: int
    event_times: FloatArray
    """The time of each event row the residuals belong to, in row order of the residuals."""
    schoenfeld: FloatArray | None
    """The scaled Schoenfeld residual of each event row, a column per predictor."""
    schoenfeld_correlation: FloatArray
    schoenfeld_p_value: FloatArray


def _coding(request: SurvivalRequest) -> _Coding:
    """Which level of the event column is the event, by the rule the request documents.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` for a column that does not hold exactly two levels, or
            a named level the column does not hold.
    """
    found = levels(request.event, minimum=2)
    if len(found) > 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the event column {request.event.name!r} holds {len(found)} labels "
            f"({', '.join(repr(label) for label in found)}), and survival takes exactly two: "
            "one for the event and one for censoring",
        )
    if request.event_level is None:
        return _Coding(event=found[1], censored=found[0])
    if request.event_level not in found:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the event level {request.event_level!r} is not one of the event column's levels "
            f"{', '.join(repr(label) for label in found)}",
        )
    censored = found[0] if found[1] == request.event_level else found[1]
    return _Coding(event=request.event_level, censored=censored)


def _spread_labels(positions: IntArray, group: GroupColumn) -> tuple[NDArray[np.bool_], StrArray]:
    """``group``'s labels laid out over ``positions``: which ones it covers, and each label."""
    covered = np.zeros(positions.size, dtype=bool)
    labels: StrArray = np.full(positions.size, "", dtype=object)
    _common, here, there = np.intersect1d(positions, group.rows, return_indices=True)
    covered[here] = True
    labels[here] = group.labels[there]
    return covered, labels


def _complete(request: SurvivalRequest, coding: _Coding) -> _Complete:
    """The rows complete in the time, the predictors and both group columns.

    A row without a time or a finite predictor, or without a label in the event or group
    column, is left out and listed once, with the first column at fault.
    """
    numeric = pairs(request.time, *request.predictors)
    positions = numeric.columns[0].rows
    reasons: dict[int, str] = {row.index: row.reason for row in numeric.excluded}
    kept = np.ones(positions.size, dtype=bool)
    given: list[GroupColumn] = [request.event]
    if request.groups is not None:
        given.append(request.groups)
    labels: list[StrArray] = []
    for group in given:
        covered, text = _spread_labels(positions, group)
        for at in np.flatnonzero(~covered & kept):
            reasons[int(positions[at])] = f"no group label in {group.name}"
        kept &= covered
        labels.append(text)
    event_text = labels[0][kept]
    event = np.asarray(event_text == coding.event, dtype=bool)
    group_text = labels[1][kept] if request.groups is not None else None
    return _Complete(
        positions=positions[kept],
        times=numeric.columns[0].values[kept],
        event=event,
        groups=group_text,
        predictors=tuple(column.values[kept] for column in numeric.columns[1:]),
        excluded=tuple(
            ExcludedRow(index=index, reason=reasons[index]) for index in sorted(reasons)
        ),
        n_offered=int(positions.size + len(numeric.excluded)),
    )


def _curve_band(
    survival: FloatArray, se: FloatArray, level: float
) -> tuple[FloatArray, FloatArray]:
    """The complementary log-log interval of each survival value, NaN where undefined.

    The interval is the complementary log-log band the survival packages draw (R's survfit
    with ``conf.type = "log-log"``): on the log(-log S) scale the standard error is
    ``se / (S * |log S|)``, and the limits map back through ``exp(-exp(theta))``. A value at 0
    or 1 has no log(-log) and so no interval.
    """
    lower = np.full(survival.size, np.nan, dtype=np.float64)
    upper = np.full(survival.size, np.nan, dtype=np.float64)
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    for index, (value, error) in enumerate(zip(survival, se, strict=True)):
        if not 0.0 < value < 1.0 or not math.isfinite(error) or error <= 0.0:
            continue
        theta = math.log(-math.log(value))
        spread = error / (value * -math.log(value))
        if not math.isfinite(spread):
            continue
        lower[index] = math.exp(-math.exp(theta + critical * spread))
        upper[index] = math.exp(-math.exp(theta - critical * spread))
    return lower, upper


def kaplan_meier(
    label: str, times: FloatArray, event: NDArray[np.bool_], level: float
) -> KaplanMeier:
    """One group's Kaplan-Meier curve, median and restricted mean survival.

    The curve is the product over the distinct event times of ``1 - d/n``, with ``n`` the
    number at risk just before the time; Greenwood's variance adds ``d / (n (n - d))`` at each
    event time. The median is the first event time the curve reaches 0.5, and its
    Brookmeyer-Crowley interval is the first time the band's lower limit reaches 0.5 and the
    first the upper limit does. The restricted mean integrates the curve to the group's
    largest observed time; its variance is the Greenwood sum with the squared area after each
    event time, so a curve that reaches 0 at its last event has no standard error.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    size = int(t.size)
    sorted_times = np.sort(t)
    event_times, counts = np.unique(t[e], return_counts=True)
    event_times = np.asarray(event_times, dtype=np.float64)
    at_risk = np.empty(event_times.size, dtype=np.float64)
    survival = np.empty(event_times.size, dtype=np.float64)
    se = np.full(event_times.size, np.nan, dtype=np.float64)
    current = 1.0
    greenwood = 0.0
    defined = True
    for index, (when, dead) in enumerate(zip(event_times, counts, strict=True)):
        n = float(size - int(np.searchsorted(sorted_times, when, side="left")))
        d = float(dead)
        current *= 1.0 - d / n
        survival[index] = current
        at_risk[index] = n
        if defined and n - d > 0.0:
            greenwood += d / (n * (n - d))
            se[index] = current * math.sqrt(greenwood)
        else:
            defined = False
    lower, upper = _curve_band(survival, se, level)

    median: float | None = None
    below = np.flatnonzero(survival <= 0.5)
    if below.size:
        median = float(event_times[below[0]])
    median_lower: float | None = None
    below_lower = np.flatnonzero(lower <= 0.5)
    if below_lower.size:
        median_lower = float(event_times[below_lower[0]])
    median_upper: float | None = None
    below_upper = np.flatnonzero(upper <= 0.5)
    if below_upper.size:
        median_upper = float(event_times[below_upper[0]])

    rmst, rmst_se = _restricted_mean(t, event_times, survival, at_risk, counts)
    return KaplanMeier(
        label=label,
        n=size,
        events=int(e.sum()),
        times=event_times,
        survival=survival,
        se=se,
        lower=lower,
        upper=upper,
        median=median,
        median_lower=median_lower,
        median_upper=median_upper,
        rmst=rmst,
        rmst_se=rmst_se,
    )


def _restricted_mean(
    times: FloatArray,
    event_times: FloatArray,
    survival: FloatArray,
    at_risk: FloatArray,
    counts: NDArray[np.integer],
) -> tuple[float, float | None]:
    """The area under the curve to the largest observed time, and its Greenwood standard error.

    The curve is a step: it holds 1 from zero to the first event time, and ``survival[i]``
    from ``event_times[i]`` to the next event time or the largest observed time. The variance
    at each event time is the squared area after it times ``d / (n (n - d))``; a step that
    ends the curve at its last event leaves no area after it, and a division by zero beyond
    it gives no standard error.
    """
    if event_times.size == 0:
        return float(np.max(times)), 0.0
    tau = float(np.max(times))
    edges = np.concatenate((np.zeros(1, dtype=np.float64), event_times))
    values = np.concatenate((np.ones(1, dtype=np.float64), survival))
    lengths = np.empty(edges.size, dtype=np.float64)
    lengths[:-1] = edges[1:] - edges[:-1]
    lengths[-1] = max(0.0, tau - edges[-1])
    segments = values * lengths
    rmst = float(segments.sum())
    after = np.cumsum(segments[::-1])[::-1]
    variance = 0.0
    defined = True
    for index, (n, d) in enumerate(zip(at_risk, counts, strict=True)):
        area = float(after[index + 1])
        if area <= 0.0:
            continue
        if n - float(d) > 0.0:
            variance += area * area * float(d) / (n * (n - float(d)))
        else:
            defined = False
    if not defined:
        return rmst, None
    return rmst, math.sqrt(variance)


def logrank(times: FloatArray, event: NDArray[np.bool_], codes: IntArray, count: int) -> LogRank:
    """The log-rank (Mantel-Haenszel) test over ``count`` groups coded 0 to ``count - 1``.

    At every distinct event time the observed events of a group are compared with the number
    the pooled risk sets imply, ``d * n_i / n``; the variance is the hypergeometric one,
    ``d (n - d) / (n - 1)`` times ``r_i (delta_ij - r_j)`` with ``r = n_i / n``. The statistic
    adds the squared differences weighted by the inverse of that covariance, on count - 1
    degrees of freedom. A time with one subject at risk has no variance and is skipped.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    g = np.asarray(codes, dtype=np.intp)
    ordered = [np.sort(t[g == index]) for index in range(count)]
    event_counts = [
        {
            float(when): int(dead)
            for when, dead in zip(*np.unique(t[(g == index) & e], return_counts=True), strict=True)
        }
        for index in range(count)
    ]
    observed = np.zeros(count, dtype=np.float64)
    expected = np.zeros(count, dtype=np.float64)
    covariance = np.zeros((count, count), dtype=np.float64)
    for when in np.unique(t[e]):
        sizes = np.array(
            [
                float(ordered[index].size - int(np.searchsorted(ordered[index], when, "left")))
                for index in range(count)
            ],
            dtype=np.float64,
        )
        deaths = np.array(
            [float(event_counts[index].get(float(when), 0)) for index in range(count)],
            dtype=np.float64,
        )
        total = float(sizes.sum())
        dead = float(deaths.sum())
        if total <= 1.0 or dead == 0.0:
            continue
        fraction = sizes / total
        observed += deaths
        expected += dead * fraction
        covariance += (np.diag(fraction) - np.outer(fraction, fraction)) * (
            dead * (total - dead) / (total - 1.0)
        )
    difference = (observed - expected)[1:]
    block = covariance[1:, 1:]
    try:
        solved = np.linalg.solve(block, difference)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(block) @ difference
    statistic = float(difference @ solved)
    statistic = max(0.0, statistic)
    df = float(count - 1)
    return LogRank(
        statistic=statistic,
        df=df,
        p_value=float(scipy_stats.chi2.sf(statistic, df)),
    )


def _require_information(information: FloatArray, names: Sequence[str]) -> None:
    """Refuse a Cox information matrix that cannot give every coefficient a variance.

    Raises:
        ToolError: ``COLLINEAR`` when an information entry is zero, or the matrix's smallest
            singular value is negligible beside its largest.
    """
    singular = np.linalg.svd(information, compute_uv=False)
    zero = np.flatnonzero(np.diag(information) <= 0.0)
    if zero.size:
        raise ToolError(
            "COLLINEAR",
            f"the predictor {names[int(zero[0])]!r} carries no information in the Cox model, "
            "so its coefficient cannot be estimated",
        )
    if singular.size == 0 or singular[-1] <= COLLINEAR_RTOL * singular[0]:
        raise ToolError(
            "COLLINEAR",
            f"the predictors {', '.join(repr(name) for name in names)} are exactly collinear "
            "in the Cox model, so their coefficients cannot be estimated",
        )


class _Evaluation(NamedTuple):
    """The partial log-likelihood, its gradient and information, and the risk-set means."""

    log_likelihood: float
    score: FloatArray
    information: FloatArray
    bars: tuple[tuple[float, FloatArray, FloatArray], ...]
    """Per event time: the time, the risk-set weighted mean covariate, and the covariates of
    the rows that died at that time."""


def _evaluate(
    times: FloatArray, event: NDArray[np.bool_], design: FloatArray, beta: FloatArray
) -> _Evaluation:
    """The Breslow-tied partial likelihood at ``beta``, by one descending pass over the times.

    Walking the distinct times from the largest down lets the risk set accumulate: before a
    time's contribution is read, every row at or after it has been added to ``S0``, ``S1`` and
    ``S2``, the sums of the weights, the weights times the covariates and the weights times the
    covariates squared. A censored row at an event time stays in the risk set, as Breslow's
    handling of a tie requires. The weights are shifted by their maximum for the exponential
    not to overflow; the sums are ratios, so the shift cancels everywhere but the
    log-likelihood, which adds it back.
    """
    count, parameters = design.shape
    eta = design @ beta
    shift = float(np.max(eta))
    weight = np.exp(eta - shift)
    order = np.argsort(-times, kind="mergesort")
    ordered_times = times[order]
    ordered_event = event[order]
    ordered_weight = weight[order]
    ordered_design = design[order]
    ordered_eta = eta[order]
    s0 = 0.0
    s1 = np.zeros(parameters, dtype=np.float64)
    s2 = np.zeros((parameters, parameters), dtype=np.float64)
    log_likelihood = 0.0
    score = np.zeros(parameters, dtype=np.float64)
    information = np.zeros((parameters, parameters), dtype=np.float64)
    bars: list[tuple[float, FloatArray, FloatArray]] = []
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and ordered_times[stop] == ordered_times[start]:
            stop += 1
        block_weight = ordered_weight[start:stop]
        block_design = ordered_design[start:stop]
        block_event = ordered_event[start:stop]
        s0 += float(block_weight.sum())
        s1 += (block_weight[:, None] * block_design).sum(axis=0)
        s2 += np.einsum("n,ni,nj->ij", block_weight, block_design, block_design)
        dead = float(block_event.sum())
        if dead > 0.0:
            mean = s1 / s0
            log_likelihood += float(ordered_eta[start:stop][block_event].sum()) - dead * (
                shift + math.log(s0)
            )
            score += block_design[block_event].sum(axis=0) - dead * mean
            information += dead * (s2 / s0 - np.outer(mean, mean))
            bars.append((float(ordered_times[start]), mean, block_design[block_event]))
        start = stop
    return _Evaluation(
        log_likelihood=log_likelihood,
        score=score,
        information=information,
        bars=tuple(bars),
    )


def _pearson(first: FloatArray, second: FloatArray) -> float | None:
    """Pearson's correlation of two arrays, or ``None`` when either does not vary."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return None
    a_centred = a - float(a.mean())
    b_centred = b - float(b.mean())
    spread = math.sqrt(float(a_centred @ a_centred) * float(b_centred @ b_centred))
    if spread <= 0.0 or not math.isfinite(spread):
        return None
    value = float(a_centred @ b_centred) / spread
    return max(-1.0, min(1.0, value))


def _concordance(risk: FloatArray, times: FloatArray, event: NDArray[np.bool_]) -> float | None:
    """Harrell's concordance index.

    A pair is comparable when the earlier time is an observed event and the later time is
    strictly larger: the order in which the two failed is known. The pair is concordant when
    the earlier of the two carries the larger risk, discordant when it carries the smaller,
    and tied at half a pair when the risks are equal. The index is the concordant share of the
    comparable pairs, so 0.5 is what a risk with no information scores.
    """
    comparable = 0
    concordant = 0.0
    for index in np.flatnonzero(event):
        later = times > times[index]
        how_many = int(np.count_nonzero(later))
        if how_many == 0:
            continue
        comparable += how_many
        others = risk[later]
        concordant += float(np.count_nonzero(risk[index] > others))
        concordant += 0.5 * float(np.count_nonzero(risk[index] == others))
    if comparable == 0:
        return None
    return concordant / comparable


def _schoenfeld(
    times: FloatArray,
    event: NDArray[np.bool_],
    design: FloatArray,
    beta: FloatArray,
    information: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The scaled Schoenfeld residuals, their correlation with the event time and its p-value.

    At each event time the residual of a row is its covariate less the risk set's weighted
    mean covariate; it is scaled by the inverse of the estimated coefficient covariance, the
    Grambsch-Therneau scaling, so that a coefficient that changes with time shows as a trend.
    The check correlates each column with the event time (Pearson) and reads a p-value from
    the t distribution on m - 2 degrees of freedom, with m the number of events. It is an
    approximation, not the full score process.
    """
    covariance = np.linalg.inv(information)
    event_times: list[float] = []
    residual_rows: list[FloatArray] = []
    for when, mean, covariates in _evaluate(times, event, design, beta).bars:
        residuals = covariates - mean
        scaled = residuals @ covariance
        for row in scaled:
            event_times.append(when)
            residual_rows.append(row)
    when_array = np.asarray(event_times, dtype=np.float64)
    residuals_array = np.vstack(residual_rows) if residual_rows else np.empty((0, design.shape[1]))
    correlations = np.full(design.shape[1], np.nan, dtype=np.float64)
    p_values = np.full(design.shape[1], np.nan, dtype=np.float64)
    size = int(when_array.size)
    for column in range(design.shape[1]):
        value = _pearson(when_array, residuals_array[:, column])
        if value is None:
            continue
        correlations[column] = value
        if size < 3:
            continue
        spread = 1.0 - value * value
        if spread <= 0.0:
            p_values[column] = 0.0
        else:
            statistic = abs(value) * math.sqrt((size - 2) / spread)
            p_values[column] = float(2.0 * scipy_stats.t.sf(statistic, size - 2))
    return when_array, residuals_array, correlations, p_values


def cox_fit(
    times: FloatArray,
    event: NDArray[np.bool_],
    design: FloatArray,
    names: Sequence[str],
    level: float,
) -> Cox:
    """Fit a Cox proportional-hazards model by Newton-Raphson on the partial likelihood.

    The predictors are centred before the fit, which leaves the coefficients and every hazard
    ratio unchanged and keeps the exponentials in range. A step solves the information for the
    score; the fit stops when no coefficient moves by more than 1e-9, or after 50 iterations,
    in which case it reports the last change rather than a silent result. The standard errors
    are the square roots of the diagonal of the inverse information at the estimate, the
    intervals the Wald intervals exp(beta +/- z SE), and the likelihood-ratio test compares the
    fit with the intercept-only model.

    Raises:
        ToolError: ``COLLINEAR`` when the information cannot give a coefficient a variance, or
            the fit leaves the finite numbers.
    """
    centred = design - design.mean(axis=0)
    parameters = centred.shape[1]
    beta = np.zeros(parameters, dtype=np.float64)
    converged = False
    last_change = math.inf
    iterations = 0
    for iteration in range(1, NEWTON_MAX_ITERATIONS + 1):
        evaluation = _evaluate(times, event, centred, beta)
        _require_information(evaluation.information, names)
        try:
            step = np.linalg.solve(evaluation.information, evaluation.score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(evaluation.information) @ evaluation.score
        beta = beta + step
        iterations = iteration
        last_change = float(np.max(np.abs(step)))
        if not np.all(np.isfinite(beta)):
            break
        if last_change < NEWTON_TOLERANCE:
            converged = True
            break
    evaluation = _evaluate(times, event, centred, beta)
    _require_information(evaluation.information, names)
    covariance = np.linalg.inv(evaluation.information)
    se = np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(se)):
        raise ToolError(
            "COLLINEAR",
            "the Cox model's coefficient or its standard error does not settle on these data",
        )
    hazard_ratio = np.exp(beta)
    if not np.all(np.isfinite(hazard_ratio)):
        raise ToolError(
            "COLLINEAR",
            "a Cox hazard ratio overflows, so the predictors separate the event times",
        )
    zero_se = np.flatnonzero(se <= 0.0)
    if zero_se.size:
        raise ToolError(
            "COLLINEAR",
            f"the predictor {names[int(zero_se[0])]!r} has no standard error in the Cox model",
        )
    z = beta / se
    p_value = 2.0 * scipy_stats.norm.sf(np.abs(z))
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    null = _evaluate(times, event, centred, np.zeros(parameters, dtype=np.float64))
    lr_statistic = max(0.0, 2.0 * (evaluation.log_likelihood - null.log_likelihood))
    lr_p_value = float(scipy_stats.chi2.sf(lr_statistic, parameters))
    risk = centred @ beta
    concordance = _concordance(risk, times, event)
    event_times, residuals, correlations, schoenfeld_p = _schoenfeld(
        times, event, centred, beta, evaluation.information
    )
    return Cox(
        beta=beta,
        se=se,
        z=z,
        p_value=p_value,
        hazard_ratio=hazard_ratio,
        lower=np.exp(beta - critical * se),
        upper=np.exp(beta + critical * se),
        log_likelihood=evaluation.log_likelihood,
        log_likelihood_null=null.log_likelihood,
        lr_statistic=lr_statistic,
        lr_p_value=lr_p_value,
        concordance=concordance,
        converged=converged,
        last_change=last_change,
        iterations=iterations,
        event_times=event_times,
        schoenfeld=residuals if residuals.size else None,
        schoenfeld_correlation=correlations,
        schoenfeld_p_value=schoenfeld_p,
    )


def _step_points(
    times: FloatArray, survival: FloatArray, tau: float
) -> tuple[tuple[float, ...], tuple[float, ...], str | None]:
    """The step curve of a Kaplan-Meier fit, as plot points, cut to the plot's bound.

    The curve is 1 until the first event time, then holds each value until the next event
    time; a curve with no event never falls.
    """
    xs: list[float] = [0.0]
    ys: list[float] = [1.0]
    previous = 1.0
    for when, value in zip(times, survival, strict=True):
        xs.append(float(when))
        ys.append(previous)
        xs.append(float(when))
        ys.append(float(value))
        previous = float(value)
    if not times.size or tau > float(times[-1]):
        xs.append(float(tau))
        ys.append(previous)
    return _bounded(xs, ys)


def _bounded(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...], str | None]:
    """Two arrays as plot points, evenly strided when longer than a plot may hold."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    note = None
    if x.size > MAX_PLOT_POINTS:
        taken = np.round(np.linspace(0.0, x.size - 1.0, MAX_PLOT_POINTS)).astype(np.intp)
        x = x[taken]
        y = y[taken]
        note = (
            f"{MAX_PLOT_POINTS:,} of {max(len(xs), len(ys)):,} points shown, taken at an even "
            "stride"
        )
    return tuple(float(value) for value in x), tuple(float(value) for value in y), note


def _survival_note(coding: _Coding) -> str:
    """The survival summary table's note: the definitions and the event's coding."""
    return (
        "The median is the first event time the Kaplan-Meier curve reaches 0.5, and its "
        "interval the Brookmeyer-Crowley interval from the complementary log-log band. The "
        "restricted mean integrates the curve to the group's own largest observed time, and "
        "its SE comes from Greenwood's variance. The event is "
        f"{coding.event!r}; {coding.censored!r} is read as censoring. A survival time of zero "
        "is allowed."
    )


def _survival_rows(curves: Sequence[KaplanMeier]) -> list[tuple[Cell, ...]]:
    """One row per group, with an empty cell where a number is undefined."""
    return [
        (
            curve.label,
            curve.n,
            curve.events,
            curve.median,
            curve.median_lower,
            curve.median_upper,
            curve.rmst,
            curve.rmst_se,
        )
        for curve in curves
    ]


def _proportional_hazards(cox: Cox | None, events: int) -> AuditCheck:
    """The approximate proportional-hazards check, as a caution or a pass."""
    if cox is None:
        return assumption(
            "proportional_hazards",
            "Proportional hazards does not apply without a Cox model",
            Verdict.NOT_APPLICABLE,
            "No Cox model was requested, so there is no proportional-hazards assumption to check.",
        )
    finite = np.isfinite(cox.schoenfeld_p_value)
    if cox.schoenfeld is None or events < SCHOENFELD_MIN_EVENTS or not finite.any():
        return assumption(
            "proportional_hazards",
            "The proportional-hazards assumption could not be checked",
            Verdict.NOT_APPLICABLE,
            f"The check correlates the scaled Schoenfeld residuals with the event time and "
            f"needs at least {SCHOENFELD_MIN_EVENTS} events; these data hold {events}.",
        )
    smallest = float(np.min(cox.schoenfeld_p_value[finite]))
    verdict = Verdict.CAUTION if smallest < CAUTION_P else Verdict.PASS
    titles = {
        Verdict.PASS: "The coefficients look constant over the follow-up",
        Verdict.CAUTION: "A coefficient may change over the follow-up",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If a coefficient changes with time, read the hazard ratio as an average over the "
        "follow-up; a time-varying coefficient or a stratified model may describe the data "
        "better."
    )
    return assumption(
        "proportional_hazards",
        titles[verdict],
        verdict,
        f"The correlation of the scaled Schoenfeld residuals with the event time gives its "
        f"smallest p as {p_text(smallest)}. This is an approximation from the residual "
        "correlations, not the full score process.",
        statistic=f"min p = {num(smallest)}",
        threshold=_CAUTION_THRESHOLD,
        next_step=next_step,
    )


def _independent_censoring() -> AuditCheck:
    """The independent-censoring assumption, which no number in the data can test."""
    return assumption(
        "independent_censoring",
        "Censoring is assumed independent of the event time",
        Verdict.NOT_APPLICABLE,
        "Whether a person's censoring is independent of their event time cannot be read from "
        "these data; it is argued from how the follow-up ended, not tested here.",
        next_step="Say in the write-up how follow-up ended, and whether the reason for "
        "censoring could be related to the risk of the event.",
    )


def _evidence(p_value: float) -> str:
    """What the log-rank p-value says about equal survival, as a clause."""
    if p_value >= CAUTION_P:
        return "the data are consistent with the same survival in every group"
    if p_value >= FAIL_P:
        return "the data show evidence that survival differs across the groups"
    return "the data show strong evidence that survival differs across the groups"


def _degrees(df: float) -> str:
    """``1 degree of freedom`` or ``3 degrees of freedom``."""
    count = num(df)
    return f"{count} degree of freedom" if df == 1.0 else f"{count} degrees of freedom"


def _interpretation(
    coding: _Coding,
    curves: Sequence[KaplanMeier],
    overall: LogRank | None,
    cox: Cox | None,
    names: Sequence[str],
) -> str:
    """One to three sentences: the medians, the log-rank test and the Cox hazard ratios."""
    medians = [
        f"{num(curve.median)} in {curve.label}"
        if curve.median is not None
        else f"not reached in {curve.label}"
        for curve in curves
    ]
    sentences = [
        f"With {coding.event!r} read as the event and {coding.censored!r} as censoring, the "
        f"median survival is {joined(medians)}."
    ]
    if overall is not None:
        sentences.append(
            f"The log-rank test on {_degrees(overall.df)} gives chi-square = "
            f"{num(overall.statistic)}, {p_text(overall.p_value)}, so "
            f"{_evidence(overall.p_value)}."
        )
    if cox is not None:
        ratios = [
            f"{num(float(cox.hazard_ratio[index]))} for {name} "
            f"({num(float(cox.lower[index]))} to {num(float(cox.upper[index]))})"
            for index, name in enumerate(names)
        ]
        sentences.append(
            f"The Cox model gives a hazard ratio of {joined(ratios)}; the likelihood-ratio "
            f"test gives chi-square = {num(cox.lr_statistic)} on "
            f"{_degrees(float(len(names)))}, {p_text(cox.lr_p_value)}."
        )
    return " ".join(sentences)


def _methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Survival was estimated by the Kaplan-Meier product-limit estimator, with Greenwood's "
        "variance; the pointwise interval uses the complementary log-log transformation, the "
        "median its Brookmeyer-Crowley interval, and the restricted mean integrates the curve "
        "to the largest observed time with the Greenwood-based variance. Groups were compared "
        "by the log-rank (Mantel-Haenszel) test on k - 1 degrees of freedom, with the pairwise "
        "comparisons adjusted by Holm's method. The Cox model was fitted by Newton-Raphson on "
        "the Breslow-tied partial likelihood, to a tolerance of 1e-9 or 50 iterations, with "
        "Wald intervals, a likelihood-ratio test and Harrell's concordance index; proportional "
        "hazards was checked by the correlation of the scaled Schoenfeld residuals with the "
        "event time. Censoring is assumed independent of the event time."
    )


def run_survival(request: BaseModel) -> ToolResult:
    """Run the survival tool on its own request."""
    if not isinstance(request, SurvivalRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"survival takes a SurvivalRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: SurvivalRequest) -> ToolResult:
    """The analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    if request.groups is None and not request.predictors:
        raise ToolError(
            "INVALID_OPTION",
            "survival needs groups for the Kaplan-Meier curves and the log-rank test, "
            "predictors for the Cox model, or both",
        )
    if request.names is not None and len(request.names) != len(request.predictors):
        raise ToolError(
            "INVALID_OPTION",
            f"survival takes one name per predictor, so {len(request.predictors)} names, not "
            f"{len(request.names)}",
        )
    coding = _coding(request)
    declared = levels(request.groups, minimum=2) if request.groups is not None else None
    complete = _complete(request, coding)
    if np.any(complete.times < 0.0):
        raise ToolError(
            "INVALID_OPTION",
            f"the time column {request.time.name!r} holds a value below zero, and a survival "
            "time must not be negative",
        )
    events = int(complete.event.sum())
    distinct = int(np.unique(complete.times).size)
    if events < MIN_EVENTS or distinct < MIN_DISTINCT_TIMES:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"survival needs at least {MIN_EVENTS} events and {MIN_DISTINCT_TIMES} distinct "
            f"times, and the rows left hold {events} "
            f"{'event' if events == 1 else 'events'} at {distinct} distinct "
            f"{'time' if distinct == 1 else 'times'}",
        )

    warnings: list[str] = []
    group_labels = complete.groups
    labels: tuple[str, ...]
    if declared is None:
        labels = (ALL_GROUP,)
        masks = [np.ones(complete.times.size, dtype=bool)]
    else:
        assert group_labels is not None
        labels = tuple(label for label in declared if bool(np.any(group_labels == label)))
        masks = [np.asarray(group_labels == label, dtype=bool) for label in labels]
        for label in declared:
            if label not in labels:
                warnings.append(f"the group {label!r} holds no complete row, so it has no curve")
    curves = [
        kaplan_meier(label, complete.times[mask], complete.event[mask], request.level)
        for label, mask in zip(labels, masks, strict=True)
    ]
    for curve in curves:
        if curve.events == 0:
            warnings.append(
                f"the group {curve.label!r} has no events, so its survival curve never falls"
            )

    overall: LogRank | None = None
    tests: list[HypothesisTest] = []
    if len(labels) >= 2:
        assert group_labels is not None
        codes = np.zeros(complete.times.size, dtype=np.intp)
        for index, label in enumerate(labels):
            codes[np.asarray(group_labels == label, dtype=bool)] = index
        overall = logrank(complete.times, complete.event, codes, len(labels))
        tests.append(
            HypothesisTest(
                name="Log-rank test",
                method="Log-rank (Mantel-Haenszel) test",
                null_hypothesis="The survival curves of every group are the same.",
                statistic=overall.statistic,
                df=overall.df,
                p_value=overall.p_value,
                p_value_text=p_text(overall.p_value),
            )
        )
        if len(labels) > 2:
            assert group_labels is not None
            tests.extend(_pairwise(complete.times, complete.event, group_labels, labels))

    names = (
        tuple(request.names)
        if request.names is not None
        else tuple(column.name for column in request.predictors)
    )
    cox: Cox | None = None
    if request.predictors:
        design = np.column_stack(complete.predictors)
        cox = cox_fit(complete.times, complete.event, design, names, request.level)
        tests.append(
            HypothesisTest(
                name="Likelihood-ratio test of the Cox model",
                method="Cox proportional hazards model, likelihood-ratio test, Breslow ties",
                null_hypothesis="Every predictor's coefficient is zero, so the predictors do "
                "not change the hazard.",
                statistic=cox.lr_statistic,
                df=float(len(names)),
                p_value=cox.lr_p_value,
                p_value_text=p_text(cox.lr_p_value),
            )
        )
        if not cox.converged:
            warnings.append(
                "the Cox model's Newton-Raphson iteration did not converge within "
                f"{NEWTON_MAX_ITERATIONS} iterations; the last change in a coefficient was "
                f"{num(cox.last_change)}"
            )

    estimates: list[Estimate] = []
    for curve in curves:
        if curve.median is not None:
            estimates.append(
                Estimate(
                    label=f"Median survival, {curve.label}",
                    value=curve.median,
                    lower=curve.median_lower,
                    upper=curve.median_upper,
                )
            )
        estimates.append(
            Estimate(
                label=f"Restricted mean survival, {curve.label}",
                value=curve.rmst,
                se=curve.rmst_se,
            )
        )
    if cox is not None:
        for index, name in enumerate(names):
            estimates.append(
                Estimate(
                    label=f"Hazard ratio, {name}",
                    value=float(cox.hazard_ratio[index]),
                    lower=float(cox.lower[index]),
                    upper=float(cox.upper[index]),
                )
            )
        if cox.concordance is not None:
            estimates.append(Estimate(label="Harrell's concordance index", value=cox.concordance))

    if events < SCHOENFELD_MIN_EVENTS:
        if request.predictors:
            warnings.append(
                f"there are {events} events, fewer than the {SCHOENFELD_MIN_EVENTS} the "
                "proportional-hazards check needs, so that check is not run and the estimates "
                "are imprecise"
            )
        else:
            warnings.append(
                f"there are {events} events, fewer than {SCHOENFELD_MIN_EVENTS}, so the "
                "survival estimates are imprecise"
            )

    assumptions = (_proportional_hazards(cox, events), _independent_censoring())
    tables = [
        table(
            "Survival summary",
            _SURVIVAL_COLUMNS,
            _survival_rows(curves),
            note=_survival_note(coding),
        )
    ]
    if cox is not None:
        tables.append(
            table(
                "Cox model",
                _COX_COLUMNS,
                (
                    (
                        name,
                        float(cox.hazard_ratio[index]),
                        float(cox.lower[index]),
                        float(cox.upper[index]),
                        float(cox.z[index]),
                        float(cox.p_value[index]),
                    )
                    for index, name in enumerate(names)
                ),
                note=(
                    "The hazard ratio is the exponential of the coefficient; its interval is "
                    "the Wald interval exp(beta +/- z SE) at the stated level. z is the "
                    "coefficient over its standard error and p its two-sided normal p-value. "
                    "Tied event times are handled by Breslow's method, and the model has no "
                    "intercept."
                ),
            )
        )
        if cox.schoenfeld is not None and events >= SCHOENFELD_MIN_EVENTS:
            tables.append(
                table(
                    "Schoenfeld check",
                    _SCHOENFELD_COLUMNS,
                    (
                        (
                            name,
                            (
                                float(cox.schoenfeld_correlation[index])
                                if math.isfinite(cox.schoenfeld_correlation[index])
                                else None
                            ),
                            (
                                float(cox.schoenfeld_p_value[index])
                                if math.isfinite(cox.schoenfeld_p_value[index])
                                else None
                            ),
                        )
                        for index, name in enumerate(names)
                    ),
                    note=(
                        "Each residual is the covariate less the risk set's weighted mean, "
                        "scaled by the inverse of the coefficient covariance; its correlation "
                        "with the event time and the p-value of that correlation are shown. "
                        "This is an approximation, not the full score process."
                    ),
                )
            )

    plots = _plots(curves, complete, request, cox, names, events)
    listed, unlisted = list_excluded(complete.excluded)
    used = int(complete.times.size)
    return ToolResult(
        tool="survival",
        title=(
            f"Survival analysis of {request.time.name} by {request.groups.name}"
            if request.groups is not None
            else f"Survival analysis of {request.time.name}"
        ),
        level=request.level,
        estimates=tuple(estimates),
        tests=tuple(tests),
        assumptions=assumptions,
        warnings=tuple(warnings),
        interpretation=_interpretation(coding, curves, overall, cox, names),
        methods_fragment=_methods(),
        counts=Counts(
            n_offered=complete.n_offered,
            n_used=used,
            n_excluded=len(complete.excluded),
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=plots,
    )


def _pairwise(
    times: FloatArray,
    event: NDArray[np.bool_],
    groups: StrArray,
    labels: Sequence[str],
) -> tuple[HypothesisTest, ...]:
    """One log-rank test per pair of groups, with Holm's adjustment on the p-values."""
    tests: list[HypothesisTest] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            left = np.asarray(groups == labels[first], dtype=bool)
            right = np.asarray(groups == labels[second], dtype=bool)
            both = left | right
            codes = np.where(left[both], 0, 1).astype(np.intp)
            found = logrank(times[both], event[both], codes, 2)
            tests.append(
                HypothesisTest(
                    name=f"Log-rank test, {labels[first]} against {labels[second]}",
                    method="Log-rank (Mantel-Haenszel) test, two groups, with Holm's adjustment",
                    null_hypothesis=(
                        f"The survival curves of {labels[first]} and {labels[second]} are the same."
                    ),
                    statistic=found.statistic,
                    df=found.df,
                    p_value=found.p_value,
                    p_value_text=p_text(found.p_value),
                )
            )
    return holm(tests)


def _plots(
    curves: Sequence[KaplanMeier],
    complete: _Complete,
    request: SurvivalRequest,
    cox: Cox | None,
    names: Sequence[str],
    events: int,
) -> tuple[MultiLinePlot | ScatterPlot, ...]:
    """The Kaplan-Meier step curves, and the Schoenfeld residuals when the check runs."""
    series: list[XYSeries] = []
    stride_note: str | None = None
    for curve in curves:
        mask = (
            np.ones(complete.times.size, dtype=bool)
            if complete.groups is None
            else np.asarray(complete.groups == curve.label, dtype=bool)
        )
        tau = float(np.max(complete.times[mask]))
        x, y, note = _step_points(curve.times, curve.survival, tau)
        if note is not None:
            stride_note = note
        series.append(XYSeries(label=curve.label, x=x, y=y))
    km_note = stride_note or "A step curve per group, built from the event times."
    plots: list[MultiLinePlot | ScatterPlot] = [
        MultiLinePlot(
            title="Kaplan–Meier",  # noqa: RUF001 - the estimator's name
            x_label=request.time.name,
            y_label="Survival probability",
            series=tuple(series),
            note=km_note,
        )
    ]
    if cox is not None and cox.schoenfeld is not None and events >= SCHOENFELD_MIN_EVENTS:
        residual = np.asarray(cox.schoenfeld[:, 0], dtype=np.float64)
        x, y, note = _bounded(
            [float(value) for value in cox.event_times], [float(value) for value in residual]
        )
        term = names[0] if names else "the predictor"
        description = (
            "The scaled Schoenfeld residual of each event against its time, for the "
            f"first predictor ({term}); the table holds every term."
            if len(names) > 1
            else "The scaled Schoenfeld residual of each event against its time."
        )
        plots.append(
            ScatterPlot(
                title="Schoenfeld residuals",
                x_label=request.time.name,
                y_label=f"Scaled Schoenfeld residual, {term}",
                x=x,
                y=y,
                note=description if note is None else f"{description} {note}",
            )
        )
    return tuple(plots)


TOOLS.register(
    ToolSpec(
        name="survival",
        family="advanced",
        title="Survival analysis",
        request=SurvivalRequest,
        run=run_survival,
    )
)
