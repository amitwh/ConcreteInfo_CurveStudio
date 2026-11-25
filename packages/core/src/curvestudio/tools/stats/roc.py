"""Receiver operating characteristic curves: the area under the curve and its DeLong interval.

The tool pairs a numeric score with a two-level outcome by row position and keeps the rows
complete in both. The outcome's level that appears second, in the order of first appearance,
is the positive class, and the result names both levels and their counts. A score can point
either way: with ``higher_is_positive`` false the scores are negated before the analysis, so a
lower score is read as positive, and the result and the methods say so.

The curve is plotted at every distinct score, walked from the highest score down: at each
threshold every row whose score is at or above it is classified positive, which gives a tied
score one threshold, the mid-rank handling. The area under the curve is the trapezoid area
beneath that curve, which equals the Mann-Whitney probability that a randomly chosen positive
row scores above a randomly chosen negative row, with ties counted as one half. Its interval
is DeLong's method: the placement values of the positive rows among the negative rows and of
the negative rows among the positive rows, their variances, and the normal interval on the
area. The Youden index, sensitivity plus specificity minus one, picks the reported cut-off;
on a tie the first threshold met walking from the highest score down is reported. The scores
are also tested with the Mann-Whitney U test, two-sided, in its tie-corrected normal
approximation.

A ROC curve ranks a score: it says how well the score orders the positive rows above the
negative ones, not that the score diagnoses anything. Fewer than six rows in either class, or
fewer than three distinct scores, are refused. Every row left out is counted and listed, never
dropped without a word.
"""

import math
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.data.dataset import IntArray
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import ExcludedRow, Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    BoolArray,
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    _Kept,
    _spread,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    MultiLinePlot,
    Table,
    XYSeries,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PER_CLASS = 6
"""The fewest rows either class needs for a curve to be built."""

MIN_DISTINCT = 3
"""The fewest distinct scores a curve needs; two is a step, not a curve."""

WARN_PER_CLASS = 10
"""Fewer rows than this in a class makes the area unstable enough to warn about."""

MAX_TABLE_ROWS = 200
"""The most thresholds the table shows; a longer curve is sampled at an even stride."""

_YOUDEN_TIE = 1e-12
"""Two Youden indices this close are one, and the first threshold met is reported."""

_THRESHOLD_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("threshold", "Threshold", "number"),
    ("sensitivity", "Sensitivity", "number"),
    ("specificity", "Specificity", "number"),
    ("youden", "Youden index", "number"),
    ("classified_positive", "Classified positive", "int"),
    ("classified_negative", "Classified negative", "int"),
)


class RocRequest(BaseModel):
    """The score, the two-level outcome, the score's direction and the level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: Column = Field(
        description=(
            "The numeric score whose ability to rank the positive rows above the negative "
            "rows the curve measures."
        ),
    )
    outcome: GroupColumn = Field(
        description=(
            "The outcome column, which must hold exactly two labels. The label that appears "
            "second, in the order of first appearance, is the positive class."
        ),
    )
    higher_is_positive: bool = Field(
        default=True,
        description=(
            "Read a higher score as pointing to the positive class, or leave false to read a "
            "lower score as positive; when false the scores are negated before the analysis."
        ),
        json_schema_extra={"label": "A higher score means positive"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of the area's interval, between 0 and 1.",
    )


class _Cases(NamedTuple):
    """The complete pairs of score and label, and the rows left out."""

    positive: str
    negative: str
    scores: FloatArray
    is_positive: BoolArray
    excluded: tuple[ExcludedRow, ...]
    offered: int


class _Curve(NamedTuple):
    """The ROC curve: its points, the rates at each distinct threshold, and the area."""

    fpr: FloatArray
    tpr: FloatArray
    thresholds: FloatArray
    """The distinct scores, descending, in the direction read as positive."""
    sensitivity: FloatArray
    specificity: FloatArray
    youden: FloatArray
    classified_positive: IntArray
    classified_negative: IntArray
    auc: float


def _require_levels(outcome: GroupColumn) -> tuple[str, ...]:
    """The outcome's labels in order of first appearance, exactly two of them."""
    found = levels(outcome, minimum=0)
    if len(found) != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the column {outcome.name!r} holds {len(found)} labels, and a receiver operating "
            "characteristic curve needs exactly two",
        )
    return found


def _complete_cases(score: Column, outcome: GroupColumn, found: tuple[str, ...]) -> _Cases:
    """The rows complete in the score and the outcome, and the rows left out.

    The score and the labels are laid out over every position either names, and a position
    missing in the outcome or in the score, or holding a non-finite score, is left out with
    the reason, named after the first column at fault: ``no group label in <outcome>``,
    ``missing in <score>`` or ``not a finite number in <score>``. The excluded rows come in
    position order.
    """
    positions = np.unique(np.concatenate([score.rows, outcome.rows]))
    rows = _Kept(positions)
    covered, labels = _spread(positions, outcome.rows, outcome.labels, "")
    rows.drop(~covered, f"no group label in {outcome.name}")
    covered, values = _spread(positions, score.rows, score.values, np.nan)
    rows.drop(~covered | np.isnan(values), f"missing in {score.name}")
    rows.drop(np.isinf(values), f"not a finite number in {score.name}")
    kept = rows.kept
    used = labels[kept]
    return _Cases(
        positive=found[1],
        negative=found[0],
        scores=values[kept],
        is_positive=used == found[1],
        excluded=rows.excluded(),
        offered=int(positions.size),
    )


def _curve(scores: FloatArray, is_positive: BoolArray) -> _Curve:
    """The ROC curve at every distinct score, walked from the highest score down.

    At each threshold every row whose score is at or above it is classified positive, so a
    tied score adds all its rows at one threshold. The points start at (0, 0), above the
    highest score, and end at (1, 1), at the lowest. The area is the trapezoid rule over the
    points; the vertical stretches of a tied block contribute no width.
    """
    n1 = int(np.count_nonzero(is_positive))
    n0 = int(is_positive.size - n1)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_positive = is_positive[order]

    fpr = [0.0]
    tpr = [0.0]
    thresholds: list[float] = []
    sensitivity: list[float] = []
    specificity: list[float] = []
    youden: list[float] = []
    classified_positive: list[int] = []
    classified_negative: list[int] = []

    true_positive = 0
    false_positive = 0
    index = 0
    size = ordered_scores.size
    while index < size:
        threshold = ordered_scores[index]
        following = index
        while following < size and ordered_scores[following] == threshold:
            if ordered_positive[following]:
                true_positive += 1
            else:
                false_positive += 1
            following += 1
        true_negative = n0 - false_positive
        false_negative = n1 - true_positive
        sens = true_positive / n1
        spec = true_negative / n0
        thresholds.append(float(threshold))
        sensitivity.append(sens)
        specificity.append(spec)
        youden.append(sens + spec - 1.0)
        classified_positive.append(true_positive + false_positive)
        classified_negative.append(true_negative + false_negative)
        fpr.append(false_positive / n0)
        tpr.append(sens)
        index = following

    x = np.asarray(fpr, dtype=np.float64)
    y = np.asarray(tpr, dtype=np.float64)
    auc = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))
    return _Curve(
        fpr=x,
        tpr=y,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        sensitivity=np.asarray(sensitivity, dtype=np.float64),
        specificity=np.asarray(specificity, dtype=np.float64),
        youden=np.asarray(youden, dtype=np.float64),
        classified_positive=np.asarray(classified_positive, dtype=np.intp),
        classified_negative=np.asarray(classified_negative, dtype=np.intp),
        auc=auc,
    )


def _placements(positive: FloatArray, negative: FloatArray) -> tuple[FloatArray, FloatArray]:
    """DeLong's placement values: V10 of the positives and V01 of the negatives.

    ``V10[i]`` is the share of the negative rows that the i-th positive row scores above, a
    tie counting one half; ``V01[j]`` is the same for the j-th negative row against the
    positives. Their means both equal the area under the curve.
    """
    n1 = positive.size
    n0 = negative.size
    sorted_negative = np.sort(negative)
    below = np.searchsorted(sorted_negative, positive, side="left")
    through = np.searchsorted(sorted_negative, positive, side="right")
    v10 = (below + 0.5 * (through - below)) / n0

    sorted_positive = np.sort(positive)
    below = np.searchsorted(sorted_positive, negative, side="left")
    through = np.searchsorted(sorted_positive, negative, side="right")
    v01 = ((n1 - through) + 0.5 * (through - below)) / n1
    return np.asarray(v10, dtype=np.float64), np.asarray(v01, dtype=np.float64)


def _delong_se(v10: FloatArray, v01: FloatArray) -> float:
    """DeLong's standard error of the area: the two placement-value variances over their counts."""
    n1, n0 = v10.size, v01.size
    s10 = float(np.var(v10, ddof=1))
    s01 = float(np.var(v01, ddof=1))
    return math.sqrt(max(s10 / n1 + s01 / n0, 0.0))


def _best_youden(curve: _Curve) -> int:
    """The index of the largest Youden index, the first met on a tie."""
    best = 0
    for index in range(1, curve.youden.size):
        if curve.youden[index] > curve.youden[best] + _YOUDEN_TIE:
            best = index
    return best


def _percent(level: float) -> str:
    return f"{num(level * 100)} %"


def _direction_words(higher_is_positive: bool) -> str:
    """The direction sentence the methods fragment carries."""
    if higher_is_positive:
        return "A higher score is read as positive."
    return (
        "A lower score is read as positive, so the scores were negated before the analysis, "
        "and the cut-off is reported as an original score value."
    )


def _threshold_table(
    curve: _Curve,
    *,
    higher_is_positive: bool,
    positive: str,
    negative: str,
    n1: int,
    n0: int,
) -> Table:
    """The rates at each distinct threshold, at most :data:`MAX_TABLE_ROWS` of them."""
    total = int(curve.thresholds.size)
    if total <= MAX_TABLE_ROWS:
        chosen = np.arange(total, dtype=np.intp)
        stride_note = None
    else:
        chosen = np.unique(np.round(np.linspace(0.0, total - 1.0, MAX_TABLE_ROWS)).astype(np.intp))
        stride_note = f"{chosen.size:,} of {total:,} thresholds shown, taken at an even stride"
    sign = 1.0 if higher_is_positive else -1.0
    rows: list[tuple[Cell, ...]] = [
        (
            sign * float(curve.thresholds[index]),
            float(curve.sensitivity[index]),
            float(curve.specificity[index]),
            float(curve.youden[index]),
            int(curve.classified_positive[index]),
            int(curve.classified_negative[index]),
        )
        for index in chosen
    ]
    classified = (
        "At each threshold a row whose score is at or above it is classified positive."
        if higher_is_positive
        else "At each threshold a row whose score is at or below it is classified positive."
    )
    details = [
        f"{n1} positive rows ({positive!r}) and {n0} negative rows ({negative!r}).",
        classified,
        "When two thresholds share the largest Youden index, the first met walking the scores "
        "from the positive end is reported.",
    ]
    if stride_note is not None:
        details.append(stride_note)
    return table("Thresholds", _THRESHOLD_COLUMNS, rows, note=" ".join(details))


def _roc_plot(curve: _Curve) -> MultiLinePlot:
    """The curve, with the chance line from (0, 0) to (1, 1)."""
    x, note_x = stride(curve.fpr)
    y, note_y = stride(curve.tpr)
    return MultiLinePlot(
        title="ROC curve",
        x_label="False-positive rate",
        y_label="True-positive rate",
        series=(
            XYSeries(
                label="ROC curve",
                x=tuple(float(value) for value in x),
                y=tuple(float(value) for value in y),
            ),
            XYSeries(
                label="Chance",
                x=(0.0, 1.0),
                y=(0.0, 1.0),
            ),
        ),
        note=note_x or note_y,
    )


def _warnings(n1: int, n0: int, distinct: int) -> tuple[str, ...]:
    """Few rows in a class, and many tied scores."""
    found: list[str] = []
    if n1 < WARN_PER_CLASS:
        found.append(
            f"Only {n1} positive rows are used, fewer than {WARN_PER_CLASS}; the area under the "
            "curve may be unstable."
        )
    if n0 < WARN_PER_CLASS:
        found.append(
            f"Only {n0} negative rows are used, fewer than {WARN_PER_CLASS}; the area under the "
            "curve may be unstable."
        )
    if 2 * distinct <= n1 + n0:
        found.append(
            f"The scores hold only {distinct} distinct values in {n1 + n0} rows, so many are "
            f"tied; the curve rests on {distinct} cut-points and ties count as one half."
        )
    return tuple(found)


def _interpretation(
    auc: Estimate,
    cut: Estimate,
    sensitivity: Estimate,
    specificity: Estimate,
    *,
    positive: str,
    negative: str,
    n1: int,
    n0: int,
    higher_is_positive: bool,
    level: float,
) -> str:
    """Three sentences: the area and what it ranks, the cut, and what a curve cannot do."""
    assert auc.lower is not None and auc.upper is not None
    first = (
        f"The area under the curve is {num(auc.value)}, with a {_percent(level)} interval from "
        f"{num(auc.lower)} to {num(auc.upper)}, over {n1} positive rows ({positive!r}) and "
        f"{n0} negative rows ({negative!r}); it is the chance that a randomly chosen positive "
        "row scores above a randomly chosen negative row, with ties counted as one half."
    )
    second = (
        f"The best cut-off by the Youden index is {num(cut.value)}, with sensitivity "
        f"{num(sensitivity.value)} and specificity {num(specificity.value)}"
    )
    if not higher_is_positive:
        second += (
            "; a lower score is read as positive, so the scores were negated and a row at or "
            "below the cut-off is classified positive"
        )
    second += "."
    third = "A ROC curve ranks a score; it does not establish a diagnosis."
    return " ".join((first, second, third))


def _methods(higher_is_positive: bool) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Receiver operating characteristic curves plot the true-positive rate against the "
        "false-positive rate at every distinct score, walked from the highest score down; a "
        "tied score is given one threshold, the mid-rank handling, so ties count as one half "
        "in the area. The area under the curve is the trapezoid area beneath the curve, which "
        "equals the Mann-Whitney probability that a randomly chosen positive row scores above "
        "a randomly chosen negative row. Its interval is DeLong's method: the placement values "
        "of the positive rows among the negative rows and of the negative rows among the "
        "positive rows, their variances, and the normal interval on the area. "
        f"{_direction_words(higher_is_positive)}"
    )


def run_roc(request: BaseModel) -> ToolResult:
    """Run the receiver operating characteristic curve tool on its own request."""
    if not isinstance(request, RocRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"roc takes a RocRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: RocRequest) -> ToolResult:
    """The curve itself, once the request is known to be this tool's own."""
    check_level(request.level)
    found = _require_levels(request.outcome)
    cases = _complete_cases(request.score, request.outcome, found)
    n1 = int(np.count_nonzero(cases.is_positive))
    n0 = int(cases.is_positive.size - n1)
    if n1 < MIN_PER_CLASS or n0 < MIN_PER_CLASS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"a receiver operating characteristic curve needs at least {MIN_PER_CLASS} rows in "
            f"each class, and {cases.positive!r} holds {n1} and {cases.negative!r} holds {n0}",
        )
    distinct = int(np.unique(cases.scores).size)
    if distinct < MIN_DISTINCT:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the column {request.score.name!r} holds {distinct} distinct "
            f"{'value' if distinct == 1 else 'values'} in the rows used, and the curve needs at "
            f"least {MIN_DISTINCT}",
        )

    scores = cases.scores if request.higher_is_positive else -cases.scores
    curve = _curve(scores, cases.is_positive)
    positive = scores[cases.is_positive]
    negative = scores[~cases.is_positive]
    v10, v01 = _placements(positive, negative)
    se = _delong_se(v10, v01)
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - request.level) / 2.0))
    area = Estimate(
        label="Area under the curve",
        value=curve.auc,
        se=se,
        lower=curve.auc - critical * se,
        upper=curve.auc + critical * se,
    )

    best = _best_youden(curve)
    sign = 1.0 if request.higher_is_positive else -1.0
    cut = Estimate(label="Best cut-off (Youden)", value=sign * float(curve.thresholds[best]))
    sensitivity = Estimate(label="Sensitivity at the cut", value=float(curve.sensitivity[best]))
    specificity = Estimate(label="Specificity at the cut", value=float(curve.specificity[best]))

    statistic, p_value = _mann_whitney(positive, negative)
    test = HypothesisTest(
        name="Mann-Whitney U test of the scores",
        method=(
            "Mann-Whitney U test, two-sided, normal approximation with ties and continuity "
            "correction"
        ),
        null_hypothesis="the scores do not rank the positive class above the negative class",
        statistic=statistic,
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=area,
        reason=None if p_value is not None else "the tie-corrected variance of the ranks is zero",
    )

    excluded, unlisted = list_excluded(cases.excluded)
    used = n1 + n0
    return ToolResult(
        tool="roc",
        title=f"Receiver operating characteristic curve of {request.score.name} by "
        f"{request.outcome.name}",
        level=request.level,
        estimates=(area, cut, sensitivity, specificity),
        tests=(test,),
        assumptions=(
            assumption(
                "independence",
                "Independence between the rows cannot be read from the columns",
                Verdict.NOT_APPLICABLE,
                "Each row pairs one score with one outcome label; the rows are taken as "
                "independent measurements, and whether they are independent is a fact "
                "about how the data were collected, which the columns cannot record.",
                next_step=(
                    "Check how the rows were collected; dependence between them changes what "
                    "the p-value means."
                ),
            ),
        ),
        warnings=_warnings(n1, n0, distinct),
        interpretation=_interpretation(
            area,
            cut,
            sensitivity,
            specificity,
            positive=cases.positive,
            negative=cases.negative,
            n1=n1,
            n0=n0,
            higher_is_positive=request.higher_is_positive,
            level=request.level,
        ),
        methods_fragment=_methods(request.higher_is_positive),
        counts=Counts(n_offered=cases.offered, n_used=used, n_excluded=len(cases.excluded)),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=(
            _threshold_table(
                curve,
                higher_is_positive=request.higher_is_positive,
                positive=cases.positive,
                negative=cases.negative,
                n1=n1,
                n0=n0,
            ),
        ),
        plots=(_roc_plot(curve),),
    )


def _mann_whitney(positive: FloatArray, negative: FloatArray) -> tuple[float | None, float | None]:
    """The Mann-Whitney U of the positives and its two-sided tie-corrected normal p-value."""
    tested = scipy_stats.mannwhitneyu(
        positive, negative, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    statistic = float(tested.statistic)
    p_value = float(tested.pvalue)
    return (
        statistic if math.isfinite(statistic) else None,
        p_value if math.isfinite(p_value) else None,
    )


TOOLS.register(
    ToolSpec(
        name="roc",
        family="measurement",
        title="Receiver operating characteristic curve",
        request=RocRequest,
        run=run_roc,
    )
)
