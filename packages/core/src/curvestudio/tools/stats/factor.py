"""Factor analysis of three to thirty numeric columns, with varimax rotation.

The tool aligns the columns on the rows complete in all of them and factors their
correlation matrix by principal-axis factoring. The communalities start at the squared
multiple correlations (each column predicted from all the others), the diagonal of the
correlation matrix is replaced by them, the matrix is decomposed again, and the loop runs
until the communalities stop moving (below 1e-6, at most 100 iterations). The loadings are
the eigenvectors of the reduced matrix scaled by the square root of their eigenvalues.

The factors may then be rotated by Kaiser's normalised varimax: each row of loadings is
scaled to unit length, every pair of factors is turned by the angle that maximises the
variance of the squared loadings, and sweeps continue until the criterion stops moving
(below 1e-8, at most 100 sweeps). The factors are then ordered by the variance they explain,
and each factor's sign is fixed so its largest absolute loading is positive, so the same
data give the same table on every surface. Factor scores use the regression method
(Thurstone): the standardised columns multiplied by the inverse correlation matrix and the
shown loadings, then standardised again.

The result reports the Kaiser-Meyer-Olkin measure, the variance the factors explain and
Kaiser's criterion; Bartlett's test of sphericity; the loadings, communalities and
uniquenesses; the variance explained by each factor; and a summary of the scores, with a
scatter plot of the first two factors when there are two. A communality at or above one is a
Heywood case: it is warned about and the numbers are still shown. The outliers of every
column are checked against Tukey's fences (more than 5 % flagged is a caution), and the
linearity the model assumes is left to the scatter plots. A factor is a direction in the
data, not a latent cause.

Fewer than ten complete rows are refused, as is a column that holds one value repeated, a
correlation matrix that is singular, and a factor count outside what the columns can carry.
Every row left out is counted and listed, never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from curvestudio.data import Column
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import joined, num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    assumption,
    check_level,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    ScatterPlot,
    Table,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.pca import _Bartlett, _bartlett, _kaiser_meyer_olkin
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_COLUMNS = 3
MAX_COLUMNS = 30
MIN_ROWS = 10
"""The fewest complete rows a factor solution is read from."""

MAX_FACTORS = 10
"""The most factors the tool extracts, whatever the number of columns."""

PAF_TOLERANCE = 1e-6
"""The largest change in a communality that ends the principal-axis iteration."""

PAF_MAX_ITERATIONS = 100
"""The most principal-axis iterations, whether or not they converge."""

VARIMAX_TOLERANCE = 1e-8
"""The smallest change in the varimax criterion that continues the sweeps."""

VARIMAX_MAX_SWEEPS = 100
"""The most varimax sweeps, whether or not they converge."""

KMO_PASS = 0.80
"""A Kaiser-Meyer-Olkin measure at or above this is good for a factor analysis."""

KMO_CAUTION = 0.60
"""A measure at or above this, but below :data:`KMO_PASS`, is middling."""

OUTLIER_SHARE = 5.0
"""More than this percentage of a column beyond Tukey's fences is a caution."""

FENCE = 1.5
"""The interquartile range multiplier of Tukey's fences, the shipped box-plot rule."""

ROWS_PER_COLUMN = 5
"""Fewer used rows than this per column makes a factor solution unstable enough to warn."""

_LOADING_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("column", "Column", "text"),
    ("communality", "Communality", "number"),
    ("uniqueness", "Uniqueness", "number"),
)

_VARIANCE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("factor", "Factor", "int"),
    ("sum_of_squares", "Sum of squared loadings", "number"),
    ("proportion", "Proportion", "number"),
    ("cumulative", "Cumulative", "number"),
)

_SCORE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("factor", "Factor", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("minimum", "Minimum", "number"),
    ("maximum", "Maximum", "number"),
)


class FactorRequest(BaseModel):
    """The columns, the number of factors, the rotation and the level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: tuple[Column, ...] = Field(
        description=(
            "The three to thirty numeric columns whose common factors the analysis "
            "describes; each row is used only where it holds a value in every column."
        )
    )
    factors: int = Field(
        default=2,
        description=(
            "The number of factors to extract, from 1 to the smaller of 10 and one fewer "
            "than the number of columns."
        ),
        json_schema_extra={"label": "Number of factors"},
    )
    rotation: Literal["varimax", "none"] = Field(
        default="varimax",
        description=(
            "Rotate the factors by Kaiser's normalised varimax, or leave the loadings "
            "unrotated with 'none'."
        ),
        json_schema_extra={"label": "Rotation"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level the result carries, between 0 and 1.",
    )


class _Solution(NamedTuple):
    """The unrotated loadings of a principal-axis fit and its communalities."""

    loadings: FloatArray
    communalities: FloatArray


def _correlation(data: FloatArray) -> FloatArray:
    """The correlation matrix of the columns of ``data``."""
    return np.asarray(np.corrcoef(data, rowvar=False), dtype=np.float64)


def _squared_multiple_correlations(correlation: FloatArray) -> FloatArray:
    """Each column's squared multiple correlation with all the others.

    It is ``1 - 1 / (R^-1)_jj``: the share of a column's variance the other columns
    explain. A singular matrix has no inverse and is refused as collinear.

    Raises:
        ToolError: ``COLLINEAR`` when the correlation matrix cannot be inverted or a
            column's multiple correlation is not positive.
    """
    try:
        inverse = np.linalg.inv(correlation)
    except np.linalg.LinAlgError as exc:
        raise ToolError(
            "COLLINEAR",
            "the correlation matrix is singular, so the squared multiple correlations and "
            "the factors are undefined",
        ) from exc
    diagonal = np.diag(inverse)
    if np.any(diagonal <= 0.0):
        raise ToolError(
            "COLLINEAR",
            "the correlation matrix has a column that the others reproduce exactly, so its "
            "squared multiple correlation and the factors are undefined",
        )
    return np.asarray(1.0 - 1.0 / diagonal, dtype=np.float64)


def _principal_axis(correlation: FloatArray, factors: int) -> _Solution:
    """The unrotated loadings of ``factors`` principal axes and their communalities.

    The communalities start at the squared multiple correlations. Each iteration replaces
    the diagonal of the correlation matrix with them, takes the ``factors`` largest
    eigenvalues and vectors, and reads the new communalities off the loadings. The loop ends
    when the largest change is below :data:`PAF_TOLERANCE`, or after
    :data:`PAF_MAX_ITERATIONS`. A factor with no positive eigenvalue is a column of zeros.
    """
    size = correlation.shape[0]
    communalities = _squared_multiple_correlations(correlation)
    loadings = np.zeros((size, factors), dtype=np.float64)
    for _ in range(PAF_MAX_ITERATIONS):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communalities)
        values, vectors = np.linalg.eigh(reduced)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        positive = int(np.count_nonzero(values > 0.0))
        selected = min(positive, factors)
        loadings = np.zeros((size, factors), dtype=np.float64)
        if selected:
            loadings[:, :selected] = vectors[:, :selected] * np.sqrt(values[:selected])
        updated = np.asarray(np.sum(loadings**2, axis=1), dtype=np.float64)
        change = float(np.max(np.abs(updated - communalities)))
        communalities = updated
        if change < PAF_TOLERANCE:
            break
    return _Solution(loadings=loadings, communalities=communalities)


def _varimax_criterion(loadings: FloatArray) -> float:
    """Kaiser's varimax criterion: the summed variance of the squared loadings."""
    squared = loadings**2
    column_sums = np.sum(squared, axis=0)
    return float(np.sum(squared**2) - np.sum(column_sums**2) / loadings.shape[0])


def _pair_criterion(first: FloatArray, second: FloatArray) -> float:
    """The varimax criterion of one pair of factors, up to a constant."""
    difference = first**2 - second**2
    total = float(np.sum(difference))
    return float(np.sum(difference**2) - total * total / first.size)


def _rotate_pair(loadings: FloatArray, first: int, second: int) -> None:
    """Turn one pair of factors to the angle that maximises its varimax criterion.

    The angle solves the published Kaiser equation; the two stationary points a quarter turn
    apart are both tried and the one with the larger criterion is kept.
    """
    left = loadings[:, first]
    right = loadings[:, second]
    difference = left**2 - right**2
    cross = 2.0 * left * right
    rows = loadings.shape[0]
    a = float(np.sum(difference))
    b = float(np.sum(cross))
    c = float(np.sum(difference**2 - cross**2))
    d = float(2.0 * np.sum(difference * cross))
    angle = 0.25 * math.atan2(-(d - 2.0 * a * b / rows), c - (a * a - b * b) / rows)
    best_score: float | None = None
    best_left = left
    best_right = right
    for turn in (0.0, math.pi / 4.0):
        turned = angle + turn
        cosine, sine = math.cos(turned), math.sin(turned)
        candidate_left = left * cosine - right * sine
        candidate_right = left * sine + right * cosine
        score = _pair_criterion(candidate_left, candidate_right)
        if best_score is None or score > best_score:
            best_score = score
            best_left = candidate_left
            best_right = candidate_right
    loadings[:, first] = best_left
    loadings[:, second] = best_right


def _varimax(loadings: FloatArray) -> FloatArray:
    """``loadings`` after Kaiser's normalised varimax rotation.

    Each row is scaled to unit length before the pairwise sweeps and scaled back after them,
    so a variable with a large communality does not dominate the criterion. Sweeps continue
    until the criterion changes by less than :data:`VARIMAX_TOLERANCE`, at most
    :data:`VARIMAX_MAX_SWEEPS`.
    """
    factors = loadings.shape[1]
    norms = np.sqrt(np.asarray(np.sum(loadings**2, axis=1), dtype=np.float64))
    normalised = np.array(loadings, dtype=np.float64, copy=True)
    usable = norms > 0.0
    normalised[usable] = normalised[usable] / norms[usable, None]
    previous = _varimax_criterion(normalised)
    for _ in range(VARIMAX_MAX_SWEEPS):
        for first in range(factors - 1):
            for second in range(first + 1, factors):
                _rotate_pair(normalised, first, second)
        current = _varimax_criterion(normalised)
        if abs(current - previous) < VARIMAX_TOLERANCE:
            break
        previous = current
    normalised[usable] = normalised[usable] * norms[usable, None]
    return normalised


def _ordered_and_signed(loadings: FloatArray) -> FloatArray:
    """``loadings`` with the factors ordered by variance and each sign fixed.

    The factors are sorted by their sum of squared loadings, largest first. Each factor is
    then negated when needed so that its largest absolute loading is positive.
    """
    sums = np.asarray(np.sum(loadings**2, axis=0), dtype=np.float64)
    order = np.argsort(-sums, kind="stable")
    ordered = np.array(loadings[:, order], dtype=np.float64, copy=True)
    for index in range(ordered.shape[1]):
        largest = int(np.argmax(np.abs(ordered[:, index])))
        if ordered[largest, index] < 0.0:
            ordered[:, index] = -ordered[:, index]
    return ordered


def _scores(data: FloatArray, correlation: FloatArray, loadings: FloatArray) -> FloatArray:
    """The regression-method factor scores, standardised to mean zero and unit SD.

    The standardised columns are multiplied by the inverse correlation matrix and the shown
    loadings (Thurstone's regression method), and each score is then standardised on its own
    sample standard deviation. A factor whose raw scores do not vary is left at zero.
    """
    mean = np.mean(data, axis=0)
    spread = np.std(data, axis=0, ddof=1)
    standardised = (data - mean) / spread
    inverse = np.linalg.inv(correlation)
    raw = np.asarray(standardised @ inverse @ loadings, dtype=np.float64)
    centred = raw - np.mean(raw, axis=0)
    scale = np.std(raw, axis=0, ddof=1)
    usable = scale > 0.0
    centred[:, usable] = centred[:, usable] / scale[usable]
    return centred


def _kmo_reading(measure: float) -> str:
    """The reading of a Kaiser-Meyer-Olkin measure: pass, caution or fail."""
    if measure >= KMO_PASS:
        return "pass"
    if measure >= KMO_CAUTION:
        return "caution"
    return "fail"


def _sphericity_test(found: _Bartlett) -> HypothesisTest:
    """Bartlett's test of sphericity as the result's one hypothesis test."""
    return HypothesisTest(
        name="Bartlett's test of sphericity",
        method="Bartlett's test of sphericity, chi-square approximation",
        null_hypothesis="the columns are uncorrelated",
        statistic=found.statistic,
        df=found.df,
        p_value=found.p_value,
        p_value_text=p_text(found.p_value),
        effect=(
            None if found.statistic is None else Estimate(label="Chi-square", value=found.statistic)
        ),
        reason=found.reason,
    )


def _outlier_check(index: int, column: Column) -> AuditCheck:
    """One column's values beyond Tukey's fences, the shipped box-plot rule.

    More than :data:`OUTLIER_SHARE` per cent flagged is a caution; otherwise the check
    passes. The evidence names the count and the share.
    """
    values = np.asarray(column.values, dtype=np.float64)
    lower, upper = (float(value) for value in np.quantile(values, [0.25, 0.75]))
    spread = upper - lower
    low = lower - FENCE * spread
    high = upper + FENCE * spread
    flagged = int(np.count_nonzero((values < low) | (values > high)))
    share = flagged / values.size * 100.0
    verdict = Verdict.CAUTION if share > OUTLIER_SHARE else Verdict.PASS
    if flagged == 0:
        title = f"No value in {column.name!r} lies beyond Tukey's fences"
    elif flagged == 1:
        title = f"1 of {values.size} values in {column.name!r} lies beyond Tukey's fences"
    else:
        title = f"{flagged} of {values.size} values in {column.name!r} lie beyond Tukey's fences"
    verb = "lies" if flagged == 1 else "lie"
    return assumption(
        f"outliers_{index + 1}",
        title,
        verdict,
        f"In {column.name!r}, {flagged} of {values.size} used values ({num(share)} %) {verb} "
        f"beyond the fences at Q1 - {num(FENCE)} IQR and Q3 + {num(FENCE)} IQR; more than "
        f"{num(OUTLIER_SHARE)} % is a caution.",
        statistic=f"{flagged} flagged",
        threshold=(
            f"more than {num(OUTLIER_SHARE)} % of a column's used values beyond Tukey's fences"
        ),
        next_step=(
            "Outliers can pull the correlations; check them before reading the factors."
            if verdict is Verdict.CAUTION
            else None
        ),
    )


def _linearity_check() -> AuditCheck:
    """The linearity the factor model assumes, which these rows cannot test."""
    return assumption(
        "linearity",
        "Linearity between the columns is assumed, not tested",
        Verdict.NOT_APPLICABLE,
        "factor analysis assumes linear relations between the columns; look at the scatter plots",
        next_step="Look at the scatter plots of the columns before reading the factors.",
    )


def _loadings_table(
    loadings: FloatArray,
    communalities: FloatArray,
    uniquenesses: FloatArray,
    names: Sequence[str],
) -> Table:
    """One row per column: its loading on each factor, its communality and uniqueness."""
    columns: tuple[tuple[str, str, ColumnKind], ...] = (
        _LOADING_COLUMNS[0],
        *tuple(
            (f"factor_{index + 1}", f"Factor {index + 1}", "number")
            for index in range(loadings.shape[1])
        ),
        _LOADING_COLUMNS[1],
        _LOADING_COLUMNS[2],
    )
    rows: list[tuple[Cell, ...]] = []
    for row, name in enumerate(names):
        rows.append(
            (
                name,
                *(float(value) for value in loadings[row, :]),
                float(communalities[row]),
                float(uniquenesses[row]),
            )
        )
    return table(
        "Factor loadings",
        columns,
        rows,
        note=(
            "A loading is a column's coefficient on a factor, and its square is the share of "
            "that column's variance the factor takes. The communality is the sum of the "
            "squared loadings over the factors shown, and the uniqueness is one minus the "
            "communality."
        ),
    )


def _variance_table(sums: FloatArray, proportions: FloatArray, cumulative: FloatArray) -> Table:
    """One row per factor: its sum of squared loadings, proportion and cumulative share."""
    rows: list[tuple[Cell, ...]] = [
        (
            index + 1,
            float(sums[index]),
            float(proportions[index]),
            float(cumulative[index]),
        )
        for index in range(sums.size)
    ]
    return table(
        "Variance explained",
        _VARIANCE_COLUMNS,
        rows,
        note=(
            "The sum of squared loadings is the variance a factor explains; the proportion is "
            "that sum over the number of columns, and the cumulative column adds the factors "
            "in the order shown."
        ),
    )


def _scores_table(scores: FloatArray) -> Table:
    """One row per factor: the mean, SD, minimum and maximum of its scores."""
    rows: list[tuple[Cell, ...]] = [
        (
            index + 1,
            float(np.mean(scores[:, index])),
            float(np.std(scores[:, index], ddof=1)),
            float(np.min(scores[:, index])),
            float(np.max(scores[:, index])),
        )
        for index in range(scores.shape[1])
    ]
    return table(
        "Factor scores summary",
        _SCORE_COLUMNS,
        rows,
        note=(
            "A score is a row's value on the factor, from the regression method: the "
            "standardised columns multiplied by the inverse correlation matrix and the shown "
            "loadings. The scores are standardised to mean zero and unit sample standard "
            "deviation, with n - 1 in the denominator."
        ),
    )


def _scores_plot(scores: FloatArray) -> ScatterPlot:
    """The scores on the first two factors, as points in the plane they span."""
    first, note_x = stride(scores[:, 0])
    second, note_y = stride(scores[:, 1])
    details = ["The loadings are in the Factor loadings table."]
    note = note_x or note_y
    if note is not None:
        details.append(note)
    return ScatterPlot(
        title="Scores on the first two factors",
        x_label="Factor 1 score",
        y_label="Factor 2 score",
        x=tuple(float(value) for value in first),
        y=tuple(float(value) for value in second),
        note=" ".join(details),
    )


def _interpretation(loadings: FloatArray, names: Sequence[str]) -> str:
    """The largest absolute loading of each factor, and what a factor is not."""
    parts = []
    for index in range(loadings.shape[1]):
        at = int(np.argmax(np.abs(loadings[:, index])))
        parts.append(f"{names[at]!r} on factor {index + 1}")
    return (
        "The column with the largest absolute loading on each factor is "
        f"{joined(parts)}. A factor is a direction in the data, not a latent cause."
    )


def _methods(rotation: Literal["varimax", "none"]) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    turned = (
        "Kaiser's normalised varimax rotation"
        if rotation == "varimax"
        else "no rotation (the unrotated loadings)"
    )
    return (
        "Factor analysis used principal-axis factoring of the correlation matrix, with the "
        "squared multiple correlations as starting communalities and iteration until they "
        f"changed by less than {PAF_TOLERANCE:g}. The loadings are shown after {turned}, "
        "ordered by the variance each factor explains and signed so that each factor's "
        "largest absolute loading is positive; the factor scores use the regression method "
        "(Thurstone) and are standardised to mean zero and unit sample standard deviation."
    )


def _warnings(
    names: Sequence[str],
    communalities: FloatArray,
    rotation: Literal["varimax", "none"],
    n_used: int,
) -> tuple[str, ...]:
    """A warning for each Heywood case, few rows per column, and an unrotated solution."""
    found = [
        f"The communality of {names[index]!r} is {num(float(communalities[index]))}, at or "
        "above 1 (a Heywood case); the numbers are shown as computed."
        for index in range(communalities.size)
        if communalities[index] >= 1.0
    ]
    if n_used < ROWS_PER_COLUMN * len(names):
        found.append(
            f"{n_used} rows are used for {len(names)} columns, fewer than "
            f"{ROWS_PER_COLUMN} rows per column, so the factors may be unstable."
        )
    if rotation == "none":
        found.append(
            "No rotation was applied (rotation='none'), so the loadings are the unrotated ones."
        )
    return tuple(found)


def run_factor(request: BaseModel) -> ToolResult:
    """Run the factor analysis tool on its own request."""
    if not isinstance(request, FactorRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"factor takes a FactorRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: FactorRequest) -> ToolResult:
    """The factor solution itself, once the request is known to be this tool's own."""
    check_level(request.level)
    offered = len(request.columns)
    if not MIN_COLUMNS <= offered <= MAX_COLUMNS:
        raise ToolError(
            "INVALID_OPTION",
            f"factor analysis takes between {MIN_COLUMNS} and {MAX_COLUMNS} columns, not {offered}",
        )
    limit = min(MAX_FACTORS, offered - 1)
    if not 1 <= request.factors <= limit:
        raise ToolError(
            "INVALID_OPTION",
            f"the number of factors must lie between 1 and {limit} for {offered} columns, "
            f"not {request.factors}",
        )
    pairing = pairs(*request.columns)
    n_used = pairing.columns[0].values.size
    if n_used < MIN_ROWS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"factor analysis needs at least {MIN_ROWS} complete rows, and {n_used} are left",
        )
    for column in pairing.columns:
        if float(np.min(column.values)) == float(np.max(column.values)):
            raise ToolError(
                "CONSTANT_COLUMN",
                f"the column {column.name!r} holds one value repeated in the rows used, so "
                "its correlation with the other columns is undefined",
            )
    names = [column.name for column in pairing.columns]
    data = np.column_stack([column.values for column in pairing.columns])
    correlation = _correlation(data)
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        raise ToolError(
            "COLLINEAR",
            "the correlation matrix is singular, because two columns are exactly collinear "
            "or there are fewer rows than columns, so the factors are undefined",
        )
    solution = _principal_axis(correlation, request.factors)
    turned = _varimax(solution.loadings) if request.rotation == "varimax" else solution.loadings
    loadings = _ordered_and_signed(turned)
    communalities = np.asarray(np.sum(loadings**2, axis=1), dtype=np.float64)
    uniquenesses = 1.0 - communalities
    scores = _scores(data, correlation, loadings)
    sums = np.asarray(np.sum(loadings**2, axis=0), dtype=np.float64)
    proportions = sums / float(len(names))
    cumulative = np.cumsum(proportions)
    eigenvalues = np.linalg.eigvalsh(correlation)
    kaiser = float(np.count_nonzero(eigenvalues > 1.0))
    sphericity = _bartlett(correlation, n_used)
    measure = _kaiser_meyer_olkin(correlation)
    estimates: list[Estimate] = []
    if measure is not None:
        estimates.append(
            Estimate(
                label=f"Kaiser-Meyer-Olkin measure ({_kmo_reading(measure)})",
                value=measure,
            )
        )
    estimates.extend(
        (
            Estimate(
                label="Variance explained by the factors %",
                value=float(np.sum(sums) / len(names) * 100.0),
            ),
            Estimate(
                label="Kaiser's criterion: factors with an eigenvalue above 1",
                value=kaiser,
            ),
        )
    )
    assumptions: list[AuditCheck] = [
        _outlier_check(index, column) for index, column in enumerate(pairing.columns)
    ]
    assumptions.append(_linearity_check())
    warnings = list(_warnings(names, communalities, request.rotation, n_used))
    if measure is None:
        warnings.append(
            "The Kaiser-Meyer-Olkin measure is undefined because the columns share no variance."
        )
    tables: list[Table] = [
        _loadings_table(loadings, communalities, uniquenesses, names),
        _variance_table(sums, proportions, cumulative),
        _scores_table(scores),
    ]
    plots: list[ScatterPlot] = []
    if request.factors >= 2:
        plots.append(_scores_plot(scores))
    listed, unlisted = list_excluded(pairing.excluded)
    n_offered = max(column.rows.size for column in request.columns)
    return ToolResult(
        tool="factor",
        title="Factor analysis",
        level=request.level,
        estimates=tuple(estimates),
        tests=(_sphericity_test(sphericity),),
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
        interpretation=_interpretation(loadings, names),
        methods_fragment=_methods(request.rotation),
        counts=Counts(
            n_offered=int(n_offered),
            n_used=int(n_used),
            n_excluded=len(pairing.excluded),
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=tuple(plots),
    )


TOOLS.register(
    ToolSpec(
        name="factor",
        family="relationships",
        title="Factor analysis",
        request=FactorRequest,
        run=run_factor,
    )
)
