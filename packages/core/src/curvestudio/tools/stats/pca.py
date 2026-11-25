"""Principal component analysis of two to thirty numeric columns.

The tool aligns the columns on the rows complete in all of them and decomposes either their
correlation matrix (the default: every column is turned into a z-score first, so a column in
large units does not drown the others) or their covariance matrix. The eigenvalue
decomposition of that matrix gives one component per column, ordered by the variance it
explains; each component is an orthogonal linear combination of the columns, and its
eigenvector is the list of loadings. The sign of every eigenvector is fixed so that its
largest absolute loading is positive, which makes the same data give the same components on
every surface. The scores are the standardised (or centred) columns projected on the
components.

The result reports the variance the first one and the first two components explain and how
many components reach 80 % of the variance, the eigenvalues with their loadings as a table,
a summary of the scores and a scree plot; the scores on the first two components are drawn
as a scatter plot when both are shown. The analysis is worth doing only where the columns
are correlated, so two checks always run: Bartlett's test of sphericity (whose null is that
the columns are uncorrelated) and the Kaiser-Meyer-Olkin measure of shared variance. A low
p-value and a high measure are what make the components worth reading; neither is a proof
that the components mean anything, and a component is a direction in the data, not a cause.

A column that holds one value repeated on the rows used has no variance and so no loading:
it is refused when the columns are standardised, and left out with a warning when they are
not. Fewer than four complete rows are refused. Every row left out is counted and listed,
never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import listing, num, p_threshold
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
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
    MultiLinePlot,
    ScatterPlot,
    Table,
    XYSeries,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_COLUMNS = 2
MAX_COLUMNS = 30
MIN_ROWS = 4
"""The fewest complete rows a component is read from."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

KMO_PASS = 0.80
"""A Kaiser-Meyer-Olkin measure at or above this is good for a component analysis."""

KMO_CAUTION = 0.60
"""A measure at or above this, but below :data:`KMO_PASS`, is middling."""

VARIANCE_TARGET = 80.0
"""The cumulative percentage of variance a person usually asks the components to reach."""

WARN_ROWS = 10
"""Fewer used rows than this makes the components unstable enough to warn about."""

_COMPONENT_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("component", "Component", "int"),
    ("eigenvalue", "Eigenvalue", "number"),
    ("variance", "Variance explained %", "number"),
    ("cumulative", "Cumulative %", "number"),
)

_SCORE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("component", "Component", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("minimum", "Minimum", "number"),
    ("maximum", "Maximum", "number"),
)


class PcaRequest(BaseModel):
    """The columns, the scaling, the most components to show and the level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: tuple[Column, ...] = Field(
        description=(
            "The two to thirty numeric columns whose common structure the components "
            "describe; each row is used only where it holds a value in every column."
        )
    )
    standardise: bool = Field(
        default=True,
        description=(
            "Turn every column into a z-score and decompose the correlation matrix, or leave "
            "false to decompose the covariance matrix of the centred columns. Standardising "
            "keeps a column in large units from dominating the components."
        ),
        json_schema_extra={"label": "Standardise the columns"},
    )
    max_components: int | None = Field(
        default=None,
        description=(
            "Show at most this many components in the tables and the plots. The variance "
            "percentages are still computed over every component. Left empty, every "
            "component is shown."
        ),
        json_schema_extra={"label": "Most components to show"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level the result carries, between 0 and 1.",
    )


class _Bartlett(NamedTuple):
    """Bartlett's test of sphericity: its statistic, degrees of freedom and p-value."""

    statistic: float | None
    df: float
    p_value: float | None
    reason: str | None


def _correlation(data: FloatArray) -> FloatArray:
    """The correlation matrix of the columns of ``data``."""
    return np.asarray(np.corrcoef(data, rowvar=False), dtype=np.float64)


def _matrix(data: FloatArray, standardise: bool) -> tuple[FloatArray, FloatArray]:
    """The matrix to decompose, and the columns the scores are projected from.

    Standardised, the matrix is the correlation of the columns and the projection takes their
    z-scores; otherwise the matrix is their covariance and the projection takes them centred.
    """
    if standardise:
        return _correlation(data), _standardised(data)
    mean = np.mean(data, axis=0)
    covariance = np.asarray(np.cov(data, rowvar=False, ddof=1), dtype=np.float64)
    return covariance, data - mean


def _standardised(data: FloatArray) -> FloatArray:
    """The columns as z-scores, each centred on its mean and divided by its sample SD."""
    mean = np.mean(data, axis=0)
    spread = np.std(data, axis=0, ddof=1)
    return (data - mean) / spread


def _decompose(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The eigenvalues and eigenvectors of a symmetric matrix, in descending order.

    The eigenvector of each component has its sign fixed so that its largest absolute entry
    is positive. An eigenvalue below zero is rounding of a positive-semidefinite matrix and
    reads as zero.
    """
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    ordered = values[order]
    components = vectors[:, order]
    for index in range(components.shape[1]):
        largest = int(np.argmax(np.abs(components[:, index])))
        if components[largest, index] < 0.0:
            components[:, index] = -components[:, index]
    return np.where(ordered < 0.0, 0.0, ordered), components


def _bartlett(correlation: FloatArray, n: int) -> _Bartlett:
    """Bartlett's test of sphericity of ``correlation`` on ``n`` rows.

    The null hypothesis is that the columns are uncorrelated, so the correlation matrix is an
    identity. The statistic is ``-(n - 1 - (2p + 5) / 6) * log(det(R))`` on ``p(p - 1) / 2``
    degrees of freedom. A singular matrix leaves the determinant at zero and the test
    undefined.
    """
    size = correlation.shape[0]
    df = float(size * (size - 1) // 2)
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        return _Bartlett(
            statistic=None,
            df=df,
            p_value=None,
            reason=(
                "The correlation matrix is singular, because two columns are exactly "
                "collinear, so its determinant and the test are undefined."
            ),
        )
    statistic = -(n - 1.0 - (2.0 * size + 5.0) / 6.0) * float(log_determinant)
    p_value = float(scipy_stats.chi2.sf(statistic, df))
    return _Bartlett(statistic=statistic, df=df, p_value=p_value, reason=None)


def _kaiser_meyer_olkin(correlation: FloatArray) -> float | None:
    """The Kaiser-Meyer-Olkin measure of the shared variance of ``correlation``.

    It is the sum of the squared correlations off the diagonal over that sum plus the same
    sum of the squared partial correlations, which the inverse of the matrix gives. A
    singular matrix, or one whose columns share no variance at all, has no measure.
    """
    size = correlation.shape[0]
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        return None
    inverse = np.linalg.inv(correlation)
    diagonal = np.sqrt(np.diag(inverse))
    if np.any(diagonal <= 0.0):
        return None
    partial = -inverse / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)
    off_diagonal = ~np.eye(size, dtype=bool)
    shared = float(np.sum(correlation[off_diagonal] ** 2))
    unique = float(np.sum(partial[off_diagonal] ** 2))
    total = shared + unique
    if total <= 0.0:
        return None
    return shared / total


def _sphericity_verdict(p_value: float | None) -> Verdict:
    """The verdict Bartlett's p-value gives the suitability of a component analysis.

    The null hypothesis is that the columns are uncorrelated, so a small p is the good news
    here: it reads as a pass, and the reverse as a failure.
    """
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < CAUTION_P:
        return Verdict.PASS
    if p_value < 2.0 * CAUTION_P:
        return Verdict.CAUTION
    return Verdict.FAIL


def _sphericity_check(found: _Bartlett) -> AuditCheck:
    """Bartlett's test as a check of whether the columns are correlated enough to summarise."""
    if found.statistic is None or found.p_value is None:
        return assumption(
            "sphericity",
            "Bartlett's test of sphericity could not be run",
            Verdict.NOT_APPLICABLE,
            found.reason or "The test could not be run on this correlation matrix.",
        )
    verdict = _sphericity_verdict(found.p_value)
    titles = {
        Verdict.PASS: "The columns are correlated enough for components to summarise them",
        Verdict.CAUTION: "The columns show only weak correlation",
        Verdict.FAIL: "The columns look close to uncorrelated, so components may add little",
    }
    next_steps = {
        Verdict.PASS: None,
        Verdict.CAUTION: "With only weak correlation among the columns, components may add "
        "little to them.",
        Verdict.FAIL: "The columns are close to uncorrelated; components may not summarise them.",
    }
    return assumption(
        "sphericity",
        titles[verdict],
        verdict,
        "Bartlett's test of sphericity gives chi-square = "
        f"{num(found.statistic)} on {num(found.df)} degrees of freedom, "
        f"{p_text(found.p_value)}.",
        statistic=f"chi-square = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_steps[verdict],
    )


def _factorability_check(measure: float | None) -> AuditCheck:
    """The Kaiser-Meyer-Olkin measure as a check of how well components can summarise."""
    if measure is None:
        return assumption(
            "factorability",
            "The Kaiser-Meyer-Olkin measure could not be computed",
            Verdict.NOT_APPLICABLE,
            "The correlation matrix is singular or its columns share no variance, so the "
            "partial correlations are undefined.",
        )
    if measure >= KMO_PASS:
        verdict = Verdict.PASS
        title = "The columns share enough common variance for components to summarise them"
        next_step = None
    elif measure >= KMO_CAUTION:
        verdict = Verdict.CAUTION
        title = "The columns share only moderate common variance"
        next_step = "The components may fit the columns only partly; consider other columns."
    else:
        verdict = Verdict.FAIL
        title = "The columns share little common variance"
        next_step = "The components may not summarise these columns; consider another method."
    return assumption(
        "factorability",
        title,
        verdict,
        f"The Kaiser-Meyer-Olkin measure is {num(measure)}.",
        statistic=f"KMO = {num(measure)}",
        threshold=(
            f"KMO >= {num(KMO_PASS)} pass; {num(KMO_CAUTION)} <= KMO < {num(KMO_PASS)} "
            f"caution; KMO < {num(KMO_CAUTION)} fail"
        ),
        next_step=next_step,
    )


def _components_table(
    eigenvalues: FloatArray,
    percentages: FloatArray,
    cumulative: FloatArray,
    components: FloatArray,
    names: Sequence[str],
    shown: int,
) -> Table:
    """The components, their eigenvalues and variance, and one loading per column."""
    columns: tuple[tuple[str, str, ColumnKind], ...] = _COMPONENT_COLUMNS + tuple(
        (f"loading_{index + 1}", name, "number") for index, name in enumerate(names)
    )
    rows: list[tuple[Cell, ...]] = []
    for index in range(shown):
        loadings = (float(value) for value in components[:, index])
        rows.append(
            (
                index + 1,
                float(eigenvalues[index]),
                float(percentages[index]),
                float(cumulative[index]),
                *loadings,
            )
        )
    return table(
        "Components",
        columns,
        rows,
        note=(
            "The eigenvalue is the variance of the component, in the units of the matrix "
            "decomposed; the variance explained is that eigenvalue as a percentage of the "
            "total, and the cumulative percentage adds the components in the order shown. A "
            "loading is a column's coefficient in the component, and its square is the share "
            "of that column's variance the component takes."
        ),
    )


def _scores_table(scores: FloatArray, shown: int) -> Table:
    """One row per shown component: the mean, SD, minimum and maximum of its scores."""
    rows: list[tuple[Cell, ...]] = []
    for index in range(shown):
        values = scores[:, index]
        rows.append(
            (
                index + 1,
                float(np.mean(values)),
                float(np.std(values, ddof=1)),
                float(np.min(values)),
                float(np.max(values)),
            )
        )
    return table(
        "Scores summary",
        _SCORE_COLUMNS,
        rows,
        note=(
            "A score is a row's value on the component: the standardised (or centred) columns "
            "projected on the component's loadings. The SD is the sample standard deviation, "
            "with n - 1 in its denominator."
        ),
    )


def _scree_plot(eigenvalues: FloatArray, shown: int) -> MultiLinePlot:
    """The eigenvalue of each shown component, against the component number."""
    values, note = stride(eigenvalues[:shown])
    numbers = np.linspace(1.0, float(shown), values.size)
    labels = tuple(str(round(float(number))) for number in numbers)
    return MultiLinePlot(
        title="Scree plot",
        x_label="Component",
        y_label="Eigenvalue",
        series=(
            XYSeries(
                label="Eigenvalue",
                x=tuple(float(number) for number in numbers),
                y=tuple(float(value) for value in values),
            ),
        ),
        x_labels=labels,
        note=note,
    )


def _scores_plot(scores: FloatArray) -> ScatterPlot:
    """The scores on the first two components, as points in the plane they span."""
    first, note_x = stride(scores[:, 0])
    second, note_y = stride(scores[:, 1])
    details = ["The loadings are in the Components table."]
    note = note_x or note_y
    if note is not None:
        details.append(note)
    return ScatterPlot(
        title="Scores on the first two components",
        x_label="Component 1 score",
        y_label="Component 2 score",
        x=tuple(float(value) for value in first),
        y=tuple(float(value) for value in second),
        note=" ".join(details),
    )


def _components_for_target(cumulative: FloatArray) -> int:
    """How many components reach :data:`VARIANCE_TARGET` per cent, at least one and at most p."""
    count = int(np.searchsorted(cumulative, VARIANCE_TARGET, side="left")) + 1
    return min(count, cumulative.size)


def _interpretation(components: FloatArray, names: Sequence[str], percentages: FloatArray) -> str:
    """Two or three sentences: the variance of the first two components, and what they load."""
    first = int(np.argmax(np.abs(components[:, 0])))
    second = int(np.argmax(np.abs(components[:, 1])))
    return (
        f"The first component explains {num(float(percentages[0]))} % of the total variance, "
        f"and the column with the largest loading on it is {names[first]!r}. The second "
        f"component explains {num(float(percentages[1]))} % of the total variance, and the "
        f"column with the largest loading on it is {names[second]!r}. A component is a "
        "direction in the data, not a cause."
    )


def _methods(standardise: bool) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    matrix = (
        "correlation matrix of the standardised columns"
        if standardise
        else "covariance matrix of the centred columns"
    )
    projected = "standardised" if standardise else "centred"
    return (
        f"Principal component analysis decomposes the {matrix} by the eigenvalue "
        "decomposition; each component is an orthogonal linear combination of the columns, "
        "ordered by the variance it explains. The sign of every eigenvector is fixed so that "
        "its largest absolute loading is positive, and the scores are the "
        f"{projected} columns projected on the components."
    )


def _warnings(dropped: Sequence[str], shown: int, available: int, n_used: int) -> tuple[str, ...]:
    """A warning for each column left out, a shortened component count, and few rows."""
    found = [
        f"The column {name!r} holds one value repeated and was left out of the decomposition."
        for name in dropped
    ]
    if shown < available:
        found.append(
            f"Only the first {num(float(shown))} of {num(float(available))} components are "
            f"shown; the variance percentages use all {num(float(available))}."
        )
    if n_used < WARN_ROWS:
        found.append(f"Only {num(float(n_used))} rows are used, so the components may be unstable.")
    return tuple(found)


def run_pca(request: BaseModel) -> ToolResult:
    """Run the principal component analysis tool on its own request."""
    if not isinstance(request, PcaRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"pca takes a PcaRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: PcaRequest) -> ToolResult:
    """The components themselves, once the request is known to be this tool's own."""
    check_level(request.level)
    offered = len(request.columns)
    if not MIN_COLUMNS <= offered <= MAX_COLUMNS:
        raise ToolError(
            "INVALID_OPTION",
            f"principal component analysis takes between {MIN_COLUMNS} and {MAX_COLUMNS} "
            f"columns, not {offered}",
        )
    if request.max_components is not None and not 1 <= request.max_components <= offered:
        raise ToolError(
            "INVALID_OPTION",
            "the most components to show must lie between 1 and the number of columns "
            f"({offered}), not {request.max_components}",
        )
    pairing = pairs(*request.columns)
    n_used = pairing.columns[0].values.size
    if n_used < MIN_ROWS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"principal component analysis needs at least {MIN_ROWS} complete rows, and "
            f"{n_used} are left",
        )
    dropped: list[str] = []
    kept: list[Column] = []
    for column in pairing.columns:
        values = column.values
        if float(np.min(values)) == float(np.max(values)):
            if request.standardise:
                raise ToolError(
                    "CONSTANT_COLUMN",
                    f"the column {column.name!r} holds one value repeated in the rows used, "
                    "so its variance and loading are undefined",
                )
            dropped.append(column.name)
        else:
            kept.append(column)
    if len(kept) < MIN_COLUMNS:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the columns {listing(dropped, 'column')} hold one value repeated, and a "
            f"principal component analysis needs at least {MIN_COLUMNS} columns that vary",
        )
    names = [column.name for column in kept]
    data = np.column_stack([column.values for column in kept])
    matrix, projected = _matrix(data, request.standardise)
    eigenvalues, components = _decompose(matrix)
    total = float(np.sum(eigenvalues))
    percentages = eigenvalues / total * 100.0
    cumulative = np.cumsum(percentages)
    scores = projected @ components
    available = len(kept)
    shown = available if request.max_components is None else min(request.max_components, available)
    correlation = matrix if request.standardise else _correlation(data)
    sphericity = _bartlett(correlation, n_used)
    factorability = _kaiser_meyer_olkin(correlation)
    tables: list[Table] = [
        _components_table(eigenvalues, percentages, cumulative, components, names, shown),
        _scores_table(scores, shown),
    ]
    plots: list[MultiLinePlot | ScatterPlot] = [_scree_plot(eigenvalues, shown)]
    if shown >= 2:
        plots.append(_scores_plot(scores))
    listed, unlisted = list_excluded(pairing.excluded)
    n_offered = max(column.rows.size for column in request.columns)
    return ToolResult(
        tool="pca",
        title="Principal component analysis",
        level=request.level,
        estimates=(
            Estimate(
                label="Variance explained by the first component",
                value=float(percentages[0]),
            ),
            Estimate(
                label="Variance explained by the first two components",
                value=float(percentages[0] + percentages[1]),
            ),
            Estimate(
                label="Number of components for 80 % of variance",
                value=float(_components_for_target(cumulative)),
            ),
        ),
        assumptions=(_sphericity_check(sphericity), _factorability_check(factorability)),
        warnings=_warnings(dropped, shown, available, n_used),
        interpretation=_interpretation(components, names, percentages),
        methods_fragment=_methods(request.standardise),
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
        name="pca",
        family="relationships",
        title="Principal component analysis",
        request=PcaRequest,
        run=run_pca,
    )
)
