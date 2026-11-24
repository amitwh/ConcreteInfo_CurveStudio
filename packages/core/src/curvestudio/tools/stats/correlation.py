"""Correlation between every pair of a chosen set of numeric columns.

The tool reads two to twenty columns, takes each pair on the rows complete in both of its
columns (a row missing in one column is left out of both sides of that pair), and reports the
coefficient with an interval and a test for every pair: Pearson's product-moment coefficient
with the t test and Fisher's z interval, Spearman's rank coefficient with the t approximation
and Fisher's z applied to the coefficient as an approximation, or Kendall's tau-b with the
normal approximation and its tie-corrected standard error.

A p-value is a statement of evidence, never a statement of proof. A coefficient measures a
linear association for Pearson and a monotone one for the rank methods, so a strong curved
relationship can read near 0 and a scatter plot is the first thing to read. A pair with fewer
than three complete rows, or with a column that does not vary, has no coefficient: its test
carries no statistic, says why, and warns, and its cells are left empty.

The result also carries the whole correlation matrix (with 1 on its diagonal) and the table of
pairs; the pairs are drawn as scatter plots when there are at most four of them.
"""

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

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
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

Method = Literal["pearson", "spearman", "kendall"]

MIN_COLUMNS = 2
MAX_COLUMNS = 20
MIN_PAIRS = 3
"""The fewest complete rows a coefficient is computed from."""

METHODS: dict[str, str] = {
    "pearson": "Pearson correlation, t test, two-sided",
    "spearman": "Spearman rank correlation, t test on the ranks",
    "kendall": "Kendall's tau-b, normal approximation",
}
"""Each coefficient's test, as the methods text names it."""

COEFFICIENT_LABELS: dict[str, str] = {
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "kendall": "Kendall's tau-b",
}
"""Each coefficient's estimate label."""

NULL_HYPOTHESES: dict[str, str] = {
    "pearson": "the two columns are not associated, so their Pearson correlation is 0",
    "spearman": "the two columns are not associated, so their Spearman correlation is 0",
    "kendall": "the two columns are not associated, so their Kendall tau-b is 0",
}
"""Each test's null hypothesis, in words."""

_UNIT_MARGIN = 1e-12
"""How far a coefficient is held from -1 and 1 before Fisher's z transformation."""


class CorrelationRequest(BaseModel):
    """The columns to correlate, the coefficient to compute and the confidence level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: tuple[Column, ...] = Field(
        description=("The two to twenty numeric columns whose pairwise correlations are computed."),
    )
    method: Method = Field(
        default="pearson",
        description=(
            "The coefficient: pearson for a linear association, spearman for a rank one with "
            "the t approximation, or kendall for tau-b with the normal approximation."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Measured(NamedTuple):
    """One pair's coefficient, with its interval and the p-value of its test."""

    coefficient: float
    lower: float | None
    upper: float | None
    se: float | None
    p_value: float


class _Pair(NamedTuple):
    """One pair as the result shows it: its rows, its coefficient and why it has none."""

    left_index: int
    right_index: int
    left: str
    right: str
    n: int
    x: FloatArray
    y: FloatArray
    coefficient: float | None
    lower: float | None
    upper: float | None
    se: float | None
    p_value: float | None
    reason: str | None


def _fisher_interval(r: float, n: int, level: float) -> tuple[float | None, float | None]:
    """The Fisher z interval for a correlation coefficient, or none from three rows.

    The coefficient is held just inside -1 and 1, so a perfect linear relationship still
    gives a finite interval. The transformation needs ``n - 3`` in its denominator, so a pair
    of three complete rows has no interval.
    """
    if n <= 3:
        return None, None
    clipped = float(np.clip(r, -1.0 + _UNIT_MARGIN, 1.0 - _UNIT_MARGIN))
    z = math.atanh(clipped)
    half = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0)) / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


def _tie_counts(values: FloatArray) -> tuple[int, int, int]:
    """The tie counts of one column: tied pairs, tied triples and the weighted sum.

    These are the ``x0``, ``x1`` and ``xtie`` counts of Kendall's normal approximation, read
    from the sizes of the groups of equal values.
    """
    _values, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    tied = counts[counts > 1].astype(np.int64)
    tied_pairs = int((tied * (tied - 1) // 2).sum())
    tied_triples = int((tied * (tied - 1) * (tied - 2)).sum())
    weighted = int((tied * (tied - 1) * (2 * tied + 5)).sum())
    return tied_pairs, tied_triples, weighted


def _tau_b_se(x: FloatArray, y: FloatArray) -> float:
    """The standard error of Kendall's tau-b under the normal approximation, with ties.

    This is the tie-corrected variance Kendall (1970) gives, as SciPy applies it, divided by
    the tau-b normalising constant.
    """
    n = int(x.size)
    total = n * (n - 1) // 2
    x_tied, x_triples, x_weighted = _tie_counts(x)
    y_tied, y_triples, y_weighted = _tie_counts(y)
    m = n * (n - 1.0)
    variance = (
        (m * (2 * n + 5) - x_weighted - y_weighted) / 18
        + (2 * x_tied * y_tied) / m
        + x_triples * y_triples / (9 * m * (n - 2))
    )
    return math.sqrt(variance / ((total - x_tied) * (total - y_tied)))


def _coefficient(x: FloatArray, y: FloatArray, method: Method, level: float) -> _Measured:
    """One pair's coefficient with its interval and the p-value of its test.

    Pearson and Spearman use Fisher's z for the interval; Kendall's tau-b uses the normal
    approximation with its tie-corrected standard error, held within -1 and 1.
    """
    check_level(level)
    if method == "pearson":
        found = scipy_stats.pearsonr(x, y)
        r = float(found.statistic)
        lower, upper = _fisher_interval(r, int(x.size), level)
        return _Measured(
            coefficient=r,
            lower=lower,
            upper=upper,
            se=None,
            p_value=float(found.pvalue),
        )
    if method == "spearman":
        found = scipy_stats.spearmanr(x, y)
        rho = float(found.statistic)
        lower, upper = _fisher_interval(rho, int(x.size), level)
        return _Measured(
            coefficient=rho,
            lower=lower,
            upper=upper,
            se=None,
            p_value=float(found.pvalue),
        )
    found = scipy_stats.kendalltau(x, y, variant="b", method="asymptotic")
    tau = float(found.statistic)
    se = _tau_b_se(x, y)
    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return _Measured(
        coefficient=tau,
        lower=max(-1.0, tau - quantile * se),
        upper=min(1.0, tau + quantile * se),
        se=se,
        p_value=float(found.pvalue),
    )


def _constant(name: str, values: FloatArray) -> str | None:
    """``name`` when ``values`` does not vary, and ``None`` when it does."""
    return name if float(np.min(values)) == float(np.max(values)) else None


def _without_coefficient(
    left_index: int,
    right_index: int,
    left: Column,
    right: Column,
    n: int,
    x: FloatArray,
    y: FloatArray,
    reason: str,
) -> _Pair:
    """A pair whose coefficient cannot be computed, with the reason and empty cells."""
    return _Pair(
        left_index=left_index,
        right_index=right_index,
        left=left.name,
        right=right.name,
        n=n,
        x=x,
        y=y,
        coefficient=None,
        lower=None,
        upper=None,
        se=None,
        p_value=None,
        reason=reason,
    )


def _pair(
    left_index: int, right_index: int, left: Column, right: Column, method: Method, level: float
) -> _Pair:
    """One pair, aligned on its complete rows, with its coefficient or the reason it has none."""
    pairing = pairs(left, right)
    x = pairing.columns[0].values
    y = pairing.columns[1].values
    n = int(x.size)
    if n < MIN_PAIRS:
        reason = (
            f"the pair {left.name!r} with {right.name!r} has {n} complete "
            f"{'row' if n == 1 else 'rows'}, and a coefficient needs at least {MIN_PAIRS}"
        )
        return _without_coefficient(left_index, right_index, left, right, n, x, y, reason)
    constant = _constant(left.name, x)
    if constant is None:
        constant = _constant(right.name, y)
    if constant is not None:
        reason = (
            f"the column {constant!r} holds one value repeated, so a correlation with it is "
            "undefined"
        )
        return _without_coefficient(left_index, right_index, left, right, n, x, y, reason)
    measured = _coefficient(x, y, method, level)
    return _Pair(
        left_index=left_index,
        right_index=right_index,
        left=left.name,
        right=right.name,
        n=n,
        x=x,
        y=y,
        coefficient=measured.coefficient,
        lower=measured.lower,
        upper=measured.upper,
        se=measured.se,
        p_value=measured.p_value,
        reason=None,
    )


def _test(pair: _Pair, method: Method) -> HypothesisTest:
    """One pair's test, with the coefficient and its interval beside the p-value."""
    effect = None
    if pair.coefficient is not None:
        effect = Estimate(
            label=COEFFICIENT_LABELS[method],
            value=pair.coefficient,
            se=pair.se,
            lower=pair.lower,
            upper=pair.upper,
        )
    return HypothesisTest(
        name=f"{pair.left} with {pair.right}",
        method=METHODS[method],
        null_hypothesis=NULL_HYPOTHESES[method],
        statistic=pair.coefficient,
        df=None if method == "kendall" or pair.coefficient is None else float(pair.n - 2),
        p_value=pair.p_value,
        p_value_text=p_text(pair.p_value),
        effect=effect,
        reason=pair.reason,
    )


def _matrix(columns: Sequence[Column], measured: Sequence[_Pair]) -> Table:
    """The correlation matrix: a text column of names, then one number column per column."""
    keys: list[tuple[str, str, ColumnKind]] = [("column", "Column", "text")]
    keys.extend((f"c{index}", column.name, "number") for index, column in enumerate(columns))
    size = len(columns)
    grid: list[list[Cell]] = [
        [1.0 if row == column else None for column in range(size)] for row in range(size)
    ]
    for pair in measured:
        grid[pair.left_index][pair.right_index] = pair.coefficient
        grid[pair.right_index][pair.left_index] = pair.coefficient
    rows = [(column.name, *grid[index]) for index, column in enumerate(columns)]
    return table("Correlation matrix", keys, rows)


def _pairs_table(measured: Sequence[_Pair]) -> Table:
    """One row per pair: its two columns, its complete rows, its coefficient and p-value."""
    columns: tuple[tuple[str, str, ColumnKind], ...] = (
        ("x", "X", "text"),
        ("y", "Y", "text"),
        ("n", "n", "int"),
        ("coefficient", "Coefficient", "number"),
        ("lower", "Lower", "number"),
        ("upper", "Upper", "number"),
        ("p", "p", "p_value"),
    )
    return table(
        "Pairs",
        columns,
        (
            (pair.left, pair.right, pair.n, pair.coefficient, pair.lower, pair.upper, pair.p_value)
            for pair in measured
        ),
    )


def _scatter(pair: _Pair) -> ScatterPlot:
    """One pair's points, cut by :func:`stride` with the same indices on both axes."""
    shown_x, note = stride(np.asarray(pair.x, dtype=np.float64))
    shown_y, _ = stride(np.asarray(pair.y, dtype=np.float64))
    return ScatterPlot(
        title=f"Scatter plot of {pair.left} against {pair.right}",
        x_label=pair.left,
        y_label=pair.right,
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        note=note,
    )


def _plots(measured: Sequence[_Pair]) -> tuple[ScatterPlot, ...]:
    """A scatter plot for each pair when there are at most four of them, none otherwise."""
    return () if len(measured) > 4 else tuple(_scatter(pair) for pair in measured)


def _assumptions(method: Method) -> tuple[AuditCheck, ...]:
    """What a coefficient cannot check: independence, and the shape it measures."""
    shape = "linear" if method == "pearson" else "monotone"
    return (
        assumption(
            "independence",
            "Independence cannot be read from two columns",
            Verdict.NOT_APPLICABLE,
            "A pair of columns records no order in time and no repeated measure, which is what "
            "independence would have to be read from.",
            next_step=(
                "Check how the rows were collected: a time order or repeated measures make them "
                "dependent, and the p-values then do not carry their usual meaning."
            ),
        ),
        assumption(
            "association_shape",
            f"A {shape} association is what the coefficient measures",
            Verdict.NOT_APPLICABLE,
            f"A correlation coefficient measures a {shape} association only; a strong curved "
            "relationship can give a coefficient near 0.",
            next_step="Look at a scatter plot of each pair before reading its coefficient.",
        ),
    )


def _evidence(p_value: float) -> str:
    """What a p-value says about the null hypothesis, as a clause of evidence."""
    if p_value >= 0.05:
        return "the data are consistent with no association"
    if p_value >= 0.001:
        return "the data show evidence of an association"
    return "the data show strong evidence of an association"


def _defined_coefficient(pair: _Pair) -> float:
    """The coefficient of a pair known to have one, for the words that read it."""
    assert pair.coefficient is not None
    return float(pair.coefficient)


def _interpretation(measured: Sequence[_Pair], method: Method, level: float) -> str:
    """One to three sentences: the strongest pair, the rest, and the pairs with no coefficient."""
    defined = [pair for pair in measured if pair.coefficient is not None]
    if not defined:
        count = len(measured)
        return (
            f"None of the {count} {'pair has' if count == 1 else 'pairs have'} the "
            f"{MIN_PAIRS} complete rows a coefficient needs, so no association is reported."
        )
    strongest = max(defined, key=lambda pair: abs(_defined_coefficient(pair)))
    coefficient = _defined_coefficient(strongest)
    assert strongest.p_value is not None
    if strongest.lower is None or strongest.upper is None:
        interval = "with no interval from so few complete rows"
    else:
        interval = (
            f"with a {num(level * 100)} % interval from {num(strongest.lower)} to "
            f"{num(strongest.upper)}"
        )
    sentences = [
        f"The {COEFFICIENT_LABELS[method]} of {strongest.left} with {strongest.right} is "
        f"{num(coefficient)}, {interval}, and the test gives {p_text(strongest.p_value)}; "
        f"{_evidence(strongest.p_value)}."
    ]
    others = [pair for pair in defined if pair is not strongest]
    if others:
        values = [_defined_coefficient(pair) for pair in others]
        sentences.append(
            f"The other {len(others)} {'pair gives' if len(others) == 1 else 'pairs give'} "
            f"coefficients from {num(min(values))} to {num(max(values))}."
        )
    missing = len(measured) - len(defined)
    if missing:
        subject = (
            "pair could not be computed and is"
            if missing == 1
            else ("pairs could not be computed and are")
        )
        sentences.append(f"{missing} {subject} listed in the warnings.")
    return " ".join(sentences)


def _methods(method: Method) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    common = (
        "Each pair is computed on the rows complete in both of its columns; the counts report "
        "the rows complete in every selected column."
    )
    texts = {
        "pearson": (
            "Pearson's product-moment correlation is tested with the t statistic on n - 2 "
            "degrees of freedom, and its interval is the Fisher z transformation of the "
            "coefficient, which needs at least four complete pairs."
        ),
        "spearman": (
            "Spearman's rank correlation is tested with the t approximation on n - 2 degrees of "
            "freedom, and its interval applies the Fisher z transformation to the coefficient, "
            "an approximation that needs at least four complete pairs."
        ),
        "kendall": (
            "Kendall's tau-b is tested with the normal approximation, using the tie-corrected "
            "standard error of the coefficient, and its interval is the coefficient plus or "
            "minus the normal quantile times that standard error, held within -1 and 1."
        ),
    }
    return f"{common} {texts[method]}"


def run_correlation(request: BaseModel) -> ToolResult:
    """Run the correlation tool on its own request."""
    if not isinstance(request, CorrelationRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"correlation takes a CorrelationRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: CorrelationRequest) -> ToolResult:
    """The analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    columns = request.columns
    size = len(columns)
    if not MIN_COLUMNS <= size <= MAX_COLUMNS:
        raise ToolError(
            "INVALID_OPTION",
            f"correlation takes between {MIN_COLUMNS} and {MAX_COLUMNS} columns, not {size}",
        )
    measured = [
        _pair(index, other, columns[index], columns[other], request.method, request.level)
        for index in range(size)
        for other in range(index + 1, size)
    ]
    complete = pairs(*columns)
    used = int(complete.columns[0].values.size)
    excluded = len(complete.excluded)
    listed, unlisted = list_excluded(complete.excluded)
    names = [column.name for column in columns]
    title = f"Correlation of {joined(names)}" if size <= 4 else f"Correlation of {size} columns"
    warnings = tuple(
        f"The pair {pair.left} with {pair.right} has no coefficient: {pair.reason}."
        for pair in measured
        if pair.reason is not None
    )
    return ToolResult(
        tool="correlation",
        title=title,
        level=request.level,
        tests=tuple(_test(pair, request.method) for pair in measured),
        assumptions=_assumptions(request.method),
        warnings=warnings,
        interpretation=_interpretation(measured, request.method, request.level),
        methods_fragment=_methods(request.method),
        counts=Counts(n_offered=used + excluded, n_used=used, n_excluded=excluded),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(_matrix(columns, measured), _pairs_table(measured)),
        plots=_plots(measured),
    )


TOOLS.register(
    ToolSpec(
        name="correlation",
        family="relationships",
        title="Correlation",
        request=CorrelationRequest,
        run=run_correlation,
    )
)
