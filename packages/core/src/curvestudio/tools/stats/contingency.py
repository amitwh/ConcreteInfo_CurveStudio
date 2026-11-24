"""Contingency tables and McNemar's test for two paired binary columns.

The contingency tool pairs a row-label column with a column-label column by row position and
tabulates the two classifications. It reports the Pearson chi-square test of independence
(with Yates' continuity correction for a 2 x 2 table, as SciPy applies it by default), Fisher's
exact test in its place for a 2 x 2 table with an expected count below 5, Cramér's V as the
effect size, and for a 2 x 2 table the odds ratio and the risk ratio with their intervals.
The four tables show the counts, the row percentages, the column percentages and each cell's
share of the total.

McNemar's test reads two label columns of exactly two levels each, paired by row position, and
tests whether the two marginal proportions are equal. It uses the exact binomial distribution
when the discordant pairs number at most 25, and the chi-square distribution with a continuity
correction otherwise. The effect is the difference in the two marginal proportions, with
Newcombe's method 10 interval for paired proportions: Wilson score limits for each marginal
combined through the phi coefficient of the paired indicators (Newcombe R.G., "Improved
confidence intervals for the difference between binomial proportions based on paired data",
Statistics in Medicine 17:2635-2650, 1998; the closed-form combination is the MOVER interval
of Zou G.Y. and Donner A., Statistics in Medicine 27:1693-1702, 2008).

A p-value is a statement of evidence, never one of proof. A row without a label in either
column is left out and listed; a level the paired rows leave empty is dropped with a warning.
"""

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import GroupColumn
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num
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
)
from curvestudio.tools.stats._display import Cell, ColumnKind, Table
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MAX_LEVELS = 20
"""The most levels a contingency table takes on either side."""

MCNEMAR_EXACT_MAX = 25
"""The most discordant pairs the exact binomial McNemar test is used for."""

MIN_EXPECTED = 5.0
"""The expected count below which the chi-square approximation is in doubt."""


class ContingencyRequest(BaseModel):
    """Two label columns paired by row position, and the confidence level of the intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: GroupColumn = Field(
        description=(
            "The row labels, paired by row position with the column labels; "
            "a row without a label in either column is left out and listed."
        ),
    )
    columns: GroupColumn = Field(
        description=(
            "The column labels, paired by row position with the row labels; "
            "a row without a label in either column is left out and listed."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for the odds and risk ratio intervals, between 0 and 1.",
    )


class McNemarRequest(BaseModel):
    """Two label columns of exactly two levels each, paired by row position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    before: GroupColumn = Field(
        description="The label before the change, paired by row position with the after label.",
    )
    after: GroupColumn = Field(
        description="The label after the change, paired by row position with the before label.",
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of the difference interval, between 0 and 1.",
    )


class _Paired(NamedTuple):
    """The labels two group columns share by position, and the positions they do not."""

    first: NDArray[np.object_]
    second: NDArray[np.object_]
    excluded: tuple[ExcludedRow, ...]


def _align_labels(first: GroupColumn, second: GroupColumn) -> _Paired:
    """The labels the two columns both hold at a position, and the positions without both."""
    positions = np.unique(np.concatenate([first.rows, second.rows]))
    first_has, first_labels = _spread(positions, first)
    second_has, second_labels = _spread(positions, second)
    kept = first_has & second_has
    excluded = tuple(
        ExcludedRow(
            index=int(positions[at]),
            reason=(
                f"no group label in {first.name}"
                if not first_has[at]
                else f"no group label in {second.name}"
            ),
        )
        for at in np.flatnonzero(~kept)
    )
    return _Paired(
        first=first_labels[kept],
        second=second_labels[kept],
        excluded=excluded,
    )


def _spread(
    positions: NDArray[np.intp], group: GroupColumn
) -> tuple[NDArray[np.bool_], NDArray[np.object_]]:
    """``group``'s labels laid out over ``positions``: which ones it covers, and each label."""
    covered = np.zeros(positions.size, dtype=bool)
    labels: NDArray[np.object_] = np.full(positions.size, "", dtype=object)
    _common, here, there = np.intersect1d(positions, group.rows, return_indices=True)
    covered[here] = True
    labels[here] = group.labels[there]
    return covered, labels


def _check_levels_unique(group: GroupColumn, found: tuple[str, ...]) -> None:
    """Refuse a column with more levels than a contingency table is built for."""
    if len(found) > MAX_LEVELS:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"the column {group.name!r} holds {len(found)} labels, and a contingency table "
            f"takes at most {MAX_LEVELS} on either side",
        )


def _grid(
    row_levels: tuple[str, ...] | list[str],
    column_levels: tuple[str, ...] | list[str],
    row_labels: NDArray[np.object_],
    column_labels: NDArray[np.object_],
) -> NDArray[np.int64]:
    """The counts of every pair of labels, rows in ``row_levels`` and columns in
    ``column_levels``."""
    rows = {label: index for index, label in enumerate(row_levels)}
    columns = {label: index for index, label in enumerate(column_levels)}
    grid: NDArray[np.int64] = np.zeros((len(rows), len(columns)), dtype=np.int64)
    for row, column in zip(row_labels, column_labels, strict=True):
        grid[rows[str(row)], columns[str(column)]] += 1
    return grid


def _count_table(
    title: str,
    row_levels: tuple[str, ...] | list[str],
    column_levels: tuple[str, ...] | list[str],
    grid: NDArray[np.int64],
) -> Table:
    """The counts with a Total row and a Total column."""
    columns: list[tuple[str, str, ColumnKind]] = [("row", "Row", "text")]
    columns.extend((f"c{index}", label, "int") for index, label in enumerate(column_levels))
    columns.append(("total", "Total", "int"))
    rows: list[tuple[Cell, ...]] = [
        (label, *(int(value) for value in grid[index]), int(grid[index].sum()))
        for index, label in enumerate(row_levels)
    ]
    rows.append(("Total", *(int(value) for value in grid.sum(axis=0)), int(grid.sum())))
    return table(title, columns, rows)


def _percent_table(
    title: str,
    row_levels: tuple[str, ...] | list[str],
    column_levels: tuple[str, ...] | list[str],
    grid: NDArray[np.int64],
    *,
    mode: str,
) -> Table:
    """The percentages of a count table, as numbers of per cent rather than fractions.

    ``mode`` is ``row`` (each row sums to 100), ``column`` (each column sums to 100) or
    ``all`` (each cell is its share of the total). A Total row or column holds the matching
    margin, so every row of the table reads consistently.
    """
    grand = float(grid.sum())
    row_totals = grid.sum(axis=1).astype(np.float64)
    column_totals = grid.sum(axis=0).astype(np.float64)
    columns: list[tuple[str, str, ColumnKind]] = [("row", "Row", "text")]
    columns.extend((f"c{index}", label, "number") for index, label in enumerate(column_levels))
    columns.append(("total", "Total", "number"))
    rows: list[tuple[Cell, ...]] = []
    for index, label in enumerate(row_levels):
        if mode == "row":
            cells = [
                100.0 * grid[index, other] / row_totals[index] for other in range(grid.shape[1])
            ]
            total = 100.0
        elif mode == "column":
            cells = [
                100.0 * grid[index, other] / column_totals[other] for other in range(grid.shape[1])
            ]
            total = 100.0 * row_totals[index] / grand
        else:
            cells = [100.0 * grid[index, other] / grand for other in range(grid.shape[1])]
            total = 100.0 * row_totals[index] / grand
        rows.append((label, *cells, total))
    if mode == "row":
        total_cells = [100.0 * value / grand for value in column_totals]
    else:
        total_cells = [100.0 for _ in column_levels]
    rows.append(("Total", *total_cells, 100.0))
    return table(title, columns, rows)


def _expected(grid: NDArray[np.int64]) -> NDArray[np.float64]:
    """The expected count of each cell under independence: its row total times its column
    total, divided by the grand total."""
    total = float(grid.sum())
    return np.outer(grid.sum(axis=1), grid.sum(axis=0)) / total


def _cramers_v(grid: NDArray[np.int64]) -> float:
    """Cramér's V from the uncorrected Pearson chi-square.

    Raises:
        ValueError: the table holds a row or column with no count, so the statistic has no
            answer (the contingency tool drops such levels before it gets here).
    """
    found = scipy_stats.chi2_contingency(grid, correction=False)
    rows, columns = grid.shape
    return float(np.sqrt(float(found.statistic) / (float(grid.sum()) * min(rows - 1, columns - 1))))


def _sample_odds_ratio(grid: NDArray[np.int64]) -> float | None:
    """The sample odds ratio of a 2 x 2 table, or ``None`` when a cell is 0."""
    first, second, third, fourth = (int(value) for value in grid.reshape(-1))
    if min(first, second, third, fourth) == 0:
        return None
    return float(first * fourth / (second * third))


def _reference_label(row_level: str, column_level: str) -> str:
    return f"(reference cell: {row_level} and {column_level})"


def _two_by_two_estimates(
    grid: NDArray[np.int64],
    row_levels: tuple[str, ...] | list[str],
    column_levels: tuple[str, ...] | list[str],
    level: float,
) -> tuple[tuple[Estimate, ...], tuple[str, ...]]:
    """The odds ratio and the risk ratio of a 2 x 2 table, each with its interval.

    The reference cell is the first row level with the first column level. The odds ratio's
    interval is Wald's on the log scale; the risk ratio's is Katz's log interval. An estimate
    the table cannot give is left out, and a warning says why.
    """
    first, second, third, fourth = (int(value) for value in grid.reshape(-1))
    row_reference = row_levels[0]
    column_reference = column_levels[0]
    reference = _reference_label(row_reference, column_reference)
    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    estimates: list[Estimate] = []
    warnings: list[str] = []

    cells = (
        (first, row_levels[0], column_levels[0]),
        (second, row_levels[0], column_levels[1]),
        (third, row_levels[1], column_levels[0]),
        (fourth, row_levels[1], column_levels[1]),
    )
    zeros = [(row, column) for count, row, column in cells if count == 0]
    if zeros:
        row, column = zeros[0]
        warnings.append(f"The odds ratio is undefined because the cell {row} and {column} holds 0.")
    else:
        odds = first * fourth / (second * third)
        error = math.sqrt(1.0 / first + 1.0 / second + 1.0 / third + 1.0 / fourth)
        lower = math.exp(math.log(odds) - quantile * error)
        upper = math.exp(math.log(odds) + quantile * error)
        estimates.append(
            Estimate(
                label=f"Odds ratio {reference}",
                value=float(odds),
                lower=lower,
                upper=upper,
            )
        )

    if third == 0:
        warnings.append(
            f"The risk ratio is undefined because the cell {row_levels[1]} and "
            f"{column_levels[0]} holds 0."
        )
    else:
        risk = (first / (first + second)) / (third / (third + fourth))
        if first == 0:
            estimates.append(Estimate(label=f"Risk ratio {reference}", value=0.0))
            warnings.append(
                f"The Katz interval for the risk ratio cannot be built because the cell "
                f"{row_levels[0]} and {column_levels[0]} holds 0."
            )
        else:
            error = math.sqrt(
                1.0 / first - 1.0 / (first + second) + 1.0 / third - 1.0 / (third + fourth)
            )
            lower = math.exp(math.log(risk) - quantile * error)
            upper = math.exp(math.log(risk) + quantile * error)
            estimates.append(
                Estimate(
                    label=f"Risk ratio {reference}",
                    value=float(risk),
                    lower=lower,
                    upper=upper,
                )
            )
    return tuple(estimates), tuple(warnings)


def _independence_test(grid: NDArray[np.int64]) -> HypothesisTest:
    """The chi-square test of independence, or Fisher's exact test for a 2 x 2 table with an
    expected count below 5, with Cramér's V as its effect."""
    two_by_two = grid.shape == (2, 2)
    smallest = float(_expected(grid).min())
    null = "the row classification and the column classification are independent"
    effect = Estimate(label="Cramér's V", value=_cramers_v(grid))
    if two_by_two and smallest < MIN_EXPECTED:
        found = scipy_stats.fisher_exact(grid)
        p_value = float(found.pvalue)
        return HypothesisTest(
            name="Fisher's exact test",
            method="Fisher's exact test, two-sided",
            null_hypothesis=null,
            statistic=_sample_odds_ratio(grid),
            p_value=p_value,
            p_value_text=p_text(p_value),
            effect=effect,
        )
    found = scipy_stats.chi2_contingency(grid, correction=two_by_two)
    p_value = float(found.pvalue)
    method = "Pearson chi-square test of independence"
    if two_by_two:
        method += " with Yates' continuity correction"
    return HypothesisTest(
        name="Pearson chi-square",
        method=method,
        null_hypothesis=null,
        statistic=float(found.statistic),
        df=float(found.dof),
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=effect,
    )


def _independence_assumptions(smallest: float, *, fisher: bool) -> tuple[AuditCheck, ...]:
    """What the test cannot check, and how the expected counts fare."""
    checks = [
        assumption(
            "independence",
            "Independence between the rows cannot be read from the table",
            Verdict.NOT_APPLICABLE,
            "Each row is one observation; whether the observations are independent is a fact "
            "about how the data were collected, which the table cannot record.",
            next_step=(
                "Check how the rows were collected; dependence between them changes what the "
                "p-value means."
            ),
        )
    ]
    if fisher:
        checks.append(
            assumption(
                "expected_counts",
                "The exact test does not rest on the expected counts",
                Verdict.NOT_APPLICABLE,
                f"The smallest expected count is {num(smallest)}; Fisher's exact test reads the "
                "table without relying on it.",
                next_step=None,
            )
        )
    else:
        enough = smallest >= MIN_EXPECTED
        checks.append(
            assumption(
                "expected_counts",
                (
                    "Every expected count reaches the rule of thumb"
                    if enough
                    else "An expected count falls below the rule of thumb"
                ),
                Verdict.PASS if enough else Verdict.CAUTION,
                f"The smallest expected count is {num(smallest)}.",
                statistic=num(smallest),
                threshold=f"every expected count is at least {num(MIN_EXPECTED)}",
                next_step=(
                    None
                    if enough
                    else "Read the chi-square p-value with care, or use a table with more rows."
                ),
            )
        )
    return tuple(checks)


def _evidence(p_value: float) -> str:
    """What a p-value says about independence, as a clause of evidence."""
    if p_value >= 0.05:
        return "the data are consistent with independence"
    if p_value >= 0.001:
        return "the data show evidence of an association"
    return "the data show strong evidence of an association"


def _percent(level: float) -> str:
    return f"{num(level * 100)} %"


def _contingency_words(
    test: HypothesisTest,
    estimates: tuple[Estimate, ...],
    level: float,
    excluded: int,
) -> str:
    """Two or three sentences: the test, the effect sizes, and the rows left out."""
    assert test.p_value is not None
    subject = test.name if test.name.startswith("Fisher") else f"The {test.name}"
    sentences = [f"{subject} gives {p_text(test.p_value)}; {_evidence(test.p_value)}."]
    cramers = next(found for found in estimates if found.label == "Cramér's V")
    odds = next((found for found in estimates if found.label.startswith("Odds ratio")), None)
    risk = next((found for found in estimates if found.label.startswith("Risk ratio")), None)
    clauses = [f"Cramér's V is {num(cramers.value)}"]
    if odds is not None and odds.lower is not None and odds.upper is not None:
        clauses.append(
            f"the odds ratio is {num(odds.value)}, with a {_percent(level)} interval from "
            f"{num(odds.lower)} to {num(odds.upper)}"
        )
    if risk is not None and risk.lower is not None and risk.upper is not None:
        clauses.append(
            f"the risk ratio is {num(risk.value)}, with a {_percent(level)} interval from "
            f"{num(risk.lower)} to {num(risk.upper)}"
        )
    if len(clauses) == 1:
        sentences.append(f"{clauses[0]}.")
    else:
        sentences.append(f"{clauses[0]}; " + ", and ".join(clauses[1:]) + ".")
    if excluded:
        noun = "row was" if excluded == 1 else "rows were"
        sentences.append(f"{excluded} {noun} left out and listed.")
    return " ".join(sentences[:3])


def _contingency_methods(two_by_two: bool, fisher: bool, level: float) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    degrees = "(r - 1)(c - 1)"
    parts = [
        "Each row pairs the row label with the column label at the same position.",
        f"The Pearson chi-square test of independence uses {degrees} degrees of freedom"
        + (" and Yates' continuity correction for the 2 x 2 table." if two_by_two else "."),
    ]
    if fisher:
        parts.append(
            "The 2 x 2 table has an expected count below 5, so Fisher's exact test, two-sided, "
            "is used instead of the chi-square approximation."
        )
    parts.append(
        "Cramér's V is the square root of the uncorrected Pearson chi-square divided by the "
        "number of rows and by the smaller of the row and column degrees of freedom."
    )
    if two_by_two:
        parts.append(
            "The odds ratio takes the first row level with the first column level as its "
            f"reference cell, and its {_percent(level)} interval is Wald's on the log scale; "
            "the risk ratio takes the same reference, and its interval is Katz's on the log "
            "scale."
        )
    return " ".join(parts)


def run_contingency(request: BaseModel) -> ToolResult:
    """Run the contingency table tool on its own request."""
    if not isinstance(request, ContingencyRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"contingency takes a ContingencyRequest, not a request of {type(request).__name__}",
        )
    return _contingency(request)


def _contingency(request: ContingencyRequest) -> ToolResult:
    check_level(request.level)
    given_rows = levels(request.rows, minimum=2)
    given_columns = levels(request.columns, minimum=2)
    _check_levels_unique(request.rows, given_rows)
    _check_levels_unique(request.columns, given_columns)

    paired = _align_labels(request.rows, request.columns)
    if paired.first.size == 0:
        raise ToolError(
            "NO_DATA",
            f"the columns {request.rows.name!r} and {request.columns.name!r} share no row "
            "with a label in both",
        )
    whole = _grid(given_rows, given_columns, paired.first, paired.second)

    warnings: list[str] = []
    row_totals = whole.sum(axis=1)
    column_totals = whole.sum(axis=0)
    kept_rows = [index for index, total in enumerate(row_totals) if total > 0]
    kept_columns = [index for index, total in enumerate(column_totals) if total > 0]
    for index, label in enumerate(given_rows):
        if index not in kept_rows:
            warnings.append(f"The row level {label!r} holds no paired observation and was dropped.")
    for index, label in enumerate(given_columns):
        if index not in kept_columns:
            warnings.append(
                f"The column level {label!r} holds no paired observation and was dropped."
            )
    if len(kept_rows) < 2 or len(kept_columns) < 2:
        raise ToolError(
            "SINGLE_GROUP",
            "after dropping the empty levels the table holds "
            f"{len(kept_rows)} row and {len(kept_columns)} column levels, and a contingency "
            "table needs at least 2 of each",
        )

    row_levels = [given_rows[index] for index in kept_rows]
    column_levels = [given_columns[index] for index in kept_columns]
    grid = whole[np.ix_(kept_rows, kept_columns)]
    two_by_two = grid.shape == (2, 2)
    smallest = float(_expected(grid).min())
    test = _independence_test(grid)
    estimates: list[Estimate] = [Estimate(label="Cramér's V", value=_cramers_v(grid))]
    if two_by_two:
        extra, extra_warnings = _two_by_two_estimates(
            grid, row_levels, column_levels, request.level
        )
        estimates.extend(extra)
        warnings.extend(extra_warnings)
    if smallest < MIN_EXPECTED:
        if two_by_two:
            warnings.append(
                f"The smallest expected count is {num(smallest)}, below {num(MIN_EXPECTED)}, so "
                "Fisher's exact test is given instead of the chi-square test."
            )
        else:
            warnings.append(
                f"The smallest expected count is {num(smallest)}, below {num(MIN_EXPECTED)}, so "
                "the chi-square approximation is in doubt."
            )

    excluded, unlisted = list_excluded(paired.excluded)
    used = int(grid.sum())
    excluded_count = len(paired.excluded)
    return ToolResult(
        tool="contingency",
        title=f"Contingency table of {request.rows.name} by {request.columns.name}",
        level=request.level,
        estimates=tuple(estimates),
        tests=(test,),
        assumptions=_independence_assumptions(smallest, fisher=test.name == "Fisher's exact test"),
        warnings=tuple(warnings),
        interpretation=_contingency_words(
            test, tuple(estimates), request.level, len(paired.excluded)
        ),
        methods_fragment=_contingency_methods(
            two_by_two, test.name == "Fisher's exact test", request.level
        ),
        counts=Counts(n_offered=used + excluded_count, n_used=used, n_excluded=excluded_count),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=(
            _count_table("Counts", row_levels, column_levels, grid),
            _percent_table("Row percentages", row_levels, column_levels, grid, mode="row"),
            _percent_table("Column percentages", row_levels, column_levels, grid, mode="column"),
            _percent_table("Of all", row_levels, column_levels, grid, mode="all"),
        ),
    )


def _wilson(count: int, total: int, quantile: float) -> tuple[float, float]:
    """The Wilson score interval for a proportion of ``count`` in ``total``."""
    proportion = count / total
    squared = quantile * quantile
    centre = (count + squared / 2.0) / (total + squared)
    half = quantile * math.sqrt(total * proportion * (1.0 - proportion) + squared / 4.0)
    half /= total + squared
    return max(0.0, centre - half), min(1.0, centre + half)


def _phi(first: int, second: int, third: int, fourth: int) -> float:
    """The phi coefficient of the paired indicators of a 2 x 2 table."""
    denominator = math.sqrt(
        (first + second) * (third + fourth) * (first + third) * (second + fourth)
    )
    if denominator == 0.0:
        return 0.0
    return (first * fourth - second * third) / denominator


def _newcombe_interval(
    first: int, second: int, third: int, fourth: int, total: int, level: float
) -> tuple[float, float]:
    """Newcombe's method 10 interval for the difference of two paired proportions.

    ``first``, ``second``, ``third`` and ``fourth`` are the cells of the 2 x 2 table of the
    paired indicators; the difference is the after marginal proportion of the first column
    level minus the before marginal proportion of the first row level. Each marginal gets a
    Wilson score limit, and the two are combined through the phi coefficient of the paired
    indicators, the closed form of the MOVER interval.
    """
    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    after = (first + third) / total
    before = (first + second) / total
    after_low, after_high = _wilson(first + third, total, quantile)
    before_low, before_high = _wilson(first + second, total, quantile)
    correlation = _phi(first, second, third, fourth)
    difference = after - before
    lower_square = (
        (after - after_low) ** 2
        + (before_high - before) ** 2
        - 2.0 * correlation * (after - after_low) * (before_high - before)
    )
    upper_square = (
        (after_high - after) ** 2
        + (before - before_low) ** 2
        - 2.0 * correlation * (after_high - after) * (before - before_low)
    )
    lower = difference - math.sqrt(max(lower_square, 0.0))
    upper = difference + math.sqrt(max(upper_square, 0.0))
    return max(-1.0, lower), min(1.0, upper)


def _marginal_difference(
    grid: NDArray[np.int64],
    level: float,
    before_levels: tuple[str, ...],
    after_levels: tuple[str, ...],
) -> Estimate:
    """The after minus before marginal proportion of the first level, with Newcombe's interval."""
    first, second, third, fourth = (int(value) for value in grid.reshape(-1))
    total = int(grid.sum())
    difference = (third - second) / total
    lower, upper = _newcombe_interval(first, second, third, fourth, total, level)
    label = (
        "Difference in marginal proportions "
        f"(after {after_levels[0]} minus before {before_levels[0]})"
    )
    return Estimate(label=label, value=difference, lower=lower, upper=upper)


def _mcnemar_test(grid: NDArray[np.int64], effect: Estimate) -> HypothesisTest:
    """McNemar's test: exact binomial at most 25 discordant pairs, chi-square beyond."""
    second = int(grid[0, 1])
    third = int(grid[1, 0])
    discordant = second + third
    null = "the two marginal proportions are equal"
    if discordant == 0:
        return HypothesisTest(
            name="McNemar's test",
            method="McNemar's test, exact binomial",
            null_hypothesis=null,
            statistic=None,
            p_value=None,
            p_value_text=p_text(None),
            effect=effect,
            reason="the two columns agree on every row, so there are no discordant pairs",
        )
    if discordant <= MCNEMAR_EXACT_MAX:
        found = scipy_stats.binomtest(min(second, third), discordant, 0.5, alternative="two-sided")
        return HypothesisTest(
            name="McNemar's test",
            method="McNemar's test, exact binomial",
            null_hypothesis=null,
            statistic=float(min(second, third)),
            p_value=float(found.pvalue),
            p_value_text=p_text(float(found.pvalue)),
            effect=effect,
        )
    statistic = (abs(second - third) - 1.0) ** 2 / discordant
    p_value = float(scipy_stats.chi2.sf(statistic, 1))
    return HypothesisTest(
        name="McNemar's test",
        method="McNemar's test, chi-square with continuity correction",
        null_hypothesis=null,
        statistic=statistic,
        df=1.0,
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=effect,
    )


def _mcnemar_evidence(p_value: float) -> str:
    """What a p-value says about equal marginal proportions, as a clause of evidence."""
    if p_value >= 0.05:
        return "the data are consistent with equal marginal proportions"
    if p_value >= 0.001:
        return "the data show evidence that the marginal proportions differ"
    return "the data show strong evidence that the marginal proportions differ"


def _mcnemar_words(test: HypothesisTest, difference: Estimate, level: float, excluded: int) -> str:
    """Two or three sentences: the test, the difference, and the rows left out."""
    if test.p_value is None:
        first = f"{test.name} could not be run: {test.reason}."
    else:
        first = f"{test.name} gives {p_text(test.p_value)}; {_mcnemar_evidence(test.p_value)}."
    assert difference.lower is not None and difference.upper is not None
    second = (
        f"The difference in marginal proportions is {num(difference.value)}, with a "
        f"{_percent(level)} interval from {num(difference.lower)} to {num(difference.upper)}."
    )
    sentences = [first, second]
    if excluded:
        noun = "row was" if excluded == 1 else "rows were"
        sentences.append(f"{excluded} {noun} left out and listed.")
    return " ".join(sentences[:3])


def _mcnemar_methods(exact: bool, level: float) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    tested = (
        "the exact binomial distribution of the discordant pairs, used when there are at most "
        f"{MCNEMAR_EXACT_MAX} discordant pairs"
        if exact
        else "the chi-square distribution with a continuity correction"
    )
    return (
        "Each row pairs the before label with the after label at the same position. McNemar's "
        f"test is two-sided, using {tested}. The effect is the difference between the after and "
        "before marginal proportions of the first level, and its "
        f"{_percent(level)} interval is Newcombe's method 10 for paired proportions: Wilson "
        "score limits for each marginal proportion combined through the phi coefficient of the "
        "paired indicators."
    )


def run_mcnemar(request: BaseModel) -> ToolResult:
    """Run McNemar's test on its own request."""
    if not isinstance(request, McNemarRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"mcnemar takes a McNemarRequest, not a request of {type(request).__name__}",
        )
    return _mcnemar(request)


def _mcnemar(request: McNemarRequest) -> ToolResult:
    check_level(request.level)
    before_levels = levels(request.before, minimum=1)
    after_levels = levels(request.after, minimum=1)
    if len(before_levels) != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the column {request.before.name!r} holds {len(before_levels)} labels, and "
            "McNemar's test needs exactly two",
        )
    if len(after_levels) != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the column {request.after.name!r} holds {len(after_levels)} labels, and "
            "McNemar's test needs exactly two",
        )
    paired = _align_labels(request.before, request.after)
    if paired.first.size == 0:
        raise ToolError(
            "NO_DATA",
            f"the columns {request.before.name!r} and {request.after.name!r} share no row "
            "with a label in both",
        )
    grid = _grid(before_levels, after_levels, paired.first, paired.second)
    difference = _marginal_difference(grid, request.level, before_levels, after_levels)
    test = _mcnemar_test(grid, difference)
    excluded, unlisted = list_excluded(paired.excluded)
    used = int(grid.sum())
    return ToolResult(
        tool="mcnemar",
        title=f"McNemar's test of {request.before.name} and {request.after.name}",
        level=request.level,
        estimates=(difference,),
        tests=(test,),
        assumptions=(
            assumption(
                "independence",
                "Independence between the pairs cannot be read from the two columns",
                Verdict.NOT_APPLICABLE,
                "Each row is one pair; whether the pairs are independent is a fact about how "
                "the data were collected, which the two columns cannot record.",
                next_step=(
                    "Check how the pairs were collected; dependence between them changes what "
                    "the p-value means."
                ),
            ),
        ),
        interpretation=_mcnemar_words(test, difference, request.level, len(paired.excluded)),
        methods_fragment=_mcnemar_methods(test.method.endswith("exact binomial"), request.level),
        counts=Counts(
            n_offered=used + len(paired.excluded),
            n_used=used,
            n_excluded=len(paired.excluded),
        ),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=(_count_table("Before by after", before_levels, after_levels, grid),),
    )


# --------------------------------------------------------------------------- registration


TOOLS.register(
    ToolSpec(
        name="contingency",
        family="counts",
        title="Contingency table",
        request=ContingencyRequest,
        run=run_contingency,
    )
)
TOOLS.register(
    ToolSpec(
        name="mcnemar",
        family="counts",
        title="McNemar's test",
        request=McNemarRequest,
        run=run_mcnemar,
    )
)
