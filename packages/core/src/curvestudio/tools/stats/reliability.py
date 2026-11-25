"""Cronbach's alpha: the internal consistency of a set of items measured on the same rows.

The tool reads two to fifty numeric items, keeps the rows complete in every one of them, and
reports the reliability of their sum. Raw alpha is the item count over one less than it, times
one minus the sum of the item variances over the variance of the item totals, every variance
taken with ``n - 1`` in its denominator. The standardised alpha replaces the variances with the
mean inter-item correlation, and the mean correlation is reported beside it. The interval is
Feldt's, from the F distribution on ``n - 1`` and ``(n - 1)(k - 1)`` degrees of freedom, and
the same F distribution tests the coefficient against zero. Every item is also reported with
its mean, its standard deviation, its corrected item-total correlation (the item against the
sum of the remaining items) and the alpha the scale would have with that item deleted.

A reverse-keyed item can be named in ``reverse``; it is then recoded as ``max + min - value``
on its own observed range, and the result says which items were recoded and on which ranges.
An item that correlates negatively with the sum of the rest but is not named is left as it is
and warned about.

Alpha is a lower bound on reliability when the items are not tau-equivalent, and
tau-equivalence cannot be tested from these rows. A high alpha is consistent with a set of
items that measure one construct; it does not establish that they do, nor that the construct is
the one the items were meant to measure. Fewer than three complete rows, a constant item, or an
item count outside the tool's range are refused with a named error. Every row left out is
counted and listed, never dropped without a word.
"""

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import ExcludedRow, Verdict
from curvestudio.tools._wording import joined, listing, num
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
from curvestudio.tools.stats._display import Cell, ColumnKind, FloatArray, Table
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_ITEMS = 2
"""The fewest items alpha is defined for."""
MAX_ITEMS = 50
"""The most items the tool takes."""
MIN_ROWS = 3
"""The fewest complete rows the tool takes."""
WARN_ROWS = 5
"""Fewer complete rows than this makes alpha unstable enough to warn about."""

_ITEM_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("item", "Item", "text"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("corrected", "Corrected item-total correlation", "number"),
    ("deleted", "Alpha if deleted", "number"),
)


class ReliabilityRequest(BaseModel):
    """The items, what they are called, which are reverse-keyed, and the level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[Column, ...] = Field(
        description=(
            "The two to fifty numeric items measured on the same rows. Cronbach's alpha "
            "measures the consistency of their sum."
        ),
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "What the items are called in the tables and the words, in the order given. Left "
            "empty, each item's own column name is used."
        ),
        json_schema_extra={"label": "Item names"},
    )
    reverse: tuple[str, ...] = Field(
        default=(),
        description=(
            "The items to recode before the analysis, named as the item names above. Each is "
            "recoded as max + min - value on its own observed range, and the result says which "
            "items were recoded."
        ),
        json_schema_extra={"label": "Reverse-keyed items"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of the interval for alpha, between 0 and 1.",
    )


def _item_names(request: ReliabilityRequest) -> tuple[str, ...]:
    """The names the tables use: the given names, or the columns' own names.

    Raises:
        ToolError: ``INVALID_OPTION`` for an item count outside two to fifty, or for names
            that do not name one item each.
    """
    size = len(request.items)
    if not MIN_ITEMS <= size <= MAX_ITEMS:
        raise ToolError(
            "INVALID_OPTION",
            f"Cronbach's alpha takes between {MIN_ITEMS} and {MAX_ITEMS} items, not {size}",
        )
    if request.names is None:
        return tuple(item.name for item in request.items)
    if len(request.names) != size:
        raise ToolError(
            "INVALID_OPTION",
            f"Cronbach's alpha takes one name per item, so {size} names, not {len(request.names)}",
        )
    return tuple(request.names)


def _reverse_indices(names: tuple[str, ...], reverse: tuple[str, ...]) -> tuple[int, ...]:
    """The positions of the items named in ``reverse``, once each, in the order given.

    Raises:
        ToolError: ``UNKNOWN_COLUMN`` for a name that is not one of the items.
    """
    found: list[int] = []
    for name in reverse:
        if name not in names:
            raise ToolError(
                "UNKNOWN_COLUMN",
                f"the reverse-keyed item {name!r} is not one of the items; the items are "
                f"{joined(list(names))}",
            )
        index = names.index(name)
        if index not in found:
            found.append(index)
    return tuple(found)


def _cases(request: ReliabilityRequest) -> tuple[FloatArray, tuple[ExcludedRow, ...], int]:
    """The values complete across every item, the rows left out, and the rows offered."""
    pairing = pairs(*request.items)
    matrix = np.column_stack([column.values for column in pairing.columns])
    offered = int(matrix.shape[0]) + len(pairing.excluded)
    return np.asarray(matrix, dtype=np.float64), pairing.excluded, offered


def _recode(
    matrix: FloatArray, indices: tuple[int, ...], names: tuple[str, ...]
) -> tuple[FloatArray, tuple[tuple[str, float, float], ...]]:
    """The matrix with the named items reversed on their own observed ranges.

    Returns the recoded matrix and, for each reversed item, its name and the range it was
    recoded on. The recoding is ``max + min - value``, so a value at the item's lowest becomes
    its highest and the other way round.
    """
    recoded = matrix.copy()
    ranges: list[tuple[str, float, float]] = []
    for index in indices:
        low = float(np.min(recoded[:, index]))
        high = float(np.max(recoded[:, index]))
        recoded[:, index] = high + low - recoded[:, index]
        ranges.append((names[index], low, high))
    return recoded, tuple(ranges)


def _require_varying(matrix: FloatArray, names: tuple[str, ...]) -> None:
    """Refuse an item that holds one value repeated, so its variance is zero.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` naming the first such item.
    """
    constant = [
        names[index]
        for index in range(matrix.shape[1])
        if float(np.min(matrix[:, index])) == float(np.max(matrix[:, index]))
    ]
    if constant:
        pronoun = "its" if len(constant) == 1 else "their"
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the {listing(constant, 'item')} hold{'s' if len(constant) == 1 else ''} one value "
            f"repeated in the rows used, so {pronoun} variance is zero and Cronbach's alpha is "
            "undefined",
        )


def _raw_alpha(matrix: FloatArray) -> float | None:
    """Raw alpha over the items in ``matrix``, or none when the item totals do not vary."""
    size = matrix.shape[1]
    if size < 2:
        return None
    total_variance = float(np.var(np.sum(matrix, axis=1), ddof=1))
    if total_variance == 0.0:
        return None
    item_variance = float(np.sum(np.var(matrix, axis=0, ddof=1)))
    value = size / (size - 1.0) * (1.0 - item_variance / total_variance)
    return value if math.isfinite(value) else None


def _mean_inter_item_correlation(matrix: FloatArray) -> float:
    """The mean of the off-diagonal Pearson correlations between the items."""
    size = matrix.shape[1]
    correlations = np.asarray(np.corrcoef(matrix, rowvar=False), dtype=np.float64)
    upper = correlations[np.triu_indices(size, 1)]
    return float(np.mean(upper))


def _pearson(item: FloatArray, other: FloatArray) -> float | None:
    """The Pearson correlation of two columns, or none when either holds one value repeated."""
    if float(np.ptp(item)) == 0.0 or float(np.ptp(other)) == 0.0:
        return None
    value = float(np.corrcoef(item, other)[0, 1])
    return value if math.isfinite(value) else None


def _feldt(alpha: float, rows: int, items: int, level: float) -> tuple[float, float]:
    """Feldt's interval for alpha, from the F distribution's two central quantiles.

    The bounds are ``1 - (1 - alpha) * F``, the upper bound taking the lower quantile and the
    lower bound the upper one, on ``n - 1`` and ``(n - 1)(k - 1)`` degrees of freedom.
    """
    numerator = float(rows - 1)
    denominator = float((rows - 1) * (items - 1))
    tail = (1.0 - level) / 2.0
    low_quantile = float(scipy_stats.f.ppf(tail, numerator, denominator))
    high_quantile = float(scipy_stats.f.ppf(1.0 - tail, numerator, denominator))
    return 1.0 - (1.0 - alpha) * high_quantile, 1.0 - (1.0 - alpha) * low_quantile


def _feldt_test(alpha: float, rows: int, items: int) -> tuple[float, float]:
    """Feldt's F statistic and its p-value: ``F = 1 / (1 - alpha)`` on the same df."""
    numerator = float(rows - 1)
    denominator = float((rows - 1) * (items - 1))
    statistic = 1.0 / (1.0 - alpha)
    p_value = float(scipy_stats.f.sf(statistic, numerator, denominator))
    return statistic, p_value


def _item_table(
    matrix: FloatArray,
    names: tuple[str, ...],
    ranges: tuple[tuple[str, float, float], ...],
) -> tuple[Table, tuple[float | None, ...], tuple[float | None, ...]]:
    """The item statistics, every corrected item-total correlation, and every alpha if deleted.

    Alpha if deleted is empty with two items, since one item has no alpha of its own, and
    empty for a set whose remaining items sum to a constant.
    """
    size = matrix.shape[1]
    totals = np.sum(matrix, axis=1)
    means = np.mean(matrix, axis=0)
    sds = np.std(matrix, axis=0, ddof=1)
    corrected: list[float | None] = []
    deleted: list[float | None] = []
    for index in range(size):
        correlation = _pearson(matrix[:, index], totals - matrix[:, index])
        corrected.append(correlation)
        deleted.append(_raw_alpha(np.delete(matrix, index, axis=1)))
    rows: list[tuple[Cell, ...]] = [
        (names[index], float(means[index]), float(sds[index]), corrected[index], deleted[index])
        for index in range(size)
    ]
    details = [
        "SD is the sample standard deviation, with n - 1 in its denominator.",
        "The corrected item-total correlation correlates an item with the sum of the remaining "
        "items; alpha if deleted is the alpha over the remaining items, and is empty with two "
        "items.",
    ]
    if ranges:
        recoded = joined(
            [f"{name} (range {num(low)} to {num(high)})" for name, low, high in ranges]
        )
        details.append(f"Recoded as max + min - value before the analysis: {recoded}.")
    return (
        table("Item statistics", _ITEM_COLUMNS, rows, note=" ".join(details)),
        tuple(corrected),
        tuple(deleted),
    )


def _warnings(
    corrected: tuple[float | None, ...], names: tuple[str, ...], rows: int
) -> tuple[str, ...]:
    """Few rows, and any item that runs against the sum of the rest."""
    found: list[str] = []
    if rows < WARN_ROWS:
        found.append(
            f"Only {rows} rows are used, fewer than {WARN_ROWS}; alpha and its interval may be "
            "unstable."
        )
    negative = [
        (names[index], value)
        for index, value in enumerate(corrected)
        if value is not None and value < 0.0
    ]
    if negative:
        items = listing([name for name, _value in negative], "item")
        values = joined([num(value) for _name, value in negative])
        if len(negative) == 1:
            found.append(
                f"The {items} correlates negatively with the sum of the other items "
                f"(r = {values}), so it may be reverse-keyed; name it in reverse to recode it."
            )
        else:
            found.append(
                f"The {items} correlate negatively with the sum of the other items "
                f"(r = {values}), so they may be reverse-keyed; name them in reverse to "
                "recode them."
            )
    return tuple(found)


def _reversal_words(ranges: tuple[tuple[str, float, float], ...]) -> str:
    """The sentence the methods fragment and the table carry about recoding."""
    if not ranges:
        return (
            "No item was reversed; an item named in reverse would be recoded as max + min - "
            "value on its own observed range."
        )
    recoded = joined([f"{name} (range {num(low)} to {num(high)})" for name, low, high in ranges])
    return (
        f"Reversed before the analysis, recoded as max + min - value on each item's own "
        f"observed range: {recoded}."
    )


def _interpretation(
    alpha: Estimate,
    deleted: tuple[float | None, ...],
    names: tuple[str, ...],
    rows: int,
    items: int,
    level: float,
) -> str:
    """Three sentences: alpha, the usual guide and the item that matters most, and the caveat."""
    assert alpha.lower is not None and alpha.upper is not None
    first = (
        f"Cronbach's alpha is {num(alpha.value)}, with a {num(level * 100)} % interval from "
        f"{num(alpha.lower)} to {num(alpha.upper)} over {rows} rows and {items} items."
    )
    defined = [(names[index], value) for index, value in enumerate(deleted) if value is not None]
    guide = "The usual guide of 0.70 is a convention, not a law"
    if not defined:
        second = f"{guide}; with two items no alpha can be computed after one is removed."
    else:
        best_name, best = max(defined, key=lambda pair: pair[1])
        if best > alpha.value:
            second = f"{guide}, and removing {best_name} raises alpha most, to {num(best)}."
        else:
            second = (
                f"{guide}, and the largest alpha after removing one item is {num(best)} with "
                f"{best_name} removed, still below the full-scale value."
            )
    third = "A high alpha does not establish that the items measure one construct."
    return " ".join((first, second, third))


def _methods(ranges: tuple[tuple[str, float, float], ...]) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Cronbach's alpha is k / (k - 1) times one minus the sum of the item variances over "
        "the variance of the item totals, with n - 1 in every denominator. The standardised "
        "alpha replaces the variances with the mean inter-item correlation, k times that mean "
        "over one plus k - 1 times it. The interval is Feldt's, one minus one minus alpha "
        "times the F quantile on n - 1 and (n - 1)(k - 1) degrees of freedom, and the same F "
        "distribution tests the coefficient against zero. The corrected item-total correlation "
        "correlates an item with the sum of the remaining items, and alpha if deleted is the "
        f"alpha over the remaining items. {_reversal_words(ranges)}"
    )


def run_reliability(request: BaseModel) -> ToolResult:
    """Run the reliability tool on its own request."""
    if not isinstance(request, ReliabilityRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"reliability takes a ReliabilityRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: ReliabilityRequest) -> ToolResult:
    """The analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    names = _item_names(request)
    reverse_indices = _reverse_indices(names, request.reverse)
    offered_matrix, excluded, offered = _cases(request)
    rows = int(offered_matrix.shape[0])
    if rows < MIN_ROWS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"Cronbach's alpha needs at least {MIN_ROWS} rows complete across every item, and "
            f"{rows} {'is' if rows == 1 else 'are'} left",
        )
    matrix, ranges = _recode(offered_matrix, reverse_indices, names)
    _require_varying(matrix, names)

    items = len(names)
    alpha_value = _raw_alpha(matrix)
    if alpha_value is None:
        raise ToolError(
            "CONSTANT_COLUMN",
            "the sum of the items holds one value repeated in the rows used, so Cronbach's "
            "alpha is undefined",
        )
    mean_correlation = _mean_inter_item_correlation(matrix)
    denominator = 1.0 + (items - 1.0) * mean_correlation
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ToolError(
            "CONSTANT_COLUMN",
            "the mean inter-item correlation is at or below the floor that makes the "
            "standardised alpha undefined",
        )
    standardised = items * mean_correlation / denominator
    lower, upper = _feldt(alpha_value, rows, items, request.level)
    statistic, p_value = _feldt_test(alpha_value, rows, items)

    item_table, corrected, deleted = _item_table(matrix, names, ranges)
    alpha = Estimate(
        label="Cronbach's alpha",
        value=alpha_value,
        lower=lower,
        upper=upper,
    )
    tests = (
        HypothesisTest(
            name="F test of the reliability coefficient (Feldt)",
            method=(
                "Feldt's F test on the two-way analysis of variance, with n - 1 and "
                "(n - 1)(k - 1) degrees of freedom; the interval is Feldt's on the same "
                "degrees of freedom"
            ),
            null_hypothesis=(
                "the reliability coefficient is zero, so the items share no common variance"
            ),
            statistic=statistic,
            df=float(rows - 1),
            df_denominator=float((rows - 1) * (items - 1)),
            p_value=p_value,
            p_value_text=p_text(p_value),
            effect=alpha,
        ),
    )
    assumptions = (
        assumption(
            "tau_equivalence",
            "Tau-equivalence cannot be tested from these rows",
            Verdict.NOT_APPLICABLE,
            "Alpha is a lower bound on reliability when the items are not tau-equivalent, and "
            "tau-equivalence cannot be tested from these rows.",
            next_step=(
                "Report alpha as a lower bound unless the items were shown to be tau-equivalent, "
                "for example by a factor analysis that finds one factor of equal loadings."
            ),
        ),
        assumption(
            "one_direction",
            "The items are treated as measuring one direction",
            Verdict.NOT_APPLICABLE,
            "Every item is read as pointing the same way; an item keyed in the opposite "
            "direction lowers alpha unless it is named in reverse.",
            next_step=(
                "Check the wording of each item, and name a reverse-keyed one in reverse so that "
                "it is recoded before the analysis."
            ),
        ),
    )
    listed, unlisted = list_excluded(excluded)
    return ToolResult(
        tool="reliability",
        title="Reliability (Cronbach's alpha)",
        level=request.level,
        estimates=(
            alpha,
            Estimate(label="Standardised alpha", value=standardised),
            Estimate(label="Mean inter-item correlation", value=mean_correlation),
        ),
        tests=tests,
        assumptions=assumptions,
        warnings=_warnings(corrected, names, rows),
        interpretation=_interpretation(alpha, deleted, names, rows, items, request.level),
        methods_fragment=_methods(ranges),
        counts=Counts(n_offered=offered, n_used=rows, n_excluded=len(excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(item_table,),
    )


TOOLS.register(
    ToolSpec(
        name="reliability",
        family="measurement",
        title="Reliability (Cronbach's alpha)",
        request=ReliabilityRequest,
        run=run_reliability,
    )
)
