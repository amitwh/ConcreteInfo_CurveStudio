"""Distribution analysis: the location, the spread and the shape of one column.

The tool reports the mean with its interval and the standard deviation, runs the two
standard tests of normality (Shapiro-Wilk and Anderson-Darling) and draws the two pictures
a person reads the shape from: a histogram with the fitted normal's expected count in each
bin, and a normal Q-Q plot at the Blom positions the core's residuals already use.

The tests come from :mod:`curvestudio.diagnostics`, so the numbers here are the core's own:
Shapiro-Wilk is SciPy's W, and Anderson-Darling is the small-sample adjusted statistic with
the Stephens (1974) piecewise p-value. A p-value is a statement about evidence, never a
statement of proof: a small one is evidence against normality, and a large one is not
evidence that the values are normal.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.diagnostics import DiagnosticTest, anderson_darling, shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import Verdict
from curvestudio.tools._wording import num
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
    MAX_PLOT_POINTS,
    FloatArray,
    HistogramPlot,
    Line,
    QQPlot,
    Series,
    stride,
)
from curvestudio.tools.stats._tables import mean_ci
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_VALUES = 3
"""The fewest values the analysis takes: Shapiro-Wilk, the weaker test here, needs three."""

MAX_SHAPIRO = 5_000
"""Above this many values Shapiro-Wilk runs on a seeded subsample; Anderson-Darling takes all."""

MAX_BINS = MAX_PLOT_POINTS - 1
"""The most histogram bins a plot holds: a histogram has one more edge than bins."""

_NORMAL_NULL = (
    "The values are drawn from a normal distribution, with its mean and variance estimated."
)
"""The null hypothesis both tests address, in the words of the core's diagnostics."""

_ANDERSON_METHOD = (
    "Anderson-Darling's A² test for normality, the statistic adjusted for the sample size, "
    "with the Stephens (1974) asymptotic p-value"
)


class DistributionRequest(BaseModel):
    """One column to describe, and the two options the description takes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(description="The column whose distribution is described.")
    level: float = Field(
        default=0.95,
        description="The confidence level of the interval for the mean, between 0 and 1.",
    )
    bins: int | None = Field(
        default=None,
        description=(
            "The number of histogram bins; the Freedman-Diaconis rule picks one when empty."
        ),
    )


def run_distribution(request: BaseModel) -> ToolResult:
    """Run the distribution analysis on its own request."""
    if not isinstance(request, DistributionRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"distribution takes a DistributionRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: DistributionRequest) -> ToolResult:
    """The analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    aligned = pairs(request.values)
    column = aligned.columns[0]
    values = column.values
    n = int(values.size)
    if n < MIN_VALUES:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the distribution analysis needs at least {MIN_VALUES} values, and the column "
            f"{column.name!r} holds {n}",
        )
    if float(np.max(values)) == float(np.min(values)):
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the column {column.name!r} holds one value repeated, so it has no distribution "
            "to describe",
        )

    mean = mean_ci(values, f"Mean of {column.name}", request.level)
    spread = float(np.std(values, ddof=1))
    warnings: list[str] = []
    tested = values
    if n > MAX_SHAPIRO:
        # Pure function: the subsample is the first draw of one fixed seed on every run.
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(n, size=MAX_SHAPIRO, replace=False))
        tested = values[chosen]
        warnings.append(
            f"Shapiro-Wilk was run on a seeded random subsample of {MAX_SHAPIRO:,} of the "
            f"{n:,} values; Anderson-Darling used all of them."
        )

    shapiro = shapiro_wilk(tested)
    anderson = anderson_darling(values)
    tests = (
        _hypothesis(
            shapiro, "Shapiro-Wilk", "Shapiro-Wilk's W test for normality, two-sided", mean
        ),
        _hypothesis(anderson, "Anderson-Darling", _ANDERSON_METHOD, mean),
    )

    independence = assumption(
        "independence",
        "Independence cannot be checked from a column",
        Verdict.NOT_APPLICABLE,
        "A single column records no pairing, order in time or repeated measure, which is "
        "what independence would have to be read from.",
        next_step=(
            "Check how the rows were collected: repeated measures, pairing or a time order "
            "make them dependent, and the p-values then do not carry their usual meaning."
        ),
    )

    listed, unlisted = list_excluded(aligned.excluded)
    return ToolResult(
        tool="distribution",
        title=f"Distribution of {column.name}",
        level=request.level,
        estimates=(
            mean,
            Estimate(label=f"Standard deviation of {column.name}", value=spread),
        ),
        tests=tests,
        assumptions=(independence,),
        warnings=tuple(warnings),
        interpretation=_interpretation(column.name, mean, spread, request.level, tests),
        methods_fragment=_methods(column.name, request.bins, n > MAX_SHAPIRO),
        counts=Counts(
            n_offered=n + len(aligned.excluded),
            n_used=n,
            n_excluded=len(aligned.excluded),
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        plots=(
            _histogram(column.name, values, request.bins, float(mean.value), spread),
            _qq_plot(column.name, values, float(mean.value), spread),
        ),
    )


def _hypothesis(
    diagnostic: DiagnosticTest, name: str, method: str, interval: Estimate
) -> HypothesisTest:
    """One test as every surface shows it, with the interval for the mean beside its p-value.

    A normality test has no effect size of its own in this family, so the test travels with
    the interval for the mean, at its confidence level, rather than with a bare p-value.
    """
    return HypothesisTest(
        name=name,
        method=method,
        null_hypothesis=_NORMAL_NULL,
        statistic=diagnostic.statistic,
        p_value=diagnostic.p_value,
        p_value_text=p_text(diagnostic.p_value),
        effect=interval,
        reason=diagnostic.reason,
    )


def _bin_edges(values: FloatArray, bins: int | None) -> FloatArray:
    """The histogram edges: ``bins`` of them when asked, Freedman-Diaconis otherwise.

    Raises:
        ToolError: ``INVALID_OPTION`` for a bin count a plot cannot hold.
    """
    if bins is not None:
        if not 1 <= bins <= MAX_BINS:
            raise ToolError(
                "INVALID_OPTION",
                f"the histogram takes between 1 and {MAX_BINS:,} bins, not {bins}",
            )
        return np.asarray(np.histogram_bin_edges(values, bins=bins), dtype=np.float64)
    # The Freedman-Diaconis rule; a column with more than half its values equal has no
    # interquartile range to divide by, and Sturges's rule stands in, as the core does.
    q1, q3 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
    rule = "fd" if q3 > q1 else "sturges"
    edges = np.asarray(np.histogram_bin_edges(values, bins=rule), dtype=np.float64)
    if edges.size - 1 > MAX_BINS:
        edges = np.asarray(np.histogram_bin_edges(values, bins=MAX_BINS), dtype=np.float64)
    return edges


def _histogram(
    name: str, values: FloatArray, bins: int | None, mean: float, spread: float
) -> HistogramPlot:
    """The histogram of ``values``, with the fitted normal's expected count in each bin.

    The outer bins take the normal's whole lower and upper tail, so the overlay's counts sum
    to the same n the histogram's do.
    """
    edges = _bin_edges(values, bins)
    counts, _ = np.histogram(values, bins=edges)
    inner = edges[1:-1]
    cdf_edges = np.concatenate((np.array([-np.inf]), inner, np.array([np.inf])))
    expected = values.size * np.diff(scipy_stats.norm.cdf(cdf_edges, loc=mean, scale=spread))
    return HistogramPlot(
        title=f"Histogram of {name}",
        x_label=name,
        edges=tuple(float(edge) for edge in edges),
        counts=tuple(float(count) for count in counts),
        overlays=(
            Series(
                label="Fitted normal",
                values=tuple(float(value) for value in expected),
            ),
        ),
    )


def _qq_plot(name: str, values: FloatArray, mean: float, spread: float) -> QQPlot:
    """The normal Q-Q plot: ordered values against the fitted quantiles at Blom positions.

    ``stride`` cuts both arrays with the same evenly spaced indices, so a pair never parts.
    """
    ordered = np.sort(values)
    n = ordered.size
    blom = (np.arange(1.0, n + 1.0) - 0.375) / (n + 0.25)
    theoretical = mean + spread * scipy_stats.norm.ppf(blom)
    shown_theoretical, note = stride(theoretical)
    shown_ordered, _ = stride(ordered)
    return QQPlot(
        title=f"Normal Q-Q plot of {name}",
        theoretical=tuple(float(value) for value in shown_theoretical),
        ordered=tuple(float(value) for value in shown_ordered),
        line=Line(
            x0=float(shown_theoretical[0]),
            y0=float(shown_theoretical[0]),
            x1=float(shown_theoretical[-1]),
            y1=float(shown_theoretical[-1]),
            label="Normal",
        ),
        note=note,
    )


def _shape_clause(p: float) -> str:
    """What the smallest of the two p-values says about the shape, as a sentence."""
    if p >= 0.05:
        return "the data are consistent with a normal distribution"
    if p >= 0.001:
        return (
            "the data show evidence against a normal distribution, so the interval is approximate"
        )
    return (
        "the data show strong evidence against a normal distribution, so the interval is "
        "approximate"
    )


def _interpretation(
    name: str, mean: Estimate, spread: float, level: float, tests: tuple[HypothesisTest, ...]
) -> str:
    """One to three sentences: the estimates, and what the normality tests say about them."""
    lower = mean.lower
    upper = mean.upper
    assert lower is not None and upper is not None
    first = (
        f"The mean of {name} is {num(mean.value)} with a {num(level * 100)} % interval from "
        f"{num(lower)} to {num(upper)}, and the standard deviation is {num(spread)}."
    )
    reported = [
        f"{test.name} gives {test.p_value_text}" for test in tests if test.p_value is not None
    ]
    if not reported:
        return f"{first} Neither normality test could be run on these values."
    smallest = min(test.p_value for test in tests if test.p_value is not None)
    return f"{first} {' and '.join(reported)}; {_shape_clause(smallest)}."


def _methods(name: str, bins: int | None, subsampled: bool) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    tested = "a seeded random subsample" if subsampled else "the values"
    rule = "the requested number of bins" if bins is not None else "the Freedman-Diaconis rule"
    return (
        f"The mean of {name} is reported with a Student t interval on n - 1 degrees of "
        "freedom, and the standard deviation is the sample standard deviation. "
        f"Shapiro-Wilk's W test was run on {tested} and Anderson-Darling's A² test on the "
        "full column; the Anderson-Darling statistic is adjusted for the sample size and its "
        "p-value is Stephens's (1974) asymptotic fit, not a table. "
        f"The histogram bins follow {rule} and carry the fitted normal's expected count in "
        "each bin, and the normal Q-Q plot uses Blom positions."
    )


TOOLS.register(
    ToolSpec(
        name="distribution",
        family="summarise",
        title="Distribution analysis",
        request=DistributionRequest,
        run=run_distribution,
    )
)
