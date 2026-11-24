"""Sample size and power for the standard tests, with no data at all.

The tool answers one of two questions about a design: how many observations per group a
target power needs, or the power a given number of observations reaches. A design is one of
the one-sample, independent or paired t test, the one-way analysis of variance, the Pearson
correlation and the difference of two proportions. Exactly one of ``power`` and ``n`` is
given; the other is solved for, the number of observations as the smallest whole number that
reaches the target.

Every number is a statement about a stated design at a stated significance level, never a
p-value and never a claim of proof. The power of the t tests comes from the non-central t
distribution (the independent test on the degrees of freedom of the equal-variance test), the
analysis of variance from the non-central F, the correlation from Fisher's z transformation
with the t correction, and the difference of two proportions from the normal approximation on
the arcsine scale (Cohen's h). The accompanying table scans the power over sample size, from
the first size whose power reaches 0.05 to the first whose power reaches 0.99, so a person
can see how quickly the design becomes useful.
"""

import math
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.tools._errors import ToolError
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import Counts, Estimate, ToolResult
from curvestudio.tools.stats._display import ColumnKind, MultiLinePlot, XYSeries
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

Design = Literal["one_sample", "independent", "paired", "anova", "correlation", "proportions"]
"""The designs the tool computes for."""

MAX_N = 1_000_000_000
"""The most observations per group the search tries before it gives up."""

LOW_POWER = 0.05
"""Where the scan by sample size starts: the first size whose power reaches this."""

HIGH_POWER = 0.99
"""Where the scan by sample size stops: the first size whose power reaches this."""

MAX_ROWS = 40
"""The most rows the scan by sample size has; a range with more sizes is thinned to this."""


class _Design(NamedTuple):
    """What one design is called, the noun its sample size takes, and its smallest size."""

    display: str
    noun: str
    minimum: int


_DESIGNS: dict[str, _Design] = {
    "one_sample": _Design("one-sample t test", "observations", 2),
    "independent": _Design("two-independent-samples t test", "observations per group", 2),
    "paired": _Design("paired t test", "pairs", 2),
    "anova": _Design("one-way analysis of variance", "observations per group", 2),
    "correlation": _Design("Pearson correlation", "observations", 5),
    "proportions": _Design("difference of two proportions", "observations per group", 2),
}
"""The display name, the sample-size noun and the smallest usable size of each design."""

_TWO_GROUP = frozenset({"independent", "proportions"})
"""The designs whose second group is ``ratio`` times the first."""


class PowerRequest(BaseModel):
    """What the tool takes: a design and the three of its numbers a person knows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: Design = Field(description="The design the sample size or power is computed for.")
    effect_size: float | None = Field(
        default=None,
        description=(
            "The effect size: Cohen's d for the one-sample and independent t tests, dz for "
            "the paired test, Cohen's f for the analysis of variance, and the correlation "
            "itself for the correlation test. Leave it out for the difference of two "
            "proportions, which takes p1 and p2 instead."
        ),
    )
    alpha: float = Field(
        default=0.05,
        description="The significance level of the two-sided test; between 0 and 1.",
    )
    power: float | None = Field(
        default=None,
        description="The target power; give exactly one of power and n, and the other is solved.",
    )
    n: int | None = Field(
        default=None,
        json_schema_extra={"label": "N per group"},
        description=(
            "The number of observations per group; give exactly one of power and n, and the "
            "other is solved."
        ),
    )
    ratio: float = Field(
        default=1.0,
        json_schema_extra={"label": "Group size ratio"},
        description=(
            "The second group's size relative to the first's, for the independent and "
            "proportions designs; above 0."
        ),
    )
    n_groups: int = Field(
        default=2,
        json_schema_extra={"label": "Number of groups"},
        description="The number of groups in the one-way analysis of variance; at least 2.",
    )
    p1: float | None = Field(
        default=None,
        json_schema_extra={"label": "First proportion"},
        description="The first proportion, between 0 and 1, for the proportions design.",
    )
    p2: float | None = Field(
        default=None,
        json_schema_extra={"label": "Second proportion"},
        description="The second proportion, between 0 and 1, for the proportions design.",
    )


RESULT_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("design", "Design", "text"),
    ("effect_size", "Effect size", "number"),
    ("alpha", "Alpha", "number"),
    ("power", "Power", "number"),
    ("n", "n per group", "int"),
    ("total", "Total N", "number"),
    ("ratio", "Ratio", "number"),
)
"""The columns of the one-row table that states the design and its answer."""

SCAN_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("n", "n per group", "int"),
    ("power", "Power", "number"),
)
"""The columns of the table that scans the power over sample size."""


class _Setup(NamedTuple):
    """A request once its rules are checked: the design and the numbers the power takes."""

    design: str
    effect: float
    alpha: float
    ratio: float
    n_groups: int


def _t_power(effect: float, df: float, nobs_eff: float, alpha: float) -> float:
    """The power of a two-sided t test on the non-central t distribution.

    ``nobs_eff`` is the number of observations the effect size is standardised against (the
    sample size for a one-sample test, and ``1 / (1/n1 + 1/n2)`` for two independent samples).
    This is statsmodels' ``ttest_power``.
    """
    nc = effect * math.sqrt(nobs_eff)
    upper = scipy_stats.t.isf(alpha / 2.0, df)
    lower = scipy_stats.t.ppf(alpha / 2.0, df)
    return float(scipy_stats.nct.sf(upper, df, nc) + scipy_stats.nct.cdf(lower, df, nc))


def _normal_power(effect: float, nobs_eff: float, alpha: float) -> float:
    """The power of a two-sided normal test on the effective sample size.

    This is statsmodels' ``normal_power``; it backs the difference of two proportions.
    """
    nc = effect * math.sqrt(nobs_eff)
    upper = scipy_stats.norm.isf(alpha / 2.0)
    lower = scipy_stats.norm.ppf(alpha / 2.0)
    return float(scipy_stats.norm.sf(upper - nc) + scipy_stats.norm.cdf(lower - nc))


def _correlation_power(r: float, n: int, alpha: float) -> float:
    """The power of a two-sided test of a Pearson correlation, by Fisher's z with a t correction.

    The critical correlation ``rc`` comes from the t distribution on ``n - 2`` degrees of
    freedom, so the small-sample test's cut-off is used; the coefficient is taken through
    Fisher's z transformation with the ``r / (2 (n - 1))`` bias correction, and the standard
    error on that scale is ``sqrt(n - 3)``. This is the formula of R's ``pwr.r.test`` and
    pingouin's ``power_corr``.
    """
    dof = n - 2
    critical_t = scipy_stats.t.ppf(1.0 - alpha / 2.0, dof)
    critical_r = math.sqrt(critical_t**2 / (critical_t**2 + dof))
    magnitude = abs(r)
    z_r = math.atanh(magnitude) + magnitude / (2.0 * (n - 1))
    z_critical = math.atanh(critical_r)
    root = math.sqrt(n - 3)
    return float(
        scipy_stats.norm.cdf((z_r - z_critical) * root)
        + scipy_stats.norm.cdf((-z_r - z_critical) * root)
    )


def _power_at(setup: _Setup, n: int) -> float:
    """The power of ``setup`` with ``n`` observations per group."""
    effect, alpha, ratio = setup.effect, setup.alpha, setup.ratio
    if setup.design in ("one_sample", "paired"):
        return _t_power(effect, n - 1, n, alpha)
    if setup.design == "independent":
        second = n * ratio
        return _t_power(effect, n + second - 2, 1.0 / (1.0 / n + 1.0 / second), alpha)
    if setup.design == "anova":
        total = n * setup.n_groups
        df_num = setup.n_groups - 1
        df_denom = total - setup.n_groups
        critical = scipy_stats.f.isf(alpha, df_num, df_denom)
        return float(scipy_stats.ncf.sf(critical, df_num, df_denom, effect**2 * total))
    if setup.design == "correlation":
        return _correlation_power(effect, n, alpha)
    second = n * ratio
    return _normal_power(effect, 1.0 / (1.0 / n + 1.0 / second), alpha)


def _smallest_n(setup: _Setup, target: float) -> int:
    """The least number of observations per group whose power reaches ``target``.

    The power rises with the sample size, so the least size is found by doubling a bracket and
    then halving it.

    Raises:
        ToolError: ``INVALID_OPTION`` when the target needs more than :data:`MAX_N`
            observations per group.
    """
    minimum = _DESIGNS[setup.design].minimum
    if _power_at(setup, minimum) >= target:
        return minimum
    low = minimum
    high = max(minimum + 1, 2 * minimum)
    while _power_at(setup, high) < target:
        low = high
        high *= 2
        if high > MAX_N:
            raise ToolError(
                "INVALID_OPTION",
                f"a power of {num(target)} needs more than {MAX_N:,} observations per group "
                f"for this effect size",
            )
    while low + 1 < high:
        middle = (low + high) // 2
        if _power_at(setup, middle) >= target:
            high = middle
        else:
            low = middle
    return high


def _grid(low: int, high: int) -> list[int]:
    """At most ``MAX_ROWS`` sample sizes from ``low`` to ``high``, both included.

    The sizes are evenly spaced integers. When the range holds no more than ``MAX_ROWS``
    sizes every one is taken; a range shorter than 20 gives the sizes it holds, since a
    whole number cannot be split.
    """
    span = high - low + 1
    count = min(MAX_ROWS, span)
    if count <= 1:
        return [low]
    return [low + round(index * (high - low) / (count - 1)) for index in range(count)]


def _setup(request: PowerRequest) -> _Setup:
    """Check the request's rules and read the numbers the power takes.

    Raises:
        ToolError: ``INVALID_OPTION`` for an alpha, a ratio or a group count out of range, for
            neither or both of power and n, for a missing or zero effect size, for a
            correlation outside (-1, 1), or for proportions that are not a proper pair.
    """
    if not 0.0 < request.alpha < 1.0:
        raise ToolError("INVALID_OPTION", f"alpha must lie between 0 and 1, not {request.alpha}")
    if request.ratio <= 0.0:
        raise ToolError("INVALID_OPTION", f"the ratio must be above 0, not {request.ratio}")
    if request.n_groups < 2:
        raise ToolError(
            "INVALID_OPTION",
            f"the number of groups must be at least 2, not {request.n_groups}",
        )
    if (request.power is None) == (request.n is None):
        given = "both power and n" if request.power is not None else "neither power nor n"
        raise ToolError(
            "INVALID_OPTION",
            f"give exactly one of power and n to solve for the other, not {given}",
        )
    if request.power is not None and not 0.0 < request.power < 1.0:
        raise ToolError(
            "INVALID_OPTION", f"the power must lie between 0 and 1, not {request.power}"
        )
    minimum = _DESIGNS[request.design].minimum
    if request.n is not None and request.n < minimum:
        raise ToolError(
            "INVALID_OPTION",
            f"the {request.design} design needs at least {minimum} observations per group, "
            f"not {request.n}",
        )

    if request.design == "proportions":
        return _proportions_setup(request)
    return _effect_setup(request)


def _proportions_setup(request: PowerRequest) -> _Setup:
    """Read the two proportions of the proportions design, as Cohen's h."""
    if request.effect_size is not None:
        raise ToolError(
            "INVALID_OPTION",
            "the proportions design takes p1 and p2, not an effect size",
        )
    if request.p1 is None or request.p2 is None:
        raise ToolError("INVALID_OPTION", "the proportions design needs both p1 and p2")
    if not 0.0 < request.p1 < 1.0 or not 0.0 < request.p2 < 1.0:
        raise ToolError(
            "INVALID_OPTION",
            f"p1 and p2 must lie between 0 and 1, not {request.p1} and {request.p2}",
        )
    if request.p1 == request.p2:
        raise ToolError(
            "INVALID_OPTION",
            f"p1 and p2 are equal ({request.p1}), so the difference is zero and no sample "
            "size reaches the target power",
        )
    h = 2.0 * (math.asin(math.sqrt(request.p1)) - math.asin(math.sqrt(request.p2)))
    return _Setup(
        design=request.design,
        effect=abs(h),
        alpha=request.alpha,
        ratio=request.ratio,
        n_groups=request.n_groups,
    )


def _effect_setup(request: PowerRequest) -> _Setup:
    """Read the effect size of the designs that take one."""
    if request.p1 is not None or request.p2 is not None:
        raise ToolError(
            "INVALID_OPTION",
            "p1 and p2 are for the proportions design; the others take an effect size",
        )
    if request.effect_size is None:
        raise ToolError("INVALID_OPTION", f"the {request.design} design needs an effect size")
    if request.effect_size == 0.0:
        raise ToolError(
            "INVALID_OPTION",
            "an effect size of 0 cannot reach a power above alpha, so give a nonzero effect",
        )
    effect = request.effect_size
    if request.design == "correlation":
        if not -1.0 < effect < 1.0:
            raise ToolError(
                "INVALID_OPTION",
                f"a correlation must lie between -1 and 1, not {effect}",
            )
        effect = abs(effect)
    return _Setup(
        design=request.design,
        effect=effect,
        alpha=request.alpha,
        ratio=request.ratio,
        n_groups=request.n_groups,
    )


def _total(setup: _Setup, n: int) -> int | float:
    """The total number of observations the design has at ``n`` per group."""
    if setup.design in ("one_sample", "correlation"):
        return n
    if setup.design == "paired":
        return 2 * n
    if setup.design == "anova":
        return n * setup.n_groups
    total = n + n * setup.ratio
    return int(total) if float(total).is_integer() else total


def _method(design: str, setup: _Setup) -> str:
    """The method the power of ``design`` is computed by, named for the methods text."""
    if design in ("one_sample", "paired"):
        return "two-sided t test; non-central t distribution"
    if design == "independent":
        return (
            "two-sided t test; non-central t distribution, with the degrees of freedom of "
            "the equal-variance test, as statsmodels' TTestIndPower"
        )
    if design == "anova":
        return f"one-way analysis of variance with {setup.n_groups} groups; non-central F"
    if design == "correlation":
        return "Pearson correlation; Fisher z with the t correction"
    return (
        "difference of two proportions; the normal approximation on the arcsine scale (Cohen's h)"
    )


def _effect_convention(design: str) -> str:
    """What the effect size means for ``design``, as a sentence."""
    return {
        "one_sample": (
            "The effect size is Cohen's d, the standardised difference between the mean and "
            "its null value."
        ),
        "independent": (
            "The effect size is Cohen's d, the difference of the two means divided by the "
            "pooled standard deviation."
        ),
        "paired": (
            "The effect size is Cohen's dz, the mean difference divided by the standard "
            "deviation of the differences, and the sample size counts the pairs."
        ),
        "anova": "The effect size is Cohen's f, with the groups equal in size.",
        "correlation": "The effect size is the Pearson correlation coefficient itself.",
        "proportions": (
            "The effect size is Cohen's h, twice the difference of the arcsine square roots "
            "of the two proportions."
        ),
    }[design]


def _interpretation(setup: _Setup, n: int, achieved: float, target: float | None) -> str:
    """One or two plain sentences: the size, the power and the design."""
    design = _DESIGNS[setup.design]
    if target is None:
        return (
            f"With {num(n)} {design.noun}, a {design.display} at alpha = {num(setup.alpha)} "
            f"has a power of {num(achieved)}."
        )
    return (
        f"A {design.display} at alpha = {num(setup.alpha)} needs {num(n)} {design.noun} to "
        f"reach a power of {num(target)}; at that size the power is {num(achieved)}."
    )


def _run(request: PowerRequest) -> ToolResult:
    """Run the power tool on its typed request."""
    setup = _setup(request)
    target = request.power
    if target is None:
        if request.n is None:  # unreachable: _setup refuses neither of the two
            raise ToolError("INVALID_OPTION", "give exactly one of power and n")
        n = request.n
        achieved = _power_at(setup, n)
    else:
        n = _smallest_n(setup, target)
        achieved = _power_at(setup, n)

    low = _smallest_n(setup, LOW_POWER)
    high = _smallest_n(setup, HIGH_POWER)
    sizes = _grid(low, high)
    scan = [(size, _power_at(setup, size)) for size in sizes]

    design = _DESIGNS[setup.design]
    applied_ratio = setup.ratio if setup.design in _TWO_GROUP else None
    result_row = (
        design.display[:1].upper() + design.display[1:],
        setup.effect,
        setup.alpha,
        achieved,
        n,
        _total(setup, n),
        applied_ratio,
    )
    table_note = (
        "Alpha is the significance level of the two-sided test. Ratio is the second group's "
        "size relative to the first's, and applies to the two-sample designs."
    )
    plot = MultiLinePlot(
        title="Power against sample size",
        x_label="n per group",
        y_label="Power",
        series=(
            XYSeries(
                label="Power",
                x=tuple(float(size) for size in sizes),
                y=tuple(power for _size, power in scan),
            ),
        ),
    )

    methods = (
        f"Sample size and power for a {design.display}. Method: {_method(setup.design, setup)}. "
        f"{_effect_convention(setup.design)} The calculations are two-sided, at the "
        f"significance level alpha = {num(setup.alpha)}."
    )
    return ToolResult(
        tool="power",
        title=f"Sample size and power for a {design.display}",
        level=1.0 - setup.alpha,
        estimates=(
            Estimate(label="n per group", value=float(n)),
            Estimate(label="Total number of observations", value=float(_total(setup, n))),
            Estimate(label="Power", value=achieved),
        ),
        interpretation=_interpretation(setup, n, achieved, target),
        methods_fragment=methods,
        counts=Counts(n_offered=0, n_used=0, n_excluded=0),
        tables=(
            table("Result", RESULT_COLUMNS, [result_row], note=table_note),
            table("Power by sample size", SCAN_COLUMNS, scan),
        ),
        plots=(plot,),
    )


def run_power(request: BaseModel) -> ToolResult:
    """The registry's entry point: the tool's own request, or ``INVALID_OPTION``.

    Raises:
        ToolError: ``INVALID_OPTION`` when ``request`` is not a :class:`PowerRequest`.
    """
    if not isinstance(request, PowerRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"power takes a PowerRequest, not a request of {type(request).__name__}",
        )
    return _run(request)


SPEC = ToolSpec(
    name="power",
    family="power",
    title="Sample size and power",
    request=PowerRequest,
    run=run_power,
)
TOOLS.register(SPEC)
