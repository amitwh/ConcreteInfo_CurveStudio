"""Bayesian versions of the Stage 1 tests: a Bayes factor and a posterior, not a p-value.

The tool answers the question a t test answers, in the currency of evidence rather than of
significance: how much more likely are these data under the alternative than under the null,
and what does the posterior say about the size of the effect. Four designs mirror the t
test's: one sample, paired observations, two independent groups, and a single proportion.

The Bayes factor for the three effect designs is the JZS Bayes factor of Rouder, Speckman,
Sun, Morey and Iverson (2009), *Psychonomic Bulletin & Review* 16(2), 225-237. Their
Equation 1 integrates the likelihood ratio over the inverse-chi-square mixing variable ``g``
of the Jeffreys-Zellner-Siow prior; the same equation serves the two-sample design with the
effective sample size ``n1 n2 / (n1 + n2)`` and ``n1 + n2 - 2`` degrees of freedom (their
"Extension to Two-Sample Designs"). The prior on the standardised effect is a Cauchy
distribution with the stated scale, and the posterior reported here is that prior's
posterior, not a flat-prior one.

The proportion design is the conjugate beta-binomial model: a Beta prior on the proportion
and a Beta posterior, with the point null compared by the Savage-Dickey ratio of the prior
density to the posterior density at the null (Dickey, 1971; Wagenmakers et al., 2010).

A Bayes factor is evidence under the stated prior; it is not a p-value, and it is not a
demonstration that either hypothesis is true.
"""

import math
import warnings
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import integrate
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    aligned,
    assumption,
    check_level,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    ColumnKind,
    FloatArray,
    MultiLinePlot,
    Table,
    XYSeries,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

Design = Literal["one_sample", "paired", "independent", "proportion"]
"""The four designs the tool computes for."""

MIN_OBSERVATIONS = 2
"""The fewest values a one-sample, paired or independent design takes."""

MIN_TWO_SAMPLE = 3
"""The fewest values the two independent samples together take: the pooled spread needs a df."""

PRIOR_SCALE = 0.707
"""Rouder et al.'s JZS scale on the standardised effect: 1/sqrt(2)."""

ROPE: tuple[float, float] = (-0.1, 0.1)
"""The default region of practical equivalence on the standardised effect."""

INTEGRATION_TOLERANCE = 1e-10
"""The absolute and relative tolerance of the Bayes factor's integral over g."""

INTEGRATION_LIMIT = 200
"""The most subdivisions the integral over g may take before it reports non-convergence."""

GRID_POINTS = 4_001
"""The points of the posterior grid the density, interval and ROPE are read from."""

PLOT_POINTS = 200
"""The most points the posterior density plot carries."""

BF_BANDS = (3.0, 10.0, 30.0)
"""The conventional evidence bands of a Bayes factor, named as conventions only."""

_SCALE_LABEL = "Prior scale"
_ROPE_LABEL = "Null region"


class BayesRequest(BaseModel):
    """One Bayesian test: the design, its data and the prior it is read under."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: Design = Field(
        description=(
            "The design: one sample against a hypothesised mean, paired observations, two "
            "independent groups, or a single proportion from counts."
        )
    )
    a: Column | None = Field(
        default=None,
        description=(
            "The first column: the one sample's values, the first column of a pair, or the "
            "values with a group label for the independent design. Left empty for a "
            "proportion, which takes counts."
        ),
    )
    b: Column | None = Field(
        default=None,
        description=(
            "The second column of a paired design, or the second group of an independent "
            "design when no group column is given."
        ),
    )
    groups: GroupColumn | None = Field(
        default=None,
        description=(
            "For the independent design, a group column with exactly two levels; the two "
            "groups' means are compared. Left empty when the groups are two columns."
        ),
    )
    null: float = Field(
        default=0.0,
        description=(
            "The hypothesised mean or difference of the effect designs, or the hypothesised "
            "probability of the proportion design, between 0 and 1."
        ),
        json_schema_extra={"label": "Hypothesised value"},
    )
    prior_scale: float = Field(
        default=PRIOR_SCALE,
        description=(
            "The scale of the Cauchy prior on the standardised effect, above 0; the JZS "
            "default is 0.707."
        ),
        json_schema_extra={"label": _SCALE_LABEL},
    )
    rope: tuple[float, float] = Field(
        default=ROPE,
        description=(
            "The region of practical equivalence for the standardised effect, low then high, "
            "with low below high."
        ),
        json_schema_extra={"label": _ROPE_LABEL},
    )
    successes: int | None = Field(
        default=None,
        description="The number of successes of the proportion design, a whole number from 0.",
    )
    trials: int | None = Field(
        default=None,
        description="The number of trials of the proportion design, a whole number of at least 1.",
    )
    prior_alpha: float = Field(
        default=1.0,
        description="The first shape parameter of the beta prior of the proportion design.",
    )
    prior_beta: float = Field(
        default=1.0,
        description="The second shape parameter of the beta prior of the proportion design.",
    )
    level: float = Field(
        default=0.95,
        description="The credibility level of the posterior interval, between 0 and 1.",
    )


class _Posterior(NamedTuple):
    """The posterior on a grid: its summaries, the grid and the density on it."""

    mean: float
    sd: float
    lower: float
    upper: float
    probability: float
    """The posterior probability of the null region (effects) or above the null (proportion)."""
    grid: FloatArray
    density: FloatArray


class _Sample(NamedTuple):
    """One group reduced to its finite values, with its mean and sample standard deviation."""

    label: str
    values: FloatArray
    mean: float
    sd: float
    n: int


class _Effect(NamedTuple):
    """What the three effect designs hand the Bayes factor and the posterior."""

    t: float
    df: float
    n_eff: float
    title: str


def run_bayes(request: BaseModel) -> ToolResult:
    """Run the Bayesian test on its own request.

    Raises:
        ToolError: ``INVALID_OPTION`` when the request is not a :class:`BayesRequest`.
    """
    if not isinstance(request, BayesRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"bayes takes a BayesRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: BayesRequest) -> ToolResult:
    """The test itself, once the request is known to be this tool's own."""
    check_level(request.level)
    _check_options(request)
    _check_fields(request)
    if request.design == "proportion":
        return _proportion(request)
    return _effect(request)


def _check_options(request: BayesRequest) -> None:
    """Refuse an option outside what the tool takes.

    Raises:
        ToolError: ``INVALID_OPTION`` for a scale that is not above 0, a null region whose
            bounds are not finite and ordered, or a beta prior parameter that is not above 0.
    """
    if not math.isfinite(request.prior_scale) or request.prior_scale <= 0.0:
        raise ToolError(
            "INVALID_OPTION",
            f"the prior scale must be a finite number above 0, not {request.prior_scale}",
        )
    low, high = request.rope
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolError(
            "INVALID_OPTION", f"the null region must hold finite bounds, not {request.rope}"
        )
    if not low < high:
        raise ToolError(
            "INVALID_OPTION",
            f"the null region's low bound must lie below its high bound, not {low} and {high}",
        )
    if not math.isfinite(request.null):
        raise ToolError(
            "INVALID_OPTION", f"the hypothesised value must be finite, not {request.null}"
        )
    if request.design == "proportion" and (request.prior_alpha <= 0.0 or request.prior_beta <= 0.0):
        raise ToolError(
            "INVALID_OPTION",
            "the beta prior's parameters must both be above 0, not "
            f"{request.prior_alpha} and {request.prior_beta}",
        )


def _check_fields(request: BayesRequest) -> None:
    """Refuse a field the chosen design does not take.

    Raises:
        ToolError: ``INVALID_OPTION`` naming the field that does not fit the design.
    """
    if request.design == "proportion":
        given = [
            name
            for name, value in (("a", request.a), ("b", request.b), ("groups", request.groups))
            if value is not None
        ]
        if given:
            raise ToolError(
                "INVALID_OPTION",
                f"the proportion design takes counts, not a column ({', '.join(given)} was "
                "given); it needs successes and trials",
            )
        if request.successes is None or request.trials is None:
            raise ToolError(
                "INVALID_OPTION", "the proportion design needs both successes and trials"
            )
        if request.trials < 1:
            raise ToolError(
                "INVALID_OPTION", f"the number of trials must be at least 1, not {request.trials}"
            )
        if not 0 <= request.successes <= request.trials:
            raise ToolError(
                "INVALID_OPTION",
                f"successes must lie between 0 and the {request.trials} trials, not "
                f"{request.successes}",
            )
        if not 0.0 < request.null < 1.0:
            raise ToolError(
                "INVALID_OPTION",
                f"the hypothesised probability must lie between 0 and 1, not {request.null}",
            )
        if request.rope != ROPE:
            raise ToolError(
                "INVALID_OPTION",
                "the proportion design has no standardised effect, so it takes no region of "
                "practical equivalence",
            )
        if request.prior_scale != PRIOR_SCALE:
            raise ToolError(
                "INVALID_OPTION",
                "the proportion design takes a beta prior, not the JZS prior's scale",
            )
        return

    if request.successes is not None or request.trials is not None:
        raise ToolError(
            "INVALID_OPTION",
            "the effect designs take columns, not counts; successes and trials are for the "
            "proportion design",
        )
    if request.prior_alpha != 1.0 or request.prior_beta != 1.0:
        raise ToolError(
            "INVALID_OPTION",
            "the effect designs take the JZS prior, not a beta prior; prior_alpha and "
            "prior_beta are for the proportion design",
        )
    if request.a is None:
        raise ToolError("INVALID_OPTION", f"the {request.design} design needs its first column")
    if request.design == "one_sample":
        if request.b is not None:
            raise ToolError(
                "INVALID_OPTION",
                "the one-sample design takes one column; the paired design takes the second "
                "column of a pair",
            )
        if request.groups is not None:
            raise ToolError(
                "INVALID_OPTION", "the one-sample design takes one column, not a group column"
            )
    elif request.design == "paired":
        if request.b is None:
            raise ToolError("INVALID_OPTION", "the paired design needs both columns of the pair")
        if request.groups is not None:
            raise ToolError(
                "INVALID_OPTION", "the paired design pairs two columns; it takes no group column"
            )
    else:
        if request.b is None and request.groups is None:
            raise ToolError(
                "INVALID_OPTION",
                "the independent design takes a group column with two levels, or a second "
                "column, and neither was given",
            )
        if request.b is not None and request.groups is not None:
            raise ToolError(
                "INVALID_OPTION",
                "the independent design takes a group column or a second column, not both",
            )


def _sample(label: str, values: FloatArray) -> _Sample:
    """One group reduced to its finite values, with its mean and sample SD."""
    flat = np.asarray(values, dtype=np.float64)
    flat = flat[np.isfinite(flat)]
    n = int(flat.size)
    if n == 0:
        return _Sample(label=str(label), values=flat, mean=0.0, sd=0.0, n=0)
    sd = float(np.std(flat, ddof=1)) if n >= 2 else 0.0
    return _Sample(label=str(label), values=flat, mean=float(np.mean(flat)), sd=sd, n=n)


def _finite_column(column: Column) -> tuple[FloatArray, tuple[ExcludedRow, ...]]:
    """One independent column's finite values, and the positions it leaves out and why."""
    values = np.asarray(column.values, dtype=np.float64)
    finite = np.isfinite(values)
    excluded: list[ExcludedRow] = []
    for at in np.flatnonzero(~finite):
        value = float(values[at])
        reason = (
            f"missing in {column.name}"
            if math.isnan(value)
            else f"not a finite number in {column.name}"
        )
        excluded.append(ExcludedRow(index=int(column.rows[at]), reason=reason))
    return values[finite], tuple(excluded)


def _row_count(used: int, excluded: int) -> Counts:
    """The counts a result carries: what was offered, used and left out."""
    return Counts(n_offered=used + excluded, n_used=used, n_excluded=excluded)


def _two_levels(groups: GroupColumn) -> tuple[str, str]:
    """The group column's two labels, in order of first appearance.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` unless there are exactly two.
    """
    found, first = np.unique(groups.labels, return_index=True)
    ordered = found[np.argsort(first)]
    if ordered.size != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the group column {groups.name!r} holds {ordered.size} "
            f"{'label' if ordered.size == 1 else 'labels'}, and this design compares exactly "
            "two",
        )
    return str(ordered[0]), str(ordered[1])


def _require_two(sample: _Sample, where: str) -> None:
    """Refuse a sample with fewer than the two values a spread needs.

    Raises:
        ToolError: ``TOO_FEW_ROWS`` naming the sample.
    """
    if sample.n < MIN_OBSERVATIONS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"{where} needs at least {MIN_OBSERVATIONS} values, and it holds {sample.n}",
        )


def _require_one(sample: _Sample, where: str) -> None:
    """Refuse a group with no usable value.

    Raises:
        ToolError: ``TOO_FEW_ROWS`` naming the group.
    """
    if sample.n < 1:
        raise ToolError(
            "TOO_FEW_ROWS", f"{where} holds no usable value, and each group needs at least 1"
        )


def _require_variation(sd: float, label: str) -> None:
    """Refuse a spread of zero.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when the values do not vary.
    """
    if sd == 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"{label} holds one value repeated, so the standardised effect is undefined",
        )


def _prior_density(g: float) -> float:
    """The JZS prior density of the mixing variable ``g``: inverse chi-square on one df."""
    return float((2.0 * math.pi) ** -0.5 * g**-1.5 * math.exp(-1.0 / (2.0 * g)))


def _likelihood_ratio(g: float, t: float, n_eff: float, df: float, scale: float) -> float:
    """The integrand of the JZS Bayes factor: the ratio at ``g``, times the prior density."""
    variance = 1.0 + n_eff * scale * scale * g
    return float(
        variance**-0.5 * (1.0 + t * t / (variance * df)) ** (-(df + 1.0) / 2.0) * _prior_density(g)
    )


def _integrate_g(integrand: Callable[[float], float]) -> tuple[float, str | None]:
    """Integrate ``integrand`` over g in (0, infinity), split at 1, at the stated tolerance.

    Returns the value and a note when a part did not converge (``None`` when it did). The
    split keeps the peak at g = 1/3 and the long tail on separate intervals, which is what
    makes the tolerance of :data:`INTEGRATION_TOLERANCE` reachable.
    """
    total = 0.0
    notes: list[str] = []
    for low, high in ((0.0, 1.0), (1.0, np.inf)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            part, error, info = integrate.quad(
                integrand,
                low,
                high,
                epsabs=INTEGRATION_TOLERANCE,
                epsrel=INTEGRATION_TOLERANCE,
                limit=INTEGRATION_LIMIT,
                full_output=True,
            )
        total += float(part)
        message = info.get("message")
        if caught or message is not None:
            notes.append(
                f"the integration over g from {low:g} to {high:g} did not reach the "
                f"{INTEGRATION_TOLERANCE:g} tolerance (estimated error {error:.3g}); the "
                "Bayes factor is the value at the last subdivision"
            )
    return total, "; ".join(notes) if notes else None


def jzs_bayes_factor(t: float, n_eff: float, df: float, scale: float = PRIOR_SCALE) -> float:
    """The JZS Bayes factor for a two-sided t, in favour of the alternative over the null.

    This is Rouder et al.'s (2009) Equation 1: the integral over the inverse-chi-square
    mixing variable of the t likelihood ratio under a normal prior on the standardised
    effect, divided by the null model's central-t kernel. ``n_eff`` is the sample size for a
    one-sample or paired design and ``n1 n2 / (n1 + n2)`` for two independent samples, with
    ``df`` the design's degrees of freedom. ``scale`` is the Cauchy prior's scale.

    Raises:
        ValueError: ``df`` below 1, so the t distribution the factor is built on has no
            degrees of freedom.
    """
    value, _note = _bayes_factor_with_note(t, n_eff, df, scale)
    return value


def _bayes_factor_with_note(
    t: float, n_eff: float, df: float, scale: float
) -> tuple[float, str | None]:
    """The JZS Bayes factor and the integration's convergence note.

    Raises:
        ValueError: ``df`` below 1, so the t distribution the factor is built on has no
            degrees of freedom.
    """
    if df < 1.0:
        raise ValueError(f"the JZS Bayes factor needs at least one degree of freedom, not {df}")
    numerator, note = _integrate_g(lambda g: _likelihood_ratio(g, t, n_eff, df, scale))
    null = (1.0 + t * t / df) ** (-(df + 1.0) / 2.0)
    return numerator / null, note


def _normalised(grid: FloatArray, density: FloatArray) -> tuple[FloatArray, float]:
    """The density normalised to integrate to 1 on the grid, and that integral before it was.

    Raises:
        ValueError: the density carries no mass on the grid.
    """
    mass = float(np.trapezoid(density, grid))
    if not mass > 0.0:
        raise ValueError("the posterior density carries no mass on its grid")
    return density / mass, mass


def _quantile(cdf: FloatArray, grid: FloatArray, probability: float) -> float:
    """The grid point at which the cumulative mass reaches ``probability``, interpolated."""
    return float(np.interp(probability, cdf, grid))


def _posterior(
    grid: FloatArray, density: FloatArray, level: float, rope: tuple[float, float]
) -> _Posterior:
    """The summaries of a posterior given on a grid, normalised on that grid.

    The credible interval is equal-tailed: the grid's CDF is interpolated at the two tails
    the level leaves. The probability is the grid's share inside ``rope``.
    """
    normalised, _mass = _normalised(grid, density)
    mean = float(np.trapezoid(grid * normalised, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * normalised, grid))
    increments = 0.5 * (normalised[1:] + normalised[:-1]) * np.diff(grid)
    cdf = np.concatenate((np.zeros(1), np.cumsum(increments)))
    cdf = cdf / cdf[-1]
    alpha = 1.0 - level
    low, high = rope
    inside = (grid >= low) & (grid <= high)
    probability = float(np.trapezoid(normalised[inside], grid[inside])) if inside.size else 0.0
    return _Posterior(
        mean=mean,
        sd=math.sqrt(max(variance, 0.0)),
        lower=_quantile(cdf, grid, alpha / 2.0),
        upper=_quantile(cdf, grid, 1.0 - alpha / 2.0),
        probability=probability,
        grid=grid,
        density=normalised,
    )


def _effect_posterior(
    t: float, n_eff: float, df: float, scale: float, level: float, rope: tuple[float, float]
) -> _Posterior:
    """The JZS posterior of the standardised effect on a grid, summarised.

    The density is the non-central t likelihood of the observed t at non-centrality
    ``delta * sqrt(n_eff)`` times the Cauchy prior on ``delta``; that is the posterior the
    JZS prior implies, not the posterior a flat prior would give. The grid runs from
    ``min(-3 scale, ...)`` to its symmetric maximum, wide enough to hold the posterior.
    """
    centroid = t / math.sqrt(n_eff)
    spread = 1.0 / math.sqrt(n_eff)
    half = max(3.0 * scale, abs(centroid) + 8.0 * spread, 4.0)
    grid = np.linspace(-half, half, GRID_POINTS)
    density = scipy_stats.nct.pdf(t, df, grid * math.sqrt(n_eff)) * scipy_stats.cauchy.pdf(
        grid, loc=0.0, scale=scale
    )
    # SciPy gives NaN for a non-centrality so far from t that the density has underflowed;
    # the density there is 0, not undefined, so the grid carries no NaN into the result.
    density = np.where(np.isfinite(density), density, 0.0)
    return _posterior(grid, np.asarray(density, dtype=np.float64), level, rope)


def _effect_result(
    effect: _Effect,
    request: BayesRequest,
    used: int,
    excluded_rows: tuple[ExcludedRow, ...],
) -> ToolResult:
    """The result of one of the three effect designs."""
    bf10, note = _bayes_factor_with_note(effect.t, effect.n_eff, effect.df, request.prior_scale)
    posterior = _effect_posterior(
        effect.t,
        effect.n_eff,
        effect.df,
        request.prior_scale,
        request.level,
        request.rope,
    )
    low, high = request.rope
    listed, unlisted = list_excluded(excluded_rows)
    plot = _density_plot("Standardised effect delta", posterior)
    estimate = Estimate(
        label="Standardised effect delta (posterior mean; credible interval)",
        value=posterior.mean,
        lower=posterior.lower,
        upper=posterior.upper,
    )
    rope_estimate = Estimate(
        label="Posterior probability of the null region", value=posterior.probability
    )
    test = _jzs_test(effect, bf10)
    return ToolResult(
        tool="bayes",
        title=effect.title,
        level=request.level,
        estimates=(estimate, rope_estimate),
        tests=(test,),
        assumptions=(_effect_prior(request.prior_scale),),
        warnings=() if note is None else (note,),
        interpretation=_effect_words(bf10, posterior, request, low, high),
        methods_fragment=_effect_methods(request, posterior, low, high),
        counts=_row_count(used, len(excluded_rows)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(_summary_table(posterior, bf10, "ROPE probability"),),
        plots=(plot,),
    )


def _jzs_test(effect: _Effect, bf10: float) -> HypothesisTest:
    """The Bayes factor as a test block, with no p-value and the reason it has none."""
    return HypothesisTest(
        name="Bayes factor (JZS)",
        method="JZS Bayes factor for a two-sided t test (Rouder et al., 2009)",
        null_hypothesis="The standardised effect is zero.",
        statistic=effect.t,
        df=effect.df,
        p_value=None,
        p_value_text=p_text(None),
        effect=Estimate(label="Bayes factor (BF10)", value=bf10),
        reason=(
            "a Bayes factor is not a p-value: it is the ratio of the evidence under the two "
            "hypotheses, and it has no sampling distribution"
        ),
    )


def _effect_prior(scale: float) -> AuditCheck:
    """The prior as an assumption: the evidence is read under it, so it is stated."""
    return assumption(
        "prior",
        "The evidence is read under the stated prior",
        Verdict.PASS,
        f"JZS prior: a Cauchy distribution with scale {num(scale)} on the standardised effect "
        "(Rouder et al., 2009).",
    )


def _proportion_prior(alpha: float, beta: float) -> AuditCheck:
    """The beta prior as an assumption, named by its parameters."""
    return assumption(
        "prior",
        "The evidence is read under the stated prior",
        Verdict.PASS,
        f"Beta({num(alpha)}, {num(beta)}) prior on the proportion.",
    )


def _summary_table(
    posterior: _Posterior,
    bf10: float,
    probability_label: str,
    extra: tuple[tuple[str, float], ...] = (),
) -> Table:
    """The posterior summary: one quantity per row, and any design counts before them."""
    columns: tuple[tuple[str, str, ColumnKind], ...] = (
        ("quantity", "Quantity", "text"),
        ("value", "Value", "number"),
    )
    rows: list[tuple[str, float]] = [
        *extra,
        ("Posterior mean", posterior.mean),
        ("Posterior standard deviation", posterior.sd),
        ("Credible lower", posterior.lower),
        ("Credible upper", posterior.upper),
        ("Bayes factor (BF10)", bf10),
        (probability_label, posterior.probability),
    ]
    return table(
        "Posterior summary",
        columns,
        rows,
        note=(
            "The credible interval is equal-tailed: the two tails the level leaves, read from "
            "the posterior grid's cumulative mass. The Bayes factor compares the alternative "
            "with the null under the stated prior."
        ),
    )


def _density_plot(x_label: str, posterior: _Posterior) -> MultiLinePlot:
    """The posterior density, thinned to the most points a plot carries, with a note."""
    points, note = stride(posterior.grid, limit=PLOT_POINTS)
    density, density_note = stride(posterior.density, limit=PLOT_POINTS)
    labels = tuple(f"{value:.4g}" for value in points)
    return MultiLinePlot(
        title="Posterior density",
        x_label=x_label,
        y_label="Posterior density",
        series=(
            XYSeries(
                label="Posterior",
                x=tuple(float(value) for value in points),
                y=tuple(float(value) for value in density),
            ),
        ),
        x_labels=labels,
        note=note if note is not None else density_note,
    )


def _band(magnitude: float) -> str:
    """The conventional name of a Bayes factor's band, naming the bands as conventions."""
    moderate, strong, very_strong = BF_BANDS
    if magnitude < moderate:
        return "anecdotal"
    if magnitude < strong:
        return "moderate"
    if magnitude < very_strong:
        return "strong"
    return "very strong"


def _evidence_sentence(bf10: float) -> str:
    """What the Bayes factor says, with the side and the conventional bands named."""
    lower = bf10 < 1.0
    magnitude = 1.0 / bf10 if lower else bf10
    subject = "the null" if lower else "the alternative"
    return (
        f"The Bayes factor (BF10) is {num(bf10)}, which is {_band(magnitude)} evidence for "
        f"{subject} by the conventional 3, 10 and 30 bands (conventions, not cut-offs)."
    )


def _effect_words(
    bf10: float,
    posterior: _Posterior,
    request: BayesRequest,
    low: float,
    high: float,
) -> str:
    """One to three sentences: the Bayes factor, the posterior and what neither proves."""
    percent = num(request.level * 100)
    first = (
        f"{_evidence_sentence(bf10)} The posterior mean of the standardised effect is "
        f"{num(posterior.mean)}, with a {percent} % credible interval from "
        f"{num(posterior.lower)} to {num(posterior.upper)}."
    )
    second = (
        f"The posterior probability of the null region from {num(low)} to {num(high)} is "
        f"{num(posterior.probability)}, under the Cauchy prior of scale "
        f"{num(request.prior_scale)}."
    )
    third = (
        "These are evidence under the stated prior, not a demonstration that either "
        "hypothesis is true."
    )
    return f"{first} {second} {third}"


def _effect_methods(request: BayesRequest, posterior: _Posterior, low: float, high: float) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "The Bayes factor is the JZS Bayes factor for a two-sided t test (Rouder, Speckman, "
        "Sun, Morey, and Iverson, 2009, Psychonomic Bulletin and Review 16(2), 225-237), "
        f"with a Cauchy prior of scale {num(request.prior_scale)} on the standardised effect; "
        "the integral over the inverse-chi-square mixing variable was evaluated with SciPy's "
        f"adaptive quadrature at a tolerance of {INTEGRATION_TOLERANCE:g}. The posterior, its "
        f"{num(request.level * 100)} % equal-tailed credible interval and the null region's "
        "probability are read from a grid under that same prior, not under a flat prior; the "
        f"null region is the standardised effect from {num(low)} to {num(high)}."
    )


def _one_sample(request: BayesRequest) -> ToolResult:
    """The one-sample design: a column against a hypothesised mean."""
    assert request.a is not None
    pairing = pairs(request.a)
    column = pairing.columns[0]
    sample = _sample(column.name, column.values)
    _require_two(sample, f"the one-sample design on {column.name!r}")
    _require_variation(sample.sd, f"the column {column.name!r}")
    se = sample.sd / math.sqrt(sample.n)
    t = (sample.mean - request.null) / se
    effect = _Effect(
        t=t,
        df=float(sample.n - 1),
        n_eff=float(sample.n),
        title=f"Bayesian one-sample test of {sample.label}",
    )
    return _effect_result(effect, request, sample.n, pairing.excluded)


def _paired(request: BayesRequest) -> ToolResult:
    """The paired design: the differences of two columns, complete pairs only."""
    assert request.a is not None and request.b is not None
    pairing = pairs(request.a, request.b)
    left, right = pairing.columns
    differences = _sample(f"{left.name} less {right.name}", left.values - right.values)
    _require_two(differences, "the paired design")
    _require_variation(differences.sd, f"the differences of {left.name!r} and {right.name!r}")
    se = differences.sd / math.sqrt(differences.n)
    t = (differences.mean - request.null) / se
    effect = _Effect(
        t=t,
        df=float(differences.n - 1),
        n_eff=float(differences.n),
        title=f"Bayesian paired test of {left.name} and {right.name}",
    )
    return _effect_result(effect, request, differences.n, pairing.excluded)


def _independent(request: BayesRequest) -> ToolResult:
    """The independent design: two groups given as two columns or as a group column."""
    if request.groups is not None:
        samples, excluded_rows, title = _group_samples(request)
    else:
        assert request.a is not None and request.b is not None
        first_values, first_excluded = _finite_column(request.a)
        second_values, second_excluded = _finite_column(request.b)
        samples = [
            _sample(request.a.name, first_values),
            _sample(request.b.name, second_values),
        ]
        excluded_rows = tuple(
            sorted((*first_excluded, *second_excluded), key=lambda row: row.index)
        )
        title = f"Bayesian two-sample test of {request.a.name} and {request.b.name}"
    first, second = samples
    _require_one(first, f"the group {first.label!r}")
    _require_one(second, f"the group {second.label!r}")
    total = first.n + second.n
    if total < MIN_TWO_SAMPLE:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the two-sample design needs at least {MIN_TWO_SAMPLE} values together, and it "
            f"has {total}",
        )
    df = float(total - 2)
    pooled_variance = ((first.n - 1) * first.sd**2 + (second.n - 1) * second.sd**2) / df
    pooled_sd = math.sqrt(pooled_variance)
    _require_variation(
        pooled_sd,
        f"the groups {first.label!r} and {second.label!r}",
    )
    se = pooled_sd * math.sqrt(1.0 / first.n + 1.0 / second.n)
    t = (first.mean - second.mean - request.null) / se
    n_eff = first.n * second.n / total
    effect = _Effect(t=t, df=df, n_eff=float(n_eff), title=title)
    return _effect_result(effect, request, total, excluded_rows)


def _group_samples(
    request: BayesRequest,
) -> tuple[list[_Sample], tuple[ExcludedRow, ...], str]:
    """The two samples of an independent design given as a value column and a group column.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` unless the group column holds exactly two levels;
            ``TOO_FEW_ROWS`` when either level holds no value at all.
    """
    assert request.a is not None and request.groups is not None
    labels = _two_levels(request.groups)
    pairing = aligned(request.a, request.groups)
    values = pairing.values.values
    found = pairing.groups[0].labels
    samples = [_sample(label, values[found == label]) for label in labels]
    for sample in samples:
        if sample.n == 0:
            raise ToolError(
                "TOO_FEW_ROWS",
                f"the group {sample.label!r} holds no usable value, and each group needs at "
                f"least 1",
            )
    title = f"Bayesian two-sample test of {request.a.name} by {request.groups.name}"
    return samples, pairing.excluded, title


def _effect(request: BayesRequest) -> ToolResult:
    """Dispatch the three effect designs."""
    if request.design == "one_sample":
        return _one_sample(request)
    if request.design == "paired":
        return _paired(request)
    return _independent(request)


def _proportion(request: BayesRequest) -> ToolResult:
    """The proportion design: a beta-binomial posterior and its Savage-Dickey factor."""
    assert request.successes is not None and request.trials is not None
    successes, trials = request.successes, request.trials
    failures = trials - successes
    prior = scipy_stats.beta(request.prior_alpha, request.prior_beta)
    posterior = scipy_stats.beta(request.prior_alpha + successes, request.prior_beta + failures)
    mean = float(posterior.mean())
    sd = float(posterior.std())
    alpha = 1.0 - request.level
    lower = float(posterior.ppf(alpha / 2.0))
    upper = float(posterior.ppf(1.0 - alpha / 2.0))
    above = float(posterior.sf(request.null))
    prior_density = float(prior.pdf(request.null))
    posterior_density = float(posterior.pdf(request.null))
    bf10 = prior_density / posterior_density
    grid, density = _beta_grid(request.prior_alpha, request.prior_beta, successes, failures)
    plot_posterior = _posterior(grid, density, request.level, (0.0, 1.0))
    estimate = Estimate(
        label="Proportion (posterior mean; credible interval)",
        value=mean,
        lower=lower,
        upper=upper,
    )
    above_estimate = Estimate(label="Posterior probability that p exceeds the null", value=above)
    test = HypothesisTest(
        name="Bayes factor (Savage-Dickey)",
        method="Savage-Dickey Bayes factor at the point null of a beta-binomial model",
        null_hypothesis=f"The proportion is {num(request.null)}.",
        statistic=None,
        p_value=None,
        p_value_text=p_text(None),
        effect=Estimate(label="Bayes factor (BF10)", value=bf10),
        reason=(
            "a Bayes factor is not a p-value: it is the ratio of the evidence under the two "
            "hypotheses, and it has no sampling distribution"
        ),
    )
    summary = _Posterior(
        mean=mean,
        sd=sd,
        lower=lower,
        upper=upper,
        probability=above,
        grid=plot_posterior.grid,
        density=plot_posterior.density,
    )
    title = f"Bayesian test of a proportion: {successes} of {trials}"
    return ToolResult(
        tool="bayes",
        title=title,
        level=request.level,
        estimates=(estimate, above_estimate),
        tests=(test,),
        assumptions=(_proportion_prior(request.prior_alpha, request.prior_beta),),
        interpretation=_proportion_words(bf10, summary, request),
        methods_fragment=_proportion_methods(request, successes, trials),
        counts=Counts(n_offered=0, n_used=0, n_excluded=0),
        tables=(
            _summary_table(
                summary,
                bf10,
                "Probability that p exceeds the null",
                extra=(("Successes", float(successes)), ("Trials", float(trials))),
            ),
        ),
        plots=(_density_plot("Proportion p", plot_posterior),),
    )


def _beta_grid(
    prior_alpha: float, prior_beta: float, successes: int, failures: int
) -> tuple[FloatArray, FloatArray]:
    """The beta posterior's density on a grid strictly inside (0, 1), so it is always finite."""
    posterior = scipy_stats.beta(prior_alpha + successes, prior_beta + failures)
    low = max(1e-4, float(posterior.ppf(1e-4)))
    high = min(1.0 - 1e-4, float(posterior.ppf(1.0 - 1e-4)))
    if not low < high:
        low, high = 1e-4, 1.0 - 1e-4
    grid = np.linspace(low, high, GRID_POINTS)
    return grid, np.asarray(posterior.pdf(grid), dtype=np.float64)


def _proportion_words(bf10: float, posterior: _Posterior, request: BayesRequest) -> str:
    """One to three sentences: the Bayes factor, the posterior and what neither proves."""
    percent = num(request.level * 100)
    first = (
        f"{_evidence_sentence(bf10)} The posterior proportion is {num(posterior.mean)}, with "
        f"a {percent} % credible interval from {num(posterior.lower)} to "
        f"{num(posterior.upper)}."
    )
    second = (
        f"The posterior probability that the proportion exceeds {num(request.null)} is "
        f"{num(posterior.probability)}, under the Beta({num(request.prior_alpha)}, "
        f"{num(request.prior_beta)}) prior."
    )
    third = (
        "These are evidence under the stated prior, not a demonstration that either "
        "hypothesis is true."
    )
    return f"{first} {second} {third}"


def _proportion_methods(request: BayesRequest, successes: int, trials: int) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        f"A Beta({num(request.prior_alpha)}, {num(request.prior_beta)}) prior on the "
        f"proportion, updated by {successes:,} successes in {trials:,} trials, gives the "
        f"Beta({num(request.prior_alpha + successes)}, "
        f"{num(request.prior_beta + trials - successes)}) posterior; its credible interval is "
        "the equal-tailed beta interval. The Bayes factor at the point null compares the "
        "prior density with the posterior density there (the Savage-Dickey ratio), and is "
        "reported in favour of the alternative."
    )


SPEC = ToolSpec(
    name="bayes",
    family="bayes",
    title="Bayesian test",
    request=BayesRequest,
    run=run_bayes,
)
TOOLS.register(SPEC)
