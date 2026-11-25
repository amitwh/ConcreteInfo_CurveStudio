"""Meta-analysis: several studies' effects pooled by inverse variance.

Every study gives one effect on an analysis scale and its standard error. The fixed-effect
pool is the inverse-variance weighted mean ``sum(w_i e_i) / sum(w_i)`` with ``w_i = 1 /
se_i^2``; its interval is the normal interval from ``sqrt(1 / sum(w_i))``. Cochran's
heterogeneity statistic is ``Q = sum(w_i (e_i - e_fixed)^2)``, read against the chi-square
distribution on ``k - 1`` degrees of freedom.

The random-effects pool uses the DerSimonian and Laird (1986) estimate of the between-study
variance, ``tau^2 = max(0, (Q - (k - 1)) / (sum(w_i) - sum(w_i^2) / sum(w_i)))``, which is
added to every study's variance before the weights are recomputed; the same normal interval
follows from the reweighted pool. I^2 (Higgins and Thompson 2002) is ``max(0, (Q - (k - 1)) /
Q) * 100`` per cent, the share of the variation between studies that is not sampling error.
A prediction interval for a new study's effect, ``pool +/- t * sqrt(tau^2 + se_pool^2)`` on
``k - 2`` degrees of freedom (Higgins, Thompson and Spiegelhalter 2009), is reported from
three studies; below three it is not applicable.

Egger's test (Egger et al. 1997) bears on publication bias and runs only from ten studies:
the effects are regressed on their standard errors with weights ``1 / se_i^2``, and the
intercept's t test on ``k - 2`` degrees of freedom is reported. A funnel plot shows the
effects against the standard errors, with the axis inverted as a funnel plot is read.

The measure names what the effects are. ``generic``, ``mean_difference``,
``standardised_mean_difference`` and ``correlation`` analyse the given effects on their own
scale. ``odds_ratio`` and ``risk_ratio`` instead read two-by-two tables (events and totals
in two groups): the study effect is the log ratio and its standard error the standard
formula, and a zero or complete cell is refused rather than continuity-corrected. No
transformation is applied to a correlation.

The studies are assumed independent, which a column cannot record, and publication bias
cannot be ruled out from a funnel plot. A row missing any of the columns is left out and
listed, never dropped without a word.
"""

import math
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
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
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    Line,
    MultiLinePlot,
    Plot,
    ScatterPlot,
    Table,
    XYSeries,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

Measure = Literal[
    "generic",
    "mean_difference",
    "standardised_mean_difference",
    "odds_ratio",
    "risk_ratio",
    "correlation",
]

MIN_STUDIES = 2
"""The fewest studies a pool is defined for."""
MAX_STUDIES = 200
"""The most studies the tool takes."""
EGGER_MIN_STUDIES = 10
"""Egger's test is not read from fewer studies than this."""
PREDICTION_MIN_STUDIES = 3
"""The prediction interval needs t on k - 2 degrees of freedom, so three studies."""
WARN_STUDIES = 5
"""Fewer studies than this make tau squared and I squared unstable enough to warn about."""
HIGH_HETEROGENEITY = 75.0
"""I squared above this percentage is most of the variation between studies."""
MODERATE_HETEROGENEITY = 25.0
"""I squared above this percentage is a substantial part of the variation."""

_RATIO_MEASURES = ("odds_ratio", "risk_ratio")

_MEASURE_NOUNS: dict[str, str] = {
    "odds_ratio": "odds ratio",
    "risk_ratio": "risk ratio",
}

_MEASURE_ARTICLES: dict[str, str] = {
    "odds_ratio": "an",
    "risk_ratio": "a",
}

_EFFECT_LABELS: dict[str, str] = {
    "generic": "Effect",
    "mean_difference": "Mean difference",
    "standardised_mean_difference": "Standardised mean difference",
    "odds_ratio": "Log odds ratio",
    "risk_ratio": "Log risk ratio",
    "correlation": "Correlation",
}

_STUDY_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("study", "Study", "text"),
    ("effect", "Effect", "number"),
    ("se", "SE", "number"),
    ("weight_fixed", "Weight fixed %", "number"),
    ("weight_random", "Weight random %", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)

_POOLED_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("model", "Model", "text"),
    ("estimate", "Estimate", "number"),
    ("se", "SE", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
    ("tau_squared", "Tau squared", "number"),
)


class MetaRequest(BaseModel):
    """The studies, their effects and standard errors, or their two-by-two tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    effects: tuple[Column, ...] = Field(
        default=(),
        description=(
            "The one column of study effects, one row per study, on the analysis scale. "
            "Leave it empty when measure is odds_ratio or risk_ratio."
        ),
    )
    standard_errors: tuple[Column, ...] = Field(
        default=(),
        description=(
            "The one column of standard errors, in the same rows as the effects. Leave it "
            "empty when measure is odds_ratio or risk_ratio."
        ),
    )
    labels: GroupColumn | None = Field(
        default=None,
        description=(
            "An optional column of study names, aligned with the effect rows by position. "
            "Left empty, the studies are numbered in the order they are used."
        ),
        json_schema_extra={"label": "Study labels"},
    )
    measure: Measure = Field(
        default="generic",
        description=(
            "What the effects measure. generic, mean_difference, "
            "standardised_mean_difference and correlation read the effects as given. "
            "odds_ratio and risk_ratio read the four event and total columns instead and "
            "analyse the log ratio."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for every interval, between 0 and 1.",
    )
    events_a: Column | None = Field(
        default=None,
        description="The events in group A, one row per study; for a ratio measure only.",
        json_schema_extra={"label": "Events (group A)"},
    )
    n_a: Column | None = Field(
        default=None,
        description=(
            "The patients or units in group A, one row per study; for a ratio measure only."
        ),
        json_schema_extra={"label": "Total (group A)"},
    )
    events_b: Column | None = Field(
        default=None,
        description="The events in group B, one row per study; for a ratio measure only.",
        json_schema_extra={"label": "Events (group B)"},
    )
    n_b: Column | None = Field(
        default=None,
        description=(
            "The patients or units in group B, one row per study; for a ratio measure only."
        ),
        json_schema_extra={"label": "Total (group B)"},
    )


class _Raw(NamedTuple):
    """The aligned columns a request reads, the studies they name, and the rows left out."""

    values: tuple[FloatArray, ...]
    names: tuple[str, ...]
    offered: int
    excluded: tuple[ExcludedRow, ...]


class _Studies(NamedTuple):
    """The analysis-scale effects and errors of the studies that take part."""

    effects: FloatArray
    errors: FloatArray
    names: tuple[str, ...]
    offered: int
    excluded: tuple[ExcludedRow, ...]
    effect_label: str
    measure: str
    log_scale: bool


class _Pool(NamedTuple):
    """Both pools, their intervals, the heterogeneity numbers and the weights."""

    fixed: float
    fixed_se: float
    fixed_lower: float
    fixed_upper: float
    random: float
    random_se: float
    random_lower: float
    random_upper: float
    tau2: float
    i2: float
    q: float
    q_p: float
    weights_fixed: FloatArray
    weights_random: FloatArray
    prediction_lower: float | None
    prediction_upper: float | None


class _Egger(NamedTuple):
    """Egger's weighted regression: its intercept, its standard error and the t test."""

    intercept: float
    intercept_se: float
    statistic: float
    p_value: float


# --------------------------------------------------------------------------- the reading


def _percent(level: float) -> str:
    return f"{num(level * 100)} %"


def _z(level: float) -> float:
    """The two-sided normal quantile at ``level``."""
    return float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))


def _measure_noun(measure: str) -> str:
    return _MEASURE_NOUNS[measure]


def _offered(columns: tuple[Column, ...], labels: GroupColumn | None) -> int:
    """The distinct positions the given columns and the labels cover."""
    arrays = [column.rows for column in columns]
    if labels is not None:
        arrays.append(labels.rows)
    return int(np.unique(np.concatenate(arrays)).size)


def _aligned(columns: tuple[Column, ...], labels: GroupColumn | None) -> _Raw:
    """The value columns aligned across the rows they all hold, and the rows left out.

    A row missing a value in any column is listed as ``missing in <name>`` or ``not a finite
    number in <name>``. With labels, a row without one is listed as ``no group label in
    <name>``, a label-only row as ``missing in`` the first value column, and the labels name
    the studies; without labels the studies are numbered in the order they are used.
    """
    pairing = pairs(*columns)
    positions = pairing.columns[0].rows
    values = tuple(np.asarray(column.values, dtype=np.float64) for column in pairing.columns)
    excluded = list(pairing.excluded)
    offered = _offered(columns, labels)
    if labels is None:
        names = tuple(f"Study {index + 1}" for index in range(positions.size))
        return _Raw(values, names, offered, tuple(excluded))
    by_position = dict(zip(labels.rows.tolist(), labels.labels.tolist(), strict=True))
    covered = np.array([position in by_position for position in positions.tolist()], dtype=bool)
    label_only = sorted(set(labels.rows.tolist()) - set(positions.tolist()))
    excluded.extend(
        ExcludedRow(index=int(position), reason=f"missing in {columns[0].name}")
        for position in label_only
    )
    excluded.extend(
        ExcludedRow(index=int(position), reason=f"no group label in {labels.name}")
        for position in positions[~covered]
    )
    excluded.sort(key=lambda row: row.index)
    kept = tuple(value[covered] for value in values)
    names = tuple(str(by_position[int(position)]) for position in positions[covered])
    return _Raw(kept, names, offered, tuple(excluded))


def _read(request: MetaRequest) -> _Raw:
    """The request's columns, read as the measure says.

    Raises:
        ToolError: ``INVALID_OPTION`` for the wrong set of columns for the measure, or a
            column count other than one effect and one standard error.
    """
    if request.measure in _RATIO_MEASURES:
        if request.effects or request.standard_errors:
            raise ToolError(
                "INVALID_OPTION",
                "a ratio measure reads events and totals, so leave effects and "
                "standard_errors empty",
            )
        events_a, n_a, events_b, n_b = request.events_a, request.n_a, request.events_b, request.n_b
        if events_a is None or n_a is None or events_b is None or n_b is None:
            given = (
                ("events_a", events_a),
                ("n_a", n_a),
                ("events_b", events_b),
                ("n_b", n_b),
            )
            missing = [name for name, column in given if column is None]
            raise ToolError(
                "INVALID_OPTION",
                f"the {_measure_noun(request.measure)} measure needs the four columns "
                f"events_a, n_a, events_b and n_b; {joined(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing",
            )
        return _aligned((events_a, n_a, events_b, n_b), request.labels)
    if any(
        column is not None
        for column in (request.events_a, request.n_a, request.events_b, request.n_b)
    ):
        raise ToolError(
            "INVALID_OPTION",
            f"the {request.measure} measure reads effects and standard errors, so leave the "
            "event and total columns empty",
        )
    if len(request.effects) != 1 or len(request.standard_errors) != 1:
        raise ToolError(
            "INVALID_OPTION",
            "meta-analysis takes one effects column and one standard-error column, and it "
            f"was given {len(request.effects)} and {len(request.standard_errors)}",
        )
    return _aligned((request.effects[0], request.standard_errors[0]), request.labels)


def _ratios(
    values: tuple[FloatArray, ...], names: tuple[str, ...], measure: str
) -> tuple[FloatArray, FloatArray]:
    """Every study's log ratio and its standard error.

    Raises:
        ToolError: ``INVALID_OPTION`` naming the study whose two-by-two table has a zero cell
            or every unit as an event, in either group.
    """
    events_a, n_a, events_b, n_b = values
    for index, name in enumerate(names):
        pairs_of_counts = (
            ("A", float(events_a[index]), float(n_a[index])),
            ("B", float(events_b[index]), float(n_b[index])),
        )
        for group, events, total in pairs_of_counts:
            if not 0.0 < events < total:
                raise ToolError(
                    "INVALID_OPTION",
                    f"study {name!r} has {num(events)} events out of {num(total)} in group "
                    f"{group}, so its {_measure_noun(measure)} is undefined; a zero, negative "
                    "or complete cell is refused rather than corrected",
                )
    if measure == "odds_ratio":
        vs_a, vs_b = n_a - events_a, n_b - events_b
        effects = np.log(events_a * vs_b / (vs_a * events_b))
        errors = np.sqrt(1.0 / events_a + 1.0 / vs_a + 1.0 / events_b + 1.0 / vs_b)
    else:
        effects = np.log((events_a / n_a) / (events_b / n_b))
        errors = np.sqrt(1.0 / events_a - 1.0 / n_a + 1.0 / events_b - 1.0 / n_b)
    return np.asarray(effects, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def _studies(raw: _Raw, measure: str) -> _Studies:
    """The analysis-scale effects and errors, once every study is valid.

    Raises:
        ToolError: ``INVALID_OPTION`` naming a study whose standard error is zero or below,
            or whose standard error is too small to weigh.
    """
    if measure in _RATIO_MEASURES:
        effects, errors = _ratios(raw.values, raw.names, measure)
        log_scale = True
    else:
        effects = raw.values[0]
        errors = raw.values[1]
        log_scale = False
        for index, name in enumerate(raw.names):
            if errors[index] <= 0.0:
                raise ToolError(
                    "INVALID_OPTION",
                    f"study {name!r} has a standard error of {num(float(errors[index]))}; a "
                    "meta-analysis needs a positive standard error for every study",
                )
        weights = 1.0 / errors**2
        if not np.isfinite(weights).all():
            at = int(np.flatnonzero(~np.isfinite(weights))[0])
            raise ToolError(
                "INVALID_OPTION",
                f"study {raw.names[at]!r} has a standard error too small to weigh",
            )
    return _Studies(
        effects=effects,
        errors=errors,
        names=raw.names,
        offered=raw.offered,
        excluded=raw.excluded,
        effect_label=_EFFECT_LABELS[measure],
        measure=measure,
        log_scale=log_scale,
    )


# --------------------------------------------------------------------------- the method


def _pool(effects: FloatArray, errors: FloatArray, level: float) -> _Pool:
    """The fixed and DerSimonian-Laird random pools, and the heterogeneity numbers."""
    size = int(effects.size)
    variance = errors**2
    weights = 1.0 / variance
    total = float(np.sum(weights))
    fixed = float(np.sum(weights * effects) / total)
    fixed_se = math.sqrt(1.0 / total)
    q = float(np.sum(weights * (effects - fixed) ** 2))
    df = float(size - 1)
    q_p = float(scipy_stats.chi2.sf(q, df))
    denominator = total - float(np.sum(weights**2)) / total
    tau2 = max(0.0, (q - df) / denominator)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0.0 else 0.0
    random_weights = 1.0 / (variance + tau2)
    random_total = float(np.sum(random_weights))
    random = float(np.sum(random_weights * effects) / random_total)
    random_se = math.sqrt(1.0 / random_total)
    z = _z(level)
    if size >= PREDICTION_MIN_STUDIES:
        t = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, size - 2))
        spread = math.sqrt(tau2 + random_se**2)
        prediction_lower: float | None = random - t * spread
        prediction_upper: float | None = random + t * spread
    else:
        prediction_lower, prediction_upper = None, None
    return _Pool(
        fixed=fixed,
        fixed_se=fixed_se,
        fixed_lower=fixed - z * fixed_se,
        fixed_upper=fixed + z * fixed_se,
        random=random,
        random_se=random_se,
        random_lower=random - z * random_se,
        random_upper=random + z * random_se,
        tau2=tau2,
        i2=i2,
        q=q,
        q_p=q_p,
        weights_fixed=weights / total * 100.0,
        weights_random=random_weights / random_total * 100.0,
        prediction_lower=prediction_lower,
        prediction_upper=prediction_upper,
    )


def _egger(effects: FloatArray, errors: FloatArray) -> _Egger | None:
    """Egger's weighted regression of the effect on its standard error, or none when it is
    singular (every study's standard error equal, so the slope has nothing to read)."""
    size = int(effects.size)
    weights = 1.0 / errors**2
    design = np.column_stack([np.ones(size, dtype=np.float64), errors])
    normal = design.T @ (weights[:, np.newaxis] * design)
    if np.linalg.matrix_rank(normal) < 2:
        return None
    inverse = np.linalg.inv(normal)
    beta = inverse @ (design.T @ (weights * effects))
    residual = effects - design @ beta
    residual_df = float(size - 2)
    sigma2 = float(np.sum(weights * residual**2)) / residual_df
    covariance = sigma2 * inverse
    intercept_se = math.sqrt(float(covariance[0, 0]))
    if intercept_se <= 0.0 or not math.isfinite(intercept_se):
        return None
    statistic = float(beta[0]) / intercept_se
    p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), residual_df))
    if not (math.isfinite(statistic) and math.isfinite(p_value)):
        return None
    return _Egger(
        intercept=float(beta[0]),
        intercept_se=intercept_se,
        statistic=statistic,
        p_value=p_value,
    )


# --------------------------------------------------------------------------- the blocks


def _measure_sentence(measure: str) -> str:
    """What scale the effects are analysed on."""
    if measure == "generic":
        return (
            "The effects are analysed on the scale they were given, with their standard "
            "errors as the sampling errors."
        )
    if measure == "mean_difference":
        return (
            "The effects are mean differences on their own units, with their standard "
            "errors as the sampling errors."
        )
    if measure == "standardised_mean_difference":
        return (
            "The effects are standardised mean differences as given, with their standard "
            "errors as the sampling errors."
        )
    if measure == "correlation":
        return (
            "The effects are correlations on the correlation scale as given; no Fisher z "
            "transformation is applied, so the standard errors are the given ones."
        )
    if measure == "odds_ratio":
        return (
            "Each odds ratio is analysed as its natural logarithm, with standard error "
            "sqrt(1/a + 1/(n_a - a) + 1/b + 1/(n_b - b)) from its two-by-two table."
        )
    return (
        "Each risk ratio is analysed as its natural logarithm, with standard error "
        "sqrt(1/a - 1/n_a + 1/b - 1/n_b) from its two-by-two table."
    )


def _studies_table(studies: _Studies, pool: _Pool, level: float) -> Table:
    """One row per study, with its interval and its share of each pool's weight."""
    z = _z(level)
    rows: list[tuple[Cell, ...]] = []
    for index, name in enumerate(studies.names):
        effect = float(studies.effects[index])
        error = float(studies.errors[index])
        rows.append(
            (
                name,
                effect,
                error,
                float(pool.weights_fixed[index]),
                float(pool.weights_random[index]),
                effect - z * error,
                effect + z * error,
            )
        )
    note = (
        "The interval of a study is its effect plus or minus z times its standard error. "
        "The weights are percentages of the total fixed-effect and random-effects weight."
    )
    if studies.log_scale:
        note += " The effect is a log ratio, so its interval is on the log scale."
    return table("Studies", _STUDY_COLUMNS, rows, note=note)


def _pooled_table(size: int, pool: _Pool) -> Table:
    """Both pools in one table, with the prediction interval where it is defined."""
    rows: list[tuple[Cell, ...]] = [
        (
            "Fixed effect",
            pool.fixed,
            pool.fixed_se,
            pool.fixed_lower,
            pool.fixed_upper,
            None,
        ),
        (
            "Random effects (DerSimonian-Laird)",
            pool.random,
            pool.random_se,
            pool.random_lower,
            pool.random_upper,
            pool.tau2,
        ),
    ]
    if pool.prediction_lower is not None and pool.prediction_upper is not None:
        rows.append(
            (
                "Prediction interval (random)",
                pool.random,
                None,
                pool.prediction_lower,
                pool.prediction_upper,
                pool.tau2,
            )
        )
    note = (
        "Tau squared is the DerSimonian-Laird estimate of the between-study variance; the "
        "fixed-effect model has none."
    )
    if pool.prediction_lower is None:
        note += (
            f" The prediction interval is not applicable with {size} studies, since it needs "
            f"at least {PREDICTION_MIN_STUDIES}."
        )
    else:
        note += (
            " The prediction interval is the random-effects estimate plus or minus t on "
            "k - 2 degrees of freedom times the square root of tau squared plus the pooled "
            "variance."
        )
    return table("Pooled", _POOLED_COLUMNS, rows, note=note)


def _forest(studies: _Studies, pool: _Pool, level: float) -> MultiLinePlot:
    """One line per study's interval at its own row, and the pooled interval below them."""
    z = _z(level)
    series = [
        XYSeries(
            label=name,
            x=(float(lower), float(upper)),
            y=(float(index), float(index)),
        )
        for index, (name, lower, upper) in enumerate(
            zip(
                studies.names,
                studies.effects - z * studies.errors,
                studies.effects + z * studies.errors,
                strict=True,
            )
        )
    ]
    series.append(
        XYSeries(
            label="Pooled (random)",
            x=(float(pool.random_lower), float(pool.random_upper)),
            y=(float(len(studies.names)), float(len(studies.names))),
        )
    )
    note = (
        "Each study's interval is drawn as a line from its lower to its upper bound at its "
        "own row. The pooled random-effects diamond is drawn as a line across its interval, "
        "since a plot here has no marker-only series, so the point estimates are not drawn "
        "as separate points. The studies run from row 0 to row "
        f"{len(studies.names) - 1}, with the pool below them."
    )
    return MultiLinePlot(
        title="Forest plot",
        x_label=studies.effect_label,
        y_label="Study",
        series=tuple(series),
        note=note,
    )


def _funnel(studies: _Studies, pool: _Pool) -> ScatterPlot:
    """The effects against their standard errors, on the inverted axis a funnel is read by."""
    errors = studies.errors
    return ScatterPlot(
        title="Funnel plot",
        x_label=studies.effect_label,
        y_label="Standard error (inverted axis: largest studies at the top)",
        x=tuple(float(value) for value in studies.effects),
        y=tuple(float(value) for value in errors),
        lines=(
            Line(
                x0=float(pool.random),
                y0=float(np.min(errors)),
                x1=float(pool.random),
                y1=float(np.max(errors)),
                label="Pooled (random)",
            ),
        ),
        note=(
            "The standard-error axis is inverted, as a funnel plot is read: the largest "
            "studies have the smallest standard errors and sit at the top, and the smallest "
            "studies at the bottom. The vertical line is the random-effects pool."
        ),
    )


def _assumptions(size: int) -> tuple[AuditCheck, ...]:
    """Independence, publication bias, and Egger's own minimum when it is not reached."""
    checks = [
        assumption(
            "independence",
            "Independence of the studies cannot be checked from the columns",
            Verdict.NOT_APPLICABLE,
            "Whether the studies are independent is a fact about how they were collected, "
            "which a column cannot record.",
            next_step=(
                "Check how the studies were collected; dependence between them changes what "
                "the pooled interval means."
            ),
        ),
        assumption(
            "publication_bias",
            "Publication bias cannot be ruled out",
            Verdict.NOT_APPLICABLE,
            "A funnel plot and Egger's test can suggest funnel-plot asymmetry, but neither "
            "can establish that publication bias is absent.",
            next_step=(
                "Read the funnel plot and, from ten studies, Egger's test; report that "
                "publication bias cannot be ruled out from these studies alone."
            ),
        ),
    ]
    if size < EGGER_MIN_STUDIES:
        checks.append(
            assumption(
                "egger_min_studies",
                f"Egger's test needs at least {EGGER_MIN_STUDIES} studies, and {size} are pooled",
                Verdict.NOT_APPLICABLE,
                f"Egger's test was not run because {size} studies are fewer than the "
                f"{EGGER_MIN_STUDIES} it needs.",
                next_step=(
                    "Pool at least ten studies before reading Egger's test; below that, a "
                    "funnel plot is only descriptive."
                ),
            )
        )
    return tuple(checks)


def _warnings(size: int, pool: _Pool) -> tuple[str, ...]:
    """Few studies, tau squared at its floor, and I squared above three quarters."""
    found: list[str] = []
    if size < WARN_STUDIES:
        found.append(
            f"Only {size} studies are pooled, fewer than {WARN_STUDIES}; tau squared and I "
            "squared are unstable with so few."
        )
    if pool.tau2 == 0.0:
        found.append(
            "Tau squared is estimated at zero, so the random-effects pool equals the "
            "fixed-effect pool; heterogeneity may be present but is not detectable from "
            "these studies."
        )
    if pool.i2 > HIGH_HETEROGENEITY:
        found.append(
            f"I squared is {num(pool.i2)} %, above {num(HIGH_HETEROGENEITY)} %; most of the "
            "variation between studies is heterogeneity, so the fixed-effect pool is not a "
            "description of one shared effect."
        )
    return tuple(found)


def _interpretation(studies: _Studies, pool: _Pool, level: float) -> str:
    """Both pools, I squared in words, and what heterogeneity does to the pool."""
    interval = _percent(level)
    first = (
        f"The fixed-effect pool is {num(pool.fixed)} ({interval} interval "
        f"{num(pool.fixed_lower)} to {num(pool.fixed_upper)}) and the random-effects pool is "
        f"{num(pool.random)} ({interval} interval {num(pool.random_lower)} to "
        f"{num(pool.random_upper)}) on the {studies.effect_label.lower()} scale"
    )
    if studies.log_scale:
        first += (
            f", which is {_MEASURE_ARTICLES[studies.measure]} "
            f"{_measure_noun(studies.measure)} of {num(math.exp(pool.random))} at the "
            "random-effects estimate"
        )
    first += "."
    if pool.i2 >= HIGH_HETEROGENEITY:
        second = (
            f"I squared is {num(pool.i2)} %, so most of the variation between studies is "
            "variation in the true effect rather than sampling error."
        )
    elif pool.i2 >= MODERATE_HETEROGENEITY:
        second = (
            f"I squared is {num(pool.i2)} %, so a substantial part of the variation between "
            "studies is variation in the true effect rather than sampling error."
        )
    else:
        second = (
            f"I squared is {num(pool.i2)} %, so little of the variation between studies goes "
            "beyond sampling error."
        )
    third = (
        "Heterogeneity changes what the pool means: the fixed-effect pool describes one "
        "common effect, while the random-effects pool describes the average of different "
        "true effects"
    )
    if pool.prediction_lower is not None:
        third += ", and its prediction interval shows what a new study's effect could be"
    third += "."
    return " ".join((first, second, third))


def _methods(studies: _Studies, size: int, level: float) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    parts = [
        "The studies were pooled by inverse-variance weighting on the "
        f"{studies.effect_label.lower()} scale, with w_i = 1 / SE_i^2, the fixed-effect "
        "estimate sum(w_i e_i) / sum(w_i), and Cochran's Q = sum(w_i (e_i - e_fixed)^2) read "
        "against chi-square with k - 1 degrees of freedom.",
        "The random-effects estimate uses the DerSimonian-Laird variance tau^2 = "
        "max(0, (Q - (k - 1)) / (sum(w_i) - sum(w_i^2) / sum(w_i))) added to every study's "
        "variance, with the weights recomputed; I^2 is max(0, (Q - (k - 1)) / Q) times 100 "
        "per cent.",
        _measure_sentence(studies.measure),
    ]
    if size >= PREDICTION_MIN_STUDIES:
        parts.append(
            f"The {_percent(level)} prediction interval uses t on k - 2 degrees of freedom."
        )
    if size >= EGGER_MIN_STUDIES:
        parts.append(
            "Egger's test is the weighted regression of the effect on its standard error, "
            "and its intercept's t test on k - 2 degrees of freedom bears on funnel-plot "
            "asymmetry and publication bias."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- the tool


def run_meta(request: BaseModel) -> ToolResult:
    """Run the meta-analysis on its own request."""
    if not isinstance(request, MetaRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"meta takes a MetaRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: MetaRequest) -> ToolResult:
    """The analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    raw = _read(request)
    size = int(raw.values[0].size) if raw.values else 0
    if size < MIN_STUDIES:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"a meta-analysis needs at least {MIN_STUDIES} studies, and {size} "
            f"{'is' if size == 1 else 'are'} left",
        )
    if size > MAX_STUDIES:
        raise ToolError(
            "INVALID_OPTION",
            f"a meta-analysis takes between {MIN_STUDIES} and {MAX_STUDIES} studies, not {size}",
        )
    studies = _studies(raw, request.measure)
    pool = _pool(studies.effects, studies.errors, request.level)
    egger = _egger(studies.effects, studies.errors) if size >= EGGER_MIN_STUDIES else None

    fixed = Estimate(
        label="Pooled effect (fixed)",
        value=pool.fixed,
        se=pool.fixed_se,
        lower=pool.fixed_lower,
        upper=pool.fixed_upper,
    )
    random = Estimate(
        label="Pooled effect (random)",
        value=pool.random,
        se=pool.random_se,
        lower=pool.random_lower,
        upper=pool.random_upper,
    )
    heterogeneity = HypothesisTest(
        name="Test of heterogeneity (Q)",
        method=(
            "Cochran's Q test of heterogeneity, read against chi-square with k - 1 degrees "
            "of freedom"
        ),
        null_hypothesis="the studies share one common effect",
        statistic=pool.q,
        df=float(size - 1),
        p_value=pool.q_p,
        p_value_text=p_text(pool.q_p),
    )
    tests = [heterogeneity]
    if size >= EGGER_MIN_STUDIES:
        if egger is None:
            tests.append(
                HypothesisTest(
                    name="Egger's test of funnel asymmetry",
                    method=(
                        "Egger's weighted regression of the effect on its standard error, "
                        "the intercept's t test on k - 2 degrees of freedom"
                    ),
                    null_hypothesis=(
                        "the funnel plot is symmetric, so the regression intercept is zero"
                    ),
                    statistic=None,
                    df=float(size - 2),
                    p_value=None,
                    p_value_text=p_text(None),
                    reason=(
                        "the studies' standard errors leave the weighted regression without a "
                        "slope to estimate"
                    ),
                )
            )
        else:
            tests.append(
                HypothesisTest(
                    name="Egger's test of funnel asymmetry",
                    method=(
                        "Egger's weighted regression of the effect on its standard error, "
                        "the intercept's t test on k - 2 degrees of freedom"
                    ),
                    null_hypothesis=(
                        "the funnel plot is symmetric, so the regression intercept is zero"
                    ),
                    statistic=egger.statistic,
                    df=float(size - 2),
                    p_value=egger.p_value,
                    p_value_text=p_text(egger.p_value),
                    effect=Estimate(
                        label="Egger intercept",
                        value=egger.intercept,
                        se=egger.intercept_se,
                    ),
                )
            )
    plots: list[Plot] = [_forest(studies, pool, request.level)]
    if size >= PREDICTION_MIN_STUDIES:
        plots.append(_funnel(studies, pool))
    listed, unlisted = list_excluded(studies.excluded)
    return ToolResult(
        tool="meta",
        title="Meta-analysis",
        level=request.level,
        estimates=(
            fixed,
            random,
            Estimate(label="Heterogeneity I squared %", value=pool.i2),
            Estimate(label="Tau squared", value=pool.tau2),
        ),
        tests=tuple(tests),
        assumptions=_assumptions(size),
        warnings=_warnings(size, pool),
        interpretation=_interpretation(studies, pool, request.level),
        methods_fragment=_methods(studies, size, request.level),
        counts=Counts(
            n_offered=studies.offered,
            n_used=size,
            n_excluded=studies.offered - size,
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(_studies_table(studies, pool, request.level), _pooled_table(size, pool)),
        plots=tuple(plots),
    )


TOOLS.register(
    ToolSpec(
        name="meta",
        family="evidence",
        title="Meta-analysis",
        request=MetaRequest,
        run=run_meta,
    )
)
