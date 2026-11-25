"""A linear mixed model: a fixed line with a random intercept for each group.

The tool aligns the outcome, the predictors and the group label on the rows complete in all
of them, and fits

    y = X beta + Z u + e,    u_g ~ N(0, tau squared),    e ~ N(0, sigma squared),

where each row belongs to one group and the group's random intercept ``u_g`` shifts it. The
fixed effects are estimated by restricted maximum likelihood (REML): the profile
log-likelihood is minimised over ``lambda = tau squared / sigma squared`` on the log scale
with SciPy's bounded Brent method, and at each ``lambda`` the fixed effects and the group
effects are solved by their closed forms. The fixed-effect covariance is the generalized
least squares form at the optimum, ``sigma squared (X' V^-1 X)^-1`` with
``V = sigma squared (I + lambda Z Z')``.

The denominator degrees of freedom follow the Satterthwaite approximation for a
random-intercept model: the number of groups minus one for a term that is constant within
groups, and the number of rows minus the fixed parameters minus one otherwise. The random
intercept is tested against a fixed-effect ordinary-least-squares fit by a likelihood-ratio
test, whose null distribution is the 0.5 chi-square(0) plus 0.5 chi-square(1) mixture.
Marginal and conditional R squared follow Nakagawa and Schielzeth (2013), with the fixed
part's variance the variance of the fixed-effect predictions.

A predictor that does not vary, or that is exactly collinear with the others, is refused:
its own coefficient cannot be estimated. Fewer than six rows or three groups are refused
too. Every row left out is counted and listed, never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import capped, joined, num, p_threshold
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
    Cell,
    ColumnKind,
    FloatArray,
    Line,
    ScatterPlot,
    Table,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PREDICTORS = 1
MAX_PREDICTORS = 20
MAX_PREDICTION_ROWS = 1_000
MIN_ROWS = 6
MIN_GROUPS = 3
MAX_GROUP_ROWS = 200
"""The most groups the group-effects table lists; the note says when it is shortened."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

COLLINEAR_RTOL = 1e-10
"""Columns count as collinear when the design's smallest singular value is this small a
fraction of its largest."""

LOG_LAMBDA_BOUNDS = (-30.0, 30.0)
"""The bounds of the profile search over the log of lambda = tau squared / sigma squared."""

PROFILE_XATOL = 1e-9
"""The tolerance of the bounded search over the log of lambda."""

MAX_EVALUATIONS = 200
"""The most evaluations the profile search may make."""

ZERO_LAMBDA = 1e-10
"""A lambda at or below this is reported as a between-group variance of exactly zero."""

INTERCEPT = "(Intercept)"
"""How the intercept reads as a term of the model."""

_NAMES_LABEL = "Predictor names"

_FIXED_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("estimate", "Estimate", "number"),
    ("se", "SE", "number"),
    ("df", "df", "number"),
    ("statistic", "t", "number"),
    ("p", "p", "p_value"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)

_VARIANCE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("component", "Component", "text"),
    ("estimate", "Estimate", "number"),
    ("share", "Share %", "number"),
)

_GROUP_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("rows", "Rows", "int"),
    ("blup", "BLUP", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)

_PREDICTION_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("row", "Row", "int"),
    ("predicted", "Predicted", "number"),
    ("ci_lower", "CI lower", "number"),
    ("ci_upper", "CI upper", "number"),
    ("pi_lower", "PI lower", "number"),
    ("pi_upper", "PI upper", "number"),
)


class MixedRequest(BaseModel):
    """The outcome, the predictors, the grouping column and the level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Column = Field(
        description="The numeric column the model explains; the response or dependent variable."
    )
    predictors: tuple[Column, ...] = Field(
        description=(
            "One to twenty numeric columns that explain the outcome. Each predictor contributes "
            "its own fixed coefficient to the model."
        )
    )
    group: GroupColumn = Field(
        description=(
            "The column naming each row's group. Every group gets its own random intercept, "
            "drawn from a normal distribution with mean zero and one shared variance."
        )
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "What the predictors are called in the tables and the words, in the order given. "
            "Left empty, each predictor's own column name is used."
        ),
        json_schema_extra={"label": _NAMES_LABEL},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )
    predictions: tuple[tuple[float, ...], ...] | None = Field(
        default=None,
        description=(
            "Up to 1,000 rows of predictor values to predict at; each row holds one value per "
            "predictor in the order given. Left empty, no predictions are made."
        ),
    )


class _Profile(NamedTuple):
    """One value of lambda and everything the closed forms give at it."""

    criterion: float
    """``-2`` times the REML log-likelihood, up to a constant that does not depend on lambda."""
    beta: FloatArray
    sigma2: float
    information: FloatArray
    """``X' V^-1 X`` at this lambda."""


class _Degenerate(Exception):
    """The model cannot be fit at this value: the data leave the likelihood undefined."""


class _Model(NamedTuple):
    """The fitted model and everything a result shows of it, on the rows it was fit to."""

    beta: FloatArray
    se: FloatArray
    df: FloatArray
    statistic: FloatArray
    p_value: FloatArray
    lower: FloatArray
    upper: FloatArray
    covariance: FloatArray
    """The fixed-effect covariance at the optimum, the generalized least squares form."""
    sigma2: float
    tau2: float
    icc: float
    marginal_r2: float
    conditional_r2: float
    fitted: FloatArray
    """The fixed-effect predictions, ``X beta``."""
    residuals: FloatArray
    blups: FloatArray
    blup_se: FloatArray
    group_sizes: NDArray[np.intp]
    group_labels: tuple[str, ...]
    lr: float
    lr_p: float
    converged: bool
    evaluations: int
    warnings: tuple[str, ...]


def _require_predictors(request: MixedRequest) -> None:
    """Refuse a predictor count outside one to twenty."""
    size = len(request.predictors)
    if not MIN_PREDICTORS <= size <= MAX_PREDICTORS:
        raise ToolError(
            "INVALID_OPTION",
            f"a mixed model takes between {MIN_PREDICTORS} and {MAX_PREDICTORS} predictors, "
            f"not {size}",
        )


def _require_names(request: MixedRequest) -> None:
    """Refuse names that do not name one predictor each."""
    if request.names is None:
        return
    size, names = len(request.predictors), len(request.names)
    if names != size:
        raise ToolError(
            "INVALID_OPTION",
            f"a mixed model takes one name per predictor, so {size} names, not {names}",
        )


def _require_predictions(request: MixedRequest) -> None:
    """Refuse more than 1,000 prediction rows, a row of the wrong length, or a non-finite one."""
    if request.predictions is None:
        return
    rows = len(request.predictions)
    if rows > MAX_PREDICTION_ROWS:
        raise ToolError(
            "INVALID_OPTION",
            f"a mixed model predicts at {MAX_PREDICTION_ROWS:,} rows at most, not {rows:,}",
        )
    size = len(request.predictors)
    for index, row in enumerate(request.predictions):
        if len(row) != size:
            raise ToolError(
                "INVALID_OPTION",
                f"prediction row {index + 1} holds {len(row)} values for {size} predictors",
            )
        if not all(math.isfinite(value) for value in row):
            raise ToolError(
                "INVALID_OPTION",
                f"prediction row {index + 1} holds a value that is not a finite number",
            )


def _constant(values: FloatArray) -> bool:
    """Whether ``values`` holds one value repeated."""
    return float(np.min(values)) == float(np.max(values))


def _require_varying(names: Sequence[str], columns: Sequence[FloatArray]) -> None:
    """Refuse a predictor that holds one value repeated in the rows used."""
    for name, values in zip(names, columns, strict=True):
        if _constant(values):
            raise ToolError(
                "CONSTANT_COLUMN",
                f"the predictor {name!r} holds one value repeated in the rows used, so its "
                "coefficient is undefined",
            )


def _require_outcome(name: str, values: FloatArray) -> None:
    """Refuse a constant outcome: its variation, and so R squared, is undefined."""
    if _constant(values):
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the outcome {name!r} holds one value repeated, so its variation and R squared "
            "are undefined",
        )


def _require_independent(names: Sequence[str], design: FloatArray) -> None:
    """Refuse predictors whose design matrix has dependent columns."""
    singular = np.linalg.svd(design, compute_uv=False)
    independent = int(np.count_nonzero(singular > COLLINEAR_RTOL * singular[0]))
    if independent < design.shape[1]:
        raise ToolError(
            "COLLINEAR",
            f"the predictors {joined(list(names))} are exactly collinear, so their separate "
            "coefficients cannot be estimated",
        )


def _group_order(labels: NDArray[np.object_]) -> tuple[str, ...]:
    """The distinct labels, in order of first appearance."""
    found, first = np.unique(labels, return_index=True)
    return tuple(str(label) for label in found[np.argsort(first)])


def _codes(labels: NDArray[np.object_], order: Sequence[str]) -> NDArray[np.intp]:
    """Each row's group number, in the order of first appearance."""
    position = {label: index for index, label in enumerate(order)}
    return np.array([position[str(label)] for label in labels], dtype=np.intp)


def _sizes(codes: NDArray[np.intp], groups: int) -> NDArray[np.intp]:
    """How many rows each group holds."""
    return np.bincount(codes, minlength=groups).astype(np.intp)


def _group_sums(values: FloatArray, codes: NDArray[np.intp], groups: int) -> FloatArray:
    """The sum of ``values`` inside each group."""
    return np.bincount(codes, weights=values, minlength=groups)


def _profile(
    lam: float,
    design: FloatArray,
    outcome: FloatArray,
    codes: NDArray[np.intp],
    groups: int,
) -> _Profile:
    """The closed forms at one ``lambda = tau squared / sigma squared``.

    ``V = sigma squared (I + lambda Z Z')`` is block diagonal by group: a group of ``n_g``
    rows has ``V_g = sigma squared (I + lambda J)``, whose inverse is
    ``(1/sigma squared) (I - lambda / (1 + n_g lambda) J)`` and whose determinant is
    ``sigma squared ** n_g (1 + n_g lambda)``. The fixed effects are
    ``(X' V^-1 X)^-1 X' V^-1 y``; the residual variance is
    ``r' V^-1 r / (n - p)`` with ``p`` the number of fixed parameters. The criterion is
    ``log|V| + log|X' V^-1 X| + (n - p) log(sigma squared)``, which is ``-2`` times the
    REML log-likelihood up to a constant that does not depend on ``lambda``.
    """
    n, parameters = design.shape
    group_sizes = _sizes(codes, groups)
    group_sums_design = np.zeros((groups, parameters), dtype=np.float64)
    np.add.at(group_sums_design, codes, design)
    group_sums_outcome = _group_sums(outcome, codes, groups)
    scale = 1.0 + lam * group_sizes
    shrink = lam / scale
    inverse_design = design - shrink[codes, None] * group_sums_design[codes]
    inverse_outcome = outcome - shrink[codes] * group_sums_outcome[codes]
    information = design.T @ inverse_design
    beta = np.asarray(np.linalg.solve(information, design.T @ inverse_outcome), dtype=np.float64)
    residuals = outcome - design @ beta
    sums_residuals = _group_sums(residuals, codes, groups)
    inverse_residuals = residuals - shrink[codes] * sums_residuals[codes]
    sse = float(residuals @ inverse_residuals)
    sigma2 = sse / (n - parameters)
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        raise _Degenerate(
            "the outcome is fit exactly by the fixed effects, so the residual variance is "
            "zero and the variances are undefined"
        )
    log_determinant = float(np.sum(np.log(scale)))
    sign, log_information = np.linalg.slogdet(information)
    if sign <= 0 or not math.isfinite(float(log_information)):
        raise _Degenerate("the fixed-effect information is singular at this value of lambda")
    criterion = log_determinant + float(log_information) + (n - parameters) * math.log(sigma2)
    if not math.isfinite(criterion):
        raise _Degenerate("the REML criterion is not finite at this value of lambda")
    return _Profile(
        criterion=criterion,
        beta=beta,
        sigma2=sigma2,
        information=np.asarray(information, dtype=np.float64),
    )


def _within_group_constant(values: FloatArray, codes: NDArray[np.intp], groups: int) -> bool:
    """Whether ``values`` holds one value inside every group."""
    low = np.full(groups, np.inf)
    high = np.full(groups, -np.inf)
    np.minimum.at(low, codes, values)
    np.maximum.at(high, codes, values)
    return bool(np.all(high == low))


def _degrees_of_freedom(
    design: FloatArray, codes: NDArray[np.intp], groups: int, n: int
) -> FloatArray:
    """The Satterthwaite-type denominator degrees of freedom of each fixed term.

    A term that is constant within groups is informed by the groups' means alone, so it gets
    ``groups - 1``. Every other term gets the residual degrees of freedom of the fit,
    ``n - p - 1``.
    """
    parameters = design.shape[1]
    residual_df = float(n - parameters - 1)
    if groups < 2:
        # One group carries no information about a between-group variance, so the residual
        # degrees of freedom are all there is.
        return np.full(parameters, residual_df, dtype=np.float64)
    return np.array(
        [
            float(groups - 1)
            if _within_group_constant(design[:, column], codes, groups)
            else residual_df
            for column in range(parameters)
        ],
        dtype=np.float64,
    )


def _ordinary_least_squares(
    design: FloatArray,
    outcome: FloatArray,
    codes: NDArray[np.intp],
    groups: int,
    names: tuple[str, ...],
    level: float,
) -> _Model:
    """The fit of a model whose between-group variance is zero: ordinary least squares.

    One group is this degenerate case: its random intercept cannot be told apart from the
    fixed intercept, so the between-group variance is reported at zero and the fixed effects
    are the ordinary least squares fit.
    """
    n, parameters = design.shape
    group_sizes = _sizes(codes, groups)
    beta = np.asarray(np.linalg.lstsq(design, outcome, rcond=None)[0], dtype=np.float64)
    fitted = design @ beta
    residuals = outcome - fitted
    sse = float(residuals @ residuals)
    sigma2 = sse / (n - parameters)
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            "the outcome is fit exactly by the fixed effects, so the residual variance is zero "
            "and the variances are undefined",
        )
    inverse_design = np.linalg.pinv(design)
    covariance = sigma2 * (inverse_design @ inverse_design.T)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    df = _degrees_of_freedom(design, codes, groups, n)
    statistic = beta / se
    p_value = 2.0 * scipy_stats.t.sf(np.abs(statistic), df)
    critical = scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df)
    lower = beta - critical * se
    upper = beta + critical * se
    group_variance = 0.0
    fixed_variance = float(np.var(fitted, ddof=1)) if n > 1 else 0.0
    total = fixed_variance + group_variance + sigma2
    return _Model(
        beta=beta,
        se=se,
        df=df,
        statistic=statistic,
        p_value=p_value,
        lower=lower,
        upper=upper,
        covariance=np.asarray(covariance, dtype=np.float64),
        sigma2=sigma2,
        tau2=0.0,
        icc=0.0,
        marginal_r2=fixed_variance / total,
        conditional_r2=(fixed_variance + group_variance) / total,
        fitted=fitted,
        residuals=residuals,
        blups=np.zeros(groups, dtype=np.float64),
        blup_se=np.zeros(groups, dtype=np.float64),
        group_sizes=group_sizes,
        group_labels=names,
        lr=0.0,
        lr_p=1.0,
        converged=True,
        evaluations=0,
        warnings=(),
    )


def _fit(
    design: FloatArray,
    outcome: FloatArray,
    codes: NDArray[np.intp],
    group_labels: tuple[str, ...],
    level: float,
) -> _Model:
    """Fit the random-intercept model by REML, or by ordinary least squares at tau squared 0.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when the fit is exact, so the variances are zero.
    """
    n, parameters = design.shape
    groups = len(group_labels)
    if groups == 1:
        return _ordinary_least_squares(design, outcome, codes, groups, group_labels, level)
    group_sizes = _sizes(codes, groups)
    history: list[float] = []

    def objective(log_lambda: float) -> float:
        try:
            found = _profile(math.exp(log_lambda), design, outcome, codes, groups)
        except np.linalg.LinAlgError, _Degenerate:
            return math.inf
        history.append(found.criterion)
        return found.criterion

    result = optimize.minimize_scalar(
        objective,
        bounds=LOG_LAMBDA_BOUNDS,
        method="bounded",
        options={"xatol": PROFILE_XATOL, "maxiter": MAX_EVALUATIONS},
    )
    evaluations = int(result.nfev)
    lam = float(math.exp(float(result.x)))
    try:
        optimum = _profile(lam, design, outcome, codes, groups)
    except (np.linalg.LinAlgError, _Degenerate) as exc:
        raise ToolError(
            "CONSTANT_COLUMN",
            "the REML profile could not be evaluated at its optimum, so the variances are "
            "undefined",
        ) from exc
    warnings: list[str] = []
    if not bool(result.success):
        change = (
            f"the last change in the criterion was {num(abs(history[-2] - history[-1]))}"
            if len(history) >= 2
            else "no change in the criterion was recorded"
        )
        warnings.append(
            "the REML profile did not converge within "
            f"{MAX_EVALUATIONS} evaluations; {change} at lambda = {num(lam)}"
        )
    elif lam >= math.exp(LOG_LAMBDA_BOUNDS[1]) * (1.0 - 1e-6):
        warnings.append(
            "the between-group variance reached the upper bound of the profile search, so "
            "these data do not pin it down; it is reported at the bound, not as an estimate"
        )
    if lam <= ZERO_LAMBDA:
        boundary = _ordinary_least_squares(design, outcome, codes, groups, group_labels, level)
        return boundary._replace(evaluations=evaluations, warnings=tuple(warnings))
    beta = optimum.beta
    sigma2 = optimum.sigma2
    tau2 = lam * sigma2
    covariance = sigma2 * np.linalg.inv(optimum.information)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    df = _degrees_of_freedom(design, codes, groups, n)
    statistic = beta / se
    p_value = 2.0 * scipy_stats.t.sf(np.abs(statistic), df)
    critical = scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df)
    lower = beta - critical * se
    upper = beta + critical * se
    fitted = design @ beta
    residuals = outcome - fitted
    scale = 1.0 + lam * group_sizes
    sums_residuals = _group_sums(residuals, codes, groups)
    blups = lam * sums_residuals / scale
    blup_se = np.sqrt(tau2 / scale)
    fixed_variance = float(np.var(fitted, ddof=1)) if n > 1 else 0.0
    total_variance = fixed_variance + tau2 + sigma2
    icc = tau2 / (tau2 + sigma2)
    ols_beta = np.asarray(np.linalg.lstsq(design, outcome, rcond=None)[0], dtype=np.float64)
    ols_residuals = outcome - design @ ols_beta
    ols_sigma2 = float(ols_residuals @ ols_residuals) / (n - parameters)
    _, log_information = np.linalg.slogdet(design.T @ design)
    with np.errstate(divide="ignore"):
        criterion_zero = float(log_information) + (n - parameters) * math.log(ols_sigma2)
    lr = max(0.0, criterion_zero - optimum.criterion)
    lr_p = 1.0 if lr <= 0.0 else float(0.5 * scipy_stats.chi2.sf(lr, 1))
    return _Model(
        beta=beta,
        se=se,
        df=df,
        statistic=statistic,
        p_value=p_value,
        lower=lower,
        upper=upper,
        covariance=np.asarray(covariance, dtype=np.float64),
        sigma2=sigma2,
        tau2=tau2,
        icc=float(np.clip(icc, 0.0, 1.0)),
        marginal_r2=fixed_variance / total_variance,
        conditional_r2=(fixed_variance + tau2) / total_variance,
        fitted=fitted,
        residuals=residuals,
        blups=blups,
        blup_se=blup_se,
        group_sizes=group_sizes,
        group_labels=group_labels,
        lr=lr,
        lr_p=lr_p,
        converged=bool(result.success),
        evaluations=evaluations,
        warnings=tuple(warnings),
    )


def _verdict(p_value: float | None) -> Verdict:
    """The verdict a p-value gives an assumption."""
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _normality(residuals: FloatArray) -> AuditCheck:
    """The Shapiro-Wilk check of normality of the residuals."""
    found = shapiro_wilk(residuals)
    if found.status != "ok" or found.p_value is None:
        return assumption(
            "normality",
            "The normality of the residuals could not be checked",
            Verdict.NOT_APPLICABLE,
            found.reason or "Shapiro-Wilk's W test could not be run on these residuals.",
        )
    assert found.statistic is not None
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The residuals are consistent with a normal distribution",
        Verdict.CAUTION: "The residuals leave the normal assumption in doubt",
        Verdict.FAIL: "The residuals show evidence against a normal distribution",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the residuals are far from normal, the intervals and the p-values rely on an "
        "approximation; with many rows the test also reacts to departures of no practical size."
    )
    return assumption(
        "normality",
        titles[verdict],
        verdict,
        f"Shapiro-Wilk's W test gives W = {num(found.statistic)}, {p_text(found.p_value)}.",
        statistic=f"W = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _random_intercept_assumption(group_name: str) -> AuditCheck:
    """The random intercept's normal distribution, which the data cannot check here."""
    return assumption(
        "random_intercept",
        "The group intercepts are assumed drawn from one normal distribution",
        Verdict.NOT_APPLICABLE,
        f"The model assumes each level of {group_name} shifts the outcome by an intercept "
        "drawn from a normal distribution with mean zero and one shared variance. The data "
        "cannot show whether that distribution is normal with the groups they hold.",
        next_step="If the groups are few, treat the group effect as a fixed factor instead.",
    )


def _group_warnings(model: _Model, group_name: str) -> tuple[str, ...]:
    """Warnings for thin groups, groups of one, and a between-group variance at zero."""
    warnings: list[str] = []
    small = [
        label for label, size in zip(model.group_labels, model.group_sizes, strict=True) if size < 5
    ]
    if small:
        names = capped([repr(label) for label in small], "group")
        warnings.append(
            f"{names} {'holds' if len(small) == 1 else 'hold'} fewer than 5 rows, so the "
            "between-group variance is estimated from little information."
        )
    single = [
        label
        for label, size in zip(model.group_labels, model.group_sizes, strict=True)
        if size == 1
    ]
    if single:
        names = capped([repr(label) for label in single], "group")
        warnings.append(
            f"{names} {'holds' if len(single) == 1 else 'hold'} one row, so "
            f"{'it contributes its' if len(single) == 1 else 'they contribute their'} fixed "
            "effects and no information about the between-group variance."
        )
    if model.tau2 == 0.0:
        warnings.append(
            f"the between-group variance of {group_name} is estimated at zero, so the grouping "
            "contributes nothing beyond the fixed effects."
        )
    return tuple(warnings)


def _fixed_effects_table(model: _Model, names: Sequence[str], level: float) -> Table:
    """The fixed effects: the intercept first, then one row per predictor."""
    terms = (INTERCEPT, *names)
    rows: list[tuple[Cell, ...]] = [
        (
            terms[index],
            float(model.beta[index]),
            float(model.se[index]),
            float(model.df[index]),
            float(model.statistic[index]),
            float(model.p_value[index]),
            float(model.lower[index]),
            float(model.upper[index]),
        )
        for index in range(len(terms))
    ]
    return table(
        "Fixed effects",
        _FIXED_COLUMNS,
        rows,
        note=(
            "SE is the generalized least squares standard error at the optimum, and t is the "
            "estimate over it. The denominator degrees of freedom follow the Satterthwaite "
            "approximation for a random-intercept model: the number of groups minus one for a "
            "term constant within groups, and the number of rows minus the fixed parameters "
            f"minus one otherwise. The interval is at the {level * 100:g} % level."
        ),
    )


def _variance_table(model: _Model) -> Table:
    """The variance components and each one's share of the random variation."""
    total = model.tau2 + model.sigma2
    rows = (
        ("Residual (sigma squared)", model.sigma2, 100.0 * model.sigma2 / total),
        ("Between-group (tau squared)", model.tau2, 100.0 * model.tau2 / total),
    )
    return table(
        "Variance components",
        _VARIANCE_COLUMNS,
        rows,
        note=(
            "The share of each component is its percentage of the residual plus the "
            "between-group variance, the two parts of the variation the fixed effects leave."
        ),
    )


def _group_table(model: _Model, level: float) -> Table:
    """The group effects, at most 200, with intervals at the groups' degrees of freedom."""
    shown = min(len(model.group_labels), MAX_GROUP_ROWS)
    critical = (
        0.0
        if len(model.group_labels) < 2
        else float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, len(model.group_labels) - 1))
    )
    rows: list[tuple[Cell, ...]] = []
    for index in range(shown):
        blup = float(model.blups[index])
        half = critical * float(model.blup_se[index])
        rows.append(
            (
                model.group_labels[index],
                int(model.group_sizes[index]),
                blup,
                blup - half,
                blup + half,
            )
        )
    note = (
        "The BLUP is the group's estimated intercept shift, and the interval is at the stated "
        "level on the number of groups minus one degrees of freedom."
    )
    if len(model.group_labels) > shown:
        note += (
            f" The first {shown:,} of {len(model.group_labels):,} groups are shown; the others "
            "take part in the fit and are counted, but not listed."
        )
    return table("Group effects", _GROUP_COLUMNS, rows, note=note)


def _prediction_row(
    index: int, values: Sequence[float], model: _Model, df: float, level: float
) -> tuple[Cell, ...]:
    """One prediction: the fixed-effect mean, its interval, and the interval for a new group."""
    point = np.concatenate(([1.0], np.asarray(values, dtype=np.float64)))
    predicted = float(point @ model.beta)
    variance = float(point @ model.covariance @ point)
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    half_mean = critical * math.sqrt(max(0.0, variance))
    half_new = critical * math.sqrt(max(0.0, variance + model.tau2 + model.sigma2))
    return (
        index,
        predicted,
        predicted - half_mean,
        predicted + half_mean,
        predicted - half_new,
        predicted + half_new,
    )


def _predictions_table(
    model: _Model, predictions: Sequence[Sequence[float]], level: float
) -> Table:
    """The predictions, one numbered row per requested predictor row."""
    df = len(model.residuals) - model.beta.size - 1
    return table(
        "Predictions",
        _PREDICTION_COLUMNS,
        (
            _prediction_row(index, values, model, float(df), level)
            for index, values in enumerate(predictions, start=1)
        ),
        note=(
            "The confidence interval is for the mean response of a group at the given predictor "
            "values; the prediction interval is for a single new observation in a new group, "
            "which adds the between-group and the residual variance. Both use the t "
            "distribution on rows minus fixed parameters minus one degrees of freedom, at the "
            "stated level."
        ),
    )


def _group_plot(model: _Model, group_name: str) -> ScatterPlot:
    """Each group's BLUP against its index, in order of first appearance."""
    total = int(model.blups.size)
    shown, note = stride(model.blups)
    if shown.size == total:
        taken = np.arange(total, dtype=np.intp)
    else:
        taken = np.round(np.linspace(0.0, total - 1.0, shown.size)).astype(np.intp)
    indices = tuple(float(index + 1) for index in taken)
    return ScatterPlot(
        title="Group effects",
        x_label="Group (in order of first appearance)",
        y_label="Group intercept shift (BLUP)",
        x=indices,
        y=tuple(float(value) for value in shown),
        lines=(Line(x0=0.0, y0=0.0, x1=float(total + 1), y1=0.0, label="Zero"),),
        note=(
            f"The x axis numbers the levels of {group_name} in order of first appearance; the "
            "y axis is each group's estimated intercept shift." + (f" {note}." if note else "")
        ),
    )


def _marginal_plot(
    model: _Model, predictor: FloatArray, outcome: FloatArray, *, outcome_name: str, name: str
) -> ScatterPlot:
    """The data with the marginal fitted line, from the fixed effects alone."""
    shown_x, note = stride(predictor)
    shown_y, _ = stride(outcome)
    low, high = float(predictor.min()), float(predictor.max())
    return ScatterPlot(
        title=f"{outcome_name} against {name} with the marginal fit",
        x_label=name,
        y_label=outcome_name,
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(
            Line(
                x0=low,
                y0=float(model.beta[0] + model.beta[1] * low),
                x1=high,
                y1=float(model.beta[0] + model.beta[1] * high),
                label="Marginal fit",
            ),
        ),
        note=(
            "The line is the fixed-effect fit, so it is the response of an average group."
            + (f" {note}." if note else "")
        ),
    )


def _fixed_tests(
    model: _Model,
    names: Sequence[str],
    design: FloatArray,
    codes: NDArray[np.intp],
    groups: int,
) -> tuple[HypothesisTest, ...]:
    """One t test per fixed effect, each naming its denominator degrees of freedom rule."""
    terms = (INTERCEPT, *names)
    tests: list[HypothesisTest] = []
    for index, term in enumerate(terms):
        constant = _within_group_constant(design[:, index], codes, groups)
        rule = (
            "the number of groups minus one, for a term constant within groups"
            if constant
            else "rows minus fixed parameters minus one"
        )
        p_value = float(model.p_value[index])
        tests.append(
            HypothesisTest(
                name=f"t test of {term}",
                method=f"Linear mixed model by REML; t test with {num(model.df[index])} degrees "
                f"of freedom ({rule})",
                null_hypothesis=f"The fixed effect of {term} is zero.",
                statistic=float(model.statistic[index]),
                df=float(model.df[index]),
                p_value=p_value,
                p_value_text=p_text(p_value),
                effect=Estimate(
                    label=term,
                    value=float(model.beta[index]),
                    se=float(model.se[index]),
                    lower=float(model.lower[index]),
                    upper=float(model.upper[index]),
                ),
            )
        )
    return tuple(tests)


def _random_intercept_test(model: _Model) -> HypothesisTest:
    """The likelihood-ratio test of the random intercept against ordinary least squares."""
    mixture = "0.5 chi-square(0) + 0.5 chi-square(1) mixture"
    return HypothesisTest(
        name="Likelihood-ratio test of the random intercept",
        method=(
            "REML likelihood-ratio test against a fixed-effect ordinary-least-squares fit, "
            f"with the {mixture} as its null distribution"
        ),
        null_hypothesis=(
            "The between-group variance of the intercept is zero, so the groups do not shift "
            "the outcome."
        ),
        statistic=model.lr,
        df=1.0,
        p_value=model.lr_p,
        p_value_text=f"{p_text(model.lr_p)} ({mixture})",
        effect=Estimate(label="Between-group variance (tau squared)", value=model.tau2),
    )


def _largest_fixed_effect(
    model: _Model, names: Sequence[str], predictors: Sequence[FloatArray]
) -> str:
    """The predictor whose standardised coefficient is largest, in words."""
    sizes = [
        abs(float(model.beta[index])) * float(np.std(predictors[index - 1], ddof=1))
        for index in range(1, len(model.beta))
    ]
    best = int(np.argmax(sizes))
    name = names[best]
    coefficient = float(model.beta[best + 1])
    return (
        f"The largest fixed effect is {name}: its coefficient of {num(coefficient)} moves the "
        f"outcome by about {num(sizes[best])} per one standard deviation of {name}."
    )


def _interpretation(
    model: _Model,
    names: Sequence[str],
    predictors: Sequence[FloatArray],
    group_name: str,
    outcome_name: str,
) -> str:
    """One to three sentences: the ICC, the largest fixed effect, and the grouping's share."""
    sentences = [
        f"The intraclass correlation is {num(model.icc)}, so {num(model.icc * 100)} % of the "
        f"variation the fixed effects leave lies between the levels of {group_name}."
    ]
    sentences.append(_largest_fixed_effect(model, names, predictors))
    if model.tau2 == 0.0:
        sentences.append(
            f"The grouping adds nothing beyond the fixed effects: the between-group variance "
            f"of {group_name} is estimated at zero."
        )
    else:
        sentences.append(
            f"The grouping raises the explained share of the variation in {outcome_name} from "
            f"{num(model.marginal_r2 * 100)} % (marginal) to "
            f"{num(model.conditional_r2 * 100)} % (conditional)."
        )
    return " ".join(sentences)


def _methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "A linear mixed model with a random intercept for each group fits the outcome as a "
        "fixed intercept plus one coefficient per predictor plus a group-level intercept drawn "
        "from a normal distribution with mean zero and variance tau squared, with normal "
        "residuals of variance sigma squared. The model is fit by restricted maximum likelihood "
        "(REML), profiling the REML log-likelihood over lambda = tau squared / sigma squared on "
        "the log scale; at each lambda the fixed effects and the group effects are solved by "
        "their closed forms, and the fixed-effect standard errors are the generalized least "
        "squares standard errors at the optimum. The denominator degrees of freedom follow the "
        "Satterthwaite approximation for a random-intercept model: the number of groups minus "
        "one for a term constant within groups, and the number of rows minus the fixed "
        "parameters minus one otherwise. The random intercept is tested against a fixed-effect "
        "ordinary-least-squares fit by a likelihood-ratio test whose null distribution is the "
        "0.5 chi-square(0) plus 0.5 chi-square(1) mixture. Marginal and conditional R squared "
        "follow Nakagawa and Schielzeth: marginal is the variance of the fixed-effect "
        "predictions over that variance plus tau squared plus sigma squared, and conditional is "
        "the variance of the fixed-effect predictions plus tau squared over the same total."
    )


def run_mixed(request: BaseModel) -> ToolResult:
    """Run the mixed model tool on its own request."""
    if not isinstance(request, MixedRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"mixed takes a MixedRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: MixedRequest) -> ToolResult:
    """The fit itself, once the request is known to be this tool's own."""
    check_level(request.level)
    _require_predictors(request)
    _require_names(request)
    _require_predictions(request)
    pairing = pairs(request.outcome, *request.predictors)
    numeric = pairing.columns
    fitting = aligned(numeric[0], request.group)
    positions = fitting.values.rows
    mask = np.isin(numeric[0].rows, positions)
    outcome = fitting.values.values
    predictor_values = tuple(column.values[mask] for column in numeric[1:])
    labels = fitting.groups[0].labels
    names = (
        tuple(request.names)
        if request.names is not None
        else tuple(column.name for column in numeric[1:])
    )
    size = len(predictor_values)
    n = int(outcome.size)
    if n < MIN_ROWS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"a mixed model needs at least {MIN_ROWS} complete rows, and {n} are left",
        )
    order = _group_order(labels)
    groups = len(order)
    if groups < MIN_GROUPS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"a mixed model needs at least {MIN_GROUPS} groups to estimate a between-group "
            f"variance, and {groups} are left in {request.group.name!r}",
        )
    _require_outcome(request.outcome.name, outcome)
    _require_varying(names, predictor_values)
    design = np.column_stack([np.ones(n), *predictor_values]).astype(np.float64)
    _require_independent(names, design)
    parameters = design.shape[1]
    if n - parameters - 1 < 1:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the model needs at least {parameters + 2} complete rows for {parameters} fixed "
            f"parameters, and {n} are left",
        )
    codes = _codes(labels, order)
    model = _fit(design, outcome, codes, order, request.level)
    excluded = sorted([*pairing.excluded, *fitting.excluded], key=lambda row: row.index)
    listed, unlisted = list_excluded(excluded)
    estimates = (
        Estimate(label="Residual variance (sigma squared)", value=model.sigma2),
        Estimate(label="Between-group variance (tau squared)", value=model.tau2),
        Estimate(label="Intraclass correlation", value=model.icc),
        Estimate(label="Marginal R squared", value=model.marginal_r2),
        Estimate(label="Conditional R squared", value=model.conditional_r2),
    )
    tables: list[Table] = [
        _fixed_effects_table(model, names, request.level),
        _variance_table(model),
        _group_table(model, request.level),
    ]
    if request.predictions:
        tables.append(_predictions_table(model, request.predictions, request.level))
    plots: list[ScatterPlot] = [_group_plot(model, request.group.name)]
    if size == 1:
        plots.append(
            _marginal_plot(
                model,
                predictor_values[0],
                outcome,
                outcome_name=request.outcome.name,
                name=names[0],
            )
        )
    warnings = (*model.warnings, *_group_warnings(model, request.group.name))
    title = (
        f"Linear mixed model of {request.outcome.name} on {joined(list(names))} with a random "
        f"intercept for {request.group.name}"
        if size <= 3
        else f"Linear mixed model of {request.outcome.name} on {size} predictors with a random "
        f"intercept for {request.group.name}"
    )
    return ToolResult(
        tool="mixed",
        title=title,
        level=request.level,
        estimates=estimates,
        tests=(
            *_fixed_tests(model, names, design, codes, groups),
            _random_intercept_test(model),
        ),
        assumptions=(
            _normality(model.residuals),
            _random_intercept_assumption(request.group.name),
        ),
        warnings=warnings,
        interpretation=_interpretation(
            model, names, predictor_values, request.group.name, request.outcome.name
        ),
        methods_fragment=_methods(),
        counts=Counts(
            n_offered=int(n + len(excluded)),
            n_used=n,
            n_excluded=len(excluded),
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=tuple(plots),
    )


TOOLS.register(
    ToolSpec(
        name="mixed",
        family="advanced",
        title="Linear mixed model",
        request=MixedRequest,
        run=run_mixed,
    )
)
