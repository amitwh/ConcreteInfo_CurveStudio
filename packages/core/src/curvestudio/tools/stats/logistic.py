"""Logistic regression of a two-level outcome on one to twenty numeric predictors.

The tool pairs the outcome labels with the predictors by row position, keeps the rows complete
in all of them, and fits the log odds of the event as an intercept plus one coefficient per
predictor by iteratively reweighted least squares (Newton-Raphson). The event is the outcome
level that appears second in the order of first appearance; the result names it. The fit
reports the coefficients with their standard errors, z statistics, p-values, Wald intervals
and odds ratios, the likelihood ratio test of the model, McFadden's and Nagelkerke's pseudo R
squared, the AIC, the area under the ROC curve, sensitivity, specificity and accuracy at the
0.5 cut-off, the Hosmer-Lemeshow calibration test, a calibration plot, and predictions at
predictor values a person gives.

A p-value is a statement about evidence, never a statement of proof. An odds ratio measures
association in these rows; it does not show that a predictor causes the outcome. The
assumptions the model makes are checked where the data can show them: the Hosmer-Lemeshow test
reads the calibration of the fitted probabilities, and linearity of the logit and independence
of the rows are stated as things no single number can read.

Complete or quasi-complete separation is reported as a warning naming the predictor, with the
coefficients still shown; a fit that does not converge is reported with the last change in the
log-likelihood. A predictor that does not vary, or that is exactly collinear with the others,
is refused: its own coefficient cannot be estimated. Every row left out is counted and listed,
never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import special as scipy_special
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import listing, num, p_threshold
from curvestudio.tools.stats._base import (
    BoolArray,
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    _Kept,
    _spread,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    Line,
    ScatterPlot,
    Table,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PREDICTORS = 1
MAX_PREDICTORS = 20
MAX_PREDICTION_ROWS = 1_000
"""The most predictor rows a prediction request may carry."""

MAX_ITERATIONS = 50
"""The most Newton-Raphson steps the fit takes."""

CONVERGENCE = 1e-10
"""The fit has converged when the log-likelihood changes by less than this."""

SEPARATION_PROBABILITY = 1e-8
"""A fitted probability this close to 0 or 1 suggests complete or quasi-complete separation."""

SEPARATION_COEFFICIENT = 25.0
"""A coefficient larger than this in magnitude suggests separation."""

COLLINEAR_RTOL = 1e-10
"""A predictor counts as collinear with the others when the design's smallest singular value
is this small a fraction of its largest."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

MAX_GROUPS = 10
"""The most groups the Hosmer-Lemeshow test and the calibration plot use."""

SMALL_GROUP = 5
"""A Hosmer-Lemeshow group below this many observations makes the check coarse."""

FEW_ROWS = 10
"""Fewer used rows than this warrants a warning."""

INTERCEPT = "(Intercept)"
"""How the intercept reads as a term of the model."""

_NAMES_LABEL = "Predictor names"

_COEFFICIENT_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("estimate", "Estimate", "number"),
    ("se", "SE", "number"),
    ("z", "z", "number"),
    ("p", "p", "p_value"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
    ("odds_ratio", "Odds ratio", "number"),
    ("or_lower", "OR lower", "number"),
    ("or_upper", "OR upper", "number"),
)

_CLASSIFICATION_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("outcome", "Outcome", "text"),
    ("predicted_event", "Predicted event", "int"),
    ("predicted_no_event", "Predicted no event", "int"),
)

_PREDICTION_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("row", "Row", "int"),
    ("predicted", "Predicted probability", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)


class LogisticRequest(BaseModel):
    """The outcome, the predictors, their names, the level and any prediction points."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: GroupColumn = Field(
        description=(
            "The outcome column, which must hold exactly two labels. The label that appears "
            "second, in the order of first appearance, is the event the model explains."
        ),
    )
    predictors: tuple[Column, ...] = Field(
        description=(
            "One to twenty numeric columns that explain the outcome. Each predictor contributes "
            "its own coefficient to the log odds of the event."
        ),
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "What the predictors are called in the tables and the words, in the order given. "
            "Left empty, each predictor's own column name is used."
        ),
        json_schema_extra={"label": _NAMES_LABEL},
    )
    predictions: tuple[tuple[float, ...], ...] | None = Field(
        default=None,
        description=(
            "Up to 1,000 rows of predictor values to predict at; each row holds one value per "
            "predictor in the order given. Left empty, no predictions are made."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Cases(NamedTuple):
    """The rows complete in the outcome and every predictor, and the rows left out."""

    event: str
    other: str
    y: FloatArray
    predictors: tuple[FloatArray, ...]
    excluded: tuple[ExcludedRow, ...]
    offered: int


class _Model(NamedTuple):
    """The fitted model and everything a result shows of it, on the rows it was fit to."""

    n: int
    k: int
    beta: FloatArray
    se: FloatArray
    statistic: FloatArray
    p_value: FloatArray
    lower: FloatArray
    upper: FloatArray
    defined: BoolArray
    bounded: BoolArray
    log_likelihood: float
    null_log_likelihood: float
    aic: float
    fitted: FloatArray
    covariance: FloatArray
    converged: bool
    last_change: float | None


def _require_predictors(request: LogisticRequest) -> None:
    """Refuse a predictor count outside one to twenty."""
    size = len(request.predictors)
    if not MIN_PREDICTORS <= size <= MAX_PREDICTORS:
        raise ToolError(
            "INVALID_OPTION",
            f"logistic regression takes between {MIN_PREDICTORS} and {MAX_PREDICTORS} "
            f"predictors, not {size}",
        )


def _require_names(request: LogisticRequest) -> None:
    """Refuse names that do not name one predictor each."""
    if request.names is None:
        return
    size, names = len(request.predictors), len(request.names)
    if names != size:
        raise ToolError(
            "INVALID_OPTION",
            f"logistic regression takes one name per predictor, so {size} names, not {names}",
        )


def _require_predictions(request: LogisticRequest) -> None:
    """Refuse more than 1,000 prediction rows, a row of the wrong length, or a non-finite one."""
    if request.predictions is None:
        return
    rows = len(request.predictions)
    if rows > MAX_PREDICTION_ROWS:
        raise ToolError(
            "INVALID_OPTION",
            f"logistic regression predicts at {MAX_PREDICTION_ROWS:,} rows at most, not {rows:,}",
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


def _require_levels(outcome: GroupColumn) -> tuple[str, ...]:
    """The outcome's labels in order of first appearance, exactly two of them."""
    found = levels(outcome, minimum=0)
    if len(found) != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the column {outcome.name!r} holds {len(found)} labels, and logistic regression "
            "needs exactly two",
        )
    return found


def _complete_cases(
    outcome: GroupColumn, predictors: Sequence[Column], found: tuple[str, ...]
) -> _Cases:
    """The rows complete in the outcome and every predictor, and the rows left out.

    The outcome's labels and the predictors' values are laid out over every position any of
    them names, and a position missing in any of them is left out with the reason, named after
    the first column at fault: ``no group label in <outcome>``, ``missing in <predictor>`` or
    ``not a finite number in <predictor>``. The excluded rows come in position order.
    """
    positions = np.unique(
        np.concatenate([outcome.rows, *(predictor.rows for predictor in predictors)])
    )
    rows = _Kept(positions)
    covered, labels = _spread(positions, outcome.rows, outcome.labels, "")
    rows.drop(~covered, f"no group label in {outcome.name}")
    spread: list[FloatArray] = []
    for predictor in predictors:
        covered, values = _spread(positions, predictor.rows, predictor.values, np.nan)
        spread.append(values)
        rows.drop(~covered | np.isnan(values), f"missing in {predictor.name}")
        rows.drop(np.isinf(values), f"not a finite number in {predictor.name}")
    kept = rows.kept
    used = labels[kept]
    present = {str(label) for label in used}
    if present != set(found):
        missing = [label for label in found if label not in present]
        raise ToolError(
            "LEVELS_REQUIRED",
            f"after leaving out the incomplete rows the outcome {outcome.name!r} no longer "
            f"holds the {'label' if len(missing) == 1 else 'labels'} {listing(missing, 'label')}",
        )
    event, other = found[1], found[0]
    y = (used == event).astype(np.float64)
    return _Cases(
        event=event,
        other=other,
        y=y,
        predictors=tuple(values[kept] for values in spread),
        excluded=rows.excluded(),
        offered=int(positions.size),
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


def _require_independent(names: Sequence[str], design: FloatArray) -> None:
    """Refuse predictors whose design matrix has dependent columns."""
    singular = np.linalg.svd(design, compute_uv=False)
    independent = int(np.count_nonzero(singular > COLLINEAR_RTOL * singular[0]))
    if independent < design.shape[1]:
        raise ToolError(
            "COLLINEAR",
            f"the predictors {listing(list(names), 'predictor')} are exactly collinear, so "
            "their separate coefficients cannot be estimated",
        )


def _log_likelihood(design: FloatArray, outcome: FloatArray, beta: FloatArray) -> float:
    """``sum(y * eta - log(1 + exp(eta)))``, computed without overflow."""
    eta = design @ beta
    return float(np.sum(outcome * eta - np.logaddexp(0.0, eta)))


def _null_log_likelihood(outcome: FloatArray) -> float:
    """The log-likelihood of the intercept-only model, at the outcome's own event rate."""
    rate = float(np.mean(outcome))
    eta = math.log(rate / (1.0 - rate))
    return float(np.sum(outcome * eta - math.log1p(math.exp(eta))))


def _fit(design: FloatArray, outcome: FloatArray, level: float) -> _Model:
    """Fit the log odds of ``outcome`` on the ``design`` (whose first column is the intercept).

    Iteratively reweighted least squares (Newton-Raphson), at most :data:`MAX_ITERATIONS`
    steps, converged when the log-likelihood changes by less than :data:`CONVERGENCE`. The
    standard errors are the square roots of the diagonal of the inverse information matrix at
    the last iterate, by the pseudo-inverse when that matrix is singular.

    Raises:
        ToolError: ``INVALID_OPTION`` for a confidence level outside (0, 1).
    """
    check_level(level)
    n, parameters = design.shape
    k = parameters - 1
    beta = np.zeros(parameters, dtype=np.float64)
    log_likelihood = _log_likelihood(design, outcome, beta)
    converged = False
    last_change: float | None = None
    for _ in range(MAX_ITERATIONS):
        eta = design @ beta
        probability = scipy_special.expit(eta)
        weights = probability * (1.0 - probability)
        information = design.T @ (design * weights[:, None])
        gradient = design.T @ (outcome - probability)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        if not bool(np.all(np.isfinite(step))):
            break
        candidate = beta + step
        if not bool(np.all(np.isfinite(candidate))):
            break
        updated = _log_likelihood(design, outcome, candidate)
        if not math.isfinite(updated):
            break
        beta = candidate
        last_change = updated - log_likelihood
        log_likelihood = updated
        if abs(last_change) < CONVERGENCE:
            converged = True
            break

    eta = design @ beta
    probability = scipy_special.expit(eta)
    weights = probability * (1.0 - probability)
    information = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(information, hermitian=True)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    defined = (se > 0.0) & np.isfinite(se) & np.isfinite(beta)
    statistic = np.zeros(parameters, dtype=np.float64)
    statistic[defined] = beta[defined] / se[defined]
    p_value = np.ones(parameters, dtype=np.float64)
    p_value[defined] = 2.0 * scipy_stats.norm.sf(np.abs(statistic[defined]))
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    lower = beta - critical * se
    upper = beta + critical * se
    bounded = defined & np.isfinite(lower) & np.isfinite(upper)
    return _Model(
        n=n,
        k=k,
        beta=beta,
        se=se,
        statistic=statistic,
        p_value=p_value,
        lower=lower,
        upper=upper,
        defined=defined,
        bounded=bounded,
        log_likelihood=log_likelihood,
        null_log_likelihood=_null_log_likelihood(outcome),
        aic=2.0 * parameters - 2.0 * log_likelihood,
        fitted=probability,
        covariance=covariance,
        converged=converged,
        last_change=last_change,
    )


def _pseudo_r_squared(model: _Model) -> tuple[float, float]:
    """McFadden's and Nagelkerke's pseudo R squared."""
    mcfadden = 1.0 - model.log_likelihood / model.null_log_likelihood
    cox_snell = 1.0 - math.exp((2.0 / model.n) * (model.null_log_likelihood - model.log_likelihood))
    maximum = 1.0 - math.exp((2.0 / model.n) * model.null_log_likelihood)
    nagelkerke = cox_snell / maximum if maximum > 0.0 else 0.0
    return mcfadden, nagelkerke


def _auc(outcome: FloatArray, fitted: FloatArray) -> float:
    """The area under the ROC curve, as the Mann-Whitney probability with mid-ranks for ties."""
    positive = outcome == 1.0
    events = int(np.count_nonzero(positive))
    others = outcome.size - events
    if events == 0 or others == 0:
        return 0.5
    ranks = scipy_stats.rankdata(fitted)
    return float((ranks[positive].sum() - events * (events + 1) / 2.0) / (events * others))


def _threshold_rates(outcome: FloatArray, fitted: FloatArray) -> tuple[float, float, float]:
    """Sensitivity, specificity and accuracy at the 0.5 cut-off."""
    predicted = fitted >= 0.5
    positive = outcome == 1.0
    true_positive = int(np.count_nonzero(predicted & positive))
    false_negative = int(np.count_nonzero(~predicted & positive))
    true_negative = int(np.count_nonzero(~predicted & ~positive))
    false_positive = int(np.count_nonzero(predicted & ~positive))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    accuracy = (true_positive + true_negative) / outcome.size
    return sensitivity, specificity, accuracy


def _deciles(fitted: FloatArray) -> list[NDArray[np.intp]]:
    """The row indices sorted by fitted probability, split into at most ten nearly equal groups."""
    groups = min(MAX_GROUPS, fitted.size)
    order = np.argsort(fitted, kind="stable")
    return [np.asarray(group, dtype=np.intp) for group in np.array_split(order, groups)]


class _Calibration(NamedTuple):
    """The Hosmer-Lemeshow statistic and the groups it was read from."""

    statistic: float
    df: int
    p_value: float
    groups: int


def _hosmer_lemeshow(
    groups: Sequence[NDArray[np.intp]], outcome: FloatArray, fitted: FloatArray
) -> _Calibration | None:
    """The Hosmer-Lemeshow chi-square, or ``None`` when a group leaves no expected event."""
    statistic = 0.0
    for group in groups:
        size = float(group.size)
        observed = float(outcome[group].sum())
        expected = float(fitted[group].sum())
        if expected <= 0.0 or expected >= size:
            return None
        statistic += (observed - expected) ** 2 / (expected * (1.0 - expected / size))
    df = len(groups) - 2
    if df < 1:
        return None
    p_value = float(scipy_stats.chi2.sf(statistic, df))
    return _Calibration(
        statistic=statistic,
        df=df,
        p_value=p_value,
        groups=len(groups),
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


def _calibration_check(found: _Calibration | None) -> AuditCheck:
    """The Hosmer-Lemeshow check of the fit's calibration."""
    if found is None:
        return assumption(
            "hosmer_lemeshow",
            "The calibration of the fitted probabilities could not be checked",
            Verdict.NOT_APPLICABLE,
            "A group sorted by fitted probability holds no expected event or no expected "
            "non-event, so the Hosmer-Lemeshow statistic is undefined.",
            next_step="Look at the observed proportions against the fitted probabilities.",
        )
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The fitted probabilities match the observed proportions by decile",
        Verdict.CAUTION: "The fitted probabilities leave the calibration in doubt",
        Verdict.FAIL: "The fitted probabilities do not match the observed proportions by decile",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "A small p means poor calibration: compare the observed proportion with the mean "
        "fitted probability in each group, and consider a different model."
    )
    return assumption(
        "hosmer_lemeshow",
        titles[verdict],
        verdict,
        f"The Hosmer-Lemeshow test on {found.groups} groups gives "
        f"chi-square = {num(found.statistic)} on {found.df} degrees of freedom, "
        f"{p_text(found.p_value)}.",
        statistic=f"chi-square = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _linearity_check() -> AuditCheck:
    """Linearity of the logit, which only the fitted probabilities against a predictor show."""
    return assumption(
        "linearity",
        "A straight line in the log odds is what the model assumes",
        Verdict.NOT_APPLICABLE,
        "The linearity of the logit in each predictor cannot be read from one number.",
        next_step="Check the fitted probabilities against each predictor.",
    )


def _independence_check() -> AuditCheck:
    """Independence of the rows, a fact about how the data were collected."""
    return assumption(
        "independence",
        "Independence between the rows cannot be read from the columns",
        Verdict.NOT_APPLICABLE,
        "Each row is one observation; whether the rows are independent is a fact about how the "
        "data were collected, which the columns cannot record.",
        next_step="Check how the rows were collected; dependence between them changes what the "
        "p-value means.",
    )


def _optional(values: FloatArray, mask: BoolArray, index: int) -> float | None:
    """``values[index]`` when ``mask`` marks it defined, else ``None``."""
    return float(values[index]) if bool(mask[index]) else None


def _odds(value: float | None) -> float | None:
    """``exp(value)`` when it is finite, else ``None``."""
    if value is None:
        return None
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _coefficients(model: _Model, names: Sequence[str]) -> Table:
    """The coefficients table: the intercept first, then one row per predictor."""
    rows: list[tuple[Cell, ...]] = []
    for index, term in enumerate((INTERCEPT, *names)):
        lower = _optional(model.lower, model.bounded, index)
        upper = _optional(model.upper, model.bounded, index)
        rows.append(
            (
                term,
                float(model.beta[index]),
                _optional(model.se, model.defined, index),
                _optional(model.statistic, model.defined, index),
                _optional(model.p_value, model.defined, index),
                lower,
                upper,
                _odds(float(model.beta[index])),
                _odds(lower),
                _odds(upper),
            )
        )
    return table(
        "Coefficients",
        _COEFFICIENT_COLUMNS,
        rows,
        note=(
            "The estimate is the change in the log odds of the event for a one-unit change in "
            "the term; the odds ratio is its exponential. SE is the square root of the inverse "
            "information matrix's diagonal, and the interval is Wald's at the stated level. "
            "The odds ratio interval is the exponential of that interval."
        ),
    )


def _classification(model: _Model, cases: _Cases) -> Table:
    """The classification at the 0.5 cut-off, by observed outcome, with a Total row."""
    predicted = model.fitted >= 0.5
    positive = cases.y == 1.0
    true_positive = int(np.count_nonzero(predicted & positive))
    false_negative = int(np.count_nonzero(~predicted & positive))
    false_positive = int(np.count_nonzero(predicted & ~positive))
    true_negative = int(np.count_nonzero(~predicted & ~positive))
    return table(
        "Classification at 0.5",
        _CLASSIFICATION_COLUMNS,
        (
            (cases.event, true_positive, false_negative),
            (cases.other, false_positive, true_negative),
            ("Total", true_positive + false_positive, false_negative + true_negative),
        ),
        note=(
            "Predicted event means a fitted probability of 0.5 or more. The event row's diagonal "
            "is the count of correctly predicted events, and the other row's second cell the "
            "count of correctly predicted non-events."
        ),
    )


def _predictions(
    model: _Model,
    points: Sequence[Sequence[float]],
    level: float,
) -> Table:
    """One row per requested point: the fitted probability and its Wald interval."""
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    rows: list[tuple[Cell, ...]] = []
    for index, values in enumerate(points, start=1):
        row = np.concatenate(([1.0], np.asarray(values, dtype=np.float64)))
        eta = float(row @ model.beta)
        variance = float(row @ model.covariance @ row)
        if variance > 0.0 and math.isfinite(variance):
            half = critical * math.sqrt(variance)
            lower = _optional_finite(float(scipy_special.expit(eta - half)))
            upper = _optional_finite(float(scipy_special.expit(eta + half)))
        else:
            lower = upper = None
        rows.append((index, float(scipy_special.expit(eta)), lower, upper))
    return table(
        "Predictions",
        _PREDICTION_COLUMNS,
        rows,
        note=(
            "The predicted probability is the fitted value at the given predictor values; its "
            "interval is Wald's on the logit scale, back-transformed to probabilities, at the "
            "stated level."
        ),
    )


def _optional_finite(value: float) -> float | None:
    """``value`` when it is finite, else ``None``."""
    return value if math.isfinite(value) else None


def _calibration_plot(
    groups: Sequence[NDArray[np.intp]], outcome: FloatArray, fitted: FloatArray
) -> ScatterPlot:
    """The observed proportion against the mean fitted probability, with the identity line."""
    x = tuple(float(np.mean(fitted[group])) for group in groups)
    y = tuple(float(np.mean(outcome[group])) for group in groups)
    return ScatterPlot(
        title="Calibration",
        x_label="Mean predicted probability",
        y_label="Observed proportion",
        x=x,
        y=y,
        lines=(Line(x0=0.0, y0=0.0, x1=1.0, y1=1.0, label="Perfect calibration"),),
        note=(
            "Each point is one group of the rows sorted by fitted probability; a point on the "
            "line means the predicted probability matched the observed proportion."
        ),
    )


def _likelihood_ratio_test(model: _Model, mcfadden: float) -> HypothesisTest:
    """The likelihood ratio test of the model, with McFadden's pseudo R squared as its effect."""
    statistic = max(0.0, 2.0 * (model.log_likelihood - model.null_log_likelihood))
    p_value = float(scipy_stats.chi2.sf(statistic, model.k))
    return HypothesisTest(
        name="Likelihood ratio test of the model",
        method="Likelihood ratio test, chi-square",
        null_hypothesis=(
            "Every predictor's coefficient is zero, so the predictors explain none of the "
            "log odds of the event."
        ),
        statistic=statistic,
        df=float(model.k),
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=Estimate(label="McFadden's pseudo R squared", value=mcfadden),
    )


def _wald_tests(model: _Model, names: Sequence[str]) -> tuple[HypothesisTest, ...]:
    """One two-sided Wald z test per coefficient, with the coefficient and its interval."""
    tests: list[HypothesisTest] = []
    for index, term in enumerate((INTERCEPT, *names)):
        defined = bool(model.defined[index])
        effect = Estimate(
            label=term,
            value=float(model.beta[index]),
            se=_optional(model.se, model.defined, index),
            lower=_optional(model.lower, model.bounded, index),
            upper=_optional(model.upper, model.bounded, index),
        )
        tests.append(
            HypothesisTest(
                name=f"{term} Wald test, two-sided",
                method="Wald z test, two-sided",
                null_hypothesis=f"the coefficient of {term} is zero",
                statistic=_optional(model.statistic, model.defined, index),
                p_value=_optional(model.p_value, model.defined, index),
                p_value_text=p_text(_optional(model.p_value, model.defined, index)),
                effect=effect,
                reason=None
                if defined
                else "the information matrix leaves this coefficient's standard error undefined",
            )
        )
    return tuple(tests)


def _separating_predictor(model: _Model) -> int:
    """The predictor whose coefficient is largest in magnitude, as an index into ``names``."""
    return int(np.argmax(np.abs(model.beta[1:]))) if model.k else 0


def _separation_warning(model: _Model, names: Sequence[str]) -> str | None:
    """A warning when the fit looks completely or quasi-completely separated, else ``None``."""
    extreme = bool(
        np.any(model.fitted < SEPARATION_PROBABILITY)
        or np.any(model.fitted > 1.0 - SEPARATION_PROBABILITY)
    )
    largest = float(np.max(np.abs(model.beta[1:]))) if model.k else 0.0
    large = largest > SEPARATION_COEFFICIENT
    if not extreme and not large:
        return None
    index = _separating_predictor(model)
    name = names[index]
    reasons = []
    if extreme:
        reasons.append("a fitted probability comes within 1e-8 of 0 or 1")
    if large:
        reasons.append(
            f"the coefficient of {name} is {num(float(model.beta[index + 1]))} in magnitude, "
            f"above {num(SEPARATION_COEFFICIENT)}"
        )
    return (
        "The fit looks like complete or quasi-complete separation: "
        + " and ".join(reasons)
        + f"; the coefficients are shown, but the standard errors and the odds ratio interval "
        f"of {name} cannot be trusted."
    )


def _convergence_warning(model: _Model) -> str | None:
    """A warning when the fit did not converge, naming the last change in the log-likelihood."""
    if model.converged:
        return None
    if model.last_change is None:
        return (
            f"The fit did not converge in {MAX_ITERATIONS} iterations; the log-likelihood could "
            "not be evaluated at the last step."
        )
    return (
        f"The fit did not converge in {MAX_ITERATIONS} iterations; the last change in the "
        f"log-likelihood was {num(model.last_change)}."
    )


def _interpretation(
    model: _Model,
    names: Sequence[str],
    cases: _Cases,
    level: float,
) -> str:
    """Three sentences: the event, the odds ratios that exclude 1, and what association means."""
    sentences = [
        f"This model explains the log odds of {cases.event!r} against {cases.other!r} in the "
        f"{model.n} rows used."
    ]
    excluding = [
        name
        for index, name in enumerate(names, start=1)
        if bool(model.bounded[index]) and (model.lower[index] > 0.0 or model.upper[index] < 0.0)
    ]
    if excluding:
        sentences.append(
            f"The odds ratio interval of {listing(excluding, 'predictor')} excludes 1 at the "
            f"{num(level * 100)} % level."
        )
    else:
        sentences.append(
            f"No predictor's odds ratio interval excludes 1 at the {num(level * 100)} % level."
        )
    sentences.append(
        "An odds ratio measures association in these rows; it does not show that a predictor "
        "causes the outcome."
    )
    return " ".join(sentences)


def _methods(event: str, other: str) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Logistic regression models the log odds of "
        f"{event!r} against {other!r} as an intercept plus one coefficient per predictor, fit "
        "by iteratively reweighted least squares (Newton-Raphson), at most "
        f"{MAX_ITERATIONS} iterations, until the log-likelihood changes by less than "
        f"{num(CONVERGENCE)}. The standard error of a coefficient is the square root of the "
        "diagonal of the inverse information matrix at convergence, and its Wald interval is "
        "the coefficient plus or minus the normal quantile times that standard error. The odds "
        "ratio is the exponential of the coefficient. McFadden's pseudo R squared is one minus "
        "the log-likelihood of the fitted model over that of the intercept-only model; "
        "Nagelkerke's pseudo R squared divides the Cox-Snell value, one minus the exponential "
        "of two over n times the null minus the fitted log-likelihood, by its maximum, one "
        "minus the exponential of two over n times the null log-likelihood. The area under the "
        "ROC curve is the probability that a randomly chosen event row scores above a randomly "
        "chosen non-event row, with ties counted as one half. The Hosmer-Lemeshow statistic "
        "sorts the rows by fitted probability into at most ten groups and compares the observed "
        "and expected event counts in each."
    )


def run_logistic(request: BaseModel) -> ToolResult:
    """Run the logistic regression tool on its own request."""
    if not isinstance(request, LogisticRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"logistic regression takes a LogisticRequest, not a request of "
            f"{type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: LogisticRequest) -> ToolResult:
    """The fit itself, once the request is known to be this tool's own."""
    check_level(request.level)
    _require_predictors(request)
    _require_names(request)
    _require_predictions(request)
    found = _require_levels(request.outcome)
    cases = _complete_cases(request.outcome, request.predictors, found)
    names = (
        tuple(request.names)
        if request.names is not None
        else tuple(predictor.name for predictor in request.predictors)
    )
    size = len(names)
    if cases.y.size < size + 2:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"logistic regression needs at least {size + 2} complete rows for {size} "
            f"{'predictor' if size == 1 else 'predictors'}, and {cases.y.size} are left",
        )
    _require_varying(names, cases.predictors)
    design = np.column_stack([np.ones(cases.y.size), *cases.predictors])
    _require_independent(names, design)
    model = _fit(design, cases.y, request.level)
    mcfadden, nagelkerke = _pseudo_r_squared(model)
    sensitivity, specificity, accuracy = _threshold_rates(cases.y, model.fitted)
    groups = _deciles(model.fitted)
    calibration = _hosmer_lemeshow(groups, cases.y, model.fitted)
    warnings = [
        warning
        for warning in (
            _convergence_warning(model),
            _separation_warning(model, names),
        )
        if warning is not None
    ]
    smallest = min(group.size for group in groups)
    if smallest < SMALL_GROUP:
        warnings.append(
            f"A Hosmer-Lemeshow group holds only {smallest} observations, so the calibration "
            "check is coarse."
        )
    if model.n < FEW_ROWS:
        warnings.append(
            f"Only {model.n} rows are used, fewer than {FEW_ROWS}; the fit and its checks rest "
            "on very little data."
        )
    estimates = (
        Estimate(label="McFadden's pseudo R squared", value=mcfadden),
        Estimate(label="Nagelkerke's pseudo R squared", value=nagelkerke),
        Estimate(label="AIC", value=model.aic),
        Estimate(label="Area under the ROC curve", value=_auc(cases.y, model.fitted)),
        Estimate(label="Sensitivity at 0.5", value=sensitivity),
        Estimate(label="Specificity at 0.5", value=specificity),
        Estimate(label="Accuracy at 0.5", value=accuracy),
    )
    assumptions = (
        _calibration_check(calibration),
        _linearity_check(),
        _independence_check(),
    )
    tables: list[Table] = [
        _coefficients(model, names),
        _classification(model, cases),
    ]
    if request.predictions:
        tables.append(_predictions(model, request.predictions, request.level))
    listed, unlisted = list_excluded(cases.excluded)
    excluded = len(cases.excluded)
    return ToolResult(
        tool="logistic",
        title="Logistic regression",
        level=request.level,
        estimates=estimates,
        tests=(_likelihood_ratio_test(model, mcfadden), *_wald_tests(model, names)),
        assumptions=assumptions,
        warnings=tuple(warnings),
        interpretation=_interpretation(model, names, cases, request.level),
        methods_fragment=_methods(cases.event, cases.other),
        counts=Counts(
            n_offered=cases.offered,
            n_used=model.n,
            n_excluded=excluded,
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=(_calibration_plot(groups, cases.y, model.fitted),),
    )


TOOLS.register(
    ToolSpec(
        name="logistic",
        family="relationships",
        title="Logistic regression",
        request=LogisticRequest,
        run=run_logistic,
    )
)
