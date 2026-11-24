"""Linear regression of one numeric outcome on one to twenty numeric predictors.

The tool aligns the outcome and every predictor on the rows complete in all of them, fits
the model by ordinary least squares, and reports what the fit found: the coefficients with
their standard errors, t statistics, p-values and intervals, the standardised coefficients
(beta weights), R squared, adjusted R squared and the residual standard error, the F test of
the regression, the variance inflation factor of each predictor, and prediction intervals at
predictor values a person gives. One predictor is drawn as a scatter plot with the fitted
line and the confidence band of the mean response; the residuals against the fitted values
and a normal Q-Q plot of the residuals are drawn always.

A p-value is a statement about evidence, never a statement of proof. The assumptions the
model makes are checked where the data can show them: Shapiro-Wilk's W test for normality of
the residuals, the studentized Breusch-Pagan test for a constant variance, and the exact
Durbin-Watson test, whose p-values depend on the design matrix of this very model, for
independence in the order of the rows. Linearity cannot be read from one number and is left
to the residuals plotted against each predictor.

A predictor that does not vary, or that is exactly collinear with the others, is refused:
its own coefficient cannot be estimated. Fewer than one residual degree of freedom is
refused too. Every row left out is counted and listed, never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.diagnostics import breusch_pagan, shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import joined, listing, num, p_threshold
from curvestudio.tools.durbin_watson import durbin_watson_test
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
    Band,
    Cell,
    ColumnKind,
    FloatArray,
    Line,
    QQPlot,
    ScatterPlot,
    Table,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PREDICTORS = 1
MAX_PREDICTORS = 20
MAX_PREDICTION_ROWS = 1_000
"""The most predictor rows a prediction request may carry."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

COLLINEAR_RTOL = 1e-10
"""A predictor counts as collinear with the others when the design's smallest singular value
is this small a fraction of its largest."""

EXACT_RATIO = 1e-20
"""The residuals count as exactly zero when their sum of squares is this small a fraction of
the outcome's total: a perfect fit leaves only the rounding of a float behind."""

_BAND_POINTS = 100
"""How many predictor values the confidence band of the fitted line is drawn at."""

INTERCEPT = "(Intercept)"
"""How the intercept reads as a term of the model."""

_NAMES_LABEL = "Predictor names"

_COEFFICIENT_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("estimate", "Estimate", "number"),
    ("se", "SE", "number"),
    ("statistic", "Statistic", "number"),
    ("p", "p", "p_value"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
    ("standardised", "Standardised", "number"),
)

_VIF_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("term", "Term", "text"),
    ("vif", "VIF", "number"),
)

_PREDICTION_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("row", "Row", "int"),
    ("predicted", "Predicted", "number"),
    ("ci_lower", "CI lower", "number"),
    ("ci_upper", "CI upper", "number"),
    ("pi_lower", "PI lower", "number"),
    ("pi_upper", "PI upper", "number"),
)


class RegressionRequest(BaseModel):
    """The outcome, the predictors, their names, the level and any prediction points."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Column = Field(
        description="The numeric column the model explains; the response or dependent variable."
    )
    predictors: tuple[Column, ...] = Field(
        description=(
            "One to twenty numeric columns that explain the outcome. Each predictor contributes "
            "its own coefficient to the model."
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


class _Model(NamedTuple):
    """The fitted model and everything a result shows of it, on the rows it was fit to."""

    n: int
    k: int
    df_residual: int
    beta: FloatArray
    se: FloatArray
    statistic: FloatArray | None
    p_value: FloatArray | None
    lower: FloatArray
    upper: FloatArray
    standardised: FloatArray
    fitted: FloatArray
    residuals: FloatArray
    r_squared: float
    adjusted_r_squared: float
    residual_standard_error: float
    f_statistic: float | None
    f_p_value: float | None
    inverse: FloatArray
    """``(X'X)`` inverse, from the pseudo-inverse: the leverage of a predictor row."""
    variance: float
    """The residual variance, ``SSE / (n - k - 1)``."""
    vif: FloatArray
    exact: bool
    """The residuals are zero to the last bit a float holds, so the tests are undefined."""


def _require_predictors(request: RegressionRequest) -> None:
    """Refuse a predictor count outside one to twenty."""
    size = len(request.predictors)
    if not MIN_PREDICTORS <= size <= MAX_PREDICTORS:
        raise ToolError(
            "INVALID_OPTION",
            f"regression takes between {MIN_PREDICTORS} and {MAX_PREDICTORS} predictors, "
            f"not {size}",
        )


def _require_names(request: RegressionRequest) -> None:
    """Refuse names that do not name one predictor each."""
    if request.names is None:
        return
    size, names = len(request.predictors), len(request.names)
    if names != size:
        raise ToolError(
            "INVALID_OPTION",
            f"regression takes one name per predictor, so {size} names, not {names}",
        )


def _require_predictions(request: RegressionRequest) -> None:
    """Refuse more than 1,000 prediction rows, a row of the wrong length, or a non-finite one."""
    if request.predictions is None:
        return
    rows = len(request.predictions)
    if rows > MAX_PREDICTION_ROWS:
        raise ToolError(
            "INVALID_OPTION",
            f"regression predicts at {MAX_PREDICTION_ROWS:,} rows at most, not {rows:,}",
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


def _require_outcome(name: str, values: FloatArray) -> None:
    """Refuse a constant outcome: its variation, and so R squared, is undefined."""
    if _constant(values):
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the outcome {name!r} holds one value repeated, so its variation and R squared "
            "are undefined",
        )


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


def _variance_inflation(design: FloatArray) -> FloatArray:
    """The variance inflation factor of each predictor, from its own R squared.

    The factor is ``1 / (1 - R**2)`` of the regression of that predictor on the intercept and
    the other predictors. A factor that would be infinite means the predictors are collinear,
    which the fit has already refused.
    """
    k = design.shape[1] - 1
    if k < 2:
        return np.empty(0, dtype=np.float64)
    values = np.empty(k, dtype=np.float64)
    for column in range(1, k + 1):
        response = design[:, column]
        others = np.delete(design, column, axis=1)
        solution = np.linalg.lstsq(others, response, rcond=None)[0]
        residual = response - others @ solution
        sse = float(residual @ residual)
        sst = float(((response - float(np.mean(response))) ** 2).sum())
        r_squared = 1.0 - sse / sst
        remaining = 1.0 - r_squared
        if remaining <= 0.0:
            raise ToolError(
                "COLLINEAR",
                "a predictor is exactly collinear with the others, so its variance inflation "
                "factor is undefined",
            )
        values[column - 1] = 1.0 / remaining
    return values


def _fit(design: FloatArray, outcome: FloatArray, level: float) -> _Model:
    """Fit ``outcome`` on the ``design`` matrix (whose first column is the intercept).

    Raises:
        ToolError: ``COLLINEAR`` when the design's columns are dependent; ``INVALID_OPTION``
            for a confidence level outside (0, 1).
    """
    check_level(level)
    n, parameters = design.shape
    k = parameters - 1
    df_residual = n - parameters
    pinv = np.linalg.pinv(design)
    beta = pinv @ outcome
    fitted = design @ beta
    residuals = outcome - fitted
    sse = float(residuals @ residuals)
    mean = float(np.mean(outcome))
    sst = float(((outcome - mean) ** 2).sum())
    variance = sse / df_residual
    inverse = pinv @ pinv.T
    r_squared = 1.0 - sse / sst
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / df_residual
    standardised = beta[1:] * np.std(design[:, 1:], axis=0, ddof=1) / float(np.std(outcome, ddof=1))
    vif = _variance_inflation(design)
    exact = sse <= EXACT_RATIO * sst
    if exact:
        se = np.zeros(parameters, dtype=np.float64)
        return _Model(
            n=n,
            k=k,
            df_residual=df_residual,
            beta=beta,
            se=se,
            statistic=None,
            p_value=None,
            lower=beta.copy(),
            upper=beta.copy(),
            standardised=standardised,
            fitted=fitted,
            residuals=residuals,
            r_squared=r_squared,
            adjusted_r_squared=adjusted,
            residual_standard_error=0.0,
            f_statistic=None,
            f_p_value=None,
            inverse=inverse,
            variance=variance,
            vif=vif,
            exact=True,
        )
    se = np.sqrt(np.maximum(variance * np.diag(inverse), 0.0))
    statistic = beta / se
    p_value = 2.0 * scipy_stats.t.sf(np.abs(statistic), df_residual)
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_residual))
    lower = beta - critical * se
    upper = beta + critical * se
    f_statistic = ((sst - sse) / k) / variance
    f_p_value = float(scipy_stats.f.sf(f_statistic, k, df_residual))
    return _Model(
        n=n,
        k=k,
        df_residual=df_residual,
        beta=beta,
        se=se,
        statistic=statistic,
        p_value=p_value,
        lower=lower,
        upper=upper,
        standardised=standardised,
        fitted=fitted,
        residuals=residuals,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        residual_standard_error=math.sqrt(variance),
        f_statistic=float(f_statistic),
        f_p_value=f_p_value,
        inverse=inverse,
        variance=variance,
        vif=vif,
        exact=False,
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


def _normality(residuals: FloatArray, exact: bool) -> AuditCheck:
    """The Shapiro-Wilk check of normality of the residuals."""
    if exact:
        return assumption(
            "normality",
            "The normality of the residuals could not be checked",
            Verdict.NOT_APPLICABLE,
            "The fit is exact, so the residuals are zero and their distribution cannot be "
            "read from them.",
        )
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


def _constant_variance(residuals: FloatArray, fitted: FloatArray, exact: bool) -> AuditCheck:
    """The studentized Breusch-Pagan check that the residual variance is constant."""
    if exact:
        return assumption(
            "constant_variance",
            "The constancy of the residual variance could not be checked",
            Verdict.NOT_APPLICABLE,
            "The fit is exact, so the residuals are zero and their variance cannot be read "
            "from them.",
        )
    found = breusch_pagan(residuals, fitted)
    if found.status != "ok" or found.p_value is None:
        return assumption(
            "constant_variance",
            "The constancy of the residual variance could not be checked",
            Verdict.NOT_APPLICABLE,
            found.reason or "The Breusch-Pagan test could not be run on these residuals.",
        )
    assert found.statistic is not None
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The residual variance looks constant",
        Verdict.CAUTION: "The residual variance may change with the fitted value",
        Verdict.FAIL: "The residual variance changes with the fitted value",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the spread grows with the fitted value, the standard errors are too small; "
        "weighted least squares or a transformation of the outcome may help."
    )
    return assumption(
        "constant_variance",
        titles[verdict],
        verdict,
        f"The studentized Breusch-Pagan test gives LM = {num(found.statistic)}, "
        f"{p_text(found.p_value)}.",
        statistic=f"LM = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _durbin_watson_statistic(residuals: FloatArray) -> float:
    """``sum((e[t] - e[t-1])**2) / sum(e**2)``, the Durbin-Watson statistic of the residuals."""
    differences = np.diff(residuals)
    return float(differences @ differences) / float(residuals @ residuals)


def _independence(residuals: FloatArray, design: FloatArray, exact: bool) -> AuditCheck:
    """The exact Durbin-Watson check of independence in the order of the rows."""
    if exact:
        return assumption(
            "independence",
            "The independence of the residuals could not be checked",
            Verdict.NOT_APPLICABLE,
            "The residuals are all zero, so the Durbin-Watson statistic is undefined.",
        )
    statistic = _durbin_watson_statistic(residuals)
    found = durbin_watson_test(statistic, design)
    if found is None:
        return assumption(
            "independence",
            "The independence of the residuals was not tested",
            Verdict.NOT_APPLICABLE,
            f"The Durbin-Watson statistic is {num(statistic)}, but its exact p-value is not "
            "computed above 1,000 rows.",
            statistic=f"D = {num(statistic)}",
            next_step="Look at the residuals against the fitted values and in the order of the "
            "rows.",
        )
    p_value = min(1.0, 2.0 * min(found.positive, found.negative))
    verdict = _verdict(p_value)
    titles = {
        Verdict.PASS: "The residuals look independent in the order of the rows",
        Verdict.CAUTION: "The residuals may be dependent in the order of the rows",
        Verdict.FAIL: "The residuals are dependent in the order of the rows",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the residuals wander or alternate in the order of the rows, the observations "
        "are not independent and the standard errors and p-values are unreliable."
    )
    return assumption(
        "independence",
        titles[verdict],
        verdict,
        "The exact Durbin-Watson test for this design gives "
        f"D = {num(statistic)}, {p_text(p_value)}.",
        statistic=f"D = {num(statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _linearity() -> AuditCheck:
    """The linearity the model assumes, which only a residual plot can show."""
    return assumption(
        "linearity",
        "A straight-line relation with each predictor is what the model assumes",
        Verdict.NOT_APPLICABLE,
        "The linearity of the fit cannot be read from one number; a curve in the residuals "
        "against a predictor is what would show it.",
        next_step="Look at the residuals against each predictor.",
    )


def _coefficients(model: _Model, names: Sequence[str]) -> Table:
    """The coefficients table: the intercept first, then one row per predictor."""
    rows: list[tuple[Cell, ...]] = [
        (
            INTERCEPT,
            float(model.beta[0]),
            float(model.se[0]),
            None if model.statistic is None else float(model.statistic[0]),
            None if model.p_value is None else float(model.p_value[0]),
            float(model.lower[0]),
            float(model.upper[0]),
            None,
        )
    ]
    for index, name in enumerate(names, start=1):
        rows.append(
            (
                name,
                float(model.beta[index]),
                float(model.se[index]),
                None if model.statistic is None else float(model.statistic[index]),
                None if model.p_value is None else float(model.p_value[index]),
                float(model.lower[index]),
                float(model.upper[index]),
                float(model.standardised[index - 1]),
            )
        )
    return table(
        "Coefficients",
        _COEFFICIENT_COLUMNS,
        rows,
        note=(
            "SE is the standard error of the estimate; the statistic is its t value on "
            "n - k - 1 degrees of freedom, and the interval is at the stated level. The "
            "standardised coefficient is the coefficient of the standardised variables, which "
            "puts the predictors on one scale; the intercept has none."
        ),
    )


def _vif_table(model: _Model, names: Sequence[str]) -> Table:
    """The variance inflation table, one row per predictor."""
    return table(
        "Variance inflation",
        _VIF_COLUMNS,
        ((name, float(factor)) for name, factor in zip(names, model.vif, strict=True)),
        note=(
            "The variance inflation factor of a predictor is one over one minus the R squared "
            "of that predictor regressed on the other predictors; it is 1 when the predictors "
            "are uncorrelated, and large values mean the coefficient is poorly determined."
        ),
    )


def _prediction_row(
    index: int, values: Sequence[float], model: _Model, level: float
) -> tuple[Cell, ...]:
    """One prediction: the fitted value, its interval for the mean, and its interval for a
    single new observation."""
    centered = np.concatenate(([1.0], np.asarray(values, dtype=np.float64)))
    predicted = float(centered @ model.beta)
    leverage = float(centered @ model.inverse @ centered)
    if model.exact:
        return (index, predicted, predicted, predicted, predicted, predicted)
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, model.df_residual))
    se_mean = math.sqrt(max(0.0, model.variance * leverage))
    se_observation = math.sqrt(max(0.0, model.variance * (1.0 + leverage)))
    return (
        index,
        predicted,
        predicted - critical * se_mean,
        predicted + critical * se_mean,
        predicted - critical * se_observation,
        predicted + critical * se_observation,
    )


def _predictions(model: _Model, predictions: Sequence[Sequence[float]], level: float) -> Table:
    """The predictions table, one numbered row per requested predictor row."""
    return table(
        "Predictions",
        _PREDICTION_COLUMNS,
        (
            _prediction_row(index, values, model, level)
            for index, values in enumerate(predictions, start=1)
        ),
        note=(
            "The confidence interval is for the mean response at the given predictor values; "
            "the prediction interval is for a single new observation, which adds the residual "
            "variance. Both are at the stated level."
        ),
    )


def _line_plot(
    model: _Model,
    predictor: FloatArray,
    outcome: FloatArray,
    *,
    outcome_name: str,
    predictor_name: str,
    level: float,
) -> ScatterPlot:
    """The data with the fitted line and the confidence band of the mean response."""
    shown_x, note = stride(predictor)
    shown_y, _ = stride(outcome)
    low, high = float(predictor.min()), float(predictor.max())
    grid = np.linspace(low, high, _BAND_POINTS)
    fitted = model.beta[0] + model.beta[1] * grid
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, model.df_residual))
    centered = np.column_stack((np.ones(grid.size), grid))
    leverage = np.einsum("ij,jk,ik->i", centered, model.inverse, centered)
    half = critical * np.sqrt(np.maximum(0.0, model.variance * leverage))
    return ScatterPlot(
        title=f"Regression of {outcome_name} on {predictor_name}",
        x_label=predictor_name,
        y_label=outcome_name,
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(
            Line(
                x0=low,
                y0=float(model.beta[0] + model.beta[1] * low),
                x1=high,
                y1=float(model.beta[0] + model.beta[1] * high),
                label="Fitted line",
            ),
        ),
        band=Band(
            x=tuple(float(value) for value in grid),
            lower=tuple(float(value) for value in fitted - half),
            upper=tuple(float(value) for value in fitted + half),
        ),
        note=note,
    )


def _residual_plot(model: _Model) -> ScatterPlot:
    """The residuals against the fitted values, with a line at zero."""
    shown_x, note = stride(model.fitted)
    shown_y, _ = stride(model.residuals)
    low, high = float(shown_x.min()), float(shown_x.max())
    return ScatterPlot(
        title="Residuals against fitted values",
        x_label="Fitted value",
        y_label="Residual",
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(Line(x0=low, y0=0.0, x1=high, y1=0.0, label="Zero"),),
        note=note,
    )


def _residual_qq(model: _Model) -> QQPlot:
    """The normal Q-Q plot of the residuals, at Blom positions with their own mean and spread."""
    ordered = np.sort(model.residuals)
    size = ordered.size
    blom = (np.arange(1.0, size + 1.0) - 0.375) / (size + 0.25)
    mean = float(np.mean(model.residuals))
    spread = float(np.std(model.residuals, ddof=1))
    theoretical = mean + spread * scipy_stats.norm.ppf(blom)
    shown_theoretical, note = stride(theoretical)
    shown_ordered, _ = stride(ordered)
    return QQPlot(
        title="Normal Q-Q plot of the residuals",
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


def _evidence(p_value: float) -> str:
    """What the F test's p-value says about the null hypothesis, as a clause of evidence."""
    if p_value >= CAUTION_P:
        return "the data are consistent with every predictor's coefficient being zero"
    if p_value >= FAIL_P:
        return "the data show evidence that at least one predictor's coefficient is not zero"
    return "the data show strong evidence that at least one predictor's coefficient is not zero"


def _strongest(model: _Model) -> int | None:
    """The index of the predictor whose coefficient carries the smallest p-value, or ``None``."""
    if model.p_value is None or model.k < 1:
        return None
    tail = np.asarray(model.p_value[1:], dtype=np.float64)
    return int(np.argmin(tail)) + 1


def _interpretation(model: _Model, names: Sequence[str], outcome_name: str, level: float) -> str:
    """One to three sentences: the fit, the F test, and the strongest coefficient."""
    if model.exact:
        return (
            f"The fit is exact: {outcome_name} lies on the fitted plane at every row, so R "
            "squared is 1 and there is no residual variation to test."
        )
    assert model.f_p_value is not None
    sentences = [
        f"The model explains {num(model.r_squared * 100)} % of the variation in "
        f"{outcome_name}; the F test on {num(model.k)} and {num(model.df_residual)} degrees of "
        f"freedom gives {p_text(model.f_p_value)}, so {_evidence(model.f_p_value)}."
    ]
    if model.k > 1:
        index = _strongest(model)
        assert index is not None and model.p_value is not None
        sentences.append(
            f"The coefficient of {names[index - 1]} carries the strongest evidence, with "
            f"{p_text(float(model.p_value[index]))} and a {num(level * 100)} % interval from "
            f"{num(float(model.lower[index]))} to {num(float(model.upper[index]))}."
        )
    return " ".join(sentences)


def _methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Ordinary least squares fits the outcome as an intercept plus one coefficient per "
        "predictor. The F test of the regression has one numerator degree of freedom per "
        "predictor and n minus the number of parameters in the denominator; each coefficient "
        "is tested with the t statistic on those degrees of freedom, and its interval is the "
        "coefficient plus or minus t times its standard error at the stated level. The "
        "standardised coefficient is the coefficient of the standardised variables, and the "
        "variance inflation factor of a predictor is one over one minus the R squared of that "
        "predictor regressed on the others. Shapiro-Wilk's W test checks normality of the "
        "residuals, the studentized Breusch-Pagan test the constancy of their variance, and "
        "the exact Durbin-Watson test their independence in the order of the rows. A "
        "prediction interval adds the residual variance to the interval for the mean."
    )


def _f_test(model: _Model) -> HypothesisTest:
    """The F test of the regression, with R squared as its effect size."""
    return HypothesisTest(
        name="F test of the regression",
        method="Ordinary least squares regression, F test",
        null_hypothesis=(
            "Every predictor's coefficient is zero, so the predictors explain none of the "
            "variation in the outcome."
        ),
        statistic=model.f_statistic,
        df=float(model.k),
        df_denominator=float(model.df_residual),
        p_value=model.f_p_value,
        p_value_text=p_text(model.f_p_value),
        effect=Estimate(label="R squared", value=model.r_squared),
        reason=None
        if model.f_p_value is not None
        else "the residuals are zero, so the fit is exact and the F statistic is undefined",
    )


def run_regression(request: BaseModel) -> ToolResult:
    """Run the regression tool on its own request."""
    if not isinstance(request, RegressionRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"regression takes a RegressionRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: RegressionRequest) -> ToolResult:
    """The fit itself, once the request is known to be this tool's own."""
    check_level(request.level)
    _require_predictors(request)
    _require_names(request)
    _require_predictions(request)
    pairing = pairs(request.outcome, *request.predictors)
    outcome = pairing.columns[0]
    used = pairing.columns[1:]
    names = (
        tuple(request.names) if request.names is not None else tuple(column.name for column in used)
    )
    size = len(used)
    if used[0].values.size < size + 2:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the regression needs at least {size + 2} complete rows for {size} "
            f"{'predictor' if size == 1 else 'predictors'}, and {used[0].values.size} are "
            "left",
        )
    _require_outcome(outcome.name, outcome.values)
    _require_varying(names, [column.values for column in used])
    design = np.column_stack([np.ones(outcome.values.size), *[c.values for c in used]])
    _require_independent(names, design)
    model = _fit(design, outcome.values, request.level)
    estimates = (
        Estimate(label="R squared", value=model.r_squared),
        Estimate(label="Adjusted R squared", value=model.adjusted_r_squared),
        Estimate(label="Residual standard error", value=model.residual_standard_error),
    )
    assumptions = (
        _normality(model.residuals, model.exact),
        _constant_variance(model.residuals, model.fitted, model.exact),
        _independence(model.residuals, design, model.exact),
        _linearity(),
    )
    tables: list[Table] = [_coefficients(model, names)]
    if size >= 2:
        tables.append(_vif_table(model, names))
    if request.predictions:
        tables.append(_predictions(model, request.predictions, request.level))
    plots: list[ScatterPlot | QQPlot] = []
    if size == 1:
        plots.append(
            _line_plot(
                model,
                used[0].values,
                outcome.values,
                outcome_name=outcome.name,
                predictor_name=names[0],
                level=request.level,
            )
        )
    plots.extend((_residual_plot(model), _residual_qq(model)))
    listed, unlisted = list_excluded(pairing.excluded)
    excluded = len(pairing.excluded)
    title = (
        f"Linear regression of {outcome.name} on {joined(list(names))}"
        if size <= 3
        else f"Linear regression of {outcome.name} on {size} predictors"
    )
    return ToolResult(
        tool="regression",
        title=title,
        level=request.level,
        estimates=estimates,
        tests=(_f_test(model),),
        assumptions=assumptions,
        interpretation=_interpretation(model, names, outcome.name, request.level),
        methods_fragment=_methods(),
        counts=Counts(
            n_offered=used[0].values.size + excluded,
            n_used=used[0].values.size,
            n_excluded=excluded,
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=tuple(plots),
    )


TOOLS.register(
    ToolSpec(
        name="regression",
        family="relationships",
        title="Linear regression",
        request=RegressionRequest,
        run=run_regression,
    )
)
