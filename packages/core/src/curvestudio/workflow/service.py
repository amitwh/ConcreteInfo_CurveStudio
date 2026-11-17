"""``fit_dataset`` and ``predict``: the numbers both the desktop app and the API show.

Configuration mistakes (unknown model, impossible method/loss/bounds, a bad equation)
raise :class:`WorkflowError`. Numerical outcomes never raise: a fit that does not
converge comes back as a bundle whose ``result.status`` says so, with ``curve=None`` and
a reason, and an undefined band or statistic is ``None`` with a reason.
"""

import ast
import copy
import hashlib
import math
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.diagnostics import FitReport, describe_fit
from curvestudio.expressions import RESERVED_NAMES, split_equation, validate_expression
from curvestudio.expressions.grammar import parse_expression
from curvestudio.fitting import (
    FitConfig,
    FitResult,
    FitStatus,
    collect_provenance,
    fit,
    fit_polynomial,
)
from curvestudio.models import ModelSpec, Parameter
from curvestudio.statistics import (
    Band,
    FitView,
    StatisticsError,
    build_view,
    confidence_band,
    prediction_band,
)
from curvestudio.workflow.bundle import (
    BAND_LEVEL,
    COMPACT_MAX_POINTS,
    CURVE_MARGIN,
    CURVE_POINTS,
    MAX_PREDICT_POINTS,
    CurveSamples,
    DataSnapshot,
    FitBundle,
    Prediction,
)
from curvestudio.workflow.catalog import registered_spec
from curvestudio.workflow.errors import WorkflowError
from curvestudio.workflow.spec import CustomRef, FitSpec, PolynomialRef, RegisteredRef

FloatArray = NDArray[np.float64]

CUSTOM_MODEL_ID = "custom"
_POLYNOMIAL_ID = re.compile(r"poly(\d+)(_origin)?")


class _TimeLimitExceeded(Exception):
    """Raised inside model evaluation once the fit's wall-clock budget is spent."""


# -- model resolution ------------------------------------------------------------------------------


_NAME_TOKEN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*")
"""Numeric literals first, so the ``e`` of ``1e-3`` is not read as a name."""


def _names_in(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in _NAME_TOKEN.findall(text):
        if token[0].isalpha() or token[0] == "_":
            seen[token] = None
    return list(seen)


def _custom_model(ref: CustomRef, initial: dict[str, float]) -> ModelSpec:
    def guess(_x: FloatArray, _y: FloatArray) -> dict[str, float]:
        return {name: initial.get(name, 1.0) for name in ref.parameters}

    try:
        expression = split_equation(ref.equation)
        if not ref.parameters:
            # Check the text first, treating every free name as a parameter, so a grammar
            # mistake (x^2, 2x, Exp(x)) is reported before the missing parameter list.
            candidates = [
                n for n in _names_in(expression) if n != ref.variable and n not in RESERVED_NAMES
            ]
            validate_expression(expression, frozenset({ref.variable, *candidates}))
            found = f" (the equation uses {', '.join(candidates)})" if candidates else ""
            raise ValueError(
                f"no parameters are listed: list the parameter names to fit, e.g. ['a','b']{found}"
            )
        model = ModelSpec(
            id=CUSTOM_MODEL_ID,
            name="Custom equation",
            category="custom",
            expression=expression,
            parameters=tuple(Parameter(name) for name in ref.parameters),
            initial_guess=guess,
            variables=(ref.variable,),
        )
        names = frozenset({ref.variable, *ref.parameters})
        used = {
            node.id
            for node in ast.walk(parse_expression(expression, names))
            if isinstance(node, ast.Name)
        }
        for name in ref.parameters:
            if name not in used:
                raise ValueError(f"parameter {name!r} is not used in the equation")
    except ValueError as exc:
        raise WorkflowError("INVALID_EQUATION", str(exc)) from exc
    return model


def _resolve_model(spec: FitSpec) -> ModelSpec | None:
    """The model to fit, or ``None`` for a polynomial (which has no :class:`ModelSpec`)."""
    ref = spec.model
    if isinstance(ref, PolynomialRef):
        return None
    if isinstance(ref, RegisteredRef):
        return registered_spec(ref.id)
    return _custom_model(ref, spec.initial)


def _with_deadline(model: ModelSpec, limit: float) -> tuple[ModelSpec, list[bool]]:
    """A copy of ``model`` whose evaluations raise once ``limit`` seconds have passed.

    The second return value is a one-element flag that turns true when the limit was hit.
    The optimiser reports the raised error as a ``failed`` fit with its message.
    """
    deadline = time.monotonic() + limit
    hit = [False]
    evaluate, jacobian = model.evaluate, model.jacobian

    def check() -> None:
        if time.monotonic() > deadline:
            hit[0] = True
            raise _TimeLimitExceeded(f"fit exceeded the time limit of {limit:g} s")

    def timed_evaluate(x: FloatArray, **params: float) -> FloatArray:
        check()
        return evaluate(x, **params)

    def timed_jacobian(x: FloatArray, **params: float) -> FloatArray:
        check()
        return jacobian(x, **params)

    clone = copy.copy(model)  # shallow: shares the compiled expression
    object.__setattr__(clone, "evaluate", timed_evaluate)
    object.__setattr__(clone, "jacobian", timed_jacobian)
    return clone, hit


# -- fitting ---------------------------------------------------------------------------------------


def _apply_weight_choice(xy: XY, spec: FitSpec) -> tuple[XY, list[str]]:
    if spec.use_weights:
        if xy.weights is None:
            raise WorkflowError(
                "INVALID_SPEC", "use_weights is set but the data has no weights column"
            )
        return xy, []
    if xy.weights is None:
        return xy, []
    # Unweighted fit of weighted data: a different input, so a different fingerprint.
    fingerprint = hashlib.sha256(f"{xy.fingerprint}|weights ignored".encode()).hexdigest()
    plain = XY(xy.x, xy.y, None, xy.x_name, xy.y_name, fingerprint)
    return plain, ["the data has weights but use_weights is off: the fit is unweighted"]


def _fit_config(spec: FitSpec) -> FitConfig:
    bounds = {
        name: (
            -math.inf if lower is None else lower,
            math.inf if upper is None else upper,
        )
        for name, (lower, upper) in spec.bounds.items()
    }
    try:
        return FitConfig(
            method=spec.method,
            loss=spec.loss,
            initial=dict(spec.initial) or None,
            bounds=bounds or None,
            seed=spec.seed if spec.seed is not None else 0,
        )
    except ValueError as exc:
        raise WorkflowError("INVALID_SPEC", str(exc)) from exc


def _fit_polynomial(spec: FitSpec, ref: PolynomialRef, data: XY) -> FitResult:
    rejected = [
        name
        for name, given in (
            ("method", spec.method != "auto"),
            ("loss", spec.loss != "linear"),
            ("bounds", bool(spec.bounds)),
            ("initial", bool(spec.initial)),
        )
        if given
    ]
    if rejected:
        raise WorkflowError(
            "INVALID_SPEC",
            f"{', '.join(rejected)} can only be used with a registered or custom model, "
            "not with a polynomial",
        )
    try:
        return fit_polynomial(data, ref.degree)
    except ValueError as exc:
        raise WorkflowError("INVALID_SPEC", str(exc)) from exc


def fit_dataset(xy: XY, spec: FitSpec, *, compact: bool = False) -> FitBundle:
    """Fit ``spec`` to ``xy`` and gather everything needed to show the result.

    The fit always uses every row. ``compact=True`` only shrinks ``bundle.data`` to at most
    ``COMPACT_MAX_POINTS`` rows (an evenly strided subset, marked ``subsampled``); the
    result's per-row ``fitted``/``residuals`` and the summary's per-row diagnostics stay
    full, as their schema requires, so a compact bundle is smaller but still O(n).

    Raises:
        WorkflowError: ``INVALID_EQUATION``, ``UNKNOWN_MODEL`` or ``INVALID_SPEC`` (a
            specification the engine cannot run). A fit that runs but does not converge
            does not raise; see :class:`FitBundle`.
    """
    model = _resolve_model(spec)
    data, warnings = _apply_weight_choice(xy, spec)
    timed_out = [False]
    if model is None:
        assert isinstance(spec.model, PolynomialRef)  # model is None only for polynomials
        result = _fit_polynomial(spec, spec.model, data)
    else:
        config = _fit_config(spec)
        run_model = model
        if spec.time_limit is not None:
            run_model, timed_out = _with_deadline(model, spec.time_limit)
        try:
            result = fit(run_model, data, config)
        except _TimeLimitExceeded as exc:  # raised outside the optimiser, before its first step
            result = _timed_out_result(model, data, spec, str(exc))
        except ValueError as exc:  # unknown parameter names, impossible method/loss/bounds
            raise WorkflowError("INVALID_SPEC", str(exc)) from exc
    if timed_out[0]:
        warnings.append(
            f"the fit was stopped at the time limit of {spec.time_limit:g} s; "
            "the numbers are not a fit"
        )
    return _assemble(data, result, model, warnings, compact)


def _timed_out_result(model: ModelSpec, data: XY, spec: FitSpec, reason: str) -> FitResult:
    """A ``failed`` result for a fit that ran out of time before the optimiser started."""
    return FitResult(
        model_id=model.id,
        equation=f"y = {model.expression.strip()}",
        equation_fitted=None,
        parameters=(),
        covariance=None,
        correlation=None,
        stats=None,
        fitted=[],
        residuals=[],
        status=FitStatus.FAILED,
        termination_reason=reason,
        method=spec.method,
        loss=spec.loss,
        initial_params={},
        n_starts=1,
        warnings=(),
        provenance=collect_provenance(data.fingerprint, spec.seed or 0),
    )


def _curve_grid(view: FitView, x: FloatArray) -> tuple[FloatArray, list[str]]:
    """Curve x values: the data range plus a 5 % margin, minus any margin the model rejects.

    A margin on one side is dropped entirely when the model (or its Jacobian, which the
    bands need) is not finite anywhere in it, e.g. ``ln(x)`` below zero. The returned
    notes say which sides were trimmed.
    """
    low, high = float(x.min()), float(x.max())
    margin = CURVE_MARGIN * (high - low if high > low else max(abs(low), 1.0))
    grid = np.linspace(low - margin, high + margin, CURVE_POINTS)
    with np.errstate(all="ignore"):
        defined = np.isfinite(view.predict(grid)) & np.all(
            np.isfinite(view.jacobian_at(grid)), axis=1
        )
    left_ok = bool(np.all(defined[grid < low]))
    right_ok = bool(np.all(defined[grid > high]))
    if left_ok and right_ok:
        return grid, []
    notes = [
        f"the curve was not extended past the data on the {side} (x {sign} {edge:.6g}) "
        "because the model is undefined there"
        for side, sign, edge, ok in (("left", "<", low, left_ok), ("right", ">", high, right_ok))
        if not ok
    ]
    trimmed = np.linspace(
        low - (margin if left_ok else 0.0), high + (margin if right_ok else 0.0), CURVE_POINTS
    )
    return trimmed, notes


def _snapshot(data: XY, compact: bool) -> DataSnapshot:
    n = int(data.x.size)
    keep = np.arange(n)
    if compact and n > COMPACT_MAX_POINTS:
        keep = np.unique(np.linspace(0, n - 1, COMPACT_MAX_POINTS).round().astype(np.intp))
    return DataSnapshot(
        x=data.x[keep].tolist(),
        y=data.y[keep].tolist(),
        weights=None if data.weights is None else data.weights[keep].tolist(),
        x_name=data.x_name,
        y_name=data.y_name,
        n_total=n,
        subsampled=keep.size < n,
    )


def _assemble(
    data: XY,
    result: FitResult,
    model: ModelSpec | None,
    extra_warnings: list[str],
    compact: bool,
) -> FitBundle:
    summary: FitReport | None = None
    summary_reason: str | None = None
    curve: CurveSamples | None = None
    curve_reason: str | None = None
    conf: Band | None = None
    pred: Band | None = None
    bands_reason: str | None = None
    warnings = list(result.warnings)

    if result.status is FitStatus.CONVERGED:
        try:
            summary = describe_fit(data, result, model)
        except StatisticsError as exc:
            summary_reason = f"{exc.code}: {exc}"
        else:
            warnings = list(summary.warnings)
        view = build_view(data, result, model)
        grid, trim_notes = _curve_grid(view, data.x)
        extra_warnings = [*extra_warnings, *trim_notes]
        with np.errstate(all="ignore"):
            values = view.predict(grid)
        if bool(np.all(np.isfinite(values))):
            curve = CurveSamples(x=grid.tolist(), y=values.tolist())
            conf, pred, bands_reason = _bands(view, grid)
        else:
            curve_reason = (
                f"the fitted model is not finite (outside its domain) at some of the "
                f"{CURVE_POINTS} curve points between x={grid[0]:.6g} and x={grid[-1]:.6g}"
            )
            bands_reason = "no bands: the curve is not defined over the plotted range"
    else:
        why = f"the fit is {result.status.value} ({result.termination_reason})"
        summary_reason = f"no summary: {why}"
        curve_reason = f"no curve: {why}"
        bands_reason = f"no bands: {why}"

    warnings.extend(w for w in extra_warnings if w not in warnings)
    return FitBundle(
        result=result,
        summary=summary,
        summary_reason=summary_reason,
        curve=curve,
        curve_reason=curve_reason,
        confidence_band=conf,
        prediction_band=pred,
        bands_reason=bands_reason,
        equation_text=result.equation_fitted or result.equation,
        warnings=tuple(warnings),
        data=_snapshot(data, compact),
    )


def _bands(view: FitView, grid: FloatArray) -> tuple[Band | None, Band | None, str | None]:
    try:
        conf = confidence_band(view, grid, BAND_LEVEL)
        pred = prediction_band(view, grid, BAND_LEVEL)
    except StatisticsError as exc:
        return None, None, f"{exc.code}: {exc}"
    for band in (conf, pred):
        if not all(math.isfinite(v) for v in (*band.lower, *band.upper)):
            return None, None, "the bands are not finite at some curve points"
    return conf, pred, None


# -- prediction ------------------------------------------------------------------------------------


def _check_spec_matches(result: FitResult, spec: FitSpec) -> ModelSpec | None:
    ref = spec.model
    is_polynomial_result = _POLYNOMIAL_ID.fullmatch(result.model_id) is not None
    if isinstance(ref, PolynomialRef):
        if not is_polynomial_result or len(result.parameters) != ref.degree + 1:
            raise WorkflowError(
                "MODEL_MISMATCH",
                f"the fit is {result.model_id!r}, not a polynomial of degree {ref.degree}",
            )
        return None
    model = _resolve_model(spec)
    assert model is not None
    if is_polynomial_result or result.model_id != model.id:
        raise WorkflowError("MODEL_MISMATCH", f"the fit is {result.model_id!r}, not {model.id!r}")
    expected = f"y = {model.expression.strip()}"
    if _squeeze(result.equation) != _squeeze(expected):
        raise WorkflowError(
            "MODEL_MISMATCH",
            f"the fit is {result.equation!r}, but the specification is {expected!r}",
        )
    return model


def _squeeze(text: str) -> str:
    """Whitespace-free form of an equation, for comparing what was fitted with what is asked."""
    return "".join(text.split())


def _snapshot_xy(bundle: FitBundle) -> XY:
    snap = bundle.data
    return XY(
        x=np.asarray(snap.x, dtype=np.float64),
        y=np.asarray(snap.y, dtype=np.float64),
        weights=None if snap.weights is None else np.asarray(snap.weights, dtype=np.float64),
        x_name=snap.x_name,
        y_name=snap.y_name,
        fingerprint=bundle.result.provenance.dataset_fingerprint,
    )


def predict(
    source: FitBundle | FitResult,
    spec: FitSpec,
    x: Sequence[float],
    band: Literal["confidence", "prediction"] | None = None,
    *,
    data: XY | None = None,
) -> Prediction:
    """Predicted y at ``x`` from a converged fit, with an optional 95 % pointwise band.

    A :class:`FitBundle` carries its data; a bare :class:`FitResult` needs ``data=`` (the
    exact data it was fitted to). Rows where the prediction is undefined are ``None``.

    Raises:
        WorkflowError: ``FIT_NOT_CONVERGED``; ``MODEL_MISMATCH`` / ``DATA_MISMATCH`` when
            ``spec`` or ``data`` are not what the fit was computed from; ``DATA_REQUIRED``
            for a bare result without ``data``; ``INVALID_EQUATION`` / ``UNKNOWN_MODEL`` for
            a bad ``spec``.
    """
    if len(x) > MAX_PREDICT_POINTS:
        raise WorkflowError(
            "TOO_LARGE", f"{len(x):,} x values requested; the limit is {MAX_PREDICT_POINTS:,}"
        )
    if isinstance(source, FitBundle):
        result = source.result
        if data is None and source.data.subsampled:
            raise WorkflowError(
                "DATA_REQUIRED",
                "the bundle holds only a subsample of its data (compact mode): pass data= "
                "with the full data the fit was computed from",
            )
        xy = data if data is not None else _snapshot_xy(source)
    else:
        result = source
        if data is None:
            raise WorkflowError(
                "DATA_REQUIRED",
                "a bare FitResult does not carry its data: pass data= or use the FitBundle",
            )
        xy = data
    if result.status is not FitStatus.CONVERGED:
        raise WorkflowError(
            "FIT_NOT_CONVERGED",
            f"cannot predict from a {result.status.value} fit ({result.termination_reason})",
        )
    model = _check_spec_matches(result, spec)
    try:
        view = build_view(xy, result, model)
    except StatisticsError as exc:
        raise WorkflowError(exc.code, str(exc)) from exc

    points = np.asarray(list(x), dtype=np.float64)
    ok = np.isfinite(points)
    y_all = np.full(points.shape, np.nan)
    if bool(ok.any()):
        with np.errstate(all="ignore"):
            y_all[ok] = view.predict(points[ok])
    defined = np.isfinite(y_all)

    lower: list[float | None] | None = None
    upper: list[float | None] | None = None
    reason: str | None = None
    if band is not None and bool(defined.any()):
        lower, upper, reason = _band_rows(view, points, defined, band)
    elif band is not None:
        lower, upper = [None] * points.size, [None] * points.size
        reason = "no band: the prediction is undefined at every requested x"
    with np.errstate(invalid="ignore"):
        extrapolated = ((points < view.x.min()) | (points > view.x.max())).tolist()
    return Prediction(
        x=_nullable(points, ok),
        y=_nullable(y_all, defined),
        band=band,
        level=BAND_LEVEL,
        lower=lower,
        upper=upper,
        extrapolated=extrapolated,
        band_reason=reason,
    )


def _band_rows(
    view: FitView,
    points: FloatArray,
    defined: NDArray[np.bool_],
    kind: Literal["confidence", "prediction"],
) -> tuple[list[float | None], list[float | None], str | None]:
    rows = int(points.size)
    try:
        compute = confidence_band if kind == "confidence" else prediction_band
        computed = compute(view, points[defined], BAND_LEVEL)
    except StatisticsError as exc:
        return [None] * rows, [None] * rows, f"{exc.code}: {exc}"
    low = np.full(rows, np.nan)
    high = np.full(rows, np.nan)
    low[defined] = computed.lower
    high[defined] = computed.upper
    return _nullable(low, np.isfinite(low)), _nullable(high, np.isfinite(high)), None


def _nullable(values: FloatArray, keep: NDArray[np.bool_]) -> list[Any]:
    return [float(v) if k else None for v, k in zip(values, keep, strict=True)]
