"""Seeded k-fold and leave-one-out cross-validation of a fitting procedure.

Cross-validation estimates how a *fitted model* generalises: the active data are split
into folds, each fold is fitted on the remaining points, and the held-out points are
scored. It measures the fitting procedure (model form plus fitting settings), not one
particular fit, because every fold is a fresh fit on its training subset.

Conventions:

* Only active (``weight > 0``) points take part; zero-weight points never enter a
  training set or a held-out set.
* One ``numpy.random.Generator(seed).permutation`` decides the fold assignment, so the
  same seed reproduces the folds and every number exactly, and fold sizes differ by at
  most one.
* A held-out RMSE is ``sqrt(sum(w * e**2) / sum(w))`` over held-out points, using the
  data's weights; the pooled ``rmse`` applies the same formula to all held-out points.
* A fold is *failed* when its training set has at most ``k`` points (no residual degrees
  of freedom), when the fit does not return ``CONVERGED``, when the fitter raises, or
  when the held-out predictions are not finite. Every failed fold is counted in
  ``n_failed`` and named in ``reasons``; a fold is never replaced or skipped, and any
  failure leaves the pooled ``rmse`` as ``None`` rather than averaging the surviving folds.

Assumptions: the observations are exchangeable (the seeded permutation is the only source
of randomness) and the model form is fixed before the folds are seen. Limitations: the
estimate is noisy on small samples, the held-out points lie inside the observed x range so
extrapolation is not tested, and a low held-out error is evidence about predictive
accuracy, never proof that the model is correct.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit, fit_polynomial
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.models import ModelSpec, get_model
from curvestudio.statistics.errors import StatisticsError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class Fitter(Protocol):
    """A refittable model: fit it to data, then predict from the returned result.

    Structural typing: any object with these two methods is a valid fitter, which is what
    lets cross-validation, bootstrap and the report bundle share one interface over
    polynomials, nonlinear specs and test doubles.
    """

    def fit(self, data: XY) -> FitResult:
        """Fit the model to ``data`` (the training subset of one fold)."""
        ...

    def predict(self, result: FitResult, x: FloatArray) -> FloatArray:
        """Model values at ``x`` from ``result``."""
        ...


@dataclass(frozen=True)
class PolynomialFitter:
    """Fit a polynomial by (weighted) least squares exactly as ``fit_polynomial`` does.

    ``through_origin`` fixes the intercept at zero and ``degree`` follows the fitting
    layer's rules (``>= 0``, or ``>= 1`` through the origin). Held-out predictions are
    therefore the same quantity a direct :func:`curvestudio.fitting.fit_polynomial` call
    would produce for that training subset.
    """

    degree: int
    through_origin: bool = False

    def __post_init__(self) -> None:
        minimum = 1 if self.through_origin else 0
        if (
            not isinstance(self.degree, int)
            or isinstance(self.degree, bool)
            or self.degree < minimum
        ):
            raise ValueError(f"degree must be an int >= {minimum}, got {self.degree!r}")

    def fit(self, data: XY) -> FitResult:
        """Fit the polynomial to ``data``."""
        return fit_polynomial(data, self.degree, through_origin=self.through_origin)

    def predict(self, result: FitResult, x: FloatArray) -> FloatArray:
        """Evaluate the fitted polynomial at ``x``.

        Raises:
            ValueError: ``result`` does not hold this fitter's coefficients (it was not
                produced by an identically configured ``PolynomialFitter``).
        """
        offset = 1 if self.through_origin else 0
        coefficients = np.zeros(self.degree + 1, dtype=np.float64)
        values = [p.value for p in result.parameters]
        if len(values) != coefficients.size - offset:
            raise ValueError(
                f"a result with {len(values)} parameter(s) does not come from "
                f"PolynomialFitter(degree={self.degree}, "
                f"through_origin={self.through_origin})"
            )
        coefficients[offset:] = values
        return np.asarray(
            np.polynomial.polynomial.polyval(np.asarray(x, dtype=np.float64), coefficients)
        )


@dataclass(frozen=True)
class NonlinearFitter:
    """Fit ``model`` with :func:`curvestudio.fitting.fit`, optionally from a warm start.

    ``config`` is used unchanged unless ``start`` is given, in which case ``start``
    replaces ``config.initial`` (a parameter it omits still falls back to the model's own
    ``initial_guess``). A warm start from the full-data estimate keeps every fold in the
    same basin, so cross-validation measures predictive accuracy rather than the
    optimizer's luck with a cold start.
    """

    model: ModelSpec | str
    config: FitConfig = field(default_factory=FitConfig)
    start: Mapping[str, float] | None = None

    def fit(self, data: XY) -> FitResult:
        """Fit ``model`` to ``data``, merging ``start`` over the configured initial values."""
        if self.start is None:
            config = self.config
        else:
            config = replace(self.config, initial=dict(self.start))
        return fit(self.model, data, config)

    def predict(self, result: FitResult, x: FloatArray) -> FloatArray:
        """Evaluate ``model`` at ``x`` with the parameters stored in ``result``."""
        spec = get_model(self.model) if isinstance(self.model, str) else self.model
        values = {p.name: p.value for p in result.parameters}
        return spec.evaluate(np.asarray(x, dtype=np.float64), **values)


class CrossValidation(BaseModel):
    """Outcome of one cross-validation run; ``None`` means a statistic is not defined.

    ``fold_rmse`` holds the weighted held-out RMSE of each fold in fold order, with
    ``None`` for a failed fold. ``rmse`` pools all held-out points and is ``None`` unless
    every fold succeeded. ``n_failed`` counts failed folds and ``reasons`` names each one,
    so no fold is silently dropped or replaced. ``status`` is ``"ok"`` only when no fold
    failed. Note that ``"ok"`` means the estimate is complete, not that the model is valid:
    cross-validation quantifies predictive accuracy, it does not confirm the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    folds: int
    seed: int
    rmse: FiniteFloat | None
    fold_rmse: list[FiniteFloat | None]
    n_failed: int
    status: Literal["ok", "incomplete"]
    reasons: list[str]


def _active_weights(data: XY) -> FloatArray:
    """The data's weights, or all ones when it carries none."""
    if data.weights is None:
        return np.ones(data.x.size, dtype=np.float64)
    return np.asarray(data.weights, dtype=np.float64)


def _subset(data: XY, rows: IntArray, fingerprint: str) -> XY:
    """A validated ``XY`` holding only ``rows`` of ``data``."""
    return XY(
        x=data.x[rows],
        y=data.y[rows],
        weights=None if data.weights is None else data.weights[rows],
        x_name=data.x_name,
        y_name=data.y_name,
        fingerprint=fingerprint,
    )


def _parameter_count(fitter: Fitter) -> int | None:
    """Parameters the fitter estimates, when that is knowable without fitting."""
    if isinstance(fitter, PolynomialFitter):
        return fitter.degree + 1 - (1 if fitter.through_origin else 0)
    if isinstance(fitter, NonlinearFitter):
        spec = get_model(fitter.model) if isinstance(fitter.model, str) else fitter.model
        return len(spec.parameters)
    return None


def _check_folds(folds: int, n_active: int) -> None:
    if not isinstance(folds, int) or isinstance(folds, bool) or not 2 <= folds <= n_active:
        raise StatisticsError(
            "INVALID_FOLDS",
            f"folds must be an int between 2 and the number of active points ({n_active}), "
            f"got {folds!r}",
        )


def _check_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise StatisticsError("INVALID_SEED", f"seed must be an int >= 0, got {seed!r}")


def k_fold_cv(data: XY, fitter: Fitter, folds: int = 5, seed: int = 0) -> CrossValidation:
    """Estimate held-out prediction error by ``folds``-fold cross-validation.

    Only active (weight > 0) points take part. One
    ``numpy.random.default_rng(seed).permutation`` assigns them to folds, so the same seed
    reproduces the fold assignment and every number exactly; fold sizes differ by at most
    one and each active point is held out exactly once.

    Each fold fits ``fitter`` on its training points and predicts the held-out ones. A
    fold's RMSE is ``sqrt(sum(w * e**2) / sum(w))`` over its held-out points with the
    data's weights; ``CrossValidation.rmse`` pools the same sums over every held-out point
    and is ``None`` unless every fold succeeded.

    A fold is counted as failed (``None`` in ``fold_rmse``, one entry in ``reasons``, one
    in ``n_failed``) when its training set has at most ``k`` points, when the fit does not
    return ``CONVERGED``, when the fitter raises, or when the held-out predictions are not
    finite. ``k`` is known from ``PolynomialFitter`` and ``NonlinearFitter`` before
    fitting; for any other fitter it is taken from the converged result's parameters.
    Failed folds are never substituted, and any failure makes ``status="incomplete"``.

    Assumptions: the observations are exchangeable (the permutation is the only source of
    randomness) and the model form is fixed before the folds are seen.

    Limitations: the estimate is noisy for small ``n``, the held-out points lie inside the
    observed x range so extrapolation is not tested, and a low RMSE is evidence about
    predictive accuracy, never proof that the model is correct.

    Raises:
        StatisticsError: ``INVALID_FOLDS`` unless ``folds`` is an int with
            ``2 <= folds <= n_active``; ``INVALID_SEED`` unless ``seed`` is an int >= 0;
            ``NON_FINITE_RMSE`` if the pooled sum over already-valid folds overflows.
        KeyError: the fitter names a model id that is not registered.
    """
    _check_seed(seed)
    weights_all = _active_weights(data)
    active_rows = np.flatnonzero(weights_all > 0.0)
    n_active = int(active_rows.size)
    _check_folds(folds, n_active)
    parameters = _parameter_count(fitter)

    order = np.random.default_rng(seed).permutation(n_active)
    fold_rmse: list[float | None] = []
    reasons: list[str] = []
    sse_total = 0.0
    weight_total = 0.0
    for index, held_positions in enumerate(np.array_split(order, folds)):
        in_fold = np.zeros(n_active, dtype=bool)
        in_fold[held_positions] = True
        train_rows = active_rows[~in_fold]
        held_rows = active_rows[in_fold]
        label = f"fold {index}"

        if parameters is not None and int(train_rows.size) <= parameters:
            fold_rmse.append(None)
            reasons.append(
                f"{label}: {train_rows.size} training point(s) for {parameters} "
                "parameter(s); the fit has no residual degrees of freedom"
            )
            continue

        try:
            result = fitter.fit(
                _subset(data, train_rows, f"{data.fingerprint}#cv-{seed}-fold-{index}")
            )
        except Exception as exc:  # counted and reported below, never swallowed
            fold_rmse.append(None)
            reasons.append(f"{label}: fit raised {type(exc).__name__}: {exc}")
            continue

        if result.status is not FitStatus.CONVERGED:
            fold_rmse.append(None)
            reasons.append(
                f"{label}: fit status is {result.status.value!r} ({result.termination_reason})"
            )
            continue

        if parameters is None and int(train_rows.size) <= len(result.parameters):
            fold_rmse.append(None)
            reasons.append(
                f"{label}: {train_rows.size} training point(s) for "
                f"{len(result.parameters)} parameter(s); the fit has no residual "
                "degrees of freedom"
            )
            continue

        with np.errstate(all="ignore"):
            predictions = np.asarray(fitter.predict(result, data.x[held_rows]), dtype=np.float64)
        if predictions.shape != (int(held_rows.size),):
            fold_rmse.append(None)
            reasons.append(
                f"{label}: fitter predicted shape {predictions.shape} for "
                f"{held_rows.size} held-out point(s)"
            )
            continue

        with np.errstate(all="ignore"):
            residual = data.y[held_rows] - predictions
            sse = float(np.sum(weights_all[held_rows] * residual * residual))
            fold_weight = float(np.sum(weights_all[held_rows]))
        if not (math.isfinite(sse) and math.isfinite(fold_weight)) or sse < 0.0:
            fold_rmse.append(None)
            reasons.append(f"{label}: held-out residuals are not finite")
            continue

        fold_rmse.append(math.sqrt(sse / fold_weight))
        sse_total += sse
        weight_total += fold_weight

    n_failed = sum(1 for value in fold_rmse if value is None)
    rmse: float | None = None
    if n_failed == 0:
        pooled = math.sqrt(sse_total / weight_total)
        if not math.isfinite(pooled):
            raise StatisticsError(
                "NON_FINITE_RMSE",
                "the pooled held-out RMSE is not finite; the held-out sums overflowed",
            )
        rmse = pooled
    return CrossValidation(
        folds=folds,
        seed=seed,
        rmse=rmse,
        fold_rmse=fold_rmse,
        n_failed=n_failed,
        status="ok" if n_failed == 0 else "incomplete",
        reasons=reasons,
    )


def loo_cv(data: XY, fitter: Fitter) -> CrossValidation:
    """Leave-one-out cross-validation: ``k_fold_cv`` with one fold per active point.

    Each fold trains on every active point but one. The fold order follows
    ``k_fold_cv``'s seeded permutation with ``seed=0``, so the result is deterministic;
    for an unweighted linear model the pooled RMSE equals the closed-form PRESS RMSE.

    Raises:
        StatisticsError: ``INVALID_FOLDS`` with fewer than two active points (there is
            nothing to leave out and still fit).
    """
    n_active = int(np.count_nonzero(_active_weights(data) > 0.0))
    if n_active < 2:
        raise StatisticsError(
            "INVALID_FOLDS",
            f"leave-one-out cross-validation needs at least 2 active points, got {n_active}",
        )
    return k_fold_cv(data, fitter, folds=n_active, seed=0)
