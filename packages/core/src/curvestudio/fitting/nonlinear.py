"""Nonlinear (weighted) least squares with seeded multi-start.

The engine calls :func:`scipy.optimize.least_squares` directly rather than through
``lmfit.Minimizer``. LMfit's wrapper doubles ``max_nfev`` and turns an exhausted budget
into its own abort path, and it inverts ``J'J`` with ``np.linalg.inv``; calling SciPy
directly gives an exact evaluation budget, SciPy's own termination status and message,
the model's analytic Jacobian without an adapter, and a covariance built here by SVD.

Problem set-up (only points with weight > 0 are *active*; zero-weight points are
reported in ``fitted``/``residuals`` but influence nothing):

* residuals ``r_i = sqrt(w_i) * (y_i - f(x_i))`` on active points, so ``sum(r**2)`` is the
  weighted SSE; Jacobian rows are scaled the same way and come from
  :meth:`ModelSpec.jacobian` (analytic, never finite differences);
* covariance ``s2 * pinv(J'J)`` with ``s2 = SSE / df`` and ``df = n_active - k``. The
  pseudo-inverse uses the SVD of ``J`` with the cut-off ``eps * max(J.shape) * s_max``;
  any singular value below it means the parameters are not separately identifiable, so
  the covariance is withheld (``SINGULAR_COVARIANCE``) instead of reporting huge numbers.

Non-finite values during iteration: if a trial step makes the model non-finite, every
non-finite residual is replaced by a large finite penalty (``1e100``). Both ``lm`` and
``trf``/``dogbox`` only accept steps that reduce the cost, so the penalty acts as a wall
and the optimizer backs off instead of continuing on NaN. A non-finite *Jacobian*
cannot be patched honestly (the search direction would be fiction), so it stops the run
and the run is reported as ``FAILED``. NumPy floating-point warnings are suppressed
inside the engine; their effect is always visible as a status or warning instead.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import least_squares

from curvestudio.data import XY
from curvestudio.fitting._uncertainty import correlation, finite_or_none
from curvestudio.fitting.config import FitConfig
from curvestudio.fitting.metrics import FitStatistics, fit_statistics
from curvestudio.fitting.result import (
    FitResult,
    FitStatus,
    ParameterEstimate,
    collect_provenance,
)
from curvestudio.models.spec import ModelSpec

FloatArray = NDArray[np.float64]

NON_FINITE_PENALTY = 1e100
"""Stand-in for a non-finite residual during iteration (squares stay far below overflow)."""

MULTISTART_RTOL = 1e-6
"""Relative SSE spread among converged starts above which they are said to disagree."""

_BOUND_RTOL = 1e-8
"""A parameter within this relative distance of a finite bound is reported at the bound."""


class _NonFiniteJacobianError(Exception):
    """Raised inside the optimizer callback when the Jacobian has NaN/inf entries."""


@dataclass(frozen=True)
class _Problem:
    """The active-point least-squares problem for one model and dataset."""

    model: ModelSpec
    x: FloatArray
    y: FloatArray
    root_w: FloatArray
    names: tuple[str, ...]

    def params(self, theta: FloatArray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, theta, strict=True)}

    def residuals(self, theta: FloatArray) -> FloatArray:
        """True weighted residuals; may contain NaN/inf."""
        f = self.model.evaluate(self.x, **self.params(theta))
        with np.errstate(all="ignore"):
            return self.root_w * (self.y - f)

    def jacobian(self, theta: FloatArray) -> FloatArray:
        """Analytic Jacobian of the weighted residuals; may contain NaN/inf."""
        jac = self.model.jacobian(self.x, **self.params(theta))
        with np.errstate(all="ignore"):
            return -self.root_w[:, None] * jac

    def guarded_residuals(self, theta: FloatArray) -> FloatArray:
        r = self.residuals(theta)
        return np.where(np.isfinite(r), r, NON_FINITE_PENALTY)

    def guarded_jacobian(self, theta: FloatArray) -> FloatArray:
        jac = self.jacobian(theta)
        if not bool(np.all(np.isfinite(jac))):
            raise _NonFiniteJacobianError(f"Jacobian is not finite at {self.describe(theta)}")
        return jac

    def describe(self, theta: FloatArray) -> str:
        return ", ".join(f"{n}={v:.6g}" for n, v in zip(self.names, theta, strict=True))


@dataclass(frozen=True)
class _Run:
    """Outcome of one optimizer run from one start."""

    start: FloatArray
    status: FitStatus
    message: str
    theta: FloatArray | None = None
    sse: float = math.inf
    jacobian: FloatArray | None = None
    active_mask: FloatArray | None = None


def generate_starts(
    base: FloatArray, lower: FloatArray, upper: FloatArray, n_starts: int, seed: int
) -> list[FloatArray]:
    """The base start followed by ``n_starts - 1`` seeded perturbations of it.

    Exact scheme (reproducible from ``seed`` alone): one array ``u`` of shape
    ``(n_starts - 1, k)`` is drawn with ``numpy.random.default_rng(seed).random``, and
    for each extra start and parameter ``j`` with base value ``v``:

    * both bounds finite: ``lower + u * (upper - lower)`` (uniform over the box);
    * otherwise, ``v != 0``: ``v * 10**(2u - 1)``, a log-uniform multiplier in [0.1, 10]
      that keeps the sign and the order of magnitude of the guess;
    * otherwise (``v == 0``): ``2u - 1``, uniform in [-1, 1];

    and every value is finally clipped into ``[lower, upper]``.

    Raises:
        ValueError: ``n_starts < 1``.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    base = np.clip(np.asarray(base, dtype=np.float64), lower, upper)
    starts = [base.copy()]
    if n_starts == 1:
        return starts
    u = np.random.default_rng(seed).random((n_starts - 1, base.size))
    boxed = np.isfinite(lower) & np.isfinite(upper)
    for row in u:
        with np.errstate(invalid="ignore"):  # inf - inf in the unused branch of np.where
            start = np.where(
                boxed,
                lower + row * (upper - lower),
                np.where(base != 0.0, base * 10.0 ** (2.0 * row - 1.0), 2.0 * row - 1.0),
            )
        starts.append(np.clip(start, lower, upper))
    return starts


def _run_once(
    problem: _Problem,
    start: FloatArray,
    method: str,
    lower: FloatArray,
    upper: FloatArray,
    config: FitConfig,
) -> _Run:
    if not bool(np.all(np.isfinite(problem.residuals(start)))):
        return _Run(start, FitStatus.FAILED, "model not finite at starting values")
    try:
        with np.errstate(all="ignore"):
            ret = least_squares(
                problem.guarded_residuals,
                start,
                jac=problem.guarded_jacobian,
                bounds=(lower, upper),
                method=method,
                loss=config.loss,
                x_scale="jac",
                ftol=config.ftol,
                xtol=config.xtol,
                gtol=config.gtol,
                max_nfev=config.max_nfev,
            )
        theta = np.asarray(ret.x, dtype=np.float64)
        resid = problem.residuals(theta)
        jac = problem.jacobian(theta)
    except _NonFiniteJacobianError as exc:
        return _Run(start, FitStatus.FAILED, str(exc))
    except Exception as exc:  # reported as FAILED with its message, never swallowed
        return _Run(start, FitStatus.FAILED, f"optimizer raised {type(exc).__name__}: {exc}")

    with np.errstate(all="ignore"):
        sse = float(np.sum(resid * resid))
    problems: list[str] = []
    if not bool(np.all(np.isfinite(theta))):
        problems.append("parameters are not finite")
    if not bool(np.all(np.isfinite(jac))):
        problems.append("Jacobian is not finite at the solution")
    if not math.isfinite(sse):
        problems.append("SSE is not finite at the solution")
    converged = bool(ret.success) and not problems
    message = "; ".join([str(ret.message), *problems])
    return _Run(
        start=start,
        status=FitStatus.CONVERGED if converged else FitStatus.NOT_CONVERGED,
        message=message,
        theta=theta if bool(np.all(np.isfinite(theta))) else None,
        sse=sse if math.isfinite(sse) else math.inf,
        jacobian=jac,
        active_mask=np.asarray(ret.active_mask),
    )


def _pick_winner(runs: list[_Run]) -> _Run:
    """Lowest-SSE converged run; else the lowest-SSE unconverged one; else the first."""
    for status in (FitStatus.CONVERGED, FitStatus.NOT_CONVERGED):
        candidates = [r for r in runs if r.status is status]
        if candidates:
            return min(candidates, key=lambda r: r.sse)  # min() keeps the first on ties
    return runs[0]


def _multistart_warning(runs: list[_Run], scale: float) -> str | None:
    """``MULTISTART_DISAGREE`` if converged SSEs spread beyond ``MULTISTART_RTOL``.

    ``scale`` (the uncentred weighted sum of squares of ``y``) sets an absolute floor so
    that two essentially exact fits (SSE 1e-30 vs 1e-28) are not called a disagreement.
    """
    sses = [r.sse for r in runs if r.status is FitStatus.CONVERGED]
    if len(sses) < 2:
        return None
    if math.isclose(max(sses), min(sses), rel_tol=MULTISTART_RTOL, abs_tol=1e-12 * scale):
        return None
    listed = ", ".join(f"{r.sse:.6g}" if math.isfinite(r.sse) else r.status.value for r in runs)
    return f"MULTISTART_DISAGREE: sse=[{listed}]"


def _at_bounds(
    run: _Run, names: tuple[str, ...], lower: FloatArray, upper: FloatArray
) -> list[str]:
    assert run.theta is not None
    flagged: list[str] = []
    for j, (name, value) in enumerate(zip(names, run.theta, strict=True)):
        mask = 0.0 if run.active_mask is None else float(run.active_mask[j])
        near = any(
            math.isfinite(b) and abs(value - b) <= _BOUND_RTOL * max(1.0, abs(b))
            for b in (float(lower[j]), float(upper[j]))
        )
        if mask != 0.0 or near:
            flagged.append(f"PARAMETER_AT_BOUND: {name}")
    return flagged


def _uncertainties(
    jac: FloatArray, stats: FitStatistics
) -> tuple[FloatArray | None, list[list[float]] | None, list[list[float]] | None]:
    """Standard errors, covariance and correlation from the weighted Jacobian.

    Returns ``(None, None, None)`` when ``J`` is rank deficient. Exact-fit data
    (``sse == 0``) gives zero covariance and standard errors and no correlation.
    """
    k = jac.shape[1]
    _, svals, vt = np.linalg.svd(jac, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(jac.shape) * (svals[0] if svals.size else 0.0)
    if svals.size < k or not bool(np.all(svals > cutoff)) or not svals[0] > 0.0:
        return None, None, None
    cov_unit = (vt.T / svals**2) @ vt
    if stats.sse == 0.0:
        return np.zeros(k), [[0.0] * k for _ in range(k)], None
    with np.errstate(all="ignore"):
        cov = (stats.sse / stats.df) * cov_unit
        covariance = finite_or_none(cov)
        if covariance is None:
            return None, None, None
        se = np.sqrt(np.diag(cov))
    return se, covariance, correlation(cov, se)


def fit_nonlinear(
    model: ModelSpec,
    data: XY,
    config: FitConfig,
    *,
    method: str,
    lower: FloatArray,
    upper: FloatArray,
) -> FitResult:
    """Run the fit described in the module docstring and package it as a :class:`FitResult`.

    ``method`` is already resolved and checked, ``lower``/``upper`` are the effective
    bounds in parameter order; use :func:`curvestudio.fitting.fit` instead of calling
    this directly.
    """
    names = tuple(p.name for p in model.parameters)
    k = len(names)
    weighted = data.weights is not None
    warnings: list[str] = []

    def build(
        status: FitStatus,
        reason: str,
        *,
        initial: dict[str, float] | None = None,
        parameters: tuple[ParameterEstimate, ...] = (),
        covariance: list[list[float]] | None = None,
        correlation_: list[list[float]] | None = None,
        stats: FitStatistics | None = None,
        fitted: list[float] | None = None,
        residuals: list[float] | None = None,
        equation_fitted: str | None = None,
    ) -> FitResult:
        return FitResult(
            model_id=model.id,
            equation=f"y = {model.expression.strip()}",
            equation_fitted=equation_fitted,
            parameters=parameters,
            covariance=covariance,
            correlation=correlation_,
            stats=stats,
            fitted=fitted or [],
            residuals=residuals or [],
            status=status,
            termination_reason=reason,
            method=method,
            loss=config.loss,
            initial_params=initial or {},
            n_starts=config.n_starts,
            warnings=tuple(warnings),
            provenance=collect_provenance(data.fingerprint, seed=config.seed),
        )

    x = np.asarray(data.x, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64)
    w = np.ones_like(y) if data.weights is None else np.asarray(data.weights, dtype=np.float64)
    active = w > 0.0
    n_active = int(np.count_nonzero(active))
    if n_active <= k:
        return build(
            FitStatus.INVALID_INPUT,
            f"{n_active} active observation(s) for {k} parameter(s): need more observations "
            "than parameters",
        )
    reasons = model.applicability(x, y)
    if reasons:
        return build(
            FitStatus.INVALID_INPUT, f"model {model.id!r} does not apply: {'; '.join(reasons)}"
        )

    problem = _Problem(model, x[active], y[active], np.sqrt(w[active]), names)
    base, start_error = _base_start(model, problem, config)
    if base is None:
        return build(FitStatus.FAILED, start_error)
    base = np.clip(base, lower, upper)

    if not bool(np.all(np.isfinite(problem.residuals(base)))):
        return build(
            FitStatus.FAILED,
            f"model not finite at starting values ({problem.describe(base)})",
            initial=problem.params(base),
        )
    starts = generate_starts(base, lower, upper, config.n_starts, config.seed)
    runs = [_run_once(problem, s, method, lower, upper, config) for s in starts]
    winner = _pick_winner(runs)
    initial = problem.params(winner.start)
    if winner.status is FitStatus.FAILED or winner.theta is None:
        return build(winner.status, winner.message, initial=initial)

    theta = winner.theta
    values = problem.params(theta)
    fitted_all = model.evaluate(x, **values)
    with np.errstate(all="ignore"):
        residuals_all = y - fitted_all
    status, reason = winner.status, winner.message
    finite_fit = bool(np.all(np.isfinite(fitted_all)) and np.all(np.isfinite(residuals_all)))
    if status is FitStatus.CONVERGED and not finite_fit:
        status = FitStatus.NOT_CONVERGED
        reason = f"{reason}; model not finite at the solution for some observations"

    warnings.extend(_at_bounds(winner, names, lower, upper))
    with np.errstate(all="ignore"):
        scale = float(np.sum(w[active] * y[active] ** 2))
    disagreement = _multistart_warning(runs, scale)
    if disagreement is not None:
        warnings.append(disagreement)

    fitted = fitted_all.tolist() if finite_fit else None
    residuals = residuals_all.tolist() if finite_fit else None
    stats = (
        fit_statistics(y[active], fitted_all[active], k, weights=w[active] if weighted else None)
        if finite_fit
        else None
    )
    if status is not FitStatus.CONVERGED or stats is None or winner.jacobian is None:
        estimates = tuple(ParameterEstimate(name=n, value=v) for n, v in values.items())
        return build(
            status,
            reason,
            initial=initial,
            parameters=estimates,
            stats=stats,
            fitted=fitted,
            residuals=residuals,
        )

    se, covariance, corr = _uncertainties(winner.jacobian, stats)
    if se is None:
        warnings.append("SINGULAR_COVARIANCE")
    estimates = _estimates(values, se, stats.df, config.confidence_level)
    return build(
        status,
        reason,
        initial=initial,
        parameters=estimates,
        covariance=covariance,
        correlation_=corr,
        stats=stats,
        fitted=fitted,
        residuals=residuals,
        equation_fitted=model.equation_text(values),
    )


def _estimates(
    values: dict[str, float], se: FloatArray | None, df: int, confidence_level: float
) -> tuple[ParameterEstimate, ...]:
    if se is None:
        return tuple(ParameterEstimate(name=n, value=v) for n, v in values.items())
    t = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))
    out: list[ParameterEstimate] = []
    for (name, value), err in zip(values.items(), se.tolist(), strict=True):
        margin = t * err
        out.append(
            ParameterEstimate(
                name=name, value=value, std_err=err, ci_low=value - margin, ci_high=value + margin
            )
        )
    return tuple(out)


def _base_start(
    model: ModelSpec, problem: _Problem, config: FitConfig
) -> tuple[FloatArray | None, str]:
    """``config.initial`` over ``model.initial_guess``; ``(None, reason)`` if unusable.

    The initial guess is model code run on the active data; anything it gets wrong
    (raising, missing parameters, non-finite values) makes the fit ``FAILED`` with the
    reason, rather than an exception from deep inside ``fit``.
    """
    override = dict(config.initial or {})
    guess: dict[str, float] = {}
    if any(name not in override for name in problem.names):
        guess_fn: Callable[[FloatArray, FloatArray], dict[str, float]] = model.initial_guess
        try:
            with np.errstate(all="ignore"):
                guess = dict(guess_fn(problem.x.copy(), problem.y.copy()))
        except Exception as exc:  # reported as FAILED with its message
            return None, f"initial_guess raised {type(exc).__name__}: {exc}"
    merged = {**guess, **override}
    missing = [n for n in problem.names if n not in merged]
    if missing:
        return None, f"initial values missing for parameter(s) {missing}"
    try:
        start = np.array([float(merged[n]) for n in problem.names], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return None, f"initial values are not numbers: {exc}"
    bad = [n for n, v in zip(problem.names, start, strict=True) if not math.isfinite(v)]
    if bad:
        return None, f"initial values are not finite for parameter(s) {bad}"
    return start, ""
