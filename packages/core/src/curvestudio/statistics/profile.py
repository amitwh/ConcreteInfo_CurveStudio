"""Profile-likelihood confidence intervals for nonlinear least-squares parameters.

A Wald interval ``estimate +- t * SE`` is exact only when the model is linear in the
parameters. The profile interval instead asks directly how far one parameter can move
before the fit gets significantly worse: for parameter ``i``, fix ``theta_i`` at a trial
value, re-optimise all the others, and record the smallest weighted SSE reachable,
``SSE_p(theta_i)``. The ``level`` interval is the set where (Bates & Watts, *Nonlinear
Regression Analysis and Its Applications*, 1988, sec. 6.1)::

    SSE_p(theta_i) <= SSE_min * (1 + F_level(1, df) / df)

It follows the curvature of the model, so it is asymmetric where the Wald interval is not,
and for a linear model it coincides with the Wald t-interval exactly (SSE is then quadratic).

Each end is found by walking away from the estimate in multiples of the Wald standard error
(at most :data:`MAX_SE_MULTIPLE`) until the profile exceeds the threshold, then refining the
crossing with Brent's method. A side that reaches a finite parameter bound while still under
the threshold is ``at_bound`` (the interval is cut by the bound, which is reported as the
limit); a side that cannot be bracketed, or whose profile cannot be computed reliably, is
``not_found`` with no limit rather than an invented number.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import stats as scipy_stats
from scipy.optimize import brentq, least_squares

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.fitting.minimize import SCALAR_METHODS
from curvestudio.models import ModelSpec
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView, build_view

FloatArray = NDArray[np.float64]

LimitStatus = Literal["found", "at_bound", "not_found"]

MAX_SE_MULTIPLE = 50.0
"""Farthest distance from the estimate searched, in Wald standard errors."""

_SE_STEPS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 50.0)
"""Bracketing schedule in standard errors; ends at :data:`MAX_SE_MULTIPLE`."""

_ROOT_XTOL_SE = 1e-10
"""Absolute tolerance of the crossing, as a fraction of one standard error."""

_ROOT_RTOL = 1e-12
_ROOT_MAXITER = 200
_VERIFY_RTOL = 1e-6
"""A located crossing is accepted only when the profile there is within this relative
distance of the threshold (it is not, e.g., a jump where the model leaves its domain)."""

_NON_FINITE_PENALTY = 1e100
"""Stand-in for a non-finite residual, as in the fitting engine (squares stay finite)."""


class ProfileInterval(BaseModel):
    """Profile-likelihood interval of one parameter.

    ``ci_low``/``ci_high`` are ``None`` exactly when the matching status is ``not_found``.
    For ``at_bound`` the limit is the parameter bound itself. ``n_evals`` counts the
    constrained re-optimisations spent on both sides.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    estimate: FiniteFloat
    ci_low: FiniteFloat | None
    ci_high: FiniteFloat | None
    low_status: LimitStatus
    high_status: LimitStatus
    n_evals: int


class ProfileResult(BaseModel):
    """Profile intervals of every parameter at one confidence ``level``.

    ``threshold_sse`` is the weighted SSE a constrained fit may reach at an interval end.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: FiniteFloat
    threshold_sse: FiniteFloat
    intervals: tuple[ProfileInterval, ...]


class _NonFiniteJacobianError(Exception):
    """A constrained re-optimisation met a non-finite Jacobian."""


@dataclass(frozen=True)
class _Constrained:
    """The weighted least-squares problem of one fit, with parameters fixable one at a time."""

    model: ModelSpec
    x: FloatArray
    y: FloatArray
    root_w: FloatArray
    names: tuple[str, ...]
    lower: FloatArray
    upper: FloatArray
    config: FitConfig

    def solve(self, fixed: int, value: float, warm: FloatArray) -> tuple[float, FloatArray, bool]:
        """Minimise the weighted SSE over every parameter but ``fixed``, held at ``value``.

        Returns ``(sse, free_parameters, ok)``; ``ok`` is false when the optimiser did not
        report success or met a non-finite Jacobian, so ``sse`` cannot be trusted.
        """
        free = [j for j in range(len(self.names)) if j != fixed]

        def theta_of(free_values: FloatArray) -> dict[str, float]:
            theta = dict.fromkeys(self.names, 0.0)
            theta[self.names[fixed]] = value
            theta.update(zip((self.names[j] for j in free), free_values.tolist(), strict=True))
            return theta

        def residuals(free_values: FloatArray) -> FloatArray:
            f = self.model.evaluate(self.x, **theta_of(free_values))
            with np.errstate(all="ignore"):
                r = self.root_w * (self.y - f)
            return np.where(np.isfinite(r), r, _NON_FINITE_PENALTY)

        if not free:
            r = residuals(np.empty(0))
            return float(r @ r), warm, True

        def jacobian(free_values: FloatArray) -> FloatArray:
            jac = self.model.jacobian(self.x, **theta_of(free_values))[:, free]
            with np.errstate(all="ignore"):
                jac = -self.root_w[:, None] * jac
            if not bool(np.all(np.isfinite(jac))):
                raise _NonFiniteJacobianError
            return np.asarray(jac, dtype=np.float64)

        lower, upper = self.lower[free], self.upper[free]
        bounded = bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
        start = np.clip(warm, lower, upper)
        try:
            ret = least_squares(
                residuals,
                start,
                jac=jacobian,
                bounds=(lower, upper),
                method="trf" if bounded else "lm",
                x_scale="jac",
                ftol=self.config.ftol,
                xtol=self.config.xtol,
                gtol=self.config.gtol,
                max_nfev=self.config.max_nfev,
            )
        except _NonFiniteJacobianError:
            return math.inf, warm, False
        solution = np.asarray(ret.x, dtype=np.float64)
        r = residuals(solution)
        return float(r @ r), solution, bool(ret.success)


def _effective_bounds(model: ModelSpec, config: FitConfig) -> tuple[FloatArray, FloatArray]:
    """Model bounds overridden per parameter by ``config.bounds`` (as :func:`fit` applies them)."""
    overrides = config.bounds or {}
    known = {p.name for p in model.parameters}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"config.bounds names unknown parameter(s) {unknown} for model {model.id!r}"
        )
    lower = np.array([overrides.get(p.name, (p.lower, p.upper))[0] for p in model.parameters])
    upper = np.array([overrides.get(p.name, (p.lower, p.upper))[1] for p in model.parameters])
    return lower, upper


class _Side:
    """One end of one parameter's interval: bracket the threshold crossing, then refine it."""

    def __init__(
        self,
        problem: _Constrained,
        view: FitView,
        index: int,
        direction: float,
        sse_min: float,
        threshold: float,
    ) -> None:
        self._problem = problem
        self._index = index
        self._direction = direction
        self._threshold = threshold
        self._scale = math.sqrt(float(view.require_covariance()[index, index]))
        self._estimate = float(view.theta[index])
        free = [j for j in range(len(view.names)) if j != index]
        self._known: list[tuple[float, FloatArray]] = [(self._estimate, view.theta[free].copy())]
        # Exactly SSE_min - threshold at the estimate, which is below zero by construction.
        self._values: dict[float, float] = {self._estimate: sse_min - threshold}
        self.n_evals = 0
        self._all_ok = True

    def _excess(self, t: float) -> float:
        """Profile SSE at ``t`` minus the threshold, with the nearest known start as warm start."""
        cached = self._values.get(t)
        if cached is not None:
            return cached
        _, warm = min(self._known, key=lambda item: abs(item[0] - t))
        sse, free_values, ok = self._problem.solve(self._index, t, warm)
        self.n_evals += 1
        self._all_ok = self._all_ok and ok
        if ok:
            self._known.append((t, free_values))
        value = sse - self._threshold
        self._values[t] = value
        return value

    def find(self) -> tuple[float | None, LimitStatus]:
        direction = self._direction
        bound = float((self._problem.upper if direction > 0 else self._problem.lower)[self._index])
        previous = self._estimate
        for multiple in _SE_STEPS:
            target = self._estimate + direction * multiple * self._scale
            at_bound = math.isfinite(bound) and (target - bound) * direction >= 0.0
            if at_bound:
                target = bound
            if target == previous:  # the estimate sits on the bound: nothing to search
                return bound, "at_bound"
            if self._excess(target) >= 0.0:
                return self._refine(previous, target)
            if at_bound:
                return (None, "not_found") if not self._all_ok else (bound, "at_bound")
            previous = target
        return None, "not_found"

    def _refine(self, inside: float, outside: float) -> tuple[float | None, LimitStatus]:
        low, high = sorted((inside, outside))
        root, info = brentq(
            self._excess,
            low,
            high,
            xtol=_ROOT_XTOL_SE * self._scale,
            rtol=_ROOT_RTOL,
            maxiter=_ROOT_MAXITER,
            full_output=True,
            disp=False,
        )
        if not info.converged or not self._all_ok:
            return None, "not_found"
        if abs(self._excess(float(root))) > _VERIFY_RTOL * self._threshold:
            return None, "not_found"
        return float(root), "found"


def _unsupported(message: str) -> StatisticsError:
    return StatisticsError("PROFILE_NOT_SUPPORTED", message)


# A frozen FitConfig is immutable, so sharing one default instance is safe (B008).
def profile_likelihood(
    data: XY,
    model: ModelSpec | str,
    result: FitResult,
    config: FitConfig = FitConfig(),  # noqa: B008
    level: float = 0.95,
) -> ProfileResult:
    """Profile-likelihood confidence intervals for every parameter of a nonlinear fit.

    ``config`` must be the configuration the fit was run with: its bounds are the
    constraints the constrained re-optimisations respect and its tolerances drive them.
    Parameter ``i`` is stepped away from its estimate in each direction, the other
    parameters are re-optimised with ``scipy.optimize.least_squares`` (warm-started from
    the nearest earlier profile point), and the interval ends are the points where the
    profile weighted SSE equals ``SSE_min * (1 + F_level(1, df) / df)``. The computation
    is deterministic (no random numbers). Noiseless data (``SSE = 0``) give a zero-width
    interval at the estimate.

    Raises:
        StatisticsError: ``FIT_NOT_CONVERGED`` / ``DATA_MISMATCH`` / ``MODEL_MISMATCH``
            (from :func:`~curvestudio.statistics.build_view`); ``NO_DEGREES_OF_FREEDOM`` /
            ``NO_COVARIANCE`` (the Wald scale of the search is missing);
            ``PROFILE_NOT_SUPPORTED`` for a polynomial result (linear in its coefficients,
            so the Wald interval is already exact), a robust ``loss`` (the profile is
            defined for the least-squares SSE) or a scalar-minimiser method.
        ValueError: ``level`` is not strictly between 0 and 1, or ``config.bounds`` names
            an unknown parameter.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    view = build_view(data, result, model)
    if view.model is None:
        raise _unsupported(
            "profile likelihood is not defined for polynomial fits: they are linear in their "
            "coefficients, so the Wald (t) interval is already exact"
        )
    if config.loss != "linear" or result.loss != "linear":
        raise _unsupported(
            f"profile likelihood needs the least-squares ('linear') loss, not "
            f"{config.loss if config.loss != 'linear' else result.loss!r}"
        )
    if config.method in SCALAR_METHODS or result.method in SCALAR_METHODS:
        raise _unsupported(
            "profile likelihood is not supported for scalar-minimiser methods "
            f"({result.method!r}); refit with 'lm', 'trf' or 'dogbox'"
        )
    covariance = view.require_covariance()
    assert result.stats is not None  # build_view guarantees statistics on a converged fit

    sse_min = result.stats.sse
    threshold = sse_min * (1.0 + float(scipy_stats.f.ppf(level, 1, view.df)) / view.df)
    lower, upper = _effective_bounds(view.model, config)

    if sse_min == 0.0:  # noiseless: any move makes the SSE positive, so the interval is a point
        intervals = tuple(
            ProfileInterval(
                name=name,
                estimate=float(estimate),
                ci_low=float(estimate),
                ci_high=float(estimate),
                low_status="found",
                high_status="found",
                n_evals=0,
            )
            for name, estimate in zip(view.names, view.theta, strict=True)
        )
        return ProfileResult(level=level, threshold_sse=threshold, intervals=intervals)

    problem = _Constrained(
        model=view.model,
        x=view.x,
        y=view.y,
        root_w=np.sqrt(view.w),
        names=view.names,
        lower=lower,
        upper=upper,
        config=config,
    )
    intervals_list: list[ProfileInterval] = []
    for index, name in enumerate(view.names):
        if not covariance[index, index] > 0.0:
            raise StatisticsError(
                "NO_COVARIANCE",
                f"parameter {name!r} has no positive variance; the profile search has no scale",
            )
        low_side = _Side(problem, view, index, -1.0, sse_min, threshold)
        high_side = _Side(problem, view, index, 1.0, sse_min, threshold)
        ci_low, low_status = low_side.find()
        ci_high, high_status = high_side.find()
        intervals_list.append(
            ProfileInterval(
                name=name,
                estimate=float(view.theta[index]),
                ci_low=ci_low,
                ci_high=ci_high,
                low_status=low_status,
                high_status=high_status,
                n_evals=low_side.n_evals + high_side.n_evals,
            )
        )
    return ProfileResult(level=level, threshold_sse=threshold, intervals=tuple(intervals_list))
