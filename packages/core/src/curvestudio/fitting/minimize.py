"""Scalar-objective optimizers: derivative-free, quasi-Newton and global methods.

``nelder``, ``powell``, ``bfgs``, ``lbfgsb``, ``differential_evolution`` and
``dual_annealing`` minimise the weighted sum of squares ``sum(w (y - f)^2)`` through
SciPy's scalar optimizers. They exist for problems where ``least_squares`` struggles (rough
surfaces, many local minima) at the price of speed and of the least-squares structure, so
they take no robust loss (that is a ``least_squares`` feature) and report a plain
:class:`MinimizeOutcome`; the caller (:mod:`curvestudio.fitting.nonlinear`) applies the same
finite checks, covariance and warnings as for the least-squares methods.

``bfgs`` and ``lbfgsb`` use the analytic gradient ``2 J_r' r`` built from the engine's
residual Jacobian (never finite differences). Global methods use SciPy's own polishing:
``differential_evolution(polish=True)`` finishes with an L-BFGS-B local search and
``dual_annealing`` runs its built-in local search; no custom polish is added here. The
global methods are seeded from ``config.seed`` and run single-threaded, so an identical
configuration gives an identical result. Their population/annealing is also started from
the (clipped) initial guess via ``x0``.

How ``FitConfig`` limits map onto each method (a dash means the setting is ignored):

============================ ================ ============== ============== ==============
method                       ``max_nfev``     ``ftol``       ``xtol``       ``gtol``
============================ ================ ============== ============== ==============
nelder (Nelder-Mead)         ``maxfev``       ``fatol``      ``xatol``      -
powell                       ``maxfev``       ``ftol``       ``xtol``       -
bfgs                         ``maxiter`` [1]_ -              -              ``gtol`` [4]_
lbfgsb (L-BFGS-B)            ``maxfun``       ``ftol``       -              ``gtol`` [4]_
differential_evolution       ``maxiter`` [2]_ ``tol`` [3]_   -              -
dual_annealing               ``maxfun``       -              -              -
============================ ================ ============== ============== ==============

.. [1] BFGS has no evaluation cap, so ``max_nfev`` caps iterations (one gradient and at
   least one function evaluation each).
.. [2] ``max_nfev // (popsize * k)`` generations (at least 1); each generation costs about
   ``popsize * k`` evaluations, plus the polish.
.. [3] Differential evolution's relative population-spread tolerance; ``atol`` stays 0.
.. [4] Relative to the starting gradient: ``gtol * max(1, |g0|_inf)``, see
   :func:`minimize_sse`.

Success follows SciPy's ``success`` flag. For ``dual_annealing`` SciPy sets it only when
the annealing finished normally; hitting the evaluation budget is reported as not
converged with SciPy's message.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, differential_evolution, dual_annealing, minimize

from curvestudio.fitting.config import FitConfig

FloatArray = NDArray[np.float64]

SCALAR_METHODS = frozenset(
    {"nelder", "powell", "bfgs", "lbfgsb", "differential_evolution", "dual_annealing"}
)
"""Names handled by :func:`minimize_sse`."""

_DE_POPSIZE = 15
"""SciPy's default population multiplier (population = popsize * number of parameters)."""

_MINIMIZE_NAMES = {
    "nelder": "Nelder-Mead",
    "powell": "Powell",
    "bfgs": "BFGS",
    "lbfgsb": "L-BFGS-B",
}


@dataclass(frozen=True)
class MinimizeOutcome:
    """What a scalar optimizer returned: the point, SciPy's success flag and message."""

    x: FloatArray
    success: bool
    message: str


def _minimize_options(
    method: str, config: FitConfig, gradient_scale: float
) -> dict[str, float | int]:
    """Options for ``scipy.optimize.minimize`` per the mapping table in the module docstring.

    ``gradient_scale`` multiplies ``gtol`` for ``bfgs``/``lbfgsb`` (see :func:`minimize_sse`).
    """
    options: dict[str, float | int] = {}
    if method == "nelder":
        options.update(xatol=config.xtol, fatol=config.ftol)
        if config.max_nfev is not None:
            options["maxfev"] = config.max_nfev
    elif method == "powell":
        options.update(xtol=config.xtol, ftol=config.ftol)
        if config.max_nfev is not None:
            options["maxfev"] = config.max_nfev
    elif method == "bfgs":
        options["gtol"] = config.gtol * gradient_scale
        if config.max_nfev is not None:
            options["maxiter"] = config.max_nfev
    else:  # lbfgsb
        options.update(ftol=config.ftol, gtol=config.gtol * gradient_scale)
        if config.max_nfev is not None:
            options["maxfun"] = config.max_nfev
    return options


def minimize_sse(
    objective: Callable[[FloatArray], float],
    gradient: Callable[[FloatArray], FloatArray],
    start: FloatArray,
    method: str,
    lower: FloatArray,
    upper: FloatArray,
    config: FitConfig,
) -> MinimizeOutcome:
    """Minimise ``objective`` from ``start`` with the scalar optimizer ``method``.

    ``gradient`` is only called by ``bfgs`` and ``lbfgsb``. Their ``gtol`` is *relative*:
    the tolerance handed to SciPy is ``config.gtol * max(1, |gradient(start)|_inf)``. The
    gradient of a sum of squares scales with the data, so an absolute ``1e-10`` is below
    what floating point can reach for data of ordinary size and made BFGS stop with
    "precision loss" (not converged) on some platforms. Bounds are passed to SciPy only
    when at least one is finite (``bfgs`` never receives any: the method check rejects it);
    ``differential_evolution`` and ``dual_annealing`` need finite bounds on every
    parameter, which :func:`curvestudio.fitting.methods.check_method_compatibility`
    guarantees. Exceptions from the objective or from SciPy propagate to the caller, which
    reports them as a failed run.
    """
    bounds = Bounds(lower, upper)
    if method in _MINIMIZE_NAMES:
        finite = bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
        gradient_scale = 1.0
        if method in ("bfgs", "lbfgsb"):
            gradient_scale = max(1.0, float(np.max(np.abs(gradient(start)))))
        ret = minimize(
            objective,
            start,
            jac=gradient if method in ("bfgs", "lbfgsb") else None,
            method=_MINIMIZE_NAMES[method],
            bounds=bounds if finite and method != "bfgs" else None,
            options=_minimize_options(method, config, gradient_scale),
        )
    elif method == "differential_evolution":
        maxiter = 1000
        if config.max_nfev is not None:
            maxiter = max(1, config.max_nfev // (_DE_POPSIZE * start.size))
        ret = differential_evolution(
            objective,
            bounds,
            x0=start,
            seed=config.seed,
            tol=config.ftol,
            maxiter=maxiter,
            popsize=_DE_POPSIZE,
            polish=True,
            workers=1,
        )
    elif method == "dual_annealing":
        ret = dual_annealing(
            objective,
            bounds,
            x0=start,
            seed=config.seed,
            maxfun=config.max_nfev if config.max_nfev is not None else 10_000_000,
        )
    else:
        raise ValueError(f"unknown scalar method {method!r}")
    x = np.asarray(ret.x, dtype=np.float64)
    return MinimizeOutcome(x=x, success=bool(ret.success), message=str(ret.message))
