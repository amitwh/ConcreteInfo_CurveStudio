"""The single public entry point for nonlinear curve fitting.

``fit`` separates *configuration* errors from *numerical* outcomes: anything the caller
got wrong (unknown model id, unknown parameter names, impossible method/loss/bounds
combinations) raises before any computation, while everything that depends on the data
(too few points, a model that does not apply, an optimizer that fails or does not
converge) comes back as a :class:`FitResult` with the matching status.
"""

import numpy as np
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.fitting.config import FitConfig
from curvestudio.fitting.methods import check_method_compatibility, resolve_method
from curvestudio.fitting.nonlinear import fit_nonlinear
from curvestudio.fitting.result import FitResult
from curvestudio.models.registry import get_model
from curvestudio.models.spec import ModelSpec

FloatArray = NDArray[np.float64]


def _effective_bounds(model: ModelSpec, config: FitConfig) -> tuple[FloatArray, FloatArray]:
    """Model bounds overridden per parameter by ``config.bounds``.

    Raises:
        ValueError: a bound names an unknown parameter, or a parameter ends up with
            ``lower == upper`` (fixing a parameter is not supported by this engine).
    """
    overrides = config.bounds or {}
    lower = np.empty(len(model.parameters))
    upper = np.empty(len(model.parameters))
    for j, p in enumerate(model.parameters):
        lower[j], upper[j] = overrides.get(p.name, (p.lower, p.upper))
        if not lower[j] < upper[j]:
            raise ValueError(
                f"parameter {p.name!r} has bounds ({lower[j]!r}, {upper[j]!r}); "
                "lower must be strictly below upper"
            )
    return lower, upper


# A frozen FitConfig is immutable, so sharing one default instance is safe (B008).
def fit(model: ModelSpec | str, data: XY, config: FitConfig = FitConfig()) -> FitResult:  # noqa: B008
    """Fit ``model`` (a spec or a registered id) to ``data`` by nonlinear least squares.

    Starts come from ``config.initial`` per parameter, else ``model.initial_guess`` on the
    active (weight > 0) points, clipped into the effective bounds; ``n_starts > 1`` adds
    seeded restarts and the lowest-SSE converged run wins. ``data.weights`` multiply the
    squared residuals. ``config.method`` may also be a scalar minimiser of the weighted SSE
    (``nelder``, ``powell``, ``bfgs``, ``lbfgsb``, ``differential_evolution``,
    ``dual_annealing``; see :mod:`curvestudio.fitting.minimize`) which takes no robust loss.
    Polynomials are not special-cased; use ``fit_polynomial`` for them.

    Status: ``INVALID_INPUT`` when there are no more active points than parameters or
    ``model.applicability`` objects; ``FAILED`` when the start is unusable or the optimizer
    raises (message kept); ``NOT_CONVERGED`` when the optimizer stops without success or
    the solution is not finite; otherwise ``CONVERGED``. See
    :mod:`curvestudio.fitting.nonlinear` for the covariance and warning conventions.

    Raises:
        KeyError: ``model`` is a string that is not a registered model id.
        ValueError: ``config`` names unknown parameters, asks for an impossible
            method/loss/bounds combination, or ``model`` has more than one variable.
    """
    spec = get_model(model) if isinstance(model, str) else model
    if len(spec.variables) != 1:
        raise ValueError(
            f"model {spec.id!r} has variables {spec.variables}; fit() takes single-variable "
            "models (XY data)"
        )
    known = {p.name for p in spec.parameters}
    for option in ("initial", "bounds"):
        unknown = sorted(set(getattr(config, option) or {}) - known)
        if unknown:
            raise ValueError(
                f"config.{option} names unknown parameter(s) {unknown} for model {spec.id!r}"
            )
    lower, upper = _effective_bounds(spec, config)
    has_bounds = bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
    check_method_compatibility(
        config.method,
        config.loss,
        has_bounds,
        has_jacobian=True,
        all_bounds_finite=bool(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))),
        n_starts=config.n_starts,
    )
    method = resolve_method(config.method, config.loss, has_bounds)
    return fit_nonlinear(spec, data, config, method=method, lower=lower, upper=upper)
