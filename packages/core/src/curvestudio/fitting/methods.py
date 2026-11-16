"""Which optimisation methods and loss functions may be combined, and how ``auto`` resolves.

The checks run before any fitting so that an impossible request (for example
Levenberg-Marquardt with bounds, which MINPACK cannot honour) is a clear ``ValueError``
naming the offending option instead of an obscure SciPy error halfway through a fit.
"""

from dataclasses import dataclass

AUTO = "auto"
LINEAR_LOSS = "linear"

LOSSES = frozenset({"linear", "soft_l1", "huber", "cauchy", "arctan"})
"""Loss functions understood by ``scipy.optimize.least_squares``."""


@dataclass(frozen=True)
class _MethodInfo:
    supports_bounds: bool
    supports_robust_loss: bool
    needs_jacobian: bool


_METHODS: dict[str, _MethodInfo] = {
    # MINPACK Levenberg-Marquardt: unconstrained, plain least squares only.
    "lm": _MethodInfo(supports_bounds=False, supports_robust_loss=False, needs_jacobian=True),
    "trf": _MethodInfo(supports_bounds=True, supports_robust_loss=True, needs_jacobian=True),
    "dogbox": _MethodInfo(supports_bounds=True, supports_robust_loss=True, needs_jacobian=True),
}
"""Gradient-based methods. They all need the analytic Jacobian here: this engine does not
fall back to finite differences, because the covariance is built from the same Jacobian
and a numerically differenced one would silently degrade the reported uncertainties."""

METHODS = frozenset({AUTO, *_METHODS})


def resolve_method(method: str, loss: str, has_bounds: bool) -> str:
    """Concrete method name for ``method``; ``auto`` picks ``lm`` when it is allowed.

    ``auto`` becomes ``trf`` when there are bounds or a robust (non-``linear``) loss,
    because ``lm`` supports neither; otherwise ``lm``. Call
    :func:`check_method_compatibility` first so unknown names never reach this point.
    """
    if method != AUTO:
        return method
    return "trf" if has_bounds or loss != LINEAR_LOSS else "lm"


def check_method_compatibility(
    method: str, loss: str, has_bounds: bool, has_jacobian: bool
) -> None:
    """Raise ``ValueError`` naming the offending option if the combination is impossible.

    Checked in order: unknown method, unknown loss, ``bounds`` with a method that cannot
    honour them, a robust ``loss`` with ``lm``, and a missing ``jacobian`` for a method
    that needs one. ``auto`` is checked as the method it resolves to.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {sorted(METHODS)}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; choose one of {sorted(LOSSES)}")
    resolved = resolve_method(method, loss, has_bounds)
    info = _METHODS[resolved]
    if has_bounds and not info.supports_bounds:
        raise ValueError(
            f"method {resolved!r} does not support bounds; use 'trf' or 'dogbox', "
            "or remove the finite parameter bounds"
        )
    if loss != LINEAR_LOSS and not info.supports_robust_loss:
        raise ValueError(
            f"method {resolved!r} supports only the 'linear' loss, not loss {loss!r}; "
            "use 'trf' or 'dogbox' for robust losses"
        )
    if info.needs_jacobian and not has_jacobian:
        raise ValueError(f"method {resolved!r} needs an analytic jacobian, none was supplied")
