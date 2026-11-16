"""Settings for one nonlinear fit.

Everything that can be checked without the model and data is checked at construction,
so a misconfigured fit fails immediately with ``ValueError`` rather than producing a
``FAILED`` result that looks like a numerical problem. Method/loss names are checked by
:func:`curvestudio.fitting.methods.check_method_compatibility` once bounds are known.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FitConfig:
    """How to run a nonlinear fit.

    ``initial`` overrides the model's initial guess per parameter, ``bounds`` overrides
    the model's parameter bounds per parameter (``(lower, upper)``, ``lower < upper``;
    infinite values mean unbounded). ``n_starts > 1`` adds seeded random restarts (see
    :func:`curvestudio.fitting.nonlinear.generate_starts`). ``max_nfev`` caps function
    evaluations per start (``None`` = SciPy's default); the tolerances go to
    ``scipy.optimize.least_squares`` unchanged.

    ``ftol`` defaults to 1e-14, not 1e-10: it bounds the *relative SSE* change, and in an
    ill-conditioned valley the parameters are only resolved to about the square root of
    it, so 1e-10 let fits report ``CONVERGED`` with ~5 correct digits (NIST MGH09).
    """

    method: str = "auto"
    loss: str = "linear"
    initial: dict[str, float] | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    n_starts: int = 1
    seed: int = 0
    max_nfev: int | None = None
    ftol: float = 1e-14
    xtol: float = 1e-10
    gtol: float = 1e-10
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not _is_int(self.n_starts) or self.n_starts < 1:
            raise ValueError(f"n_starts must be an int >= 1, got {self.n_starts!r}")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"seed must be an int >= 0, got {self.seed!r}")
        if self.max_nfev is not None and (not _is_int(self.max_nfev) or self.max_nfev < 1):
            raise ValueError(f"max_nfev must be None or an int >= 1, got {self.max_nfev!r}")
        for name in ("ftol", "xtol", "gtol"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level!r}")
        for name, value in (self.initial or {}).items():
            if not math.isfinite(value):
                raise ValueError(f"initial value for {name!r} must be finite, got {value!r}")
        for name, (lower, upper) in (self.bounds or {}).items():
            if math.isnan(lower) or math.isnan(upper) or not lower < upper:
                raise ValueError(
                    f"bounds for {name!r} must satisfy lower < upper, got ({lower!r}, {upper!r})"
                )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
