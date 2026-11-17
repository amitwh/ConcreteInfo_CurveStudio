"""Decide which candidates the data can support, with reasons for the rest."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery.candidates import MIN_RESIDUAL_DF, parameter_count, resolve_spec
from curvestudio.discovery.types import AutoFitError, CandidateRef, PolynomialRef

MIN_ACTIVE_POINTS = 4
"""Fewer active points than this refuses the whole run."""

CONSTANT_RESPONSE_RTOL = 1e-12
"""A y range at or below this fraction of ``max(1, |y|max)`` is a constant response."""


@dataclass(frozen=True)
class Screening:
    """Outcome of :func:`screen`.

    ``skipped`` maps a candidate id to every reason it cannot be fitted to this data, so a
    skipped model is reported rather than silently dropped.
    """

    n_active: int
    runnable: list[CandidateRef] = field(default_factory=list)
    skipped: dict[str, tuple[str, ...]] = field(default_factory=dict)


def screen(data: XY, candidates: Sequence[CandidateRef]) -> Screening:
    """Split ``candidates`` into those that can be fitted to ``data`` and those that cannot.

    Only active points (weight > 0) count and are shown to each model's applicability
    check. A candidate with ``k`` parameters needs ``k + 3`` active points and ``k``
    distinct x values (``k`` distinct non-zero values for a polynomial through the origin).

    Raises:
        AutoFitError: ``TOO_FEW_POINTS`` with fewer than 4 active points, ``NO_X_VARIATION``
            when every active x is the same, ``CONSTANT_RESPONSE`` when every active y is.
    """
    active = np.ones(data.x.size, dtype=bool) if data.weights is None else data.weights > 0.0
    x, y = data.x[active], data.y[active]
    n_active = int(x.size)
    if n_active < MIN_ACTIVE_POINTS:
        raise AutoFitError(
            "TOO_FEW_POINTS",
            f"Auto-fit needs at least {MIN_ACTIVE_POINTS} active points (weight > 0), "
            f"found {n_active}",
        )
    distinct = int(np.unique(x).size)
    if distinct < 2:
        raise AutoFitError(
            "NO_X_VARIATION",
            f"every active x value is {float(x[0]):.6g}: a curve cannot be fitted to a single "
            "x, so there is nothing to rank; check that the right column was chosen as x",
        )
    if float(np.ptp(y)) <= CONSTANT_RESPONSE_RTOL * max(1.0, float(np.max(np.abs(y)))):
        raise AutoFitError(
            "CONSTANT_RESPONSE",
            f"every active y value is {float(y[0]):.6g}: a constant response is fitted "
            "perfectly by any model with an offset, so no ranking would mean anything; check "
            "that the right column was chosen as y",
        )
    distinct_nonzero = int(np.unique(x[x != 0.0]).size)

    runnable: list[CandidateRef] = []
    skipped: dict[str, tuple[str, ...]] = {}
    for ref in candidates:
        k = parameter_count(ref)
        reasons: list[str] = []
        if n_active < k + MIN_RESIDUAL_DF:
            reasons.append(f"needs at least {k + MIN_RESIDUAL_DF} active points, has {n_active}")
        if isinstance(ref, PolynomialRef) and ref.through_origin:
            if distinct_nonzero < k:
                reasons.append(
                    f"polynomial through the origin needs at least {k} distinct non-zero "
                    f"x values, has {distinct_nonzero}"
                )
        elif distinct < k:
            reasons.append(f"needs at least {k} distinct x values, has {distinct}")
        if not isinstance(ref, PolynomialRef):
            spec = resolve_spec(ref)
            assert spec is not None
            if spec.variables != ("x",):
                reasons.append(
                    f"model uses variables {', '.join(spec.variables)}; Auto-fit fits y against "
                    "a single x"
                )
            else:
                reasons.extend(spec.applicability(x, y))
        if reasons:
            skipped[ref.id] = tuple(reasons)
        else:
            runnable.append(ref)
    return Screening(n_active=n_active, runnable=runnable, skipped=skipped)
