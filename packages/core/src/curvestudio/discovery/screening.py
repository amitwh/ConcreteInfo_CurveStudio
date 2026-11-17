"""Decide which candidates the data can support, with reasons for the rest."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery.candidates import MIN_RESIDUAL_DF, parameter_count, resolve_spec
from curvestudio.discovery.types import AutoFitError, CandidateRef, PolynomialRef

MIN_ACTIVE_POINTS = 4
"""Fewer active points than this refuses the whole run."""


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
        AutoFitError: ``TOO_FEW_POINTS`` with fewer than 4 active points.
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
