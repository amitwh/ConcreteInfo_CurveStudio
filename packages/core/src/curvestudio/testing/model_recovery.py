"""Noiseless parameter-recovery check: can the engine find a model's own example back?

A model that cannot recover its true parameters from its own noise-free curve is either
badly parametrised (two parameters trade off against each other) or has a poor initial
guess. :func:`check_model_recovery` fits the model's example data from its own
``initial_guess`` and demands the true parameters back to ``1e-6`` relative accuracy.
"""

import math
from dataclasses import dataclass

import numpy as np

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit
from curvestudio.models import ModelSpec

MAX_CONDITION = 1e8
"""Largest allowed condition number of the column-normalised Jacobian at the truth."""

TOLERANCE = 1e-6
"""Allowed miss per parameter: ``TOLERANCE * max(1, |true value|)``."""

SINGLE_START_TOLERANCE = 1e-4
"""Allowed miss for the single-start fit, on the same ``max(1, |true value|)`` scale."""


@dataclass(frozen=True)
class RecoveryReport:
    """What :func:`check_model_recovery` measured for a model that passed."""

    converged: bool
    max_relative_error: float
    single_start_converged: bool
    single_start_recovered: bool
    jacobian_condition: float
    """Condition number of the Jacobian at the truth, columns scaled to unit length."""


def check_model_recovery(
    spec: ModelSpec, *, n_starts: int = 12, seed: int = 0, require_single_start: bool = False
) -> RecoveryReport:
    """Fit ``spec.example_data()`` and require the example parameters back.

    The checks run in this order, and the first failure raises ``AssertionError`` naming
    the model (and the parameters where that applies):

    1. the model has an ``example``;
    2. the Jacobian at the true parameters is well conditioned: its columns are scaled to
       unit length (so the number does not depend on parameter units) and the condition
       number must not exceed :data:`MAX_CONDITION`; otherwise the parameters that cannot
       be told apart are named;
    3. the multistart fit (``n_starts`` seeded starts, the first from ``initial_guess``)
       has status ``CONVERGED``;
    4. every parameter is within ``1e-6 * max(1, |true|)`` of its true value.

    For information, a single start from ``initial_guess`` is fitted too:
    ``single_start_converged`` is its status, ``single_start_recovered`` additionally
    requires every parameter within ``1e-4 * max(1, |true|)`` of the truth (a single start
    can converge to a wrong local minimum). With ``require_single_start=True`` a single
    start that did not recover the parameters is an ``AssertionError`` too.
    """
    if spec.example is None:
        raise AssertionError(f"{spec.id}: the model has no example, so recovery cannot be checked")
    x, y = spec.example_data()
    if not np.isfinite(y).all():
        raise AssertionError(f"{spec.id}: the example curve is not finite")
    truth = {p.name: float(spec.example.params[p.name]) for p in spec.parameters}
    condition = _check_conditioning(spec, x, truth)

    data = XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint=f"example:{spec.id}")
    multi = fit(spec, data, FitConfig(n_starts=n_starts, seed=seed))
    if multi.status is not FitStatus.CONVERGED:
        raise AssertionError(
            f"{spec.id}: the multistart fit on the example did not converge "
            f"({multi.status.value}: {multi.termination_reason})"
        )
    worst, worst_name = _worst_miss(multi, truth)
    if worst > TOLERANCE:
        raise AssertionError(
            f"{spec.id}: parameter {worst_name!r} recovered as "
            f"{_value(multi, worst_name)!r}, true value {truth[worst_name]!r} "
            f"(relative miss {worst:.3g} > {TOLERANCE:g})"
        )
    single = fit(spec, data, FitConfig(n_starts=1, seed=seed))
    single_converged = single.status is FitStatus.CONVERGED
    single_miss = _worst_miss(single, truth) if single_converged else (math.inf, "")
    single_recovered = single_converged and single_miss[0] <= SINGLE_START_TOLERANCE
    if require_single_start and not single_recovered:
        raise AssertionError(
            f"{spec.id}: a single start from initial_guess did not recover the parameters "
            f"({single.status.value}; worst parameter {single_miss[1]!r}, "
            f"relative miss {single_miss[0]:.3g} > {SINGLE_START_TOLERANCE:g})"
        )
    return RecoveryReport(
        converged=True,
        max_relative_error=worst,
        single_start_converged=single_converged,
        single_start_recovered=single_recovered,
        jacobian_condition=condition,
    )


def _worst_miss(result: FitResult, truth: dict[str, float]) -> tuple[float, str]:
    worst, worst_name = 0.0, ""
    for estimate in result.parameters:
        error = abs(estimate.value - truth[estimate.name]) / max(1.0, abs(truth[estimate.name]))
        if error > worst:
            worst, worst_name = error, estimate.name
    return worst, worst_name


def _value(result: FitResult, name: str) -> float:
    return next(p.value for p in result.parameters if p.name == name)


def _check_conditioning(spec: ModelSpec, x: np.ndarray, truth: dict[str, float]) -> float:
    jac = spec.jacobian(x, **truth)
    names = [p.name for p in spec.parameters]
    if not np.isfinite(jac).all():
        raise AssertionError(f"{spec.id}: the Jacobian is not finite at the example parameters")
    if jac.shape[0] < len(names):
        raise AssertionError(
            f"{spec.id}: the example has {jac.shape[0]} points for {len(names)} parameters"
        )
    norms = np.linalg.norm(jac, axis=0)
    dead = [name for name, norm in zip(names, norms, strict=True) if norm == 0.0]
    if dead:
        raise AssertionError(
            f"{spec.id}: the example does not constrain parameter(s) {dead} "
            "(their Jacobian column is zero)"
        )
    _, singular, vt = np.linalg.svd(jac / norms, full_matrices=False)
    smallest = float(singular[-1])
    condition = math.inf if smallest == 0.0 else float(singular[0]) / smallest
    if condition > MAX_CONDITION:
        weak = vt[-1]
        involved = [n for n, v in zip(names, weak, strict=True) if abs(v) > 0.1]
        raise AssertionError(
            f"{spec.id}: parameters {involved} are not separately identifiable from the example "
            f"(Jacobian condition number {condition:.3g} > {MAX_CONDITION:g}); "
            "reparametrise the model or change the example range"
        )
    return condition
