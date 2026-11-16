"""Contract every model must satisfy; Phase 4 runs this against the whole catalogue."""

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelSpec

FloatArray = NDArray[np.float64]

_STEP = 1e-6
_RTOL = 1e-6
_NOISE_FRACTION = 0.01


def check_model_contract(
    spec: ModelSpec,
    x: FloatArray,
    true_params: dict[str, float],
    *,
    seed: int = 0,
) -> None:
    """Raise ``AssertionError`` if ``spec`` breaks the model contract at ``true_params``.

    Checks: ``latex()`` is non-empty; ``evaluate`` returns finite float64 of shape
    ``(n,)``; the analytic Jacobian equals central finite differences (rtol 1e-6); and
    ``initial_guess`` — fed data drawn from the model plus 1 % noise (seeded by ``seed``)
    — names every parameter with a finite value inside its bounds.
    """
    names = [p.name for p in spec.parameters]
    if set(true_params) != set(names):
        raise AssertionError(f"true_params keys {sorted(true_params)} must equal {names}")
    x = np.asarray(x, dtype=np.float64)

    if not spec.latex().strip():
        raise AssertionError(f"{spec.id}: latex() is empty")

    y = spec.evaluate(x, **true_params)
    if y.dtype != np.float64 or y.shape != (x.shape[0],):
        raise AssertionError(f"{spec.id}: evaluate returned {y.dtype} {y.shape}, want float64 (n,)")
    if not np.isfinite(y).all():
        raise AssertionError(f"{spec.id}: evaluate is not finite at true_params {true_params}")

    _check_jacobian(spec, x, true_params, names)
    _check_initial_guess(spec, x, y, seed)


def _check_jacobian(
    spec: ModelSpec, x: FloatArray, true_params: dict[str, float], names: list[str]
) -> None:
    analytic = spec.jacobian(x, **true_params)
    if analytic.shape != (x.shape[0], len(names)):
        raise AssertionError(
            f"{spec.id}: Jacobian shape {analytic.shape}, want {(x.shape[0], len(names))}"
        )
    numeric = np.empty_like(analytic)
    for j, name in enumerate(names):
        h = _STEP * max(1.0, abs(true_params[name]))
        up = {**true_params, name: true_params[name] + h}
        down = {**true_params, name: true_params[name] - h}
        numeric[:, j] = (spec.evaluate(x, **up) - spec.evaluate(x, **down)) / (2.0 * h)
    atol = 1e-8 * max(1.0, float(np.max(np.abs(numeric))))
    if not np.allclose(analytic, numeric, rtol=_RTOL, atol=atol):
        worst = float(np.max(np.abs(analytic - numeric)))
        raise AssertionError(
            f"{spec.id}: analytic Jacobian differs from central finite differences "
            f"(max abs difference {worst:.3g})"
        )


def _check_initial_guess(spec: ModelSpec, x: FloatArray, y: FloatArray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(y))) or 1.0
    noisy = y + rng.normal(0.0, _NOISE_FRACTION * scale, size=y.shape)
    guess = spec.initial_guess(x, noisy)
    expected = {p.name for p in spec.parameters}
    if set(guess) != expected:
        raise AssertionError(
            f"{spec.id}: initial_guess keys {sorted(guess)} must equal {sorted(expected)}"
        )
    for p in spec.parameters:
        value = guess[p.name]
        if not np.isfinite(value) or not p.lower <= value <= p.upper:
            raise AssertionError(
                f"{spec.id}: initial_guess {p.name}={value!r} outside bounds [{p.lower}, {p.upper}]"
            )
