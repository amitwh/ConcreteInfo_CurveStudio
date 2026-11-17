"""Every generated model matches an independently written NumPy reference.

The references below are written from the mathematical definition of each family (not from the
expression strings of ``curvestudio_models.generated``), so a wrong power, a sign slip or a
mis-ordered parameter in a generated expression is caught. Deliberately not named
``golden_*.py``: those files feed the hand-written catalogue test.
"""

import re
from collections.abc import Callable

import numpy as np
import pytest

from curvestudio_models import generated

FloatArray = np.ndarray
Params = dict[str, float]

_BASES: dict[str, Callable[[FloatArray], FloatArray]] = {
    "x": lambda x: x,
    "inv": np.reciprocal,
    "ln": np.log,
    "sqrt": np.sqrt,
    "sqr": np.square,
    "expneg": lambda x: np.exp(-x),
    "cube": lambda x: x**3,
    "invsqr": lambda x: np.reciprocal(x**2),
    "lnsq": lambda x: np.log(x) ** 2,
    "xlnx": lambda x: x * np.log(x),
    "invsqrt": lambda x: np.reciprocal(np.sqrt(x)),
}
_TRANSFORMS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "id": lambda eta: eta,
    "inv": np.reciprocal,
    "exp": np.exp,
    "sqr": np.square,
    "sqrtof": np.sqrt,
}


def _rat(x: FloatArray, p: Params, m: int, n: int) -> FloatArray:
    numerator = sum(p[f"a{i}"] * x**i for i in range(m + 1))
    denominator = 1.0 + sum(p[f"c{j}"] * x**j for j in range(1, n + 1))
    return np.asarray(numerator / denominator)


def _invpoly(x: FloatArray, p: Params, d: int) -> FloatArray:
    return np.asarray(1.0 / sum(p[f"a{i}"] * x**i for i in range(d + 1)))


def _recip(x: FloatArray, p: Params, d: int) -> FloatArray:
    return np.asarray(sum(p[f"a{i}"] * x ** (-i) for i in range(d + 1)))


def _exp_sum(x: FloatArray, p: Params, n: int, offset: bool) -> FloatArray:
    rate, total = p["r1"], p["a1"] * np.exp(-p["r1"] * x)
    for i in range(2, n + 1):
        rate = rate + p[f"d{i}"]
        total = total + p[f"a{i}"] * np.exp(-rate * x)
    return np.asarray(total + (p["c"] if offset else 0.0))


def _gauss_sum(x: FloatArray, p: Params, n: int) -> FloatArray:
    centre, total = p["mu1"], np.zeros_like(x)
    for i in range(1, n + 1):
        if i > 1:
            centre = centre + p[f"d{i}"]
        total = total + p[f"a{i}"] * np.exp(-((x - centre) ** 2) / (2.0 * p[f"s{i}"] ** 2))
    return total


def _pow_sum(x: FloatArray, p: Params, n: int) -> FloatArray:
    exponent, total = p["p1"], p["a1"] * x ** p["p1"]
    for i in range(2, n + 1):
        exponent = exponent + p[f"d{i}"]
        total = total + p[f"a{i}"] * x**exponent
    return np.asarray(total)


def _fourier(x: FloatArray, p: Params, n: int) -> FloatArray:
    total = np.full_like(x, p["a0"])
    for i in range(1, n + 1):
        total = total + p[f"a{i}"] * np.cos(i * p["w"] * x) + p[f"b{i}"] * np.sin(i * p["w"] * x)
    return total


def _tc(x: FloatArray, p: Params, transform: str, bases: list[str]) -> FloatArray:
    eta = p["a"] + sum(p[name] * _BASES[b](x) for name, b in zip("bc", bases, strict=False))
    return _TRANSFORMS[transform](np.asarray(eta))


def _reference(model_id: str) -> Callable[[FloatArray, Params], FloatArray]:
    parts = model_id.split("_")
    if match := re.fullmatch(r"rat_(\d+)_(\d+)", model_id):
        return lambda x, p: _rat(x, p, int(match[1]), int(match[2]))
    if match := re.fullmatch(r"invpoly_(\d+)", model_id):
        return lambda x, p: _invpoly(x, p, int(match[1]))
    if match := re.fullmatch(r"recip_(\d+)", model_id):
        return lambda x, p: _recip(x, p, int(match[1]))
    if match := re.fullmatch(r"exp_sum_(\d)(_offset)?", model_id):
        return lambda x, p: _exp_sum(x, p, int(match[1]), match[2] is not None)
    if match := re.fullmatch(r"gauss_sum_(\d)", model_id):
        return lambda x, p: _gauss_sum(x, p, int(match[1]))
    if match := re.fullmatch(r"pow_sum_(\d)", model_id):
        return lambda x, p: _pow_sum(x, p, int(match[1]))
    if match := re.fullmatch(r"fourier_(\d)", model_id):
        return lambda x, p: _fourier(x, p, int(match[1]))
    assert parts[0] in ("tc", "tc3"), model_id
    return lambda x, p: _tc(x, p, parts[1], parts[2:])


@pytest.mark.parametrize("model_id", tuple(generated._table()))
def test_generated_model_matches_its_reference(model_id: str) -> None:
    spec = generated.build(model_id)
    assert spec.example is not None
    params = dict(spec.example.params)
    low, high = spec.example.x_range
    x = np.linspace(low, high, 7)
    np.testing.assert_allclose(
        spec.evaluate(x, **params), _reference(model_id)(x, params), rtol=1e-12, atol=1e-12
    )
