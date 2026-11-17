"""Every catalogue model (and every core starter) satisfies the model contract."""

import hashlib
import re

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio.models.starter import STARTER_MODELS
from curvestudio.testing.model_contract import check_model_contract
from curvestudio_models import catalogue, family_failures

pytestmark = pytest.mark.real_plugins

ALL_MODELS: tuple[ModelSpec, ...] = (*catalogue(), *STARTER_MODELS)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_model_contract(spec: ModelSpec) -> None:
    assert spec.example is not None
    x, _ = spec.example_data()
    check_model_contract(spec, x, dict(spec.example.params), strict_metadata=True)


def test_ids_are_unique_and_snake_case() -> None:
    ids = [m.id for m in ALL_MODELS]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    assert [i for i in ids if not _SNAKE_CASE.match(i)] == []


def test_catalogue_excludes_the_core_starters() -> None:
    assert {m.id for m in catalogue()}.isdisjoint(m.id for m in STARTER_MODELS)


def test_rational_family_is_present() -> None:
    rational = {m.id for m in catalogue() if m.category == "rational"}
    assert len(rational) >= 8
    assert {
        "rational_reciprocal_offset",
        "rational_reciprocal_linear",
        "rational_x_over_linear",
        "rational_reciprocal_quadratic_offset",
        "rational_1_1",
        "rational_2_1",
        "rational_agnesi",
        "rational_1_2",
    } <= rational


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_example_parameters_lie_strictly_inside_their_bounds(spec: ModelSpec) -> None:
    assert spec.example is not None
    for p in spec.parameters:
        value = spec.example.params[p.name]
        assert p.lower < value < p.upper, (
            f"{p.name}={value} not strictly inside {p.lower}..{p.upper}"
        )


def test_no_family_module_was_skipped() -> None:
    assert family_failures() == {}


_DEGENERATE = {
    "empty": (np.array([]), np.array([])),
    "one point": (np.array([2.0]), np.array([3.0])),
    "constant y": (np.linspace(1.0, 5.0, 8), np.full(8, 2.5)),
    "repeated x": (np.full(6, 2.0), np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])),
    "all zero": (np.linspace(0.0, 4.0, 7), np.zeros(7)),
    "non-finite": (np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, np.inf, 4.0])),
    "huge": (np.linspace(1.0, 2.0, 5) * 1e150, np.linspace(1.0, 2.0, 5) * 1e150),
}


@pytest.mark.parametrize("case", sorted(_DEGENERATE))
@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_initial_guess_is_finite_and_inside_bounds_for_degenerate_data(
    spec: ModelSpec, case: str
) -> None:
    x, y = _DEGENERATE[case]
    guess = spec.initial_guess(x, y)
    assert set(guess) == {p.name for p in spec.parameters}
    for p in spec.parameters:
        assert np.isfinite(guess[p.name]), (p.name, guess[p.name])
        assert p.lower <= guess[p.name] <= p.upper, (p.name, guess[p.name])


GUESS_MISS_EXACT = 1e-2
"""On exact example data the guess curve may leave at most this fraction of the variance."""
GUESS_MISS_NOISY = 1e-1
"""The same with 2 % multiplicative noise: the guess must be a sound start, not an estimate."""


def _unexplained(spec: ModelSpec, x: np.ndarray, y: np.ndarray) -> float:
    curve = spec.evaluate(x, **spec.initial_guess(x, y))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return float(np.sum((y - curve) ** 2)) / sst


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_initial_guess_curve_explains_the_example_data(spec: ModelSpec) -> None:
    """A garbage guess can still converge on an easy model, so judge the guess itself."""
    x, y = spec.example_data()
    assert _unexplained(spec, x, y) <= GUESS_MISS_EXACT
    seed = int.from_bytes(hashlib.sha256(f"guess:{spec.id}".encode()).digest()[:4], "big")
    noisy = y * (1.0 + 0.02 * np.random.default_rng(seed).standard_normal(y.size))
    assert _unexplained(spec, x, noisy) <= GUESS_MISS_NOISY
