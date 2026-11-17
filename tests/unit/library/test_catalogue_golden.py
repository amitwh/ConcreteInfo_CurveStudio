"""Every catalogue model matches an independently written NumPy reference.

The other catalogue tests derive everything from the expression string itself, so a sign flip
or a wrong power would pass them. Here each model's value at its example parameters is compared
with a plain-NumPy callable from ``golden/golden_<family>.py`` (see ``golden_rational.py`` for
the pattern). A catalogue model without a golden entry fails.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

GOLDEN_DIR = Path(__file__).parent / "golden"
RTOL = 1e-12
DENOMINATOR_RATIO = 1e-3
"""A denominator must stay above this fraction of its largest magnitude over the example."""


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect(attribute: str) -> dict[str, Callable[..., np.ndarray]]:
    merged: dict[str, Callable[..., np.ndarray]] = {}
    for path in sorted(GOLDEN_DIR.glob("golden_*.py")):
        entries = getattr(_load(path), attribute, {})
        clash = sorted(merged.keys() & entries.keys())
        assert not clash, f"{path.name} repeats {attribute} entries {clash}"
        merged.update(entries)
    return merged


GOLDEN = _collect("GOLDEN")
DENOMINATORS = _collect("DENOMINATORS")
CATALOGUE = catalogue()
BY_ID = {m.id: m for m in CATALOGUE}


def test_every_catalogue_model_has_a_golden_reference() -> None:
    missing = sorted(m.id for m in CATALOGUE if m.id not in GOLDEN)
    assert not missing, f"add GOLDEN entries in tests/unit/library/golden/ for: {missing}"


def test_no_golden_entry_is_stale() -> None:
    assert sorted(set(GOLDEN) - BY_ID.keys()) == []
    assert sorted(set(DENOMINATORS) - BY_ID.keys()) == []


def _grid(spec: ModelSpec, n: int) -> np.ndarray:
    assert spec.example is not None
    low, high = spec.example.x_range
    return np.linspace(low, high, n)


@pytest.mark.parametrize("spec", [m for m in CATALOGUE if m.id in GOLDEN], ids=lambda s: s.id)
def test_model_matches_its_golden_reference(spec: ModelSpec) -> None:
    assert spec.example is not None
    params = dict(spec.example.params)
    x = _grid(spec, 5)
    np.testing.assert_allclose(
        spec.evaluate(x, **params), GOLDEN[spec.id](x, **params), rtol=RTOL, atol=0.0
    )


def test_every_rational_model_declares_its_denominator() -> None:
    missing = sorted(
        m.id for m in CATALOGUE if m.category == "rational" and m.id not in DENOMINATORS
    )
    assert not missing, f"add DENOMINATORS entries for: {missing}"


@pytest.mark.parametrize("spec", [m for m in CATALOGUE if m.id in DENOMINATORS], ids=lambda s: s.id)
def test_example_range_stays_away_from_every_pole(spec: ModelSpec) -> None:
    assert spec.example is not None
    magnitude = np.abs(DENOMINATORS[spec.id](_grid(spec, 2001), **dict(spec.example.params)))
    assert np.isfinite(magnitude).all()
    assert float(magnitude.min()) > DENOMINATOR_RATIO * float(magnitude.max()), (
        f"denominator reaches {magnitude.min():.3g} against a maximum of {magnitude.max():.3g}"
    )
