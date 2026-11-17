"""Every engineering model satisfies the model contract and an independent golden reference.

The engineering catalogue is an optional package with its own families; these tests run the
same harness as the general catalogue (strict metadata, Jacobian against finite differences,
seeded initial guess, conditioning at the example) plus the checks that are specific to it:
the catalogue is large enough, its ids are unique and snake_case, every parameter documents
its unit, every model is a one-variable ``x`` model so Auto-fit can use it, and no two models
repeat an expression. A model without a golden NumPy reference fails, and so does a golden
entry left behind by a removed model.
"""

import importlib.metadata
import importlib.util
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from curvestudio.models import ModelSpec, list_model_ids, load_plugins, plugin_failures
from curvestudio.models.registry import PLUGIN_GROUP, _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.testing.model_contract import check_model_contract
from curvestudio_engineering import catalogue, family_failures
from curvestudio_engineering.register import equivalent_pairs

pytestmark = pytest.mark.real_plugins

ALL_MODELS: tuple[ModelSpec, ...] = catalogue()

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

ENGINEERING_FLOOR = 30
"""The plan's agreed minimum size of the concrete and civil catalogue."""

GOLDEN_DIR = Path(__file__).parent / "golden"
RTOL = 1e-12
DENOMINATOR_RATIO = 1e-3
"""A denominator must stay above this fraction of its largest magnitude over the example."""

POLE_MODELS = frozenset(
    {
        "engineering_strength_hyperbolic_age",
        "engineering_strength_aci209",
        "engineering_properties_shrinkage_hyperbolic",
        "engineering_properties_creep",
        "engineering_cementitious_ggbs_replacement",
        "engineering_cementitious_pozzolanic_reactivity",
        "engineering_durability_permeability",
    }
)
"""Models whose curve divides by a parameter-dependent function of x and can have a pole."""


# -- the catalogue ----------------------------------------------------------------------


def test_count_at_least_30() -> None:
    assert len(ALL_MODELS) >= ENGINEERING_FLOOR, (
        f"the engineering catalogue has {len(ALL_MODELS)} models; the plan requires "
        f"{ENGINEERING_FLOOR}"
    )


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_model_contract(spec: ModelSpec) -> None:
    assert spec.example is not None
    x, _ = spec.example_data()
    check_model_contract(spec, x, dict(spec.example.params), strict_metadata=True)


def test_ids_are_unique_and_snake_case() -> None:
    ids = [m.id for m in ALL_MODELS]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    assert [i for i in ids if not _SNAKE_CASE.match(i)] == []


def test_every_id_names_its_engineering_subcategory() -> None:
    assert {m.category for m in ALL_MODELS} <= {
        "engineering.strength",
        "engineering.properties",
        "engineering.cementitious",
        "engineering.durability",
        "engineering.packing",
    }


def test_every_model_is_a_one_variable_x_model() -> None:
    """Auto-fit fits y against x; a model with another variable is screened out."""
    assert [m.id for m in ALL_MODELS if m.variables != ("x",)] == []


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_every_parameter_documents_its_unit(spec: ModelSpec) -> None:
    for p in spec.parameters:
        assert p.unit.strip(), f"{spec.id}: parameter {p.name!r} has no unit"


def test_no_two_models_repeat_an_expression() -> None:
    owner: dict[str, str] = {}
    for spec in ALL_MODELS:
        assert spec.expression not in owner, (
            f"{spec.id} repeats the expression of {owner[spec.expression]}: {spec.expression!r}"
        )
        owner[spec.expression] = spec.id


def test_no_family_module_was_skipped() -> None:
    assert family_failures() == {}


def test_declared_equivalences_name_models_that_exist() -> None:
    register_starter_models()  # another test may have cleared the process-wide registry
    load_plugins()
    known = set(list_model_ids()) | {m.id for m in ALL_MODELS}
    unknown = sorted({i for pair in equivalent_pairs() for i in pair if i not in known})
    assert unknown == [], f"EQUIVALENT_PAIRS names unknown model ids: {unknown}"


# -- registration through the entry point -----------------------------------------------


@pytest.fixture
def empty_registry() -> Iterator[None]:
    """A clean registry for a test that decides for itself which plugins may load."""
    _clear_registry()
    yield
    _clear_registry()
    register_starter_models()


def test_the_engineering_entry_point_registers_the_catalogue(empty_registry: None) -> None:
    loaded = load_plugins()
    assert "engineering" in loaded, loaded
    assert plugin_failures() == ()
    ids = set(list_model_ids())
    assert {m.id for m in ALL_MODELS} <= ids


def test_models_are_not_in_core(empty_registry: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the entry point the core registers none of the engineering ids."""
    real = importlib.metadata.entry_points

    def hidden(**selection: object) -> object:
        if selection.get("group") == PLUGIN_GROUP:
            return []
        return real(**selection)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.metadata, "entry_points", hidden)
    import curvestudio  # noqa: F401  (the core package must not reach for the catalogue)

    assert load_plugins() == []
    assert set(list_model_ids()).isdisjoint(m.id for m in ALL_MODELS)


def test_importing_the_core_does_not_import_the_engineering_package() -> None:
    """The core never imports a catalogue package; registration is an entry-point callback."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import curvestudio\n"
        "assert 'curvestudio_engineering' not in sys.modules, sorted(sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


# -- golden references ------------------------------------------------------------------


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
BY_ID = {m.id: m for m in ALL_MODELS}


def test_every_catalogue_model_has_a_golden_reference() -> None:
    missing = sorted(m.id for m in ALL_MODELS if m.id not in GOLDEN)
    assert not missing, f"add GOLDEN entries in tests/unit/engineering/golden/ for: {missing}"


def test_no_golden_entry_is_stale() -> None:
    assert sorted(set(GOLDEN) - BY_ID.keys()) == []
    assert sorted(set(DENOMINATORS) - BY_ID.keys()) == []


def _grid(spec: ModelSpec, n: int) -> np.ndarray:
    assert spec.example is not None
    low, high = spec.example.x_range
    return np.linspace(low, high, n)


@pytest.mark.parametrize("spec", [m for m in ALL_MODELS if m.id in GOLDEN], ids=lambda s: s.id)
def test_model_matches_its_golden_reference(spec: ModelSpec) -> None:
    assert spec.example is not None
    params = dict(spec.example.params)
    x = _grid(spec, 5)
    np.testing.assert_allclose(
        spec.evaluate(x, **params), GOLDEN[spec.id](x, **params), rtol=RTOL, atol=0.0
    )


def test_every_pole_model_declares_its_denominator() -> None:
    assert DENOMINATORS.keys() >= POLE_MODELS


@pytest.mark.parametrize(
    "spec", [m for m in ALL_MODELS if m.id in DENOMINATORS], ids=lambda s: s.id
)
def test_example_range_stays_away_from_every_pole(spec: ModelSpec) -> None:
    assert spec.example is not None
    magnitude = np.abs(DENOMINATORS[spec.id](_grid(spec, 2001), **dict(spec.example.params)))
    assert np.isfinite(magnitude).all()
    assert float(magnitude.min()) > DENOMINATOR_RATIO * float(magnitude.max()), (
        f"denominator reaches {magnitude.min():.3g} against a maximum of {magnitude.max():.3g}"
    )


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_example_parameters_lie_strictly_inside_their_bounds(spec: ModelSpec) -> None:
    assert spec.example is not None
    for p in spec.parameters:
        value = spec.example.params[p.name]
        assert p.lower < value < p.upper, (
            f"{p.name}={value} not strictly inside {p.lower}..{p.upper}"
        )


# -- the initial guess ------------------------------------------------------------------

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
    import hashlib

    x, y = spec.example_data()
    assert _unexplained(spec, x, y) <= GUESS_MISS_EXACT
    seed = int.from_bytes(hashlib.sha256(f"guess:{spec.id}".encode()).digest()[:4], "big")
    noisy = y * (1.0 + 0.02 * np.random.default_rng(seed).standard_normal(y.size))
    assert _unexplained(spec, x, noisy) <= GUESS_MISS_NOISY
