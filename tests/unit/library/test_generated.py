"""The generated model families: id table, contract, listings, registration, applicability."""

import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from curvestudio.models import ModelSpec, get_model, list_model_ids, register, register_family
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.testing.model_contract import check_model_contract
from curvestudio.testing.model_recovery import MAX_CONDITION
from curvestudio_models import family_failures, generated
from curvestudio_models import register as register_module
from curvestudio_models.generated import (
    DEFAULT_PARAMETER_CAP,
    EXHAUSTIVE_PARAMETER_CAP,
    GeneratedInfo,
    build,
    generated_info,
    ids,
    register_generated,
)
from curvestudio_models.register import register_all

ALL_IDS = tuple(generated._table())
DEFAULT_IDS = tuple(ids())
EXHAUSTIVE_IDS = tuple(ids(exhaustive=True))
LONG_TAIL_IDS = tuple(i for i in ALL_IDS if i not in set(DEFAULT_IDS))
SNAPSHOT = Path(__file__).parent / "golden" / "generated_ids.txt"

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_PREFIX_OF_FAMILY = {
    "rat": "rat_",
    "invpoly": "invpoly_",
    "recip": "recip_",
    "exp_sum": "exp_sum_",
    "gauss_sum": "gauss_sum_",
    "pow_sum": "pow_sum_",
    "fourier": "fourier_",
    "tc": "tc_",
    "tc3": "tc3_",
}


@pytest.fixture
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()
    register_starter_models()


# -- the id table -----------------------------------------------------------------------


def test_ids_are_stable_and_unique_and_parse_back() -> None:
    assert len(ALL_IDS) == len(set(ALL_IDS))
    assert [i for i in ALL_IDS if not _SNAKE_CASE.match(i)] == []
    for model_id in ALL_IDS:
        info = generated_info(model_id)
        assert isinstance(info, GeneratedInfo)
        assert model_id.startswith(_PREFIX_OF_FAMILY[info.family]), model_id
        assert info.description.strip()
    assert set(DEFAULT_IDS) <= set(EXHAUSTIVE_IDS) <= set(ALL_IDS)
    for model_id in ALL_IDS:  # the parsed (family, k) is what the built model really is
        spec = build(model_id)
        assert spec.id == model_id
        assert generated_info(model_id).k == len(spec.parameters)
        assert generated_info(model_id).description == spec.description


def test_id_table_matches_the_snapshot() -> None:
    """An id never changes meaning and never disappears; additions are made deliberately.

    Regenerate with ``uv run python -c "from curvestudio_models import generated;
    print('\\n'.join(generated._table()))" > tests/unit/library/golden/generated_ids.txt``.
    """
    assert list(ALL_IDS) == SNAPSHOT.read_text(encoding="utf-8").split()


def test_default_count_at_least_100_and_exhaustive_at_least_300(
    capsys: pytest.CaptureFixture[str],
) -> None:
    by_family = Counter(generated_info(i).family for i in DEFAULT_IDS)
    with capsys.disabled():
        print(
            f"\ngenerated models: {len(DEFAULT_IDS)} by default, {len(EXHAUSTIVE_IDS)} exhaustive, "
            f"{len(ALL_IDS)} addressable by id; default by family {dict(by_family)}"
        )
    assert len(DEFAULT_IDS) >= 100
    assert len(EXHAUSTIVE_IDS) >= 300


def test_generation_is_deterministic() -> None:
    def snapshot() -> list[tuple[str, str, tuple[str, ...], tuple[tuple[str, float], ...]]]:
        out = []
        for model_id in ALL_IDS:
            spec = build(model_id)
            assert spec.example is not None
            out.append(
                (
                    spec.id,
                    spec.expression,
                    tuple(p.name for p in spec.parameters),
                    tuple(spec.example.params.items()),
                )
            )
        return out

    first = snapshot()
    generated._table.cache_clear()
    assert list(generated._table()) == list(ALL_IDS)
    assert snapshot() == first


def test_parameter_cap() -> None:
    assert max(generated_info(i).k for i in DEFAULT_IDS) <= DEFAULT_PARAMETER_CAP == 5
    assert max(generated_info(i).k for i in EXHAUSTIVE_IDS) <= EXHAUSTIVE_PARAMETER_CAP == 10
    hidden = set(ALL_IDS) - set(EXHAUSTIVE_IDS)
    assert hidden == {"fourier_5", "fourier_6"}
    assert all(generated_info(i).k > EXHAUSTIVE_PARAMETER_CAP for i in hidden)


def test_extended_tc3_models_are_listed_only_exhaustively() -> None:
    assert "tc3_id_x_ln" in DEFAULT_IDS
    assert "tc3_id_x_cube" not in DEFAULT_IDS
    assert "tc3_id_x_cube" in EXHAUSTIVE_IDS
    extended = {"cube", "invsqr", "lnsq", "xlnx", "invsqrt"}
    small_but_hidden = [
        i for i in set(EXHAUSTIVE_IDS) - set(DEFAULT_IDS) if generated_info(i).k <= 5
    ]
    assert small_but_hidden
    for model_id in small_but_hidden:
        assert generated_info(model_id).family == "tc3"
        assert extended & set(model_id.split("_")[2:]), model_id


@pytest.mark.parametrize(
    ("model_id", "original"),
    [
        ("rat_1_1", "rational_1_1"),
        ("rat_0_1", ""),
        ("invpoly_1", "rational_reciprocal_linear"),
        ("recip_2", "rational_reciprocal_quadratic_offset"),
        ("tc_id_ln", "log_natural"),
        ("tc_exp_x", "exp_growth"),
        ("tc3_inv_x_sqr", "invpoly_2"),
    ],
)
def test_ids_that_repeat_an_existing_curve_are_not_generated(model_id: str, original: str) -> None:
    for call in (build, generated_info):
        with pytest.raises(KeyError) as caught:
            call(model_id)
        if original:
            assert original in str(caught.value)
        else:
            assert "not a generated model id" in str(caught.value)
    assert model_id not in ALL_IDS


def test_unknown_ids_raise_key_error() -> None:
    for model_id in ("", "rat_", "tc_id", "tc_foo_x", "fourier_0", "fourier_7", "exp_sum_4"):
        with pytest.raises(KeyError):
            build(model_id)


# -- contract, conditioning, guesses ----------------------------------------------------


@pytest.mark.parametrize("model_id", DEFAULT_IDS)
def test_every_default_model_passes_the_contract(model_id: str) -> None:
    spec = build(model_id)
    assert spec.example is not None
    x, _ = spec.example_data()
    check_model_contract(spec, x, dict(spec.example.params), strict_metadata=True)
    assert "generated" in spec.references[-1]  # the family note replaces a literature reference


@pytest.mark.parametrize("model_id", LONG_TAIL_IDS)
def test_every_long_tail_model_passes_the_contract(model_id: str) -> None:
    spec = build(model_id)
    assert spec.example is not None
    x, _ = spec.example_data()
    check_model_contract(spec, x, dict(spec.example.params), strict_metadata=True)


@pytest.mark.parametrize("model_id", ALL_IDS)
def test_example_is_well_conditioned_and_inside_the_bounds(model_id: str) -> None:
    spec = build(model_id)
    assert spec.example is not None
    for p in spec.parameters:
        assert p.lower < spec.example.params[p.name] < p.upper, p.name
    x, y = spec.example_data()
    assert np.isfinite(y).all()
    jac = spec.jacobian(x, **dict(spec.example.params))
    jac = jac / np.linalg.norm(jac, axis=0)
    assert np.linalg.cond(jac) <= MAX_CONDITION


_DEGENERATE = {
    "empty": (np.array([]), np.array([])),
    "one point": (np.array([2.0]), np.array([3.0])),
    "constant y": (np.linspace(1.0, 5.0, 8), np.full(8, 2.5)),
    "repeated x": (np.full(6, 2.0), np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])),
    "all zero": (np.linspace(0.0, 4.0, 7), np.zeros(7)),
    "non-finite": (np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, np.inf, 4.0])),
    "negative": (np.linspace(-3.0, -1.0, 9), -np.linspace(1.0, 4.0, 9)),
    "huge": (np.linspace(1.0, 2.0, 5) * 1e150, np.linspace(1.0, 2.0, 5) * 1e150),
}


@pytest.mark.parametrize("case", sorted(_DEGENERATE))
@pytest.mark.parametrize("model_id", ALL_IDS)
def test_initial_guess_is_finite_and_inside_bounds_for_degenerate_data(
    model_id: str, case: str
) -> None:
    spec = build(model_id)
    x, y = _DEGENERATE[case]
    guess = spec.initial_guess(x, y)
    assert set(guess) == {p.name for p in spec.parameters}
    for p in spec.parameters:
        assert np.isfinite(guess[p.name]), (p.name, guess[p.name])
        assert p.lower <= guess[p.name] <= p.upper, (p.name, guess[p.name])


@pytest.mark.parametrize("model_id", DEFAULT_IDS)
def test_initial_guess_curve_explains_the_example_data(model_id: str) -> None:
    spec = build(model_id)
    x, y = spec.example_data()
    sst = float(np.sum((y - np.mean(y)) ** 2))
    curve = spec.evaluate(x, **spec.initial_guess(x, y))
    assert float(np.sum((y - curve) ** 2)) / sst <= 1e-2


# -- applicability ----------------------------------------------------------------------


def _reasons(model_id: str, x: list[float], y: list[float]) -> list[str]:
    return build(model_id).applicability(np.array(x), np.array(y))


def test_domain_checks_name_the_problem() -> None:
    assert _reasons("tc_id_sqrt", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == []
    assert any("non-negative" in r for r in _reasons("tc_id_sqrt", [-1.0, 2.0], [1.0, 2.0]))
    assert any("strictly positive" in r for r in _reasons("tc_sqr_ln", [0.0, 2.0], [1.0, 2.0]))
    assert any("not be zero" in r for r in _reasons("tc_sqr_inv", [0.0, 2.0], [1.0, 2.0]))
    assert any("strictly positive" in r for r in _reasons("tc_exp_inv", [1.0, 2.0], [0.0, 2.0]))
    assert any("not be negative" in r for r in _reasons("tc_sqr_x", [1.0, 2.0], [-1.0, 2.0]))
    assert any("changes sign" in r for r in _reasons("tc_inv_ln", [1.0, 2.0], [-1.0, 2.0]))
    assert any("must be finite" in r for r in _reasons("rat_3_1", [1.0, np.nan], [1.0, 2.0]))
    assert any("non-negative" in r for r in _reasons("rat_3_1", [-1.0, 2.0], [1.0, 2.0]))
    assert any("not be zero" in r for r in _reasons("recip_3", [0.0, 1.0], [1.0, 2.0]))
    assert any("strictly positive" in r for r in _reasons("pow_sum_2", [0.0, 1.0], [1.0, 2.0]))
    assert _reasons("fourier_1", [-5.0, 0.0, 5.0], [-1.0, 0.0, 1.0]) == []


# -- registration -----------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
def test_register_generated_makes_the_families_addressable_and_listed() -> None:
    register_generated()
    assert set(DEFAULT_IDS) <= set(list_model_ids())
    assert set(EXHAUSTIVE_IDS) - set(DEFAULT_IDS) <= set(list_model_ids(exhaustive=True))
    assert (set(EXHAUSTIVE_IDS) - set(DEFAULT_IDS)) & set(list_model_ids()) == set()
    assert get_model("tc3_exp_inv_ln").id == "tc3_exp_inv_ln"
    assert get_model("fourier_6").category == "harmonic"  # addressable though never listed
    assert "fourier_6" not in list_model_ids(exhaustive=True)
    with pytest.raises(KeyError):
        get_model("rat_1_1")  # repeats rational_1_1
    assert {get_model(i).category for i in list_model_ids(category="rational")} >= {"rational"}


@pytest.mark.usefixtures("clean_registry")
def test_register_generated_is_idempotent() -> None:
    register_generated()
    before = list_model_ids(exhaustive=True)
    register_generated()
    assert list_model_ids(exhaustive=True) == before


@pytest.mark.usefixtures("clean_registry")
def test_register_generated_refuses_a_prefix_someone_else_owns() -> None:
    register_family("tc_", lambda _e: [], lambda i: build("tc_id_sqrt"))
    with pytest.raises(ValueError, match="already registered"):
        register_generated()


@pytest.mark.real_plugins
@pytest.mark.usefixtures("clean_registry")
def test_register_all_registers_the_generated_families_and_skips_them_on_a_clash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_all()
    assert "tc3_id_x_ln" in list_model_ids()
    assert family_failures() == {}
    _clear_registry()
    template = build("tc_id_sqrt")
    register(
        ModelSpec(
            id="tc_hand_written_clash",
            name="Clash",
            category="rational",
            expression=template.expression,
            parameters=template.parameters,
            initial_guess=template.initial_guess,
        )
    )
    caplog.set_level("WARNING")
    try:
        register_all()
        assert "generated" in family_failures()
        assert any("'generated'" in r.getMessage() for r in caplog.records)
    finally:
        register_module._loaded()[1].pop("generated", None)
