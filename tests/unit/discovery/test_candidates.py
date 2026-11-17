"""Tests for Auto-fit candidate enumeration."""

from collections.abc import Iterator

import numpy as np
import pytest

from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    PolynomialRef,
    RegisteredRef,
    enumerate_candidates,
)
from curvestudio.models import ModelSpec, Parameter, register, register_family
from curvestudio.models.registry import _clear_registry


def _spec(model_id: str, k: int, category: str = "growth") -> ModelSpec:
    names = [f"p{i}" for i in range(k)]
    expression = " + ".join(f"{n}*x**{i}" for i, n in enumerate(names))
    return ModelSpec(
        id=model_id,
        name=model_id,
        category=category,
        expression=expression,
        parameters=tuple(Parameter(n) for n in names),
        initial_guess=lambda x, y: dict.fromkeys(names, 1.0),
    )


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(_spec("zeta", 2, "growth"))
    register(_spec("alpha", 3, "saturation"))
    register(_spec("big7", 7, "peak"))
    register(_spec("eng_strength", 2, "engineering.strength"))
    yield
    _clear_registry()


def _ids(config: AutoFitConfig, n_active: int) -> list[str]:
    return [c.id for c in enumerate_candidates(config, n_active)]


def test_default_excludes_models_above_the_parameter_cap_and_exhaustive_includes_them() -> None:
    # 40 points: default cap min(6, 10) = 6, exhaustive cap min(10, 37) = 10
    default = _ids(AutoFitConfig(include_polynomials=False), 40)
    exhaustive = _ids(AutoFitConfig(include_polynomials=False, exhaustive=True), 40)
    assert "big7" not in default
    assert {"alpha", "zeta"} <= set(default)
    assert "big7" in exhaustive


def test_default_cap_scales_with_n_active() -> None:
    # 8 points: cap max(1, 2) = 2 -> only two-parameter models
    assert _ids(AutoFitConfig(include_polynomials=False), 8) == ["eng_strength", "zeta"]


def test_order_is_deterministic() -> None:
    config = AutoFitConfig()
    first = enumerate_candidates(config, 40)
    assert first == enumerate_candidates(config, 40)
    ids = [c.id for c in first]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_polynomial_degrees_follow_the_cap() -> None:
    ids = _ids(AutoFitConfig(max_params=3), 40)
    polys = {i for i in ids if i.startswith("poly")}
    # k = degree + 1 (free) or degree (through the origin), both <= 3
    assert polys == {"poly1", "poly2", "poly1_origin", "poly2_origin", "poly3_origin"}


def test_polynomials_can_be_excluded() -> None:
    assert not [i for i in _ids(AutoFitConfig(include_polynomials=False), 40) if "poly" in i]


def test_families_filter() -> None:
    assert _ids(AutoFitConfig(families=("growth",), include_polynomials=False), 40) == ["zeta"]
    # a family name matches its dotted sub-categories
    assert _ids(AutoFitConfig(families=("engineering",), include_polynomials=False), 40) == [
        "eng_strength"
    ]
    # "polynomial" selects the polynomial family only
    only_poly = _ids(AutoFitConfig(families=("polynomial",), max_params=2), 40)
    assert only_poly == ["poly1", "poly1_origin", "poly2_origin"]


@pytest.mark.parametrize("n_active", [4, 5, 6, 9, 13, 40])
@pytest.mark.parametrize("exhaustive", [False, True])
def test_k_never_exceeds_n_minus_3(n_active: int, exhaustive: bool) -> None:
    config = AutoFitConfig(exhaustive=exhaustive, max_params=10)
    for ref in enumerate_candidates(config, n_active):
        if isinstance(ref, PolynomialRef):
            k = ref.parameter_count
        else:
            k = {"zeta": 2, "alpha": 3, "big7": 7, "eng_strength": 2}[ref.id]
        assert k <= n_active - 3, (ref.id, k, n_active)


def test_explicit_max_params_overrides_the_default_cap() -> None:
    ids = _ids(AutoFitConfig(include_polynomials=False, max_params=7), 40)
    assert "big7" in ids


def test_generated_family_members_follow_include_generated_and_exhaustive() -> None:
    register_family(
        "gen_",
        lambda exhaustive: ["gen_a", "gen_b", "gen_tail"] if exhaustive else ["gen_a", "gen_b"],
        lambda model_id: _spec(model_id, 2, "rational"),
    )
    base = AutoFitConfig(include_polynomials=False)
    assert {"gen_a", "gen_b"} <= set(_ids(base, 40))
    assert "gen_tail" not in _ids(base, 40)
    assert "gen_tail" in _ids(base.model_copy(update={"exhaustive": True}), 40)
    assert not [
        i for i in _ids(base.model_copy(update={"include_generated": False}), 40) if "gen_" in i
    ]


def test_family_member_that_cannot_be_built_is_left_out_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def build(model_id: str) -> ModelSpec:
        if model_id == "gen_broken":
            raise ValueError("cannot build")
        return _spec(model_id, 2)

    register_family("gen_", lambda exhaustive: ["gen_ok", "gen_broken"], build)
    with caplog.at_level("WARNING"):
        ids = _ids(AutoFitConfig(include_polynomials=False), 40)
    assert "gen_ok" in ids and "gen_broken" not in ids
    assert "gen_broken" in caplog.text


def test_refs_are_discriminated_values() -> None:
    assert PolynomialRef(degree=3).id == "poly3"
    assert PolynomialRef(degree=3, through_origin=True).id == "poly3_origin"
    assert RegisteredRef(model_id="zeta").id == "zeta"
    assert PolynomialRef(degree=2).kind == "polynomial"
    assert RegisteredRef(model_id="zeta").kind == "registered"
    assert PolynomialRef(degree=2) == PolynomialRef(degree=2)
    assert len({PolynomialRef(degree=2), PolynomialRef(degree=2)}) == 1


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        AutoFitConfig(families=())
    with pytest.raises(ValueError):
        AutoFitConfig(cv_folds=1)
    with pytest.raises(ValueError):
        AutoFitConfig(time_limit=0.0)
    with pytest.raises(ValueError):
        AutoFitConfig(workers=0)
    with pytest.raises(ValueError):
        AutoFitConfig(unknown_field=1)  # type: ignore[call-arg]
    assert AutoFitConfig(cv_folds=None).cv_folds is None
    assert np.isfinite(AutoFitConfig().time_limit)


def test_errors_survive_pickling() -> None:
    import pickle

    from curvestudio.statistics import StatisticsError

    for error in (AutoFitError("TOO_FEW_POINTS", "need more"), StatisticsError("X", "msg")):
        again = pickle.loads(pickle.dumps(error))
        assert again.code == error.code
        assert str(again) == str(error)
