"""Tests for the model registry."""

from collections.abc import Iterator

import numpy as np
import pytest

from curvestudio.models import ModelSpec, Parameter, get_model, list_models, register
from curvestudio.models.registry import _clear_registry


def _guess(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"a": 1.0, "b": 1.0}


def _spec(model_id: str, category: str = "growth") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=model_id.title(),
        category=category,
        expression="a*exp(b*x)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=_guess,
    )


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()


def test_register_and_get_round_trip() -> None:
    spec = _spec("exponential")
    register(spec)
    assert get_model("exponential") is spec


def test_duplicate_register_raises() -> None:
    register(_spec("exponential"))
    with pytest.raises(ValueError, match="exponential"):
        register(_spec("exponential"))


def test_get_unknown_suggests_close_match() -> None:
    register(_spec("exponential"))
    with pytest.raises(KeyError) as excinfo:
        get_model("expo")
    assert "exponential" in str(excinfo.value)


def test_get_unknown_without_close_match_still_raises_key_error() -> None:
    with pytest.raises(KeyError, match="zzz"):
        get_model("zzz")


def test_list_models_filters_by_category_and_keeps_registration_order() -> None:
    register(_spec("beta", "growth"))
    register(_spec("alpha", "decay"))
    register(_spec("gamma", "growth"))
    assert [m.id for m in list_models()] == ["beta", "alpha", "gamma"]
    assert [m.id for m in list_models("growth")] == ["beta", "gamma"]
    assert list_models("nothing") == []


def test_list_models_returns_a_copy() -> None:
    register(_spec("exponential"))
    listing = list_models()
    listing.clear()
    assert len(list_models()) == 1
