"""Tests for the workflow model catalogue."""

from collections.abc import Iterator

import pytest

from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.workflow import ModelInfo, WorkflowError, get_model, list_models


@pytest.fixture(autouse=True)
def starter_registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


def test_list_models_is_non_empty_and_sorted_by_category_then_name() -> None:
    infos = list_models()
    assert len(infos) > 5
    keys = [(m.category, m.name) for m in infos]
    assert keys == sorted(keys)


def test_list_models_includes_the_polynomial_family_and_every_registered_model() -> None:
    infos = {m.id: m for m in list_models()}
    assert "polynomial" in infos
    assert infos["polynomial"].is_family is True
    assert {"exp_growth", "gaussian", "michaelis_menten"} <= infos.keys()


def test_model_info_describes_parameters() -> None:
    info = get_model("exp_growth")
    assert isinstance(info, ModelInfo)
    assert info.name == "Exponential growth"
    assert info.category == "exponential"
    assert info.equation == "y = a*exp(b*x)"
    assert [p.name for p in info.parameters] == ["a", "b"]
    assert info.parameters[0].description.startswith("Value of y at x = 0")
    assert info.parameters[0].lower is None
    assert info.parameters[0].upper is None
    assert info.variables == ("x",)
    assert info.references
    assert info.is_family is False
    assert "e" in info.latex


def test_bounded_parameters_report_their_bounds() -> None:
    register(
        ModelSpec(
            id="wf_bounded",
            name="Bounded",
            category="test",
            expression="a*x",
            parameters=(Parameter("a", lower=0.0, upper=5.0),),
            initial_guess=lambda x, y: {"a": 1.0},
        )
    )
    info = get_model("wf_bounded")
    assert (info.parameters[0].lower, info.parameters[0].upper) == (0.0, 5.0)


def test_catalogue_is_json_safe() -> None:
    for info in list_models():
        dumped = info.model_dump(mode="json")
        assert ModelInfo.model_validate(dumped) == info


def test_polynomial_descriptor() -> None:
    info = get_model("polynomial")
    assert info.category == "polynomial"
    assert info.is_family is True
    assert info.equation.startswith("y = b0")


def test_unknown_model_is_a_workflow_error_with_a_hint() -> None:
    with pytest.raises(WorkflowError) as caught:
        get_model("exp_growht")
    assert caught.value.code == "UNKNOWN_MODEL"
    assert "exp_growth" in str(caught.value)


def test_empty_registry_still_lists_the_polynomial_family() -> None:
    _clear_registry()
    assert [m.id for m in list_models()] == ["polynomial"]
