"""The engineering library's honesty contract: the notice, and no claim of code acceptance.

Every engineering model must end its ``assumptions`` with :data:`ENGINEERING_NOTICE`, and
nothing in its name, description or model-specific assumptions may present it as an
acceptance equation of a code. The factory that builds the models enforces the category,
references and example, so a model added later cannot quietly drop them.
"""

import re
from typing import Any

import pytest

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_engineering import ENGINEERING_NOTICE, catalogue, engineering_model

pytestmark = pytest.mark.real_plugins

ALL_MODELS: tuple[ModelSpec, ...] = catalogue()

_CODE_CLAIM = re.compile(
    r"\b(?:"
    r"code[- ]compliant|code compliance|"
    r"complies with (?:IS|ACI|EN|ASTM|BS)|"
    r"in accordance with (?:IS|ACI|EN|ASTM|BS)|"
    r"as per (?:IS|ACI|EN|ASTM|BS)|"
    r"meets (?:the )?(?:requirements|limits|acceptance)|"
    r"acceptance (?:criterion|criteria|equation|limit)|"
    r"passes (?:IS|ACI|EN|ASTM|BS)"
    r")\b",
    re.IGNORECASE,
)
"""Phrases that would present a model as a code acceptance equation."""


def _model_specific_assumptions(spec: ModelSpec) -> str:
    assert spec.assumptions.endswith(ENGINEERING_NOTICE), spec.id
    return spec.assumptions[: -len(ENGINEERING_NOTICE)].strip()


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_every_engineering_model_ends_with_the_notice(spec: ModelSpec) -> None:
    assert spec.assumptions.endswith(ENGINEERING_NOTICE)
    assert spec.assumptions.count(ENGINEERING_NOTICE) == 1


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_no_engineering_model_claims_code_compliance(spec: ModelSpec) -> None:
    specific = _model_specific_assumptions(spec)
    for text in (spec.name, spec.description, specific):
        match = _CODE_CLAIM.search(text)
        assert match is None, f"{spec.id} claims code acceptance: {match.group(0)!r} in {text!r}"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_the_notice_is_not_used_as_a_substitute_for_real_assumptions(spec: ModelSpec) -> None:
    assert len(_model_specific_assumptions(spec)) >= 40


# -- the factory -----------------------------------------------------------------------

_REFERENCE = "Neville A.M. (2011) Properties of Concrete, 5th ed., Pearson Education"


def _valid_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "engineering_strength_probe",
        "name": "Probe",
        "category": "engineering.strength",
        "description": "A probe model used only by the factory tests.",
        "expression": "a*x",
        "parameters": (Parameter("a", "Slope of the probe line", "MPa per day", lower=0.0),),
        "initial_guess": lambda x, y: {"a": 1.0},
        "references": (_REFERENCE,),
        "assumptions": "A straight line through the origin over a non-negative x.",
        "example": ModelExample({"a": 2.0}, (0.0, 1.0)),
    }
    fields.update(overrides)
    return fields


def test_the_factory_appends_the_notice_exactly_once() -> None:
    spec = engineering_model(**_valid_fields())
    assert spec.assumptions.endswith(ENGINEERING_NOTICE)
    assert spec.assumptions.count(ENGINEERING_NOTICE) == 1
    assert spec.assumptions.startswith("A straight line through the origin")

    again = engineering_model(**_valid_fields(assumptions=spec.assumptions))
    assert again.assumptions == spec.assumptions


def test_the_factory_fills_the_notice_when_assumptions_are_empty() -> None:
    spec = engineering_model(**_valid_fields(assumptions=""))
    assert spec.assumptions == ENGINEERING_NOTICE


@pytest.mark.parametrize(
    "category",
    ["strength", "engineering", "engineering.unknown", "engineering.strength.extra", ""],
)
def test_the_factory_rejects_a_category_outside_engineering(category: str) -> None:
    with pytest.raises(ValueError, match="engineering"):
        engineering_model(**_valid_fields(category=category))


def test_the_factory_accepts_every_engineering_subcategory() -> None:
    for sub in ("strength", "properties", "cementitious", "durability", "packing"):
        spec = engineering_model(**_valid_fields(category=f"engineering.{sub}"))
        assert spec.category == f"engineering.{sub}"


@pytest.mark.parametrize("references", [(), ("",), ("   ",), "Neville (2011)"])
def test_the_factory_requires_non_empty_references(references: Any) -> None:
    with pytest.raises(ValueError, match="references"):
        engineering_model(**_valid_fields(references=references))


def test_the_factory_requires_the_references_field() -> None:
    fields = _valid_fields()
    del fields["references"]
    with pytest.raises(ValueError, match="references"):
        engineering_model(**fields)


@pytest.mark.parametrize("example", [None, "not an example"])
def test_the_factory_requires_an_example(example: Any) -> None:
    with pytest.raises(ValueError, match="example"):
        engineering_model(**_valid_fields(example=example))


def test_the_factory_requires_the_example_field() -> None:
    fields = _valid_fields()
    del fields["example"]
    with pytest.raises(ValueError, match="example"):
        engineering_model(**fields)


def test_the_factory_requires_the_category_field() -> None:
    fields = _valid_fields()
    del fields["category"]
    with pytest.raises(ValueError, match="category"):
        engineering_model(**fields)
