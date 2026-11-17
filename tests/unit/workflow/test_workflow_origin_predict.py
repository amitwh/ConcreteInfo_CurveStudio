"""Prediction from an Auto-fit row: a polynomial through the origin, and custom definitions
that carry explicit parameters (which must stay hashable: the planner caches them)."""

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitResult, CustomRef, resolve_spec
from curvestudio.models import CustomModelDefinition, ParameterDefinition
from curvestudio.workflow import FitSpec, bundle_for_candidate, predict


def _xy() -> XY:
    x = np.arange(1.0, 21.0)
    return XY(
        x=x,
        y=3.0 * x + 0.1 * np.sin(x),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="fp-origin",
    )


def _shell(config: AutoFitConfig) -> AutoFitResult:
    return AutoFitResult(
        config=config,
        dataset_fingerprint="fp-origin",
        n_active=0,
        ranked=(),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def test_a_polynomial_through_the_origin_matches_a_polynomial_spec_of_its_degree() -> None:
    xy = _xy()
    config = AutoFitConfig(n_starts=1)
    for candidate, degree in (("poly1_origin", 1), ("poly2_origin", 2), ("poly2", 2)):
        bundle = bundle_for_candidate(xy, _shell(config), candidate, compact=True)
        spec = FitSpec.model_validate({"model": {"kind": "polynomial", "degree": degree}})
        prediction = predict(bundle, spec, [2.0, 4.0], data=xy)
        assert prediction.y[0] is not None
        assert abs(prediction.y[0] - 6.0) < 0.5

    wrong = FitSpec.model_validate({"model": {"kind": "polynomial", "degree": 3}})
    bundle = bundle_for_candidate(xy, _shell(config), "poly2_origin", compact=True)
    try:
        predict(bundle, wrong, [1.0], data=xy)
    except Exception as exc:
        assert getattr(exc, "code", "") == "MODEL_MISMATCH"
    else:
        raise AssertionError("a degree-3 specification must not match a degree-2 origin fit")


def test_a_custom_definition_with_explicit_parameters_is_hashable_and_builds() -> None:
    definition = CustomModelDefinition(
        id="my_line",
        name="My line",
        expression="a*x + b",
        parameters=(
            ParameterDefinition(name="a", initial=1.0),
            ParameterDefinition(name="b", initial=0.0, lower=-5.0),
        ),
    )
    assert hash(definition) == hash(definition.model_copy())
    spec = resolve_spec(CustomRef(definition=definition))
    assert spec is not None and [p.name for p in spec.parameters] == ["a", "b"]
