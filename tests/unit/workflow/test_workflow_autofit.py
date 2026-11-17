"""Tests for turning one Auto-fit row into a full ``FitBundle``."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitResult, autofit
from curvestudio.models import CustomModelDefinition
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.workflow import FitBundle, WorkflowError, bundle_for_candidate


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()


def make_data() -> XY:
    x = np.linspace(0.0, 6.0, 40)
    rng = np.random.default_rng(5)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, 40)
    return XY(x=x, y=y, weights=None, x_name="t", y_name="signal", fingerprint="bundle-test")


def run(xy: XY) -> AutoFitResult:
    mine = CustomModelDefinition(id="my_decay", name="My decay", expression="a*exp(-b*x) + c")
    cfg = AutoFitConfig(
        workers=1,
        n_starts=3,
        cv_folds=4,
        families=("exponential", "polynomial"),
        max_params=3,
        custom_models=(mine,),
    )
    return autofit(xy, cfg)


def params(bundle: FitBundle) -> dict[str, float]:
    return {p.name: p.value for p in bundle.result.parameters}


@pytest.mark.parametrize("candidate_id", ["exp_decay_offset", "poly2", "poly1_origin", "my_decay"])
def test_bundle_reproduces_the_row_it_came_from(candidate_id: str) -> None:
    xy = make_data()
    result = run(xy)
    bundle = bundle_for_candidate(xy, result, candidate_id)
    assert isinstance(bundle, FitBundle)
    row = next(o for o in result.outcomes if o.candidate_id == candidate_id)
    assert row.fit is not None
    assert bundle.result.model_id == row.fit.model_id
    stored = {p.name: p.value for p in row.fit.parameters}
    assert params(bundle) == pytest.approx(stored, rel=1e-9)
    assert bundle.result.stats is not None and row.metrics is not None
    assert bundle.result.stats.r2 == pytest.approx(row.metrics.r2)
    assert bundle.curve is not None and len(bundle.curve.x) == 200
    assert bundle.confidence_band is not None
    assert bundle.data.x_name == "t"


def test_bundle_needs_the_same_data() -> None:
    xy = make_data()
    result = run(xy)
    other = XY(xy.x, xy.y, None, "t", "signal", "another-dataset")
    with pytest.raises(WorkflowError) as info:
        bundle_for_candidate(other, result, "poly2")
    assert info.value.code == "DATA_MISMATCH"


def test_unknown_candidate_is_an_error_naming_it() -> None:
    xy = make_data()
    result = run(xy)
    with pytest.raises(WorkflowError) as info:
        bundle_for_candidate(xy, result, "no_such_candidate")
    assert info.value.code == "UNKNOWN_MODEL"
    assert "no_such_candidate" in str(info.value)


def test_bundle_of_a_slim_result_still_refits() -> None:
    xy = make_data()
    slim: Any = AutoFitResult.from_json(run(xy).to_json(include_fits=False))
    bundle = bundle_for_candidate(xy, slim, "exp_decay_offset")
    assert bundle.result.status.value == "converged"
