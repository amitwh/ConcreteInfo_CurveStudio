"""Applicability rules of the rational family."""

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

BY_ID: dict[str, ModelSpec] = {m.id: m for m in catalogue()}
X = np.linspace(0.0, 4.0, 5)


def test_reciprocal_linear_accepts_one_signed_data() -> None:
    model = BY_ID["rational_reciprocal_linear"]
    assert model.applicability(X, np.array([2.0, 1.5, 1.0, 0.8, 0.5])) == []
    assert model.applicability(X, -np.array([2.0, 1.5, 1.0, 0.8, 0.5])) == []


@pytest.mark.parametrize(
    ("y", "reason"),
    [
        (np.array([2.0, 1.0, 0.0, 1.0, 2.0]), "never reaches zero"),
        (np.array([2.0, 1.0, -1.0, -2.0, -3.0]), "changes sign"),
        (np.array([2.0, 1.0, 1e10, 1.0, 2.0]), "pole"),
    ],
    ids=["zero", "sign change", "near pole"],
)
def test_reciprocal_linear_rejects_data_with_a_pole(y: np.ndarray, reason: str) -> None:
    reasons = BY_ID["rational_reciprocal_linear"].applicability(X, y)
    assert any(reason in r for r in reasons), reasons


def test_x_over_linear_cannot_be_zero_over_zero() -> None:
    model = BY_ID["rational_x_over_linear"]
    assert all(p.lower > 0.0 for p in model.parameters)
    value = model.evaluate(
        np.array([0.0]), a=model.parameters[0].lower, b=model.parameters[1].lower
    )
    assert np.isfinite(value).all()


def test_reciprocal_models_reject_zero_x() -> None:
    for model_id in ("rational_reciprocal_offset", "rational_reciprocal_quadratic_offset"):
        reasons = BY_ID[model_id].applicability(np.array([0.0, 1.0, 2.0]), np.ones(3))
        assert any("must not be zero" in r for r in reasons), model_id
