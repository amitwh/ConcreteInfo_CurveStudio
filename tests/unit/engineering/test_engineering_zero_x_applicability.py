"""Models whose parameter derivative is undefined at x <= 0 must be skipped, not fail mid-fit."""

import numpy as np
import pytest

from curvestudio.models import get_model

pytestmark = pytest.mark.real_plugins

POWER_OF_X = (
    "engineering_cementitious_pozzolanic_reactivity",
    "engineering_durability_carbonation",
    "engineering_packing_modified_andreasen",
    "engineering_packing_modified_fuller",
    "engineering_properties_creep",
    "engineering_properties_shrinkage_hyperbolic",
    "engineering_properties_upv_strength_power",
)


@pytest.mark.parametrize("model_id", POWER_OF_X)
def test_zero_x_is_refused_with_a_reason_and_positive_x_is_accepted(model_id: str) -> None:
    spec = get_model(model_id)
    x = np.linspace(0.0, 6.0, 20)
    y = np.linspace(1.0, 5.0, 20)
    reasons = spec.applicability(x, y)
    assert reasons and "strictly positive" in reasons[0]
    assert spec.applicability(x[1:] + 0.5, y[1:]) == []
