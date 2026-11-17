"""Slow checks of the generated families: parameter recovery, label order, no repeated curves.

* ``test_recovery_sample``: every fifth default id plus every listed ``fourier_*``, ``rat_*``
  and ``exp_sum_*`` model recovers its own example parameters (multistart, noiseless).
* ``test_ordered_parametrisation_removes_label_switching``: with ordered rates the fit of
  ``exp_sum_2`` never returns the two terms swapped.
* ``test_defaults_are_not_the_same_curve_as_each_other_or_the_catalogue``: the default
  generated models do not repeat a catalogue, starter or other generated curve.
"""

from collections import defaultdict
from itertools import permutations

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import ModelSpec
from curvestudio.models.starter import STARTER_MODELS
from curvestudio.testing.model_recovery import check_model_recovery
from curvestudio_models import catalogue, generated
from curvestudio_models.equivalence import KNOWN_EQUIVALENCES

pytestmark = [pytest.mark.slow, pytest.mark.real_plugins]

DEFAULT_IDS = generated.ids()
SAMPLE = sorted(
    {
        *DEFAULT_IDS[::5],
        *(
            i
            for i in generated.ids(exhaustive=True)
            if i.startswith(("fourier_", "rat_", "exp_sum_"))
        ),
    }
)


@pytest.mark.parametrize("model_id", SAMPLE)
def test_recovery_sample(model_id: str) -> None:
    report = check_model_recovery(generated.build(model_id))
    assert report.converged
    assert report.max_relative_error <= 1e-6


@pytest.mark.parametrize("model_id", ["exp_sum_2", "exp_sum_2_offset", "pow_sum_2"])
def test_ordered_parametrisation_removes_label_switching(model_id: str) -> None:
    spec = generated.build(model_id)
    assert spec.example is not None
    truth = dict(spec.example.params)
    x, y = spec.example_data()
    for seed in range(5):
        noisy = y * (1.0 + 0.002 * np.random.default_rng(seed).standard_normal(y.size))
        data = XY(
            x=x, y=noisy, weights=None, x_name="x", y_name="y", fingerprint=f"{model_id}{seed}"
        )
        result = fit(spec, data, FitConfig(n_starts=12, seed=seed))
        assert result.status is FitStatus.CONVERGED, result.termination_reason
        fitted = {e.name: e.value for e in result.parameters}
        gap = "d2"
        assert fitted[gap] >= 0.0  # second term at or beyond the first: r2 >= r1, p2 >= p1
        first = "p1" if model_id == "pow_sum_2" else "r1"
        assert fitted[first] == pytest.approx(truth[first], rel=0.2)
        assert fitted[gap] == pytest.approx(truth[gap], rel=0.2)


EXACT = 1e-12
PROMISING = 1e-4


def _reproduces(source: ModelSpec, target: ModelSpec) -> bool:
    x, y = source.example_data()
    if target.applicability(x, y):
        return False
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst == 0.0:
        return False
    data = XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint=f"eq:{source.id}")
    for n_starts, limit in ((1, PROMISING), (6, EXACT)):
        result = fit(target, data, FitConfig(n_starts=n_starts, seed=0))
        if result.status is not FitStatus.CONVERGED:
            return False
        if float(np.sum(np.square(result.residuals))) > limit * sst:
            return False
    return True


_EXISTING = (*catalogue(), *STARTER_MODELS)
_NEW = [generated.build(i) for i in DEFAULT_IDS]
_GROUPS: dict[int, list[ModelSpec]] = defaultdict(list)
for _spec in (*_EXISTING, *_NEW):
    _GROUPS[len(_spec.parameters)].append(_spec)
_NEW_IDS = {m.id for m in _NEW}


@pytest.mark.parametrize("k", sorted(_GROUPS), ids=lambda k: f"{k}_parameters")
def test_defaults_are_not_the_same_curve_as_each_other_or_the_catalogue(k: int) -> None:
    def declared(a: ModelSpec, b: ModelSpec) -> bool:
        return (
            frozenset({a.id, b.id}) in KNOWN_EQUIVALENCES or b.id in a.aliases or a.id in b.aliases
        )

    repeated = sorted(
        {
            " == ".join(sorted((a.id, b.id)))
            for a, b in permutations(_GROUPS[k], 2)
            if (a.id in _NEW_IDS or b.id in _NEW_IDS) and not declared(a, b) and _reproduces(a, b)
        }
    )
    assert not repeated, f"generated models that repeat another curve: {repeated}"
