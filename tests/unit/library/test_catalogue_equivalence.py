"""No two models are secretly the same curve unless that is declared.

The check is global: every model against every other model with the same number of parameters
(whatever the category), in both directions, because a reparametrisation of one model may fit
the curve of another only one way round. Each direction is cheap first: skipped when the target
model's ``applicability`` rejects the source data, then one fit from the target's own
``initial_guess``; only a fit that explains all but 1e-4 of the variance is repeated as a
multistart and judged against the exactness threshold.

A pair is *declared* when it is in ``KNOWN_EQUIVALENCES`` (an ``EQUIVALENT_PAIRS`` entry of a
family module, or a core-starter pair) or one model lists the other as an alias. Declared pairs
naming a model that does not exist yet are ignored until it does.
"""

from collections import defaultdict
from itertools import permutations

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import ModelSpec
from curvestudio.models.starter import STARTER_MODELS
from curvestudio_models import catalogue
from curvestudio_models.equivalence import KNOWN_EQUIVALENCES

pytestmark = [pytest.mark.slow, pytest.mark.real_plugins]

ALL_MODELS: tuple[ModelSpec, ...] = (*catalogue(), *STARTER_MODELS)
BY_ID = {m.id: m for m in ALL_MODELS}
GROUPS: dict[int, list[ModelSpec]] = defaultdict(list)
for _model in ALL_MODELS:
    GROUPS[len(_model.parameters)].append(_model)

EXACT = 1e-12
"""A fit leaving at most this fraction of the total variance unexplained is exact."""
PROMISING = 1e-4
"""A first single-start fit below this fraction earns the multistart confirmation."""


def reproduces(source: ModelSpec, target: ModelSpec) -> bool:
    """True when ``target`` fits the noiseless example curve of ``source`` exactly."""
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


def _declared(a: ModelSpec, b: ModelSpec) -> bool:
    return (
        frozenset({a.id, b.id}) in KNOWN_EQUIVALENCES
        or b.id in a.aliases
        or a.id in b.aliases
        or bool(set(a.aliases) & set(b.aliases))
    )


@pytest.mark.parametrize("k", sorted(GROUPS), ids=lambda k: f"{k}_parameters")
def test_no_two_models_are_the_same_curve(k: int) -> None:
    undeclared = sorted(
        {
            " == ".join(sorted((a.id, b.id)))
            for a, b in permutations(GROUPS[k], 2)
            if not _declared(a, b) and reproduces(a, b)
        }
    )
    assert not undeclared, (
        "these pairs are the same curve; merge them with aliases or declare them in the "
        f"EQUIVALENT_PAIRS of a family module: {undeclared}"
    )


def test_known_equivalences_that_name_existing_models_are_real() -> None:
    present = [tuple(sorted(pair)) for pair in KNOWN_EQUIVALENCES if pair <= BY_ID.keys()]
    assert ("michaelis_menten", "rational_x_over_linear") in present
    for first, second in present:
        assert reproduces(BY_ID[first], BY_ID[second]) or reproduces(BY_ID[second], BY_ID[first]), (
            f"{first} and {second} are declared equivalent but no fit reproduces one by the other"
        )
