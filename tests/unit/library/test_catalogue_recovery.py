"""Every catalogue model recovers its own parameters, and its initial guess is a good start.

Three checks per model, all seeded:

* a multistart fit of the noiseless example returns the example parameters (1e-6);
* a *single* start from ``initial_guess`` on the noiseless example does so too (1e-4), so the
  guess alone is enough when the data are exact;
* on the example with 2 % multiplicative noise, a single start from ``initial_guess`` converges
  to the global least-squares minimum found by a 12-start fit, so the guess lands in the right
  basin for realistic data, not only for exact data.
"""

import hashlib

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import ModelSpec
from curvestudio.models.starter import STARTER_MODELS
from curvestudio.testing.model_recovery import check_model_recovery
from curvestudio_models import catalogue

pytestmark = [pytest.mark.slow, pytest.mark.real_plugins]

NOISE_FRACTION = 0.02
SSE_MATCH = 1e-6
"""Relative slack allowed between the single-start SSE and the multistart SSE."""

ALL_MODELS = (*catalogue(), *STARTER_MODELS)


def _seed(model_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"recovery:{model_id}".encode()).digest()[:4], "big")


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_recovers_its_own_parameters(spec: ModelSpec) -> None:
    report = check_model_recovery(spec, require_single_start=True)
    assert report.converged
    assert report.max_relative_error <= 1e-6
    assert report.single_start_recovered


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.id)
def test_single_start_from_the_guess_reaches_the_noisy_optimum(spec: ModelSpec) -> None:
    x, y = spec.example_data()
    rng = np.random.default_rng(_seed(spec.id))
    noisy = y * (1.0 + NOISE_FRACTION * rng.standard_normal(y.size))
    data = XY(x=x, y=noisy, weights=None, x_name="x", y_name="y", fingerprint=f"noisy:{spec.id}")
    best = fit(spec, data, FitConfig(n_starts=12, seed=0))
    single = fit(spec, data, FitConfig(n_starts=1, seed=0))
    assert best.status is FitStatus.CONVERGED, best.termination_reason
    assert single.status is FitStatus.CONVERGED, (
        f"single start from initial_guess: {single.termination_reason}"
    )
    sse_best = float(np.sum(np.square(best.residuals)))
    sse_single = float(np.sum(np.square(single.residuals)))
    assert sse_single <= sse_best * (1.0 + SSE_MATCH), (
        f"single start stuck at SSE {sse_single:.6g}; the global optimum is {sse_best:.6g}"
    )
