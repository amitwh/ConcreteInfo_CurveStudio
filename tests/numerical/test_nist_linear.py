"""NIST StRD linear least-squares benchmarks against the vendored certified values.

Norris, Pontius and Wampler1 must reproduce the certified coefficients to LRE >= 9.
Wampler4 and Filip are deliberately ill-conditioned: they must reach LRE >= 6 or carry an
``ILL_CONDITIONED`` warning, never a silent inaccurate answer. NIST names the coefficients
``B0..Bk``; the engine names them ``b0..bk`` in the same order. Norris and Wampler1 are the
fast subset CI runs; the rest is marked ``slow``.
"""

import warnings

import numpy as np
import pytest
from nist_models import LINEAR_DIR

from curvestudio.data import XY
from curvestudio.fitting import FitResult, FitStatus, fit_polynomial
from curvestudio.testing.nist import NistLinearProblem, load_lls, lre

ACCURATE_MIN_LRE = 9.0
ILL_CONDITIONED_MIN_LRE = 6.0
ACCURATE = (
    pytest.param("Norris", id="Norris"),
    pytest.param("Pontius", id="Pontius", marks=pytest.mark.slow),
    pytest.param("Wampler1", id="Wampler1"),
)
ILL_CONDITIONED = (
    pytest.param("Wampler4", id="Wampler4", marks=pytest.mark.slow),
    pytest.param("Filip", id="Filip", marks=pytest.mark.slow),
)


def _problem(name: str) -> NistLinearProblem:
    return load_lls(LINEAR_DIR / f"{name}.dat")


def _fit(p: NistLinearProblem) -> FitResult:
    data = XY(
        x=np.array(p.x),
        y=np.array(p.y),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint=f"nist-{p.name}",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit_polynomial(data, p.degree)


def _coefficient_lres(p: NistLinearProblem, result: FitResult) -> list[float]:
    names = [est.name for est in result.parameters]
    assert names == [f"b{i}" for i in range(p.degree + 1)], names
    return [lre(est.value, p.certified[f"B{i}"]) for i, est in enumerate(result.parameters)]


@pytest.mark.parametrize("name", ACCURATE)
def test_well_conditioned_problem_matches_certified_coefficients(name: str) -> None:
    p = _problem(name)
    result = _fit(p)

    assert result.status is FitStatus.CONVERGED, result.termination_reason
    lres = _coefficient_lres(p, result)
    assert min(lres) >= ACCURATE_MIN_LRE, (name, lres, result.warnings)


def test_norris_matches_certified_statistics() -> None:
    p = _problem("Norris")
    result = _fit(p)

    assert result.status is FitStatus.CONVERGED
    for i, est in enumerate(result.parameters):
        assert est.std_err is not None
        assert lre(est.std_err, p.certified_se[f"B{i}"]) >= ACCURATE_MIN_LRE
    assert result.stats is not None and result.stats.residual_se is not None
    assert lre(result.stats.residual_se, p.residual_sd) >= ACCURATE_MIN_LRE
    assert result.stats.r2 is not None
    assert lre(result.stats.r2, p.r2) >= ACCURATE_MIN_LRE
    assert result.warnings == ()


def test_wampler1_exact_fit_has_finite_covariance() -> None:
    result = _fit(_problem("Wampler1"))

    assert result.stats is not None
    assert result.stats.r2 == 1.0
    assert result.covariance is not None
    assert np.all(np.isfinite(np.array(result.covariance)))


@pytest.mark.parametrize("name", ILL_CONDITIONED)
def test_ill_conditioned_problem_is_accurate_or_warned(name: str) -> None:
    p = _problem(name)
    result = _fit(p)

    lres = _coefficient_lres(p, result) if result.parameters else [0.0]
    accurate = result.status is FitStatus.CONVERGED and min(lres) >= ILL_CONDITIONED_MIN_LRE
    assert accurate or "ILL_CONDITIONED" in result.warnings, (name, lres, result.warnings)
