"""The laws behind the kinetics and physics families, stated as physics.

The catalogue machinery (golden references, recovery, equivalence) checks a model against
its own expression, so it cannot judge whether an expression is the law its name claims.
These tests state a few defining properties directly: the Einstein heat capacity
saturates at the Dulong-Petit value instead of growing as T**2, Moseley's law increases
with the atomic number, the VFT viscosity is exponential in the reciprocal distance to
the Vogel temperature, Cabrera-Mott growth is logarithmic while the Deal-Grove film
satisfies its quadratic, the Debye relaxation curve belongs to the frequency axis, and
the Bateman pair stays finite at the corner of its ordered rates.
"""

import numpy as np
import pytest

from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

BY_ID = {m.id: m for m in catalogue()}


def _curve(model_id: str, n: int = 2001) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spec = BY_ID[model_id]
    assert spec.example is not None
    low, high = spec.example.x_range
    x = np.linspace(low, high, n)
    return x, spec.evaluate(x, **dict(spec.example.params)), dict(spec.example.params)


# -- physics: heat capacities ---------------------------------------------------------------


def test_einstein_heat_capacity_saturates_at_the_dulong_petit_value() -> None:
    _, y, p = _curve("physics_einstein_heat_capacity")
    assert np.all(np.diff(y) > 0.0), "the heat capacity rises monotonically with temperature"
    assert y.max() < p["a"], "the heat capacity approaches the Dulong-Petit limit a from below"
    assert y[-1] > 0.9 * p["a"], "the example range reaches the saturation"
    assert y[0] < 0.2 * p["a"], "the example range starts well below the saturation"


def test_einstein_heat_capacity_is_the_true_law_not_richardson_dushman() -> None:
    # The true law saturates: Cv(600 K)/Cv(50 K) is ~11 at theta = 300 K; the T**2*exp(-theta/T)
    # misformula gives ~3.5e4 on the same temperatures.
    _, y, _ = _curve("physics_einstein_heat_capacity")
    assert y[-1] / y[0] < 100.0


def test_debye_heat_capacity_is_absent_and_a_two_term_sellmeier_takes_its_place() -> None:
    assert "physics_debye_heat_capacity" not in BY_ID
    _, y, _ = _curve("physics_sellmeier_2term")
    assert np.isfinite(y).all()
    assert np.all(y > 1.0), "a Sellmeier index stays above one away from the resonances"


# -- physics: spectroscopy and occupation ---------------------------------------------------


def test_moseley_law_increases_with_the_atomic_number() -> None:
    x, y, p = _curve("physics_moseley_law")
    assert np.all(np.diff(y) > 0.0), "the K-alpha frequency grows with the atomic number"
    assert p["c"] < float(x[0]), "the vertex (screening constant) lies below the measured range"


def test_vogel_fulcher_tammann_is_exponential_in_the_reciprocal_distance() -> None:
    x, y, p = _curve("physics_vogel_fulcher_tammann")
    assert np.all(np.diff(y) < 0.0), "the viscosity falls as the temperature rises"
    # log(eta) = a + b/(T - c): a straight line in 1/(T - c) to numerical precision.
    column = 1.0 / (x - p["c"])
    slope, intercept = np.polyfit(column, np.log(y), 1)
    residual = np.log(y) - (slope * column + intercept)
    variance = float(np.sum((np.log(y) - np.mean(np.log(y))) ** 2))
    assert float(np.sum(residual**2)) < 1e-20 * variance


def test_debye_dielectric_relaxes_along_the_frequency_axis() -> None:
    spec = BY_ID["physics_debye_dielectric"]
    assert spec.example is not None
    static = spec.evaluate(np.array([0.0]), **dict(spec.example.params))[0]
    p = dict(spec.example.params)
    assert static == pytest.approx(p["a"] + p["b"] / p["c"] ** 2, rel=1e-12)
    _, y, _ = _curve("physics_debye_dielectric")
    assert np.all(np.diff(y) < 0.0), "the permittivity falls as the frequency grows"


# -- kinetics: film growth ------------------------------------------------------------------


def test_deal_grove_film_satisfies_its_quadratic() -> None:
    x, y, p = _curve("kinetics_deal_grove_oxidation")
    residual = y**2 + p["A"] * y - p["B"] * (x + p["tau"])
    assert np.max(np.abs(residual)) < 1e-9 * float(np.max(np.abs(y) ** 2))


def test_cabrera_mott_film_grows_logarithmically_from_zero() -> None:
    spec = BY_ID["kinetics_cabrera_mott_oxidation"]
    assert spec.example is not None
    p = dict(spec.example.params)
    at_zero = spec.evaluate(np.array([0.0]), **p)[0]
    assert at_zero == pytest.approx(0.0, abs=1e-12)
    _, y, _ = _curve("kinetics_cabrera_mott_oxidation")
    assert np.all(np.diff(y) > 0.0), "the film thickens with time"
    assert np.all(np.diff(y, 2) < 0.0), "and its growth slows: the law is logarithmic"


# -- kinetics: rate laws --------------------------------------------------------------------


def test_haldane_peak_matches_its_documented_maximum() -> None:
    # The documented maximum vmax*sqrt(km*ki)/(2*km + sqrt(km*ki)) at x = sqrt(km*ki).
    x, y, p = _curve("kinetics_haldane_substrate_inhibition", n=20001)
    root = np.sqrt(p["km"] * p["ki"])
    peak = p["vmax"] * root / (2.0 * p["km"] + root)
    assert float(y.max()) == pytest.approx(peak, rel=1e-3)
    assert float(x[np.argmax(y)]) == pytest.approx(root, rel=1e-2)


def test_bateman_pair_is_finite_at_the_corner_of_its_ordered_rates() -> None:
    for model_id in ("kinetics_bateman_intermediate", "kinetics_bateman_product"):
        spec = BY_ID[model_id]
        corner = {q.name: q.lower for q in spec.parameters}
        x = np.linspace(0.0, 10.0, 101)
        y = spec.evaluate(x, **corner)
        assert np.isfinite(y).all(), (model_id, corner, y[:4])
