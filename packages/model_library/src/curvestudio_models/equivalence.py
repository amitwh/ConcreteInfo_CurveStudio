"""Pairs of model ids that describe the same curve on purpose.

Two models of the same curve family but with different parameters (a rate against a time
constant, ``x/(a+b*x)`` against ``vmax*x/(km+x)``) may both ship, because each is the form
people look for. They must be declared so the catalogue test that hunts for accidental
duplicates accepts them, and so Auto-fit can collapse identical curves into one ranking tie.
Models that are the same equation under another name use ``ModelSpec.aliases`` instead.

Where to declare a pair:

* In the family module that defines one of the two models, as the optional module attribute
  ``EQUIVALENT_PAIRS: tuple[tuple[str, str], ...]``. A pair may name a model of another family
  that does not exist yet: pairs with an unknown id are ignored (and re-checked as soon as both
  ids exist), so a family can declare an equivalence ahead of its partner.
* Here, in ``CORE_STARTER_PAIRS``, only for pairs among the core starter models, which belong to
  no family module.

``KNOWN_EQUIVALENCES`` is the union of both, as unordered pairs.
"""

from curvestudio_models.register import families

CORE_STARTER_PAIRS: tuple[tuple[str, str], ...] = (
    # a*exp(-b*x) + c is a + b*exp(c*x) with the sign of the exponent folded into c.
    ("exp_decay_offset", "exp_offset"),
    # a + b*log10(x) is a + (b/ln 10)*ln(x): the same line in log-x, a different unit of b.
    ("log_natural", "log_base10"),
)

# Same curve under different parametrisations in different family modules (found by the global
# duplicate hunt after the families were merged); declared here so Auto-fit ties them.
CROSS_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (
    ("peak_lorentzian", "physics_lorentzian_resonance"),
    ("distribution_gamma", "modified_power"),
    ("distribution_gamma", "kinetics_power_exponential"),
    ("distribution_cauchy", "physics_lorentzian_resonance"),
    ("distribution_maxwell_boltzmann", "physics_maxwell_boltzmann_speed"),
    ("exponential_saturation", "kinetics_first_order_conversion"),
    ("kinetics_cabrera_mott_oxidation", "log_one_plus"),
    ("allometric_offset", "generalised_power"),
    ("biexponential_difference", "kinetics_bateman_intermediate"),
    ("dose_response_emax", "rectangular_hyperbola_offset"),
    ("dose_response_weibull1_3", "stretched_exponential"),
    ("exponential_power_growth", "modified_power"),
    ("haldane", "kinetics_haldane_substrate_inhibition"),
    ("kinetics_power_exponential", "modified_power"),
    ("dose_response_4pl", "hill_offset"),
    ("exponential_decay_double", "kinetics_two_exponential_decay"),
    ("hill_offset", "morgan_mercer_flodin"),
    ("stretched_exponential_offset", "weibull_growth"),
)


def _union() -> frozenset[frozenset[str]]:
    pairs = list(CORE_STARTER_PAIRS) + list(CROSS_FAMILY_PAIRS)
    for family in families():
        pairs.extend(family.pairs)
    return frozenset(frozenset(pair) for pair in pairs)


KNOWN_EQUIVALENCES: frozenset[frozenset[str]] = _union()
