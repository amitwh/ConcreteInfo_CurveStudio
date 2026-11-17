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


def _union() -> frozenset[frozenset[str]]:
    pairs = list(CORE_STARTER_PAIRS)
    for family in families():
        pairs.extend(family.pairs)
    return frozenset(frozenset(pair) for pair in pairs)


KNOWN_EQUIVALENCES: frozenset[frozenset[str]] = _union()
