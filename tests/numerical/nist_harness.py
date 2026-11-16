"""Shared NIST StRD benchmark harness: fit configurations, thresholds and LRE helpers.

This is the single source of truth for *how* a NIST benchmark fit is run — start values,
the multistart configuration, the warnings-as-errors wrapper and the acceptance
thresholds — so that ``test_nist_nonlinear``, ``test_nist_linear`` and
``scripts/nist_report.py`` all grade exactly the same runs. The helpers and threshold
values were moved here from those test modules without change; the assertions that use
them still live in the tests.

The module imports no pytest, so the report script can use it as a plain module (it
inserts this directory on ``sys.path``); under pytest the rootdir insertion already
provides it, which is how ``nist_models`` is imported too.
"""

import warnings

from nist_models import LINEAR_DIR, model_spec, problem

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, fit, fit_polynomial
from curvestudio.testing.nist import NistLinearProblem, NistProblem, load_lls, lre

STARTS: tuple[str, str] = ("start1", "start2")

# Acceptance thresholds (values unchanged from the test modules they came from).
CONVERGED_MIN_PARAM_LRE = 5.0
CONVERGED_MIN_SSE_LRE = 7.0
LOWER_MIN_PARAM_LRE = 6.0
LINEAR_ACCURATE_MIN_LRE = 9.0
LINEAR_ILL_CONDITIONED_MIN_LRE = 6.0

MULTISTART_N = 20
MULTISTART_SEED = 0
MULTISTART_BOUND_FACTOR = 1000.0

# Eckerle4, (b1/b2) * exp(-0.5*((x-b3)/b2)**2), is invariant under (b1, b2) -> (-b1, -b2)
# with b3 unchanged. With seed 0 the 20-start run from Start 1 lands on that mirror minimum
# (b1, b2 = -1.5544, -4.0888, b3 = 451.54): it fits the data exactly as well, so the data
# cannot tell the two apart and that case is checked against either sign.
MIRRORED_CASES = frozenset({("Eckerle4", "start1")})
MIRROR_NEGATED = ("b1", "b2")

# The linear problems split into the two acceptance tiers of test_nist_linear: the
# well-conditioned set must reproduce the certified coefficients to LRE >= 9, the
# deliberately ill-conditioned set to LRE >= 6 or with an ILL_CONDITIONED warning.
LINEAR_ACCURATE_NAMES: tuple[str, ...] = ("Norris", "Pontius", "Wampler1")
LINEAR_ILL_CONDITIONED_NAMES: tuple[str, ...] = ("Wampler4", "Filip")


def xy_from_problem(p: NistProblem | NistLinearProblem) -> XY:
    """The problem's data as :class:`XY` (unweighted, fingerprinted by problem name)."""
    return XY(
        x=p.x,
        y=p.y,
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint=f"nist-{p.name}",
    )


def start_values(p: NistProblem, start: str) -> dict[str, float]:
    """The NIST ``start`` ("start1"/"start2") values as a fresh dict."""
    return dict(p.start1 if start == "start1" else p.start2)


def run_fit(name: str, config: FitConfig) -> FitResult:
    """Fit nonlinear problem ``name``; no NumPy/SciPy warning may escape the engine."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(model_spec(name), xy_from_problem(problem(name)), config)


def multistart_config(p: NistProblem, start: str) -> FitConfig:
    """The multistart configuration from the benchmarks: 20 seeded starts inside
    +-1000 x |start value| bounds per parameter (+-1000 when the start value is 0)."""
    base = start_values(p, start)
    bounds = {
        k: (-MULTISTART_BOUND_FACTOR * abs(v), MULTISTART_BOUND_FACTOR * abs(v))
        if v != 0.0
        else (-MULTISTART_BOUND_FACTOR, MULTISTART_BOUND_FACTOR)
        for k, v in base.items()
    }
    return FitConfig(initial=base, bounds=bounds, n_starts=MULTISTART_N, seed=MULTISTART_SEED)


def param_lres(
    p: NistProblem, result: FitResult, reference: dict[str, float] | None = None
) -> dict[str, float]:
    """LRE of each parameter against ``reference`` (default: the certified values)."""
    values = {est.name: est.value for est in result.parameters}
    want = p.certified if reference is None else reference
    return {k: lre(values[k], c) if k in values else 0.0 for k, c in want.items()}


def se_lres(p: NistProblem, result: FitResult) -> dict[str, float] | None:
    """LRE of each standard error against the certified ones; ``None`` if any is missing."""
    out: dict[str, float] = {}
    for est in result.parameters:
        if est.std_err is None:
            return None
        certified = p.certified_se.get(est.name)
        if certified is None:
            return None
        out[est.name] = lre(est.std_err, certified)
    return out or None


def sse_lre(p: NistProblem, result: FitResult) -> float:
    """LRE of the fit's SSE against the certified SSE (0.0 when no statistics exist)."""
    return 0.0 if result.stats is None else lre(result.stats.sse, p.certified_sse)


def mirror_reference(p: NistProblem) -> dict[str, float]:
    """The certified values with ``MIRROR_NEGATED`` negated (the Eckerle4 sign mirror)."""
    return {k: -c if k in MIRROR_NEGATED else c for k, c in p.certified.items()}


def linear_problem(name: str) -> NistLinearProblem:
    """The parsed vendored ``.dat`` file for linear problem ``name``."""
    return load_lls(LINEAR_DIR / f"{name}.dat")


def run_linear_fit(p: NistLinearProblem) -> FitResult:
    """Fit the problem as a polynomial of its certified degree, warnings as errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit_polynomial(xy_from_problem(p), p.degree)


def coefficient_lres(p: NistLinearProblem, result: FitResult) -> list[float]:
    """LRE of each coefficient against the certified ``B0..Bk`` (engine order checked)."""
    names = [est.name for est in result.parameters]
    assert names == [f"b{i}" for i in range(p.degree + 1)], names
    return [lre(est.value, p.certified[f"B{i}"]) for i, est in enumerate(result.parameters)]
