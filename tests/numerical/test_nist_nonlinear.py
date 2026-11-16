"""NIST StRD nonlinear least-squares benchmarks against the vendored certified values.

Acceptance rules (LRE = log relative error, :func:`curvestudio.testing.nist.lre`):

1. No silently wrong answers: whenever a fit reports ``CONVERGED``, the minimum LRE over the
   parameters is >= 5 and the SSE LRE is >= 7.
2. Lower difficulty: both NIST starts converge with a minimum parameter LRE >= 6.
3. Average/higher difficulty: each start either converges (rule 1 applies) or says it did
   not (``NOT_CONVERGED``, or ``FAILED`` with a reason). Each is also run with 20 seeded
   starts inside +-1000 x |start| bounds, where only rule 1 is asserted.

No NumPy/SciPy warning may escape the engine on these problems. Misra1a and DanWood are
the fast subset CI runs; the full sweep is marked ``slow``.
"""

import re
import warnings
from typing import Any

import numpy as np
import pytest
from nist_models import NIST_EXPRESSIONS, NONLINEAR_DIR, NONLINEAR_NAMES, model_spec, problem

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit
from curvestudio.testing.model_contract import check_model_contract
from curvestudio.testing.nist import NistProblem, lre

FAST = frozenset({"Misra1a", "DanWood"})
STARTS = ("start1", "start2")

CONVERGED_MIN_PARAM_LRE = 5.0
CONVERGED_MIN_SSE_LRE = 7.0
LOWER_MIN_PARAM_LRE = 6.0

MULTISTART_N = 20
MULTISTART_SEED = 0
MULTISTART_BOUND_FACTOR = 1000.0

ECKERLE4_MIRROR = (
    "Eckerle4 is invariant under (b1, b2) -> (-b1, -b2); with seed 0 the 20-start run from "
    "Start 1 finishes on the mirror minimum (b1, b2 = -1.5544, -4.0888, SSE LRE 10.7), which "
    "fits the data exactly as well but is not the certified sign, so parameter LRE is 0"
)
"""Why the one xfail exists: the data cannot tell the two solutions apart."""


def _slow_unless_fast(name: str) -> list[pytest.MarkDecorator]:
    return [] if name in FAST else [pytest.mark.slow]


def _cases(names: tuple[str, ...]) -> list[Any]:
    return [
        pytest.param(name, start, id=f"{name}-{start}", marks=_slow_unless_fast(name))
        for name in names
        for start in STARTS
    ]


def _multistart_cases() -> list[Any]:
    out: list[Any] = []
    for name in NONLINEAR_NAMES:
        if problem(name).difficulty == "lower":
            continue
        for start in STARTS:
            marks = [pytest.mark.slow]
            if (name, start) == ("Eckerle4", "start1"):
                marks.append(pytest.mark.xfail(strict=True, reason=ECKERLE4_MIRROR))
            out.append(pytest.param(name, start, id=f"{name}-{start}", marks=marks))
    return out


def _xy(p: NistProblem) -> XY:
    return XY(
        x=np.array(p.x),
        y=np.array(p.y),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint=f"nist-{p.name}",
    )


def _start(p: NistProblem, start: str) -> dict[str, float]:
    return p.start1 if start == "start1" else p.start2


def _fit(name: str, config: FitConfig) -> FitResult:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(model_spec(name), _xy(problem(name)), config)


def _param_lres(p: NistProblem, result: FitResult) -> dict[str, float]:
    values = {est.name: est.value for est in result.parameters}
    return {k: lre(values[k], c) if k in values else 0.0 for k, c in p.certified.items()}


def _sse_lre(p: NistProblem, result: FitResult) -> float:
    return 0.0 if result.stats is None else lre(result.stats.sse, p.certified_sse)


def _summary(p: NistProblem, result: FitResult) -> str:
    lres = ", ".join(f"{k}={v:.2f}" for k, v in _param_lres(p, result).items())
    return (
        f"{p.name}: status={result.status.value} param LRE [{lres}] "
        f"SSE LRE {_sse_lre(p, result):.2f}; {result.termination_reason}; {result.warnings}"
    )


def _assert_rule_1(p: NistProblem, result: FitResult) -> None:
    """A CONVERGED result must be right: min param LRE >= 5 and SSE LRE >= 7."""
    if result.status is not FitStatus.CONVERGED:
        return
    assert result.stats is not None, _summary(p, result)
    assert min(_param_lres(p, result).values()) >= CONVERGED_MIN_PARAM_LRE, _summary(p, result)
    assert _sse_lre(p, result) >= CONVERGED_MIN_SSE_LRE, _summary(p, result)


def _assert_honest_status(result: FitResult) -> None:
    """Anything not CONVERGED must say so: NOT_CONVERGED, or FAILED with a reason."""
    if result.status is FitStatus.FAILED:
        assert result.termination_reason.strip()
    else:
        assert result.status in (FitStatus.CONVERGED, FitStatus.NOT_CONVERGED)


def _normalise(formula: str) -> str:
    """NIST ``y = f [ ... ] + e`` notation -> bare right-hand side without whitespace."""
    text = re.sub(r"\s+", "", formula).replace("[", "(").replace("]", ")")
    text = text.removeprefix("y=")
    assert text.endswith("+e"), formula
    return text.removesuffix("+e")


def test_all_vendored_nonlinear_problems_are_mapped() -> None:
    vendored = sorted(path.stem for path in NONLINEAR_DIR.glob("*.dat"))
    assert sorted(NONLINEAR_NAMES) == vendored
    assert len(vendored) == 12


@pytest.mark.parametrize("name", NONLINEAR_NAMES)
def test_model_expression_matches_the_nist_file(name: str) -> None:
    spec = model_spec(name)
    p = problem(name)
    assert _normalise(p.model_formula) == re.sub(r"\s+", "", NIST_EXPRESSIONS[name])
    assert spec.expression == NIST_EXPRESSIONS[name]
    assert [q.name for q in spec.parameters] == [f"b{i}" for i in range(1, len(p.certified) + 1)]


@pytest.mark.parametrize("name", NONLINEAR_NAMES)
def test_model_jacobian_matches_finite_differences_at_certified_values(name: str) -> None:
    p = problem(name)
    check_model_contract(model_spec(name), np.array(p.x), dict(p.certified))


@pytest.mark.parametrize(("name", "start"), _cases(NONLINEAR_NAMES))
def test_nist_start(name: str, start: str) -> None:
    p = problem(name)
    result = _fit(name, FitConfig(initial=_start(p, start)))
    print(_summary(p, result))

    _assert_honest_status(result)
    _assert_rule_1(p, result)
    if p.difficulty == "lower":
        assert result.status is FitStatus.CONVERGED, _summary(p, result)
        assert min(_param_lres(p, result).values()) >= LOWER_MIN_PARAM_LRE, _summary(p, result)


@pytest.mark.parametrize(("name", "start"), _multistart_cases())
def test_nist_multistart_within_wide_bounds(name: str, start: str) -> None:
    p = problem(name)
    base = _start(p, start)
    bounds = {
        k: (-MULTISTART_BOUND_FACTOR * abs(v), MULTISTART_BOUND_FACTOR * abs(v))
        if v != 0.0
        else (-MULTISTART_BOUND_FACTOR, MULTISTART_BOUND_FACTOR)
        for k, v in base.items()
    }
    config = FitConfig(initial=base, bounds=bounds, n_starts=MULTISTART_N, seed=MULTISTART_SEED)
    result = _fit(name, config)
    print(_summary(p, result))

    _assert_honest_status(result)
    _assert_rule_1(p, result)
