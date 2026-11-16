"""NIST StRD nonlinear least-squares benchmarks against the vendored certified values.

Acceptance rules (LRE = log relative error, :func:`curvestudio.testing.nist.lre`):

1. No silently wrong answers: whenever a fit reports ``CONVERGED``, the minimum LRE over the
   parameters is >= 5 and the SSE LRE is >= 7.
2. Lower difficulty: both NIST starts converge with a minimum parameter LRE >= 6.
3. Average/higher difficulty: each start either converges (rule 1 applies) or says it did
   not (``NOT_CONVERGED``, or ``FAILED`` with a reason). Each is also run with 20 seeded
   starts inside +-1000 x |start| bounds, where rule 1 is asserted, except the sign-mirror
   cases in ``MIRRORED_CASES`` (Eckerle4): those must reach the SSE LRE and match the
   certified parameters *or their sign mirror* to the rule-1 LREs, because the data
   cannot distinguish the two minima.

No NumPy/SciPy warning may escape the engine on these problems. Misra1a and DanWood are
the fast subset CI runs; the full sweep is marked ``slow``.
"""

import re
from typing import Any

import numpy as np
import pytest
from nist_harness import (
    CONVERGED_MIN_PARAM_LRE,
    CONVERGED_MIN_SSE_LRE,
    LOWER_MIN_PARAM_LRE,
    MIRRORED_CASES,
    STARTS,
    mirror_reference,
    multistart_config,
    param_lres,
    run_fit,
    sse_lre,
    start_values,
)
from nist_models import NIST_EXPRESSIONS, NONLINEAR_DIR, NONLINEAR_NAMES, model_spec, problem

from curvestudio.fitting import FitConfig, FitResult, FitStatus
from curvestudio.testing.model_contract import check_model_contract
from curvestudio.testing.nist import NistProblem

FAST = frozenset({"Misra1a", "DanWood"})


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
            out.append(pytest.param(name, start, id=f"{name}-{start}", marks=[pytest.mark.slow]))
    return out


def _summary(p: NistProblem, result: FitResult) -> str:
    lres = ", ".join(f"{k}={v:.2f}" for k, v in param_lres(p, result).items())
    return (
        f"{p.name}: status={result.status.value} param LRE [{lres}] "
        f"SSE LRE {sse_lre(p, result):.2f}; {result.termination_reason}; {result.warnings}"
    )


def _assert_rule_1(p: NistProblem, result: FitResult) -> None:
    """A CONVERGED result must be right: min param LRE >= 5 and SSE LRE >= 7."""
    if result.status is not FitStatus.CONVERGED:
        return
    assert result.stats is not None, _summary(p, result)
    assert min(param_lres(p, result).values()) >= CONVERGED_MIN_PARAM_LRE, _summary(p, result)
    assert sse_lre(p, result) >= CONVERGED_MIN_SSE_LRE, _summary(p, result)


def _mirror_summary(p: NistProblem, result: FitResult) -> str:
    """Failure text for the mirror check: LREs against both references, not just one."""
    lres = {
        label: param_lres(p, result, ref)
        for label, ref in (("certified", p.certified), ("mirror", mirror_reference(p)))
    }
    detail = "; ".join(
        f"{label} param LRE [{', '.join(f'{k}={v:.2f}' for k, v in against.items())}] "
        f"min {min(against.values()):.2f}"
        for label, against in lres.items()
    )
    return (
        f"{p.name}: status={result.status.value} SSE LRE {sse_lre(p, result):.2f}; {detail}; "
        f"need SSE LRE >= {CONVERGED_MIN_SSE_LRE} and a param LRE minimum >= "
        f"{CONVERGED_MIN_PARAM_LRE} against either reference; {result.termination_reason}"
    )


def _assert_certified_or_mirror(p: NistProblem, result: FitResult) -> None:
    """For a sign-symmetric problem: SSE LRE >= 7 and the parameters match the certified
    values or their mirror (``MIRROR_NEGATED`` negated) to the rule-1 LRE."""
    assert sse_lre(p, result) >= CONVERGED_MIN_SSE_LRE, _mirror_summary(p, result)
    best = max(
        min(param_lres(p, result, ref).values()) for ref in (p.certified, mirror_reference(p))
    )
    assert best >= CONVERGED_MIN_PARAM_LRE, _mirror_summary(p, result)


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
    result = run_fit(name, FitConfig(initial=start_values(p, start)))
    print(_summary(p, result))

    _assert_honest_status(result)
    _assert_rule_1(p, result)
    if p.difficulty == "lower":
        assert result.status is FitStatus.CONVERGED, _summary(p, result)
        assert min(param_lres(p, result).values()) >= LOWER_MIN_PARAM_LRE, _summary(p, result)


@pytest.mark.parametrize(("name", "start"), _multistart_cases())
def test_nist_multistart_within_wide_bounds(name: str, start: str) -> None:
    p = problem(name)
    result = run_fit(name, multistart_config(p, start))
    print(_summary(p, result))

    _assert_honest_status(result)
    if (name, start) in MIRRORED_CASES:
        _assert_certified_or_mirror(p, result)
    else:
        _assert_rule_1(p, result)
