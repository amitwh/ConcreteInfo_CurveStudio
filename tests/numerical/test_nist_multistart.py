"""NIST StRD hard-set multistart evidence: the honest outcome at 20 starts, seed 0.

The five higher-difficulty problems named by the Phase 4 plan — Eckerle4, MGH09, Thurber,
Rat43, Bennett5 — are fitted from NIST Start 1 with the benchmark multistart configuration
(``multistart_config``: 20 seeded starts inside +-1000 x |start value| bounds, seed 0).
Thurber, Rat43 and Bennett5 are vendored unchanged under
``tests/data/nist_strd/nonlinear_hard`` for this evidence; they are deliberately kept out
of the 12-problem suite in ``test_nist_nonlinear``, which pins the vendored catalogue of
``tests/data/nist_strd/nonlinear``.

The assertion is the *honest* outcome, not success: a ``CONVERGED`` result must match the
certified values — or, for the Eckerle4 sign mirror (``MIRRORED_CASES``: the model is
invariant under (b1, b2) -> (-b1, -b2) with b3 unchanged and the data cannot tell the two
minima apart) — to a minimum parameter LRE >= 6; anything else must say it did not
converge. Never a wrong "converged". The measured table lives in VALIDATION.md (Table 4).
"""

import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest
from nist_harness import (
    MIRRORED_CASES,
    mirror_reference,
    multistart_config,
    param_lres,
    sse_lre,
    xy_from_problem,
)
from nist_models import NONLINEAR_NAMES, model_spec, problem
from numpy.typing import NDArray

from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit
from curvestudio.models import ModelSpec, Parameter
from curvestudio.testing.model_contract import check_model_contract
from curvestudio.testing.nist import NistProblem, load_nls

FloatArray = NDArray[np.float64]

HARD_DIR = Path(__file__).parents[1] / "data" / "nist_strd" / "nonlinear_hard"

HARD_ONLY_EXPRESSIONS: dict[str, str] = {
    "Thurber": "(b1 + b2*x + b3*x**2 + b4*x**3) / (1 + b5*x + b6*x**2 + b7*x**3)",
    "Rat43": "b1 / ((1+exp(b2-b3*x))**(1/b4))",
    "Bennett5": "b1*(b2+x)**(-1/b3)",
}
"""Problems vendored here only; Eckerle4 and MGH09 come from ``nist_models``."""

HARD_NAMES: tuple[str, ...] = ("Eckerle4", "MGH09", "Thurber", "Rat43", "Bennett5")

MIN_PARAM_LRE = 6.0
"""The plan's honesty bar for this table: 6 correct significant digits, no exceptions."""


@cache
def hard_problem(name: str) -> NistProblem:
    """The parsed vendored ``.dat`` file for a problem that lives in ``nonlinear_hard``."""
    return load_nls(HARD_DIR / f"{name}.dat")


def _start1_guess(name: str) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        return dict(hard_problem(name).start1)

    return guess


@cache
def hard_model_spec(name: str) -> ModelSpec:
    """The ``ModelSpec`` for a hard-set problem, built exactly like ``nist_models``."""
    n_params = len(hard_problem(name).certified)
    return ModelSpec(
        id=f"nist_{name.lower()}",
        name=f"NIST StRD {name}",
        category="nist",
        expression=HARD_ONLY_EXPRESSIONS[name],
        parameters=tuple(
            Parameter(f"b{i}", description=f"NIST {name} parameter b{i}", unit="")
            for i in range(1, n_params + 1)
        ),
        initial_guess=_start1_guess(name),
        references=("https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml",),
    )


def problem_for(name: str) -> NistProblem:
    """The NIST problem for any hard-set name, wherever it is vendored."""
    return problem(name) if name in NONLINEAR_NAMES else hard_problem(name)


def spec_for(name: str) -> ModelSpec:
    """The ``ModelSpec`` for any hard-set name, wherever it is vendored."""
    return model_spec(name) if name in NONLINEAR_NAMES else hard_model_spec(name)


def run_hard_fit(name: str, config: FitConfig) -> FitResult:
    """Fit hard-set problem ``name``; no NumPy/SciPy warning may escape the engine."""
    with catch_warnings():
        simplefilter("error")
        return fit(spec_for(name), xy_from_problem(problem_for(name)), config)


def _normalise(formula: str) -> str:
    """NIST ``y = f [ ... ] + e`` notation -> bare right-hand side without whitespace."""
    text = re.sub(r"\s+", "", formula).replace("[", "(").replace("]", ")")
    text = text.removeprefix("y=")
    assert text.endswith("+e"), formula
    return text.removesuffix("+e")


def _summary(p: NistProblem, result: FitResult) -> str:
    references: tuple[dict[str, float], ...] = (p.certified,)
    if (p.name, "start1") in MIRRORED_CASES:
        references += (mirror_reference(p),)
    detail = " / ".join(
        f"[{', '.join(f'{k}={v:.2f}' for k, v in param_lres(p, result, ref).items())}]"
        for ref in references
    )
    return (
        f"{p.name}: status={result.status.value} param LRE {detail} "
        f"SSE LRE {sse_lre(p, result):.2f}; {result.termination_reason}; {result.warnings}"
    )


@pytest.mark.parametrize("name", tuple(HARD_ONLY_EXPRESSIONS))
def test_hard_vendored_problems_parse_and_transcribe(name: str) -> None:
    p = hard_problem(name)
    spec = hard_model_spec(name)
    assert p.difficulty == "higher"
    assert _normalise(p.model_formula) == re.sub(r"\s+", "", HARD_ONLY_EXPRESSIONS[name])
    assert spec.expression == HARD_ONLY_EXPRESSIONS[name]
    assert [q.name for q in spec.parameters] == [f"b{i}" for i in range(1, len(p.certified) + 1)]


@pytest.mark.parametrize("name", tuple(HARD_ONLY_EXPRESSIONS))
def test_hard_model_jacobian_matches_finite_differences_at_certified_values(name: str) -> None:
    p = hard_problem(name)
    check_model_contract(hard_model_spec(name), np.array(p.x), dict(p.certified))


@pytest.mark.slow
@pytest.mark.parametrize("name", HARD_NAMES)
def test_hard_set_multistart_from_start1_is_honest(name: str) -> None:
    p = problem_for(name)
    result = run_hard_fit(name, multistart_config(p, "start1"))
    print(_summary(p, result))

    if result.status is not FitStatus.CONVERGED:
        assert result.status in (FitStatus.NOT_CONVERGED, FitStatus.FAILED)
        assert result.termination_reason.strip()
        return
    references: tuple[dict[str, float], ...] = (p.certified,)
    if (name, "start1") in MIRRORED_CASES:
        references += (mirror_reference(p),)
    best = max(min(param_lres(p, result, ref).values()) for ref in references)
    assert best >= MIN_PARAM_LRE, _summary(p, result)
