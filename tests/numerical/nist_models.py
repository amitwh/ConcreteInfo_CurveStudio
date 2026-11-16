"""ModelSpecs for the vendored NIST StRD nonlinear problems (test-only, not registered).

Each expression is transcribed from the ``Model:`` line of the problem's ``.dat`` file, with
NIST's square brackets written as parentheses; ``test_nist_nonlinear`` checks the transcription
against the file. Parameters are ``b1..bk`` in file order. ``initial_guess`` returns the
problem's NIST "Start 1" values; the benchmarks pass explicit starts anyway.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelSpec, Parameter
from curvestudio.testing.nist import NistProblem, load_nls

FloatArray = NDArray[np.float64]

NONLINEAR_DIR = Path(__file__).parents[1] / "data" / "nist_strd" / "nonlinear"
LINEAR_DIR = Path(__file__).parents[1] / "data" / "nist_strd" / "linear"

NIST_EXPRESSIONS: dict[str, str] = {
    "Misra1a": "b1*(1-exp(-b2*x))",
    "Misra1b": "b1 * (1-(1+b2*x/2)**(-2))",
    "Misra1c": "b1 * (1-(1+2*b2*x)**(-.5))",
    "Misra1d": "b1*b2*x*((1+b2*x)**(-1))",
    "DanWood": "b1*x**b2",
    "Chwirut2": "exp(-b1*x)/(b2+b3*x)",
    "Lanczos3": "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Gauss1": ("b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 ) + b6*exp( -(x-b7)**2 / b8**2 )"),
    "BoxBOD": "b1*(1-exp(-b2*x))",
    "Rat42": "b1 / (1+exp(b2-b3*x))",
    "MGH09": "b1*(x**2+x*b2) / (x**2+x*b3+b4)",
    "Eckerle4": "(b1/b2) * exp(-0.5*((x-b3)/b2)**2)",
}
"""Problem name -> right-hand side of ``y = f(x; b1..bk)`` as in the NIST file."""

NONLINEAR_NAMES: tuple[str, ...] = tuple(NIST_EXPRESSIONS)
LINEAR_NAMES: tuple[str, ...] = ("Norris", "Pontius", "Wampler1", "Wampler4", "Filip")


@cache
def problem(name: str) -> NistProblem:
    """The parsed vendored ``.dat`` file for nonlinear problem ``name``."""
    return load_nls(NONLINEAR_DIR / f"{name}.dat")


def _start1_guess(name: str) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        return dict(problem(name).start1)

    return guess


@cache
def model_spec(name: str) -> ModelSpec:
    """The ModelSpec for nonlinear problem ``name`` (parameters ``b1..bk``, file order)."""
    n_params = len(problem(name).certified)
    return ModelSpec(
        id=f"nist_{name.lower()}",
        name=f"NIST StRD {name}",
        category="nist",
        expression=NIST_EXPRESSIONS[name],
        parameters=tuple(
            Parameter(f"b{i}", description=f"NIST {name} parameter b{i}", unit="")
            for i in range(1, n_params + 1)
        ),
        initial_guess=_start1_guess(name),
        references=("https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml",),
    )
