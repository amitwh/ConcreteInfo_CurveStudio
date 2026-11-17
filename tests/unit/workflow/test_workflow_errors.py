"""WorkflowError must survive pickling: jobs run in a process pool and re-raise in the parent."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.workflow import (
    CustomRef,
    FitSpec,
    PolynomialRef,
    WorkflowError,
    fit_dataset,
    import_dataset,
)


def _xy() -> XY:
    x = np.linspace(0.0, 5.0, 12)
    return XY(x, 2.0 * x + 1.0 + 0.01 * np.sin(7 * x), None, "x", "y", "fp-pool")


def test_pickle_round_trip_keeps_code_and_message() -> None:
    error = WorkflowError("INVALID_EQUATION", "unknown function 'Exp'")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, WorkflowError)
    assert restored.code == "INVALID_EQUATION"
    assert restored.message == "unknown function 'Exp'"
    assert str(restored) == str(error)


def test_errors_cross_a_spawned_process_pool_and_the_pool_stays_usable() -> None:
    bad_equation = FitSpec(model=CustomRef(equation="a*x^2", parameters=["a"]))
    good = FitSpec(model=PolynomialRef(degree=1))
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        failing_fit = pool.submit(fit_dataset, _xy(), bad_equation)
        failing_import = pool.submit(import_dataset, b"", "a.csv")
        for future, code in ((failing_fit, "INVALID_EQUATION"), (failing_import, "UNREADABLE")):
            with pytest.raises(WorkflowError) as caught:
                future.result(timeout=120)
            assert caught.value.code == code
        bundle = pool.submit(fit_dataset, _xy(), good).result(timeout=120)
    assert bundle.result.parameters[1].value == pytest.approx(2.0, abs=0.05)
