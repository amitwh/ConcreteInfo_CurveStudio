"""Real ``FitBundle`` fixtures for the plotting tests (no stubs: the fits actually run)."""

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    fit_dataset,
)

type BundleBuilder = Callable[..., FitBundle]


@pytest.fixture(autouse=True)
def starter_registry() -> Iterator[None]:
    """The registry may be empty by the time these tests run; register the starters."""
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


def make_xy(
    x: np.ndarray, y: np.ndarray, x_name: str = "x", y_name: str = "y", tag: str = "fig"
) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name=x_name,
        y_name=y_name,
        fingerprint=f"fp-{tag}",
    )


def quadratic_xy(x_name: str = "x", y_name: str = "y") -> XY:
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 40)
    y = 1.0 + 0.5 * x - 0.03 * x**2 + rng.normal(0.0, 0.1, 40)
    return make_xy(x, y, x_name, y_name, tag="quad")


@pytest.fixture
def poly_bundle() -> BundleBuilder:
    """Builder of a converged degree-2 polynomial bundle (optional column names)."""

    def build(x_name: str = "x", y_name: str = "y") -> FitBundle:
        return fit_dataset(quadratic_xy(x_name, y_name), FitSpec(model=PolynomialRef(degree=2)))

    return build


@pytest.fixture
def log_bundle() -> FitBundle:
    x = np.linspace(0.1, 10.0, 30)
    rng = np.random.default_rng(8)
    y = 3.0 * np.log(x) + 1.0 + rng.normal(0.0, 0.02, 30)
    return fit_dataset(make_xy(x, y, tag="logx"), FitSpec(model=RegisteredRef(id="log_natural")))


@pytest.fixture
def non_converged_bundle() -> FitBundle:
    """An unidentifiable custom fit (two parameters, one direction): never converges."""
    spec = FitSpec(
        model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]),
        initial={"a": 1.0, "b": 2.0},
    )
    return fit_dataset(quadratic_xy(), spec)


@pytest.fixture
def failed_bundle() -> FitBundle:
    """A fit stopped by its time limit before the optimiser could take a step."""
    x = np.linspace(0.0, 5.0, 40)
    rng = np.random.default_rng(3)
    y = 2.0 * np.exp(0.3 * x) + rng.normal(0.0, 0.05, 40)
    spec = FitSpec(model=RegisteredRef(id="exp_growth"), time_limit=1e-9)
    return fit_dataset(make_xy(x, y, tag="exp"), spec)


@pytest.fixture
def constant_bundle() -> FitBundle:
    x = np.linspace(0.0, 9.0, 10)
    return fit_dataset(
        make_xy(x, np.full(10, 4.0), tag="const"), FitSpec(model=PolynomialRef(degree=1))
    )
