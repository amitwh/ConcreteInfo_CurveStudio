"""Real bundles for the fit-audit tests: every fit actually runs.

Both test modules audit the same fits, so the scenarios live here: one module asserts the
verdicts, the other scans every word the audit produces.
"""

import functools
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.statistics import CrossValidation, PolynomialFitter, k_fold_cv
from curvestudio.statistics.errors import StatisticsError
from curvestudio.tools import FitAudit, audit_fit
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    fit_dataset,
)

X = np.linspace(0.0, 10.0, 40)
CURVE = 1.0 + 0.5 * X - 0.03 * X**2
BOUNDED_LINE_ID = "audit_bounded_line"


def make_xy(x: Sequence[float], y: Sequence[float], weights: Sequence[float] | None = None) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="fp-audit",
    )


def fit(xy: XY, spec: FitSpec) -> FitBundle:
    return fit_dataset(xy, spec)


@functools.cache
def clean_noise() -> tuple[float, ...]:
    """Seed-0 normal noise for the 40 points, stopped at the first sample that behaves.

    With 40 points the core's influence flag (Cook's distance above 4/n) marks at least one
    point in about 19 of 20 normal samples, and the audit then reads ``caution``. "Well
    behaved" is therefore made precise: the first sample the seed-0 generator gives whose
    quadratic fit leaves every Cook's distance below 0.075 and every residual test above
    p = 0.15, so each verdict has room on both sides of its threshold.
    """
    rng = np.random.default_rng(0)
    for _ in range(3000):
        noise = rng.normal(0.0, 0.1, 40)
        summary = fit(make_xy(X, CURVE + noise), FitSpec(model=PolynomialRef(degree=2))).summary
        assert summary is not None
        p_values = [
            test.p_value
            for test in summary.diagnostics.tests
            if test.name in ("shapiro_wilk", "ljung_box", "breusch_pagan")
        ]
        cooks = [distance for distance in summary.residuals.cooks_distance if distance is not None]
        if (
            len(cooks) == 40
            and max(cooks) < 0.075
            and len(p_values) == 3
            and all(p is not None and p > 0.15 for p in p_values)
        ):
            return tuple(float(value) for value in noise)
    raise AssertionError("no well-behaved sample in 3000 draws from the seed-0 generator")


class Scenarios:
    """Every bundle the audit tests use; each method runs a real fit."""

    QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
    LINE = FitSpec(model=PolynomialRef(degree=1))
    ZERO_WEIGHT_ROWS = (3, 10, 17, 24, 31)
    CUSTOM_LINE = CustomRef(equation="a*x + b", parameters=["a", "b"])
    PINNED_SPEC = FitSpec(model=CUSTOM_LINE, bounds={"a": (0.0, 5.0)})
    ONE_UNIT_SPEC = FitSpec(model=CUSTOM_LINE, bounds={"a": (0.0, 1.0)})

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    @staticmethod
    def clean_xy() -> XY:
        return make_xy(X, CURVE + np.asarray(clean_noise()))

    @staticmethod
    def with_missing_weight(bundle: FitBundle) -> FitBundle:
        """The same bundle with one weight blanked out: a row the audit cannot use."""
        weights = bundle.data.weights
        assert weights is not None
        blanked: list[float | None] = list(weights)
        blanked[5] = None
        data = bundle.data.model_copy(update={"weights": blanked})
        return bundle.model_copy(update={"data": data})

    @staticmethod
    def without_standard_error(bundle: FitBundle) -> FitBundle:
        """The same fit with the first parameter's uncertainty removed."""
        first, *rest = bundle.result.parameters
        bare = first.model_copy(update={"std_err": None, "ci_low": None, "ci_high": None})
        return bundle.model_copy(
            update={"result": bundle.result.model_copy(update={"parameters": (bare, *rest)})}
        )

    @staticmethod
    def without_covariance(bundle: FitBundle) -> FitBundle:
        """The same fit with its covariance matrix withheld."""
        result = bundle.result.model_copy(update={"covariance": None})
        return bundle.model_copy(update={"result": result})

    @staticmethod
    def widened_spec(model_id: str) -> FitSpec:
        """The registered bounded line fitted with its slope allowed up to 100."""
        return FitSpec(model=RegisteredRef(id=model_id), bounds={"a": (0.0, 100.0)})

    def empty(self) -> FitBundle:
        return fit(make_xy([], []), self.LINE)

    def clean(self) -> FitBundle:
        return fit(self.clean_xy(), self.QUADRATIC)

    def polynomial(self, degree: int) -> FitBundle:
        return fit(self.clean_xy(), FitSpec(model=PolynomialRef(degree=degree)))

    def non_converged(self) -> FitBundle:
        """An unidentifiable custom fit (two parameters, one direction): never converges."""
        spec = FitSpec(
            model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]),
            initial={"a": 1.0, "b": 2.0},
        )
        return fit(self.clean_xy(), spec)

    def without_summary(self) -> FitBundle:
        """A converged fit whose summary the core refused to build."""

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise StatisticsError("NO_DEGREES_OF_FREEDOM", "the summary was withheld (test double)")

        with self._monkeypatch.context() as patch:
            patch.setattr("curvestudio.workflow.service.describe_fit", refuse)
            return self.clean()

    def skewed(self) -> FitBundle:
        noise = np.random.default_rng(0).exponential(1.0, 40) - 1.0
        return fit(make_xy(X, CURVE + noise), self.QUADRATIC)

    def large_quadratic(self, points: int = 600) -> FitBundle:
        """A clean quadratic on many points: the core's 4/n flag then marks dozens of them."""
        x = np.linspace(0.0, 10.0, points)
        noise = np.random.default_rng(0).normal(0.0, 0.1, points)
        return fit(make_xy(x, 1.0 + 0.5 * x - 0.03 * x**2 + noise), self.QUADRATIC)

    def heteroscedastic(self) -> FitBundle:
        noise = np.random.default_rng(0).normal(0.0, 1.0, 40) * (0.01 + 0.3 * X)
        return fit(make_xy(X, CURVE + noise), self.QUADRATIC)

    def outliers(self, rows: Sequence[int]) -> FitBundle:
        y = CURVE + np.asarray(clean_noise())
        y[list(rows)] += 1.5
        return fit(make_xy(X, y), self.QUADRATIC)

    def curve_fitted_by_line(self) -> FitBundle:
        """A parabola is a trend the straight line leaves whole in its residuals."""
        return fit(self.clean_xy(), self.LINE)

    def six_points(self) -> FitBundle:
        x = np.linspace(0.0, 10.0, 6)
        scatter = np.array([0.1, -0.1, 0.05, -0.05, 0.08, -0.08])
        return fit(make_xy(x, 1.0 + 0.5 * x - 0.03 * x**2 + scatter), self.QUADRATIC)

    def exact(self) -> FitBundle:
        """Six points that lie exactly on a quadratic: nothing is left to test."""
        x = np.linspace(0.0, 10.0, 6)
        return fit(make_xy(x, 1.0 + 0.5 * x - 0.03 * x**2), self.QUADRATIC)

    def autocorrelated(self) -> FitBundle:
        """A straight line through noise that carries each error into the next point."""
        rng = np.random.default_rng(0)
        noise = np.zeros(40)
        for i in range(1, 40):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 0.3)
        return fit(make_xy(X, 0.5 * X + noise), self.LINE)

    @staticmethod
    def _twelve_points() -> XY:
        # With 12 points a correct line's held-out error is usually more than 1.1 times its
        # in-sample error (median 1.20 over 300 seeds); this seed's line comes in at 1.04.
        x = np.linspace(0.0, 10.0, 12)
        return make_xy(x, 0.5 * x + np.random.default_rng(22).normal(0.0, 0.5, 12))

    def overfit(self) -> tuple[FitBundle, CrossValidation]:
        """A degree-9 polynomial on 12 points, cross-validated leave-one-out.

        Five folds cannot fit it: each training set has at most as many points as the model
        has parameters, so every fold fails and there is no held-out error to compare.
        """
        xy = self._twelve_points()
        return (
            fit(xy, FitSpec(model=PolynomialRef(degree=9))),
            k_fold_cv(xy, PolynomialFitter(9), folds=12, seed=0),
        )

    def line_with_cv(self) -> tuple[FitBundle, CrossValidation]:
        xy = self._twelve_points()
        return fit(xy, self.LINE), k_fold_cv(xy, PolynomialFitter(1), seed=0)

    def zero_weights(self) -> FitBundle:
        x = np.linspace(0.0, 10.0, 45)
        y = 1.0 + 0.5 * x - 0.03 * x**2 + np.random.default_rng(0).normal(0.0, 0.1, 45)
        weights = np.ones(45)
        weights[list(self.ZERO_WEIGHT_ROWS)] = 0.0
        return fit(make_xy(x, y, weights), self.QUADRATIC.model_copy(update={"use_weights": True}))

    def pinned_slope(self) -> FitBundle:
        """A line whose slope cannot go below 0, fitted to data that fall."""
        x = np.linspace(0.0, 5.0, 12)
        y = -x + np.random.default_rng(0).normal(0.0, 0.05, 12)
        return fit(make_xy(x, y), self.PINNED_SPEC)

    @staticmethod
    def _scattered() -> XY:
        # Four scattered points on a line of slope 0.3: the slope is poorly determined and
        # the intercept (far from zero) well enough not to be flagged on its own.
        return make_xy([0.0, 1.0, 2.0, 3.0], [5.5, 5.3, 6.6, 6.1])

    def wide_interval(self) -> FitBundle:
        """Four scattered points: the slope's interval spans the whole unit range allowed."""
        return fit(self._scattered(), self.ONE_UNIT_SPEC)

    def bounded_line(self, model_id: str, *, widened: bool = False) -> FitBundle:
        """Four scattered points fitted by the registered line whose slope lies in (0, 1)."""
        spec = self.widened_spec(model_id) if widened else FitSpec(model=RegisteredRef(id=model_id))
        return fit(self._scattered(), spec)

    def slope_at_declared_bound(self, model_id: str) -> FitBundle:
        """Data that fall, fitted by the registered line whose slope cannot go below 0."""
        x = np.linspace(0.0, 5.0, 12)
        y = -x + np.random.default_rng(0).normal(0.0, 0.05, 12)
        return fit(make_xy(x, y), FitSpec(model=RegisteredRef(id=model_id)))

    def audited(self) -> dict[str, FitAudit]:
        """One audit per scenario, each given the inputs that scenario comes with."""
        overfit, overfit_cv = self.overfit()
        line, line_cv = self.line_with_cv()
        clean = self.clean()
        pinned = self.pinned_slope()
        zero_weights = self.zero_weights()
        return {
            "clean": audit_fit(clean),
            "non_converged": audit_fit(self.non_converged()),
            "without_summary": audit_fit(self.without_summary()),
            "skewed": audit_fit(self.skewed()),
            "heteroscedastic": audit_fit(self.heteroscedastic()),
            "many_flagged_points": audit_fit(self.large_quadratic()),
            "one_outlier": audit_fit(self.outliers([30])),
            "three_outliers": audit_fit(self.outliers([30, 12, 25])),
            "curve_fitted_by_line": audit_fit(self.curve_fitted_by_line()),
            "six_points": audit_fit(self.six_points()),
            "exact": audit_fit(self.exact()),
            "autocorrelated": audit_fit(self.autocorrelated()),
            "overfit_with_cv": audit_fit(overfit, cv=overfit_cv),
            "line_with_cv": audit_fit(line, cv=line_cv),
            "zero_weights": audit_fit(zero_weights),
            "missing_weight": audit_fit(self.with_missing_weight(zero_weights)),
            "pinned_slope": audit_fit(pinned, spec=self.PINNED_SPEC),
            "pinned_without_spec": audit_fit(pinned),
            "wide_interval": audit_fit(self.wide_interval(), spec=self.ONE_UNIT_SPEC),
            "without_standard_error": audit_fit(self.without_standard_error(clean)),
            "without_covariance": audit_fit(self.without_covariance(clean)),
        }


@pytest.fixture
def scenarios(monkeypatch: pytest.MonkeyPatch) -> Scenarios:
    return Scenarios(monkeypatch)


@pytest.fixture
def bounded_line() -> Iterator[str]:
    """Register a line whose slope is declared to lie between 0 and 1; yield its id."""
    _clear_registry()
    register_starter_models()

    def start(_x: np.ndarray, _y: np.ndarray) -> dict[str, float]:
        return {"a": 0.5, "b": 0.0}

    register(
        ModelSpec(
            id=BOUNDED_LINE_ID,
            name="Bounded line",
            category="test",
            expression="a*x + b",
            parameters=(Parameter("a", "Slope", lower=0.0, upper=1.0), Parameter("b", "Intercept")),
            initial_guess=start,
        )
    )
    yield BOUNDED_LINE_ID
    _clear_registry()
    register_starter_models()
