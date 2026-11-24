"""Real bundles for the fit-audit tests: every fit actually runs.

Both test modules audit the same fits, so the scenarios live here: one module asserts the
verdicts, the other scans every word the audit produces. No sample is chosen for its
verdicts: every dataset is the first draw of an ordinary seed.
"""

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.diagnostics import describe_fit
from curvestudio.fitting import fit_polynomial
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
NOISE = np.random.default_rng(0).normal(0.0, 0.1, 40)
"""The plain seed-0 sample: the generator's first draw, taken as it comes."""
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


def quadratic(x: np.ndarray) -> np.ndarray:
    return 1.0 + 0.5 * x - 0.03 * x**2


class Scenarios:
    """Every bundle the audit tests use; each method runs a real fit."""

    QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
    LINE = FitSpec(model=PolynomialRef(degree=1))
    WEIGHTED_QUADRATIC = QUADRATIC.model_copy(update={"use_weights": True})
    ZERO_WEIGHT_ROWS = (3, 10, 17, 24, 31)
    CUSTOM_LINE = CustomRef(equation="a*x + b", parameters=["a", "b"])
    PINNED_SPEC = FitSpec(model=CUSTOM_LINE, bounds={"a": (0.0, 5.0)})
    ONE_UNIT_SPEC = FitSpec(model=CUSTOM_LINE, bounds={"a": (0.0, 1.0)})

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    # -- data ----------------------------------------------------------------------------------

    @staticmethod
    def clean_xy() -> XY:
        return make_xy(X, CURVE + NOISE)

    @staticmethod
    def quadratic_xy(n: int, seed: int, sd: float = 0.1) -> XY:
        """A quadratic on ``n`` equally spaced points with normal noise of ``sd``."""
        x = np.linspace(0.0, 10.0, n)
        return make_xy(x, quadratic(x) + np.random.default_rng(seed).normal(0.0, sd, n))

    @staticmethod
    def line_xy(n: int, seed: int, sd: float = 0.5) -> XY:
        """A straight line on ``n`` equally spaced points with normal noise of ``sd``."""
        x = np.linspace(0.0, 10.0, n)
        return make_xy(x, 0.5 * x + np.random.default_rng(seed).normal(0.0, sd, n))

    @staticmethod
    def contaminated_xy(n: int, share: float, spike: float, seed: int) -> XY:
        """A quadratic with a spike of ``spike`` (either sign) on about ``share`` of its points."""
        x = np.linspace(0.0, 10.0, n)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.1, n)
        spiked = rng.random(n) < share
        signs = rng.choice(np.array([-1.0, 1.0]), n)
        return make_xy(x, quadratic(x) + noise + np.where(spiked, signs * spike, 0.0))

    # -- bundles -------------------------------------------------------------------------------

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
        result = bundle.result.model_copy(update={"parameters": (bare, *rest)})
        return bundle.model_copy(update={"result": result})

    @staticmethod
    def without_covariance(bundle: FitBundle) -> FitBundle:
        """The same fit with its covariance matrix withheld."""
        result = bundle.result.model_copy(update={"covariance": None})
        return bundle.model_copy(update={"result": result})

    @staticmethod
    def widened_spec(model_id: str) -> FitSpec:
        """The registered bounded line fitted with its slope allowed up to 100."""
        return FitSpec(model=RegisteredRef(id=model_id), bounds={"a": (0.0, 100.0)})

    @staticmethod
    def _scattered() -> XY:
        # Four scattered points on a line of slope 0.3: the slope is poorly determined and
        # the intercept (far from zero) well enough not to be flagged on its own.
        return make_xy([0.0, 1.0, 2.0, 3.0], [5.5, 5.3, 6.6, 6.1])

    def empty(self) -> FitBundle:
        return fit(make_xy([], []), self.LINE)

    def clean(self) -> FitBundle:
        return fit(self.clean_xy(), self.QUADRATIC)

    def polynomial(self, degree: int) -> FitBundle:
        return fit(self.clean_xy(), FitSpec(model=PolynomialRef(degree=degree)))

    def quadratic_fit(self, n: int, seed: int) -> FitBundle:
        """The quadratic on an ordinary seed-``seed`` sample of ``n`` points."""
        return fit(self.quadratic_xy(n, seed), self.QUADRATIC)

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

    def heteroscedastic(self) -> FitBundle:
        """A quadratic whose noise spreads from almost nothing at x = 0 to 3 at x = 10.

        On 150 points: on 40 the first draw gives a Breusch-Pagan p of 0.0036, a caution
        rather than the failure this scenario is built to show.
        """
        x = np.linspace(0.0, 10.0, 150)
        noise = np.random.default_rng(0).normal(0.0, 1.0, 150) * (0.01 + 0.3 * x)
        return fit(make_xy(x, quadratic(x) + noise), self.QUADRATIC)

    def in_range_outlier(self, sd: float = 8.0) -> FitBundle:
        """Row 31 displaced by ``sd`` noise standard deviations, at an ordinary x."""
        y = CURVE + NOISE
        y[30] += sd * 0.1
        return fit(make_xy(X, y), self.QUADRATIC)

    def far_outlier(self, sd: float = 8.0) -> FitBundle:
        """Row 31 moved far outside the other x values (to x = 30).

        It is also displaced by ``sd`` noise standard deviations from the curve.
        """
        x = X.copy()
        y = CURVE + NOISE
        x[30] = 30.0
        y[30] = quadratic(x)[30] + sd * 0.1
        return fit(make_xy(x, y), self.QUADRATIC)

    def contaminated_caution(self) -> FitBundle:
        """More points than chance over the 4/n line, none with a large Cook's distance."""
        return fit(self.contaminated_xy(100, 0.12, 0.45, seed=0), self.QUADRATIC)

    def contaminated_fail(self) -> FitBundle:
        """Far more points than chance over the 4/n line, none with a large Cook's distance."""
        return fit(self.contaminated_xy(400, 0.08, 0.4, seed=3), self.QUADRATIC)

    def curve_fitted_by_line(self) -> FitBundle:
        """A parabola is a shape the straight line leaves whole in its residuals."""
        return fit(self.clean_xy(), self.LINE)

    def cubic_by_quadratic(self) -> FitBundle:
        """The case the first trend test could not see: a cubic the quadratic cannot represent."""
        return fit(make_xy(X, 0.02 * (X - 5) ** 3 + NOISE), self.QUADRATIC)

    def quartic_by_quadratic(self) -> FitBundle:
        return fit(make_xy(X, 0.005 * (X - 5) ** 4 + NOISE), self.QUADRATIC)

    def offset_without_offset(self) -> FitBundle:
        """``exp_growth`` has no additive term; the data have an offset of 5."""
        x = np.linspace(0.0, 5.0, 40)
        noise = np.random.default_rng(0).normal(0.0, 0.05, 40)
        return fit(
            make_xy(x, 2.0 * np.exp(0.3 * x) + 5.0 + noise),
            FitSpec(model=RegisteredRef(id="exp_growth")),
        )

    def origin_quadratic_with_offset(self) -> FitBundle:
        """A quadratic through the origin fitted to data with an intercept of 5.

        The x values run symmetrically about zero, so the fit's x and x² terms cannot absorb
        the constant. ``fit_dataset`` cannot ask for a fit through the origin, so the result
        comes from the fitting layer and the summary from the diagnostics layer, on the same
        data.
        """
        x = np.linspace(-5.0, 5.0, 40)
        xy = make_xy(x, 5.0 + 0.03 * x**2 + NOISE)
        result = fit_polynomial(xy, 2, through_origin=True)
        return fit(xy, self.QUADRATIC).model_copy(
            update={"result": result, "summary": describe_fit(xy, result)}
        )

    def six_points(self) -> FitBundle:
        x = np.linspace(0.0, 10.0, 6)
        scatter = np.array([0.1, -0.1, 0.05, -0.05, 0.08, -0.08])
        return fit(make_xy(x, quadratic(x) + scatter), self.QUADRATIC)

    def exact(self) -> FitBundle:
        """Six points that lie exactly on a quadratic: nothing is left to test."""
        x = np.linspace(0.0, 10.0, 6)
        return fit(make_xy(x, quadratic(x)), self.QUADRATIC)

    def autocorrelated(self) -> FitBundle:
        """A straight line through noise that carries each error into the next point."""
        rng = np.random.default_rng(0)
        noise = np.zeros(40)
        for i in range(1, 40):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 0.3)
        return fit(make_xy(X, 0.5 * X + noise), self.LINE)

    def alternating(self) -> FitBundle:
        """A straight line through noise that flips sign from each point to the next."""
        rng = np.random.default_rng(0)
        noise = np.zeros(40)
        for i in range(1, 40):
            noise[i] = -0.8 * noise[i - 1] + rng.normal(0.0, 0.3)
        return fit(make_xy(X, 0.5 * X + noise), self.LINE)

    @staticmethod
    def subsampled(bundle: FitBundle) -> FitBundle:
        """The same bundle, marked as holding only a subsample of its data (compact mode).

        A compact bundle from the workflow has thousands of points, more than the exact
        Durbin-Watson test is computed for, so the mark is set by hand on a small fit.
        """
        return bundle.model_copy(
            update={"data": bundle.data.model_copy(update={"subsampled": True})}
        )

    def weights_ignored(self) -> tuple[FitBundle, XY, FitSpec]:
        """A line that ignored the weights of weighted data, the data as uploaded, its spec."""
        xy = make_xy(X, 0.5 * X + NOISE, np.linspace(0.5, 2.0, 40))
        spec = self.LINE.model_copy(update={"use_weights": False})
        return fit(xy, spec), xy, spec

    def large_quadratic(self, points: int = 600) -> FitBundle:
        """A clean quadratic on many points: many then have high leverage but little influence."""
        return fit(self.quadratic_xy(points, seed=0), self.QUADRATIC)

    def overfit(self) -> tuple[FitBundle, CrossValidation]:
        """A degree-9 polynomial on 12 points, cross-validated leave-one-out.

        Five folds cannot fit it: each training set has at most as many points as the model
        has parameters, so every fold fails and there is no held-out error to compare.
        """
        xy = self.line_xy(12, seed=0)
        return (
            fit(xy, FitSpec(model=PolynomialRef(degree=9))),
            k_fold_cv(xy, PolynomialFitter(9), folds=12, seed=0),
        )

    def line_with_cv(self, n: int = 12, seed: int = 0) -> tuple[FitBundle, CrossValidation]:
        """The straight line on an ordinary sample, with its five-fold cross-validation."""
        xy = self.line_xy(n, seed)
        return fit(xy, self.LINE), k_fold_cv(xy, PolynomialFitter(1), seed=0)

    def zero_weights(self) -> FitBundle:
        x = np.linspace(0.0, 10.0, 45)
        y = quadratic(x) + np.random.default_rng(0).normal(0.0, 0.1, 45)
        weights = np.ones(45)
        weights[list(self.ZERO_WEIGHT_ROWS)] = 0.0
        return fit(make_xy(x, y, weights), self.WEIGHTED_QUADRATIC)

    def pinned_slope(self) -> FitBundle:
        """A line whose slope cannot go below 0, fitted to data that fall."""
        x = np.linspace(0.0, 5.0, 12)
        y = -x + np.random.default_rng(0).normal(0.0, 0.05, 12)
        return fit(make_xy(x, y), self.PINNED_SPEC)

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
            "in_range_outlier": audit_fit(self.in_range_outlier()),
            "far_outlier": audit_fit(self.far_outlier()),
            "contaminated_caution": audit_fit(self.contaminated_caution()),
            "contaminated_fail": audit_fit(self.contaminated_fail()),
            "curve_fitted_by_line": audit_fit(self.curve_fitted_by_line()),
            "cubic_by_quadratic": audit_fit(self.cubic_by_quadratic()),
            "quartic_by_quadratic": audit_fit(self.quartic_by_quadratic()),
            "offset_without_offset": audit_fit(self.offset_without_offset()),
            "origin_quadratic_with_offset": audit_fit(self.origin_quadratic_with_offset()),
            "six_points": audit_fit(self.six_points()),
            "exact": audit_fit(self.exact()),
            "autocorrelated": audit_fit(self.autocorrelated()),
            "alternating": audit_fit(self.alternating()),
            "autocorrelated_with_spec": audit_fit(self.autocorrelated(), spec=self.LINE),
            "alternating_with_spec": audit_fit(self.alternating(), spec=self.LINE),
            "clean_with_spec": audit_fit(clean, spec=self.QUADRATIC),
            "large_quadratic": audit_fit(self.large_quadratic()),
            "large_quadratic_with_spec": audit_fit(self.large_quadratic(), spec=self.QUADRATIC),
            "overfit_with_cv": audit_fit(overfit, cv=overfit_cv),
            "line_with_cv": audit_fit(line, cv=line_cv),
            "zero_weights": audit_fit(zero_weights),
            "zero_weights_with_spec": audit_fit(zero_weights, spec=self.WEIGHTED_QUADRATIC),
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
