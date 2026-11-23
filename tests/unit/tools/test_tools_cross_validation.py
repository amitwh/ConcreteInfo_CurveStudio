"""Cross-validation for the audit, on real fits: every fold is the core's own.

``cross_validate`` refits through ``k_fold_cv`` and ``loo_cv`` with the fitter the workflow
uses for the specification, so each test compares it with a direct call of the core on the
pair ``fitter_for`` returns, and checks the refusals against the codes they carry.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitResult
from curvestudio.statistics import PolynomialFitter, k_fold_cv, loo_cv
from curvestudio.tools import ToolError, Verdict, audit_fit, cross_validate
from curvestudio.tools import cross_validation as cross_validation_module
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    bundle_for_candidate,
    fit_dataset,
    fitter_for,
)
from curvestudio.workflow.bundle import COMPACT_MAX_POINTS

QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
LINE = FitSpec(model=PolynomialRef(degree=1))
NINTH = FitSpec(model=PolynomialRef(degree=9))
GROWTH = FitSpec(model=RegisteredRef(id="exp_growth"))
UNIDENTIFIABLE = FitSpec(model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]))
"""The specification of ``Scenarios.non_converged``, which the fit recorded."""


def _xy(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float] | None = None, tag: str = "cv"
) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=f"fp-{tag}",
    )


def _origin_bundle() -> tuple[FitBundle, XY]:
    """A ``poly2_origin`` bundle, opened from an Auto-fit row the way the UI opens one."""
    x = np.arange(1.0, 21.0)
    xy = _xy(x, 3.0 * x - 0.2 * x**2 + 0.1 * np.sin(x), tag="origin")
    shell = AutoFitResult(
        config=AutoFitConfig(n_starts=1),
        dataset_fingerprint=xy.fingerprint,
        n_active=0,
        ranked=(),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )
    return bundle_for_candidate(xy, shell, "poly2_origin"), xy


def _active_points(bundle: FitBundle) -> int:
    """The points the fit used, as the core counted them."""
    stats = bundle.result.stats
    assert stats is not None
    return stats.n


def _cv_verdict(bundle: FitBundle, spec: FitSpec, folds: int) -> Verdict:
    """The audit's cross-validation verdict once it is given the result of ``folds`` folds."""
    cv = cross_validate(bundle, spec, folds=folds)
    audit = audit_fit(bundle, spec=spec, cv=cv)
    return next(check.verdict for check in audit.checks if check.id == "cross_validation")


def test_cross_validation_honours_the_spec(scenarios, bounded_line) -> None:
    # a polynomial: the core's run on the pair fitter_for returns, to the last digit
    xy = scenarios.quadratic_xy(40, seed=0)
    bundle = fit_dataset(xy, QUADRATIC)
    fitter, data = fitter_for(QUADRATIC, xy, result=bundle.result)
    seed = bundle.result.provenance.seed
    assert cross_validate(bundle, QUADRATIC) == k_fold_cv(data, fitter, folds=5, seed=seed)

    # a registered model whose declared bounds exclude the true slope: its folds are fitted
    # inside those bounds, so they predict worse than the same model let free
    x = np.linspace(0.0, 5.0, 30)
    falling = _xy(x, -x + np.random.default_rng(0).normal(0.0, 0.05, 30), tag="falling")
    declared = FitSpec(model=RegisteredRef(id=bounded_line))
    unbounded = FitSpec(model=RegisteredRef(id=bounded_line), bounds={"a": (None, None)})
    bundle = fit_dataset(falling, declared)
    assert bundle.result.status.value == "converged"
    tight = cross_validate(bundle, declared)
    free = cross_validate(bundle, unbounded)
    assert (tight.status, free.status) == ("ok", "ok")
    assert tight.fold_rmse != free.fold_rmse
    assert tight.rmse is not None and free.rmse is not None
    assert tight.rmse > free.rmse
    free_fitter, free_data = fitter_for(unbounded, falling, result=bundle.result)
    assert free == k_fold_cv(free_data, free_fitter, folds=5, seed=seed)

    # weights: used when the spec says so, dropped when it says not
    x = np.linspace(0.0, 10.0, 45)
    weights = 1.0 / (1.0 + x) ** 2
    weighted = _xy(x, 1.0 + 0.5 * x - 0.03 * x**2, weights, tag="weighted")
    used = QUADRATIC.model_copy(update={"use_weights": True})
    ignored = QUADRATIC.model_copy(update={"use_weights": False})
    bundle = fit_dataset(weighted, used)
    fitter, kept = fitter_for(used, weighted, result=bundle.result)
    assert kept.weights is not None
    np.testing.assert_array_equal(kept.weights, weights)
    with_weights = cross_validate(bundle, used)
    assert with_weights == k_fold_cv(kept, fitter, folds=5, seed=seed)
    other, dropped = fitter_for(ignored, weighted, result=bundle.result)
    assert dropped.weights is None
    without_weights = cross_validate(bundle, ignored)
    assert without_weights == k_fold_cv(dropped, other, folds=5, seed=seed)
    assert with_weights.rmse != without_weights.rmse

    # the points a fold may be dealt are the active ones: a zero weight takes a row out
    zero = scenarios.zero_weights()
    spec = scenarios.QUADRATIC.model_copy(update={"use_weights": True})
    assert len(zero.data.x) == 45
    assert _active_points(zero) == 40
    assert cross_validate(zero, spec, folds=40).status == "ok"
    with pytest.raises(ToolError) as caught:
        cross_validate(zero, spec, folds=41)
    assert caught.value.code == "INVALID_FOLDS"


def test_cross_validation_is_reproducible_and_loo_runs(scenarios) -> None:
    xy = scenarios.quadratic_xy(40, seed=0)
    bundle = fit_dataset(xy, QUADRATIC)
    fitter, data = fitter_for(QUADRATIC, xy, result=bundle.result)

    first = cross_validate(bundle, QUADRATIC, seed=7)
    assert first == cross_validate(bundle, QUADRATIC, seed=7)
    assert first == k_fold_cv(data, fitter, folds=5, seed=7)

    left_out = cross_validate(bundle, QUADRATIC, folds="loo")
    assert left_out == loo_cv(data, fitter)
    assert (left_out.folds, left_out.status) == (40, "ok")

    # the default seed is the fit's own
    x = np.linspace(0.0, 3.0, 30)
    noise = np.random.default_rng(0).normal(0.0, 0.05, 30)
    growth = _xy(x, 2.0 * np.exp(0.4 * x) + noise, tag="growth")
    spec = FitSpec(model=RegisteredRef(id="exp_growth"), seed=3)
    bundle = fit_dataset(growth, spec)
    assert bundle.result.status.value == "converged"
    fitter, data = fitter_for(spec, growth, result=bundle.result)
    own = cross_validate(bundle, spec)
    assert own.seed == 3
    assert own == k_fold_cv(data, fitter, folds=5, seed=3)
    assert cross_validate(bundle, spec, seed=5) == k_fold_cv(data, fitter, folds=5, seed=5)


def test_cross_validation_refuses_a_specification_that_is_not_the_fit_s(scenarios) -> None:
    xy = scenarios.quadratic_xy(40, seed=0)
    bundle = fit_dataset(xy, QUADRATIC)
    x = np.linspace(0.0, 3.0, 30)
    noise = np.random.default_rng(0).normal(0.0, 0.05, 30)
    grown = fit_dataset(_xy(x, 2.0 * np.exp(0.4 * x) + noise, tag="growth"), GROWTH)
    assert grown.result.status.value == "converged"
    for fitted, foreign in (
        (bundle, FitSpec(model=PolynomialRef(degree=3))),
        (grown, FitSpec(model=RegisteredRef(id="gaussian"))),
    ):
        with pytest.raises(ToolError) as caught:
            cross_validate(fitted, foreign)
        assert caught.value.code == "MODEL_MISMATCH"
        assert "MODEL_MISMATCH" in str(caught.value)
    # its own specification is accepted
    assert cross_validate(grown, GROWTH).status == "ok"

    # the check comes first: a fit that did not converge is refused for its own specification
    # only, and for a foreign one as a mismatch
    with pytest.raises(ToolError) as caught:
        cross_validate(scenarios.non_converged(), QUADRATIC)
    assert caught.value.code == "MODEL_MISMATCH"
    with pytest.raises(ToolError) as caught:
        cross_validate(scenarios.non_converged(), UNIDENTIFIABLE)
    assert caught.value.code == "FIT_NOT_CONVERGED_INPUT"

    # a polynomial through the origin is a polynomial of its degree to a specification
    origin, _ = _origin_bundle()
    assert cross_validate(origin, QUADRATIC).status == "ok"


def test_cross_validation_refuses_what_it_cannot_use(scenarios, monkeypatch) -> None:
    # a fit that did not converge has no model to refit
    with pytest.raises(ToolError) as caught:
        cross_validate(scenarios.non_converged(), UNIDENTIFIABLE)
    assert caught.value.code == "FIT_NOT_CONVERGED_INPUT"

    # a compact bundle holds a subsample: it needs the full data, and then gives the same
    # answer as the full bundle
    x = np.linspace(0.0, 10.0, COMPACT_MAX_POINTS + 1)
    big = _xy(x, 2.0 * x + 1.0 + np.sin(x), tag="big")
    compact = fit_dataset(big, LINE, compact=True)
    assert compact.data.subsampled is True
    with pytest.raises(ToolError) as caught:
        cross_validate(compact, LINE)
    assert caught.value.code == "DATA_REQUIRED"
    restored = cross_validate(compact, LINE, data=big)
    assert restored.status == "ok"
    assert restored == cross_validate(fit_dataset(big, LINE), LINE)

    # data= must be the data the fit was computed from. Another dataset of the same size
    # fails the fingerprint the fit recorded; the same data with another row count fails
    # the row count.
    points = big.x.size
    other_x = np.linspace(0.0, 10.0, points)
    other = _xy(other_x, 3.0 * other_x + 1.0 + np.sin(other_x), tag="other")
    with pytest.raises(ToolError) as caught:
        cross_validate(compact, LINE, data=other)
    assert caught.value.code == "DATA_MISMATCH"
    assert "fingerprint" in caught.value.message
    shortened = XY(big.x[:-1], big.y[:-1], None, big.x_name, big.y_name, big.fingerprint)
    with pytest.raises(ToolError) as caught:
        cross_validate(compact, LINE, data=shortened)
    assert caught.value.code == "DATA_MISMATCH"
    assert "rows" in caught.value.message

    bundle = fit_dataset(scenarios.quadratic_xy(40, seed=0), QUADRATIC)
    for folds in (0, 1, 41):
        with pytest.raises(ToolError) as caught:
            cross_validate(bundle, QUADRATIC, folds=folds)
        assert caught.value.code == "INVALID_FOLDS"
        assert "folds" in caught.value.message

    # leave-one-out above the limit is refused; at the limit it runs
    monkeypatch.setattr(cross_validation_module, "LOO_MAX_N", 39)
    with pytest.raises(ToolError) as caught:
        cross_validate(bundle, QUADRATIC, folds="loo")
    assert caught.value.code == "INVALID_FOLDS"
    monkeypatch.setattr(cross_validation_module, "LOO_MAX_N", 40)
    assert cross_validate(bundle, QUADRATIC, folds="loo").status == "ok"

    # a specification the workflow refuses comes back as the tool's own error, same code
    with pytest.raises(ToolError) as caught:
        cross_validate(bundle, FitSpec(model=RegisteredRef(id="no_such_model")))
    assert caught.value.code == "UNKNOWN_MODEL"
    with pytest.raises(ToolError) as caught:
        cross_validate(bundle, QUADRATIC.model_copy(update={"bounds": {"b0": (0.0, 1.0)}}))
    assert caught.value.code == "INVALID_SPEC"


def test_a_through_origin_polynomial_cross_validates_with_its_own_fitter() -> None:
    bundle, xy = _origin_bundle()
    fitter, data = fitter_for(QUADRATIC, xy, result=bundle.result)
    assert fitter == PolynomialFitter(degree=2, through_origin=True)

    five = cross_validate(bundle, QUADRATIC)
    assert five.status == "ok"
    assert five == k_fold_cv(data, fitter, folds=5, seed=bundle.result.provenance.seed)
    assert cross_validate(bundle, QUADRATIC, folds="loo") == loo_cv(data, fitter)
    # the ordinary fitter fits another model and gives other fold errors
    ordinary = k_fold_cv(data, PolynomialFitter(degree=2), folds=5, seed=0)
    assert five.fold_rmse != ordinary.fold_rmse


def test_with_cross_validation_the_audit_detects_overfitting(scenarios) -> None:
    xy = scenarios.line_xy(12, seed=0)
    # a degree-9 polynomial on 12 points: twelve folds expose it
    ninth = fit_dataset(xy, NINTH)
    assert ninth.result.status.value == "converged"
    assert _cv_verdict(ninth, NINTH, 12) is Verdict.FAIL

    # a straight line on the same points, five folds
    line = fit_dataset(xy, LINE)
    assert _cv_verdict(line, LINE, 5) is Verdict.PASS
    audit = audit_fit(line, spec=LINE)
    verdict = next(check.verdict for check in audit.checks if check.id == "cross_validation")
    assert verdict is Verdict.NOT_APPLICABLE
