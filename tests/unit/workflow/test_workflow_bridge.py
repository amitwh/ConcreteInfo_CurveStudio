"""The public bridge from a stored fit to its view and fitter.

``snapshot_xy``, ``fit_view`` and ``fitter_for`` let the statistics tools work on a fit the
workflow made without importing a private name of the workflow layer. Each is what
``predict`` or ``fit_dataset`` already do, so these tests compare the bridge with those two
and pin that extracting it changed no refusal.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from pydantic import ValidationError
from test_workflow_fit import big_linear_xy, exponential_xy, make_xy, poly_spec, quadratic_xy

from curvestudio import workflow
from curvestudio.data import XY
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.statistics import NonlinearFitter, PolynomialFitter, build_view, k_fold_cv
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    RegisteredRef,
    WorkflowError,
    fit_dataset,
    fit_view,
    fitter_for,
    predict,
    service,
    snapshot_xy,
)
from curvestudio.workflow.bundle import COMPACT_MAX_POINTS


@pytest.fixture(autouse=True)
def starter_registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


EXPONENTIAL = FitSpec(model=RegisteredRef(id="exp_growth"))
TIMED_OUT = FitSpec(model=RegisteredRef(id="exp_growth"), time_limit=1e-9)
UNIDENTIFIABLE = FitSpec(
    model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]), initial={"a": 1.0, "b": 2.0}
)
CONVERGED: list[tuple[FitSpec, Callable[[], XY]]] = [
    (poly_spec(2), quadratic_xy),
    (EXPONENTIAL, exponential_xy),
]


def _refusals() -> list[tuple[str, str, FitBundle, FitSpec, XY | None]]:
    """Every refusal ``predict`` makes: ``(code, start of the message, bundle, spec, data)``.

    ``data`` is the full data of a subsampled bundle and ``None`` when the bundle stands
    on its own.
    """
    big = big_linear_xy(COMPACT_MAX_POINTS + 1)
    return [
        (
            "DATA_REQUIRED",
            "the bundle holds only a subsample of its data (compact mode): pass data= with "
            "the full data the fit was computed from",
            fit_dataset(big, poly_spec(1), compact=True),
            poly_spec(1),
            big,
        ),
        (
            "FIT_NOT_CONVERGED",
            "cannot predict from a failed fit (",
            fit_dataset(exponential_xy(), TIMED_OUT),
            TIMED_OUT,
            None,
        ),
        (
            "FIT_NOT_CONVERGED",
            "cannot predict from a not_converged fit (",
            fit_dataset(quadratic_xy(), UNIDENTIFIABLE),
            UNIDENTIFIABLE,
            None,
        ),
        (
            "MODEL_MISMATCH",
            "the fit is 'poly2', not a polynomial of degree 3",
            fit_dataset(quadratic_xy(), poly_spec(2)),
            poly_spec(3),
            None,
        ),
        (
            "MODEL_MISMATCH",
            "the fit is 'exp_growth', not 'gaussian'",
            fit_dataset(exponential_xy(), EXPONENTIAL),
            FitSpec(model=RegisteredRef(id="gaussian")),
            None,
        ),
    ]


def test_snapshot_xy_round_trips_the_bundle_data() -> None:
    xy = quadratic_xy()
    bundle = fit_dataset(xy, poly_spec(2))
    plain = snapshot_xy(bundle)
    np.testing.assert_array_equal(plain.x, bundle.data.x)
    np.testing.assert_array_equal(plain.y, bundle.data.y)
    assert plain.weights is None
    assert (plain.x_name, plain.y_name) == (bundle.data.x_name, bundle.data.y_name)
    assert plain.fingerprint == bundle.result.provenance.dataset_fingerprint == xy.fingerprint

    x = np.linspace(0.0, 10.0, 30)
    weights = 1.0 / (1.0 + x)
    weighted = make_xy(x, 3.0 + 2.0 * x, weights, tag="weighted")
    restored = snapshot_xy(fit_dataset(weighted, poly_spec(1, use_weights=True)))
    assert restored.weights is not None
    np.testing.assert_array_equal(restored.weights, weights)
    assert restored.fingerprint == weighted.fingerprint


def test_fit_view_is_what_predict_builds() -> None:
    for spec, make_data in CONVERGED:
        bundle = fit_dataset(make_data(), spec)
        view = fit_view(bundle, spec)
        expected = build_view(snapshot_xy(bundle), bundle.result)
        np.testing.assert_array_equal(view.x, expected.x)
        np.testing.assert_array_equal(view.theta, expected.theta)
        np.testing.assert_array_equal(view.weighted_jacobian, expected.weighted_jacobian)
        points = np.linspace(0.5, 3.0, 6).tolist()
        assert view.predict(np.asarray(points)).tolist() == pytest.approx(
            predict(bundle, spec, points).y
        )

    refusals = _refusals()
    for code, message_start, bundle, spec, _ in refusals:
        with pytest.raises(WorkflowError) as caught:
            fit_view(bundle, spec)
        assert caught.value.code == code
        assert caught.value.message.startswith(message_start)

    # a subsampled bundle is usable again once its full data is given
    _, _, compact, spec, full_data = refusals[0]
    assert compact.data.subsampled is True
    assert fit_view(compact, spec, data=full_data).x.size == COMPACT_MAX_POINTS + 1


def test_predict_still_refuses_the_same_way() -> None:
    refusals = _refusals()
    for code, message_start, bundle, spec, _ in refusals:
        with pytest.raises(WorkflowError) as caught:
            predict(bundle, spec, [1.0])
        assert caught.value.code == code
        assert caught.value.message.startswith(message_start)
        # the bridge refuses through the same path, in the same words
        with pytest.raises(WorkflowError) as twin:
            fit_view(bundle, spec)
        assert (twin.value.code, twin.value.message) == (caught.value.code, caught.value.message)

    _, _, compact, spec, full_data = refusals[0]
    out = predict(compact, spec, [1.0], "confidence", data=full_data)
    assert out.lower is not None
    assert out.lower[0] is not None


def test_fitter_for_reproduces_the_workflow_fit() -> None:
    x = np.linspace(0.0, 10.0, 30)
    weighted = make_xy(x, 3.0 + 2.0 * x, 1.0 / (1.0 + x) ** 2, tag="weighted")
    cases: list[tuple[FitSpec, XY, type]] = [
        (poly_spec(2), quadratic_xy(), PolynomialFitter),
        (EXPONENTIAL, exponential_xy(), NonlinearFitter),
        (poly_spec(1, use_weights=True), weighted, PolynomialFitter),
        (poly_spec(1, use_weights=False), weighted, PolynomialFitter),
    ]
    for spec, xy, kind in cases:
        expected = fit_dataset(xy, spec).result
        fitter, data = fitter_for(spec, xy)
        assert isinstance(fitter, kind)
        result = fitter.fit(data)
        assert result.model_id == expected.model_id
        assert [p.name for p in result.parameters] == [p.name for p in expected.parameters]
        assert [p.value for p in result.parameters] == pytest.approx(
            [p.value for p in expected.parameters], rel=1e-9
        )
        # the pair is what the core's cross-validation takes
        assert k_fold_cv(data, fitter).status == "ok"

    # the data is the caller's after the spec's weight choice
    assert weighted.weights is not None
    assert fitter_for(poly_spec(1, use_weights=False), weighted)[1].weights is None
    kept = fitter_for(poly_spec(1, use_weights=True), weighted)[1]
    assert kept.weights is not None
    np.testing.assert_array_equal(kept.weights, weighted.weights)

    # a polynomial is fitted in closed form, so the other models' settings are refused
    bounded = poly_spec(2, bounds={"b0": (0.0, 1.0)})
    with pytest.raises(WorkflowError) as caught:
        fitter_for(bounded, quadratic_xy())
    with pytest.raises(WorkflowError) as twin:
        fit_dataset(quadratic_xy(), bounded)
    assert caught.value.code == "INVALID_SPEC"
    assert caught.value.message == twin.value.message

    # the workflow fits no polynomial through the origin: a spec cannot ask for one, so the
    # fitter is the one fit_dataset uses
    with pytest.raises(ValidationError):
        FitSpec.model_validate(
            {"model": {"kind": "polynomial", "degree": 2, "through_origin": True}}
        )
    assert fitter_for(poly_spec(2), quadratic_xy())[0] == PolynomialFitter(degree=2)


def test_the_bridge_is_exported() -> None:
    from curvestudio.workflow import fit_view, fitter_for, snapshot_xy

    assert (fitter_for, fit_view, snapshot_xy) == (
        service.fitter_for,
        service.fit_view,
        service.snapshot_xy,
    )
    assert {"fitter_for", "fit_view", "snapshot_xy"} <= set(workflow.__all__)
