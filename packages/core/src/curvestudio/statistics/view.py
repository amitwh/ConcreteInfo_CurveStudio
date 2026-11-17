"""FitView: the validated, active-point view of a fit that every statistic is computed from.

A :class:`~curvestudio.fitting.FitResult` covers every input row (zero-weight points
included) and carries plain lists. Statistics need the opposite: numpy arrays holding only
the points that influenced the fit, plus the means to evaluate the model and its Jacobian
anywhere. :func:`build_view` produces that once, after checking that the result really is a
converged fit of the supplied data, so no downstream routine re-validates or silently works
on a mismatched pair. All arrays are read-only.
"""

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.fitting import FitResult, FitStatus
from curvestudio.models import ModelSpec, get_model
from curvestudio.statistics.errors import StatisticsError

FloatArray = NDArray[np.float64]

_POLYNOMIAL_ID = re.compile(r"poly(\d+)(_origin)?")
"""Model ids written by ``fit_polynomial``: ``poly2`` or ``poly2_origin``."""


def _frozen(values: object) -> FloatArray:
    array = np.array(values, dtype=np.float64)  # always a private copy
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class FitView:
    """A converged fit seen through its active (weight > 0) points.

    ``x``, ``y``, ``w``, ``fitted`` and ``residuals`` hold the active points in input
    order (``w`` is all ones for an unweighted fit; ``residuals = y - fitted``, unweighted).
    ``theta`` and ``names`` are the estimates in parameter order, ``cov`` the covariance
    (``None`` when the fit could not provide one), ``df`` the residual degrees of freedom
    and ``s2 = SSE / df`` the residual variance (``None`` when ``df <= 0``). ``model`` is
    ``None`` for polynomials, which are evaluated from their powers (``0..d``, or ``1..d``
    for a fit through the origin, matching :func:`curvestudio.fitting.fit_polynomial`).
    """

    model: ModelSpec | None
    result: FitResult
    x: FloatArray
    y: FloatArray
    w: FloatArray
    fitted: FloatArray
    residuals: FloatArray
    names: tuple[str, ...]
    theta: FloatArray
    cov: FloatArray | None
    df: int
    s2: float | None

    @property
    def _powers(self) -> FloatArray:
        """Polynomial powers; the coefficient names are ``b<power>``."""
        return np.array([int(name[1:]) for name in self.names], dtype=np.float64)

    def jacobian_at(self, x: FloatArray) -> FloatArray:
        """Unweighted ``d f / d theta`` at ``x``, shape ``(len(x), k)``."""
        points = np.asarray(x, dtype=np.float64)
        if self.model is None:
            return np.asarray(points[:, None] ** self._powers, dtype=np.float64)
        return self.model.jacobian(
            points, **dict(zip(self.names, self.theta.tolist(), strict=True))
        )

    def predict(self, x: FloatArray) -> FloatArray:
        """Fitted curve ``f(x; theta)``."""
        points = np.asarray(x, dtype=np.float64)
        if self.model is None:
            powers = self._powers.astype(int)
            coefficients = np.zeros(int(powers.max()) + 1)
            coefficients[powers] = self.theta
            return np.asarray(np.polynomial.polynomial.polyval(points, coefficients))
        return self.model.evaluate(
            points, **dict(zip(self.names, self.theta.tolist(), strict=True))
        )

    @property
    def weighted_jacobian(self) -> FloatArray:
        """``sqrt(w)[:, None] * jacobian_at(x)`` on the active points (``J`` in ``s2 inv(J'J)``)."""
        return np.sqrt(self.w)[:, None] * self.jacobian_at(self.x)

    def require_covariance(self) -> FloatArray:
        """Return the covariance matrix, or raise why there is none.

        Raises:
            StatisticsError: ``NO_DEGREES_OF_FREEDOM`` when ``df <= 0`` (no residual variance
                can be estimated); ``NO_COVARIANCE`` when the fit withheld its covariance
                (singular or non-representable).
        """
        if self.df <= 0:
            raise StatisticsError(
                "NO_DEGREES_OF_FREEDOM",
                f"the fit has {self.df} residual degrees of freedom; this statistic needs at "
                "least one",
            )
        if self.cov is None:
            raise StatisticsError(
                "NO_COVARIANCE",
                "the fit has no covariance matrix (singular or non-representable); "
                "parameter uncertainty is not defined",
            )
        return self.cov


def build_view(data: XY, result: FitResult, model: ModelSpec | str | None = None) -> FitView:
    """Check that ``result`` is a converged fit of ``data`` and build its :class:`FitView`.

    Polynomial results are recognised from ``result.model_id`` and ignore ``model``;
    for any other result the evaluator is ``model`` (a spec or registered id), else the
    registered model named by ``result.model_id``. A missing covariance does not raise
    here; use :meth:`FitView.require_covariance` where one is needed.

    Raises:
        StatisticsError: ``FIT_NOT_CONVERGED`` (status is not ``CONVERGED``);
            ``DATA_MISMATCH`` (fingerprint differs, row count differs from the fit, or the
            active-point count differs from ``result.stats.n``); ``MODEL_MISMATCH`` (the
            model's parameters are not the result's).
        KeyError: no ``model`` was given and ``result.model_id`` is not a registered id.
    """
    if result.status is not FitStatus.CONVERGED or result.stats is None:
        raise StatisticsError(
            "FIT_NOT_CONVERGED",
            f"statistics need a converged fit; status is {result.status.value!r} "
            f"({result.termination_reason})",
        )
    if data.fingerprint != result.provenance.dataset_fingerprint:
        raise StatisticsError(
            "DATA_MISMATCH", "the data are not the data this fit was computed from (fingerprint)"
        )
    n_rows = int(data.x.size)
    if len(result.fitted) != n_rows or len(result.residuals) != n_rows:
        raise StatisticsError(
            "DATA_MISMATCH",
            f"the data have {n_rows} rows but the fit covers {len(result.fitted)}",
        )
    w_all = np.ones(n_rows) if data.weights is None else np.asarray(data.weights)
    active = w_all > 0.0
    n_active = int(np.count_nonzero(active))
    if n_active != result.stats.n:
        raise StatisticsError(
            "DATA_MISMATCH",
            f"the data have {n_active} active points but the fit used {result.stats.n}",
        )

    names = tuple(p.name for p in result.parameters)
    spec: ModelSpec | None = None
    if _POLYNOMIAL_ID.fullmatch(result.model_id) is None:
        spec = get_model(model) if isinstance(model, str) else model
        spec = get_model(result.model_id) if spec is None else spec
        if names != tuple(p.name for p in spec.parameters):
            raise StatisticsError(
                "MODEL_MISMATCH",
                f"model {spec.id!r} has parameters {[p.name for p in spec.parameters]} but the "
                f"fit estimated {list(names)}",
            )

    df = result.stats.df
    return FitView(
        model=spec,
        result=result,
        x=_frozen(data.x[active]),
        y=_frozen(data.y[active]),
        w=_frozen(w_all[active]),
        fitted=_frozen(np.asarray(result.fitted)[active]),
        residuals=_frozen(np.asarray(result.residuals)[active]),
        names=names,
        theta=_frozen([p.value for p in result.parameters]),
        cov=None if result.covariance is None else _frozen(result.covariance),
        df=df,
        s2=result.stats.sse / df if df > 0 else None,
    )
