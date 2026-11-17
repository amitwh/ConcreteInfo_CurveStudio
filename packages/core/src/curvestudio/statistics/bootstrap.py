"""Seeded residual and case bootstrap for parameter uncertainty.

The bootstrap estimates the sampling distribution of a fit's parameters by refitting the
same model on many resampled data sets. Two resampling schemes are provided:

* **Residual bootstrap** keeps the fitted values and the design fixed and rebuilds each
  response from a resampled residual: the active points' whitened residuals
  ``sqrt(w) * (y - f)`` are inflated by ``sqrt(n / (n - k))`` (raw residuals underestimate
  the error variance), drawn with replacement, and re-attached as
  ``y*_i = f_i + r*_i / sqrt(w_i)``. It assumes the model form is right and that the
  residuals are an adequate, exchangeable sample of the error distribution.
* **Case bootstrap** resamples the active ``(x, y, w)`` triples with replacement,
  including duplicated and omitted points, and assumes the active points are an
  exchangeable sample from the population.

Conventions:

* Only active (``weight > 0``) points take part; zero-weight points are never resampled.
* One ``numpy.random.Generator(seed)`` produces every draw, so the same seed reproduces
  the replicates and every number exactly. The replicate count, seed and level are stored
  on the result.
* Every replicate refits through the supplied :class:`~curvestudio.statistics.Fitter`
  (the caller supplies any warm start), and only replicates that return ``CONVERGED``
  with the expected parameters contribute. Failed replicates are counted in ``n_failed``
  and summarised, with counts, in ``reasons``; they are never dropped silently or
  replaced. A standard error needs at least two converged replicates, so if fewer than two
  converge ``StatisticsError("NO_CONVERGED_REPLICATES")`` is raised.
* Intervals are percentile intervals of the converged replicate estimates at ``level``
  (default 0.95). ``status="unreliable"`` when fewer than ``min_success`` of the
  requested replicates converged; the intervals are then still computed from what exists
  and the reason is reported.
* A fit with no residual degrees of freedom is refused: the residual inflation factor and
  the premise of a residual distribution are both undefined.

Limitations: percentile intervals are only first-order accurate, can be biased for small
samples, influential points or a misspecified model, and are not simultaneous bands. They
quantify parameter uncertainty under the fitted model; they never establish that the
model is correct, and ``status="ok"`` means the requested replicates converged, not that
the model is valid.
"""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY
from curvestudio.fitting import FitResult, FitStatus
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.validation import Fitter
from curvestudio.statistics.view import build_view

FloatArray = NDArray[np.float64]

BootstrapKind = Literal["residual", "case"]


class BootstrapParameter(BaseModel):
    """Bootstrap summary of one fitted parameter.

    ``estimate`` is the full-data fit value; ``mean`` and ``std_err`` summarise the
    converged replicate estimates (``std_err`` is their sample standard deviation with the
    ``n - 1`` denominator). ``ci_low``/``ci_high`` are the percentile interval bounds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    estimate: FiniteFloat
    mean: FiniteFloat
    std_err: FiniteFloat
    ci_low: FiniteFloat | None
    ci_high: FiniteFloat | None


class BootstrapResult(BaseModel):
    """Outcome of one bootstrap run; ``"ok"`` means the replicates converged, not validity.

    ``replicates`` holds the converged replicate parameter vectors in parameter order
    (``len(replicates) == n_converged``), so every summary can be recomputed by a caller.
    ``reasons`` lists each distinct failure reason with its count, plus a line explaining
    an ``"unreliable"`` status. Assumptions and limitations are documented in the module
    docstring: the bootstrap quantifies parameter uncertainty, it does not confirm that
    the model is correct.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: BootstrapKind
    n_requested: int
    n_converged: int
    n_failed: int
    level: float
    seed: int
    status: Literal["ok", "unreliable"]
    parameters: tuple[BootstrapParameter, ...]
    replicates: list[list[float]]
    reasons: list[str]


def _check_n_boot(n_boot: object) -> int:
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 50:
        raise StatisticsError("INVALID_N_BOOT", f"n_boot must be an int >= 50, got {n_boot!r}")
    return n_boot


def _check_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise StatisticsError("INVALID_SEED", f"seed must be an int >= 0, got {seed!r}")
    return seed


def _check_fraction(name: str, code: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise StatisticsError(code, f"{name} must lie strictly between 0 and 1, got {value!r}")
    return float(value)


def _check_kind(kind: object) -> BootstrapKind:
    if kind not in ("residual", "case"):
        raise StatisticsError("INVALID_KIND", f"kind must be 'residual' or 'case', got {kind!r}")
    return kind


def _count(counter: dict[str, int], reason: str) -> None:
    counter[reason] = counter.get(reason, 0) + 1


def _reason_lines(counter: dict[str, int]) -> list[str]:
    return [f"{count} replicate(s): {reason}" for reason, count in counter.items()]


def bootstrap(
    data: XY,
    fitter: Fitter,
    result: FitResult,
    *,
    kind: BootstrapKind = "residual",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    min_success: float = 0.9,
) -> BootstrapResult:
    """Estimate parameter uncertainty by bootstrapping ``result``'s fit of ``data``.

    ``kind="residual"`` resamples the active points' inflated whitened residuals and keeps
    the design fixed; ``kind="case"`` resamples active ``(x, y, w)`` triples. Every
    replicate is refitted with ``fitter`` (a warm start is the caller's choice, via
    ``NonlinearFitter(start=...)``). Percentile intervals at ``level`` are computed over
    the converged replicates; ``status="unreliable"`` when
    ``n_converged / n_boot < min_success``, in which case the intervals are still reported
    from what exists and ``reasons`` says so.

    Assumptions: the model form is fixed and the active residuals (residual bootstrap) or
    active points (case bootstrap) are exchangeable draws from the error distribution or
    population respectively; the fit's weights are correct up to a common scale.

    Limitations: percentile intervals are first-order accurate and can be biased in small
    samples or under a misspecified model; ``"ok"`` reports replicate convergence, never
    model validity.

    Raises:
        StatisticsError: ``INVALID_N_BOOT`` (``n_boot`` not an int >= 50);
            ``INVALID_LEVEL`` / ``INVALID_MIN_SUCCESS`` (not strictly inside (0, 1));
            ``INVALID_SEED`` (not an int >= 0); ``INVALID_KIND`` (not residual/case);
            ``FIT_NOT_CONVERGED`` (``result`` is not a converged fit of ``data``);
            ``DATA_MISMATCH`` / ``MODEL_MISMATCH`` (``result`` belongs to other data or
            another model); ``NO_DEGREES_OF_FREEDOM`` (``df <= 0``);
            ``NO_CONVERGED_REPLICATES`` (fewer than two replicates converged).
        KeyError: a nonlinear result's model id is not registered.
    """
    kind = _check_kind(kind)
    n_boot = _check_n_boot(n_boot)
    level = _check_fraction("level", "INVALID_LEVEL", level)
    min_success = _check_fraction("min_success", "INVALID_MIN_SUCCESS", min_success)
    seed = _check_seed(seed)

    view = build_view(data, result)
    if view.df <= 0:
        raise StatisticsError(
            "NO_DEGREES_OF_FREEDOM",
            f"the fit has {view.df} residual degrees of freedom; the bootstrap needs at least one",
        )

    n = int(view.x.size)
    k = len(view.names)
    sqrt_w = np.sqrt(view.w)
    # Whitened residuals, inflated to estimate the error scale: raw residuals are too small
    # by about a factor sqrt((n - k) / n).
    adjusted = sqrt_w * view.residuals * math.sqrt(n / (n - k))

    rng = np.random.default_rng(seed)
    replicates: list[list[float]] = []
    failures: dict[str, int] = {}
    for index in range(n_boot):
        draw = rng.integers(0, n, size=n)
        fingerprint = f"{data.fingerprint}#bootstrap-{kind}-{seed}-{index}"
        if kind == "residual":
            sample = XY(
                x=view.x,
                y=view.fitted + adjusted[draw] / sqrt_w,
                weights=view.w,
                x_name=data.x_name,
                y_name=data.y_name,
                fingerprint=fingerprint,
            )
        else:
            sample = XY(
                x=view.x[draw],
                y=view.y[draw],
                weights=view.w[draw],
                x_name=data.x_name,
                y_name=data.y_name,
                fingerprint=fingerprint,
            )

        try:
            replicate = fitter.fit(sample)
        except Exception as exc:  # counted and reported below, never swallowed
            _count(failures, f"fit raised {type(exc).__name__}: {exc}")
            continue

        if replicate.status is not FitStatus.CONVERGED:
            _count(
                failures,
                f"fit status is {replicate.status.value!r} ({replicate.termination_reason})",
            )
            continue

        names = tuple(parameter.name for parameter in replicate.parameters)
        if names != view.names:
            _count(
                failures,
                f"fit returned parameters {list(names)}, expected {list(view.names)}",
            )
            continue

        replicates.append([parameter.value for parameter in replicate.parameters])

    n_converged = len(replicates)
    n_failed = n_boot - n_converged
    if n_converged < 2:
        detail = "; ".join(_reason_lines(failures))
        raise StatisticsError(
            "NO_CONVERGED_REPLICATES",
            f"only {n_converged} of {n_boot} bootstrap replicates converged; a standard error "
            f"needs at least two, so no parameter uncertainty is defined ({detail})",
        )

    values = np.asarray(replicates, dtype=np.float64)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail], axis=0)
    means = values.mean(axis=0)
    std_errs = values.std(axis=0, ddof=1)

    parameters = tuple(
        BootstrapParameter(
            name=name,
            estimate=float(theta),
            mean=float(mean),
            std_err=float(std_err),
            ci_low=float(low),
            ci_high=float(high),
        )
        for name, theta, mean, std_err, low, high in zip(
            view.names, view.theta, means, std_errs, lower, upper, strict=True
        )
    )

    fraction = n_converged / n_boot
    unreliable = fraction < min_success
    reasons = _reason_lines(failures)
    if unreliable:
        reasons.append(
            f"only {n_converged} of {n_boot} replicates converged ({fraction:.1%}), below "
            f"min_success={min_success:g}; the intervals are computed from the surviving "
            "replicates and are unreliable"
        )
    return BootstrapResult(
        kind=kind,
        n_requested=n_boot,
        n_converged=n_converged,
        n_failed=n_failed,
        level=level,
        seed=seed,
        status="unreliable" if unreliable else "ok",
        parameters=parameters,
        replicates=replicates,
        reasons=reasons,
    )
