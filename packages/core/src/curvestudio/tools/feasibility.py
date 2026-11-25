"""The experiment feasibility simulator: can a planned design resolve the parameters?

The question is asked before any data exist. Given a model, the true parameter values, a
design of x locations and their replicates, a noise model and a seed, the simulator draws
dataset after dataset from the model and fits each one with the shipped
:func:`~curvestudio.workflow.fit_dataset`. It reports what a person needs to decide whether
the experiment is worth running: the bias and empirical standard error of every estimate,
the share of the fits that converged, how often each fit's interval covered the truth, and,
for a stated difference, the simulated power to detect it.

Every number is a measurement over the simulated datasets, never a promise about the real
experiment. The noise is the one the request states (a constant standard deviation, or a
coefficient of variation that scales the noise with the mean), the design is the one the
request gives, and the interval covered is the Wald interval from each fit's own standard
error. A design point outside the model's documented example range is allowed and the
result says the curve is extrapolated there; because there are no data, nothing is
extrapolated beyond anything else.

The module is pure: it fits nothing it was not asked to fit, reads no file, touches no
network, and draws every random number from one ``numpy.random.default_rng(seed)``
generator, so the same request and seed give byte-identical JSON.
"""

import hashlib
import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import stats as scipy_stats

from curvestudio.data import XY
from curvestudio.fitting import FitStatus, ParameterEstimate
from curvestudio.models import (
    CustomModelDefinition,
    ModelSpec,
    ParameterDefinition,
    build_custom_model,
)
from curvestudio.tools._errors import ToolError
from curvestudio.tools._fit_tools import CancelledFitTool, cancel_checker
from curvestudio.tools._types import Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    ToolResult,
    assumption,
    check_level,
)
from curvestudio.tools.stats._display import FloatArray, box_plot
from curvestudio.tools.stats._tables import table
from curvestudio.workflow import FitSpec, PolynomialRef, RegisteredRef, fit_dataset
from curvestudio.workflow.catalog import registered_spec

MIN_RUNS = 2
"""The fewest simulated datasets the tool draws; fewer cannot show a spread."""

MAX_RUNS = 5_000
"""The most simulated datasets one request may ask for; above it the run is too long."""

MAX_LOCATIONS = 200
"""The most distinct design points an evenly spaced design may hold."""

MAX_REPLICATES = 100
"""The most times one design point may be repeated."""

ALPHA = 0.05
"""The significance level of the power test: two-sided at 0.05."""

COVERAGE_WARNING_POINTS = 3.0
"""A coverage this many percentage points from the nominal level is warned about."""

NON_CONVERGED_WARNING_SHARE = 0.05
"""More than this share of the simulations not converging is warned about."""

FIT_LEVEL = 0.95
"""The level :func:`~curvestudio.workflow.fit_dataset` reports its parameter intervals at.

Its ``FitSpec`` carries no level (the workflow uses ``FitConfig``'s 0.95 default), so at this
level the fit's own ``ci_low``/``ci_high`` are used unchanged; at another requested level the
Wald interval is rebuilt from the fit's standard error and residual degrees of freedom.
"""


class DesignSpec(BaseModel):
    """Where to measure and how often: explicit locations, or an evenly spaced range.

    Give exactly one of ``locations`` and the ``low``/``high``/``count`` trio. A location is
    repeated ``replicates`` times in every simulated dataset. The values are checked by
    :func:`simulate`, which raises ``INVALID_OPTION`` for anything a design cannot be.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    locations: tuple[float, ...] | None = None
    low: float | None = None
    high: float | None = None
    count: int | None = None
    replicates: int = 1


class NoiseSpec(BaseModel):
    """The noise added to the model's mean: a constant SD, or a coefficient of variation.

    ``sd`` adds ``N(0, value)`` to every point, so the noise does not scale with the curve.
    ``cv`` multiplies the mean by ``1 + N(0, value)`` and fits with weights
    ``1 / (value * mean)**2`` (inverse variances), so the fit's noise model matches the draw.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["sd", "cv"] = "sd"
    value: float


class SimulationRequest(BaseModel):
    """A complete simulation: the truth, the design, the noise and the number of runs.

    ``truth`` must name exactly the model's parameters. ``runs`` is 2 to 5,000 (more is
    ``TOO_MANY_ROWS``). ``difference`` asks for the simulated power against the stated
    difference for the parameters it names; it is ``None`` when power is not wanted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    truth: dict[str, float]
    design: DesignSpec
    noise: NoiseSpec
    runs: int = 200
    seed: int = 0
    level: float = 0.95
    difference: dict[str, float] | None = None


def _check_design(design: DesignSpec) -> None:
    """Refuse a design that is not exactly one of its two forms, or is out of range."""
    if (
        isinstance(design.replicates, bool)
        or not isinstance(design.replicates, int)
        or not 1 <= design.replicates <= MAX_REPLICATES
    ):
        raise ToolError(
            "INVALID_OPTION",
            f"replicates must be a whole number from 1 to {MAX_REPLICATES}, "
            f"not {design.replicates!r}",
        )
    if design.locations is not None:
        if design.low is not None or design.high is not None or design.count is not None:
            raise ToolError(
                "INVALID_OPTION",
                "give either explicit locations or low, high and count, not both",
            )
        if not design.locations:
            raise ToolError("INVALID_OPTION", "locations is empty: give at least one point")
        for value in design.locations:
            if not math.isfinite(value):
                raise ToolError(
                    "INVALID_OPTION", f"every design location must be finite, not {value!r}"
                )
        return
    missing = [
        name
        for name, value in (("low", design.low), ("high", design.high), ("count", design.count))
        if value is None
    ]
    if missing:
        raise ToolError(
            "INVALID_OPTION",
            "give either explicit locations or all of low, high and count; "
            f"missing {', '.join(missing)}",
        )
    assert design.low is not None and design.high is not None and design.count is not None
    if not (math.isfinite(design.low) and math.isfinite(design.high)):
        raise ToolError(
            "INVALID_OPTION",
            f"low and high must be finite, not {design.low!r} and {design.high!r}",
        )
    if design.low >= design.high:
        raise ToolError(
            "INVALID_OPTION",
            f"low must lie below high, not {design.low!r} and {design.high!r}",
        )
    if (
        isinstance(design.count, bool)
        or not isinstance(design.count, int)
        or not 1 <= design.count <= MAX_LOCATIONS
    ):
        raise ToolError(
            "INVALID_OPTION",
            f"count must be a whole number from 1 to {MAX_LOCATIONS}, not {design.count!r}",
        )


def _check_request(request: SimulationRequest) -> None:
    """Refuse a request of an unknown type, an out-of-range level, run count or noise."""
    if not isinstance(request, SimulationRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"simulate takes a SimulationRequest, not a request of {type(request).__name__}",
        )
    check_level(request.level)
    if isinstance(request.runs, bool) or not isinstance(request.runs, int):
        raise ToolError("INVALID_OPTION", f"runs must be a whole number, not {request.runs!r}")
    if request.runs > MAX_RUNS:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"runs is {request.runs:,}; the simulator draws at most {MAX_RUNS:,} datasets",
        )
    if request.runs < MIN_RUNS:
        raise ToolError("INVALID_OPTION", f"runs must be at least {MIN_RUNS}, not {request.runs}")
    if isinstance(request.seed, bool) or not isinstance(request.seed, int) or request.seed < 0:
        raise ToolError(
            "INVALID_OPTION", f"seed must be a whole number of 0 or more, not {request.seed!r}"
        )
    if not math.isfinite(request.noise.value) or request.noise.value <= 0.0:
        raise ToolError(
            "INVALID_OPTION",
            f"the noise value must be a finite number above 0, not {request.noise.value!r}",
        )
    _check_design(request.design)


def _model_for(spec: FitSpec) -> ModelSpec | None:
    """The fitted model behind ``spec``, or ``None`` for a polynomial.

    A registered model comes from the catalogue; a custom equation is compiled from its
    reference so that the model can be evaluated at the true values before anything is
    fitted. The choice mirrors what :func:`~curvestudio.workflow.fit_dataset` does.
    """
    ref = spec.model
    if isinstance(ref, PolynomialRef):
        return None
    if isinstance(ref, RegisteredRef):
        return registered_spec(ref.id)
    if not ref.parameters:
        raise ToolError(
            "INVALID_OPTION",
            "the custom equation lists no parameters: list the names to fit, e.g. ['a', 'b']",
        )
    definition = CustomModelDefinition(
        id="custom",
        name="Custom equation",
        expression=ref.equation,
        variables=(ref.variable,),
        parameters=tuple(ParameterDefinition(name=name) for name in ref.parameters),
    )
    try:
        return build_custom_model(definition)
    except ValueError as exc:
        raise ToolError("INVALID_OPTION", f"the custom equation cannot be built: {exc}") from exc


def _parameter_names(spec: FitSpec, model: ModelSpec | None) -> tuple[str, ...]:
    """The names the fit reports its estimates under, in the model's own order."""
    if model is not None:
        return tuple(parameter.name for parameter in model.parameters)
    ref = spec.model
    assert isinstance(ref, PolynomialRef)  # no model only for a polynomial
    return tuple(f"b{power}" for power in range(ref.degree + 1))


def _check_truth(names: tuple[str, ...], truth: dict[str, float]) -> None:
    """Refuse true values that do not name exactly the model's parameters, or are not finite."""
    missing = sorted(set(names) - truth.keys())
    unknown = sorted(truth.keys() - set(names))
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown {', '.join(unknown)}")
        raise ToolError(
            "INVALID_OPTION",
            "truth must name exactly the model's parameters "
            f"({', '.join(names)}): {'; '.join(parts)}",
        )
    for name in names:
        value = truth[name]
        if not math.isfinite(value):
            raise ToolError(
                "INVALID_OPTION", f"the true value of {name!r} must be finite, not {value!r}"
            )


def _check_difference(names: tuple[str, ...], difference: dict[str, float]) -> None:
    """Refuse a difference naming a parameter the model does not have, or a non-finite one."""
    for name, value in difference.items():
        if name not in names:
            raise ToolError(
                "INVALID_OPTION",
                f"difference names {name!r}, which is not a parameter of the model "
                f"({', '.join(names)})",
            )
        if not math.isfinite(value):
            raise ToolError(
                "INVALID_OPTION",
                f"the difference for {name!r} must be finite, not {value!r}",
            )


def _design_points(design: DesignSpec) -> tuple[FloatArray, tuple[tuple[float, int], ...]]:
    """The design's distinct points with their replicate counts, and the x vector they give.

    An explicit location keeps the order it was given in and the replicates it shares. A
    range is filled with ``count`` evenly spaced values, ends included. A repeated location
    is listed once with its total count, so the design table has one row per point.
    """
    if design.locations is not None:
        points = np.asarray(design.locations, dtype=np.float64)
    else:
        assert design.low is not None and design.high is not None and design.count is not None
        points = np.linspace(design.low, design.high, design.count, dtype=np.float64)
    counts: dict[float, int] = {}
    for value in points:
        counts[float(value)] = counts.get(float(value), 0) + design.replicates
    x = np.repeat(points, design.replicates)
    return x, tuple((point, count) for point, count in counts.items())


def _fingerprint(x: FloatArray, y: FloatArray) -> str:
    """The dataset identity a generated ``XY`` carries: a hash of its own numbers."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _dataset(x: FloatArray, y: FloatArray, weights: FloatArray | None) -> XY:
    """One simulated dataset, named ``x`` and ``y`` (the design and the response)."""
    return XY(
        x=x,
        y=y,
        weights=weights,
        x_name="x",
        y_name="y",
        fingerprint=_fingerprint(x, y),
    )


def _curve(
    spec: FitSpec, model: ModelSpec | None, x: FloatArray, truth: dict[str, float]
) -> FloatArray:
    """The model's mean response at ``x``, with the true parameters."""
    if model is not None:
        return np.asarray(model.evaluate(x, **truth), dtype=np.float64)
    ref = spec.model
    assert isinstance(ref, PolynomialRef)  # no model only for a polynomial
    coefficients = np.array(
        [truth[f"b{power}"] for power in range(ref.degree + 1)], dtype=np.float64
    )
    return np.asarray(np.polynomial.polynomial.polyval(x, coefficients), dtype=np.float64)


def _interval(
    parameter: ParameterEstimate, df: int | None, level: float
) -> tuple[float, float] | None:
    """The Wald interval of one estimate at ``level``, or ``None`` when it cannot be built.

    At the level the workflow reports (``FIT_LEVEL``) the fit's own interval is used as it
    stands. At another level it is rebuilt from the standard error and the fit's residual
    degrees of freedom with Student's t; without a standard error or degrees of freedom
    there is no interval, never a fabricated one.
    """
    if level == FIT_LEVEL and parameter.ci_low is not None and parameter.ci_high is not None:
        return (float(parameter.ci_low), float(parameter.ci_high))
    if parameter.std_err is None or df is None or df <= 0:
        return None
    t = float(_t_quantile(level, df))
    value = float(parameter.value)
    half = t * float(parameter.std_err)
    return (value - half, value + half)


def _t_quantile(level: float, df: int) -> float:
    """The two-sided ``level`` Student-t quantile on ``df`` degrees of freedom."""
    return float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))


def _extrapolation_note(model: ModelSpec | None, points: tuple[tuple[float, int], ...]) -> str:
    """What the design table says about the model's documented x range."""
    if model is None:
        return (
            "A polynomial declares no example x range, so the design uses the points exactly "
            "as given."
        )
    if model.example is None:
        return (
            "The model declares no example x range, so the design uses the points exactly as given."
        )
    low, high = model.example.x_range
    outside = [point for point, _count in points if point < low or point > high]
    if not outside:
        return (
            f"Every design point lies within the model's example x range {num(low)} to {num(high)}."
        )
    listed = ", ".join(num(point) for point in outside[:8])
    more = "" if len(outside) <= 8 else f" and {len(outside) - 8} more"
    return (
        f"{len(outside)} design point(s) ({listed}{more}) lie outside the model's example x "
        f"range {num(low)} to {num(high)}; the curve is extrapolated there."
    )


def _noise_sentence(request: SimulationRequest) -> str:
    """The noise model in words, for the assumptions block."""
    noise = request.noise
    if noise.kind == "sd":
        return (
            f"Each point's response was the model's mean plus normal noise with a standard "
            f"deviation of {num(noise.value)}, at every design point."
        )
    return (
        f"Each point's response was the model's mean multiplied by 1 + normal noise with a "
        f"coefficient of variation of {num(noise.value)} (y = f * (1 + eps)), so the noise "
        "scales with the curve; the fits were weighted by the inverse variance 1 / (cv * mean)^2."
    )


def _methods(
    spec: FitSpec,
    request: SimulationRequest,
    model: ModelSpec | None,
    names: tuple[str, ...],
    x: FloatArray,
    points: tuple[tuple[float, int], ...],
    extrapolation: str,
) -> str:
    """The methods paragraph: design, noise, seed, fit and the interval covered."""
    design = request.design
    observed = f"({x.size} observation{'s' if x.size != 1 else ''})"
    if design.locations is not None:
        design_sentence = (
            f"The design used the locations "
            f"{', '.join(num(point) for point, _ in points)}, each repeated "
            f"{design.replicates} time{'s' if design.replicates != 1 else ''} {observed}"
        )
    else:
        assert design.low is not None and design.high is not None and design.count is not None
        design_sentence = (
            f"The design used {design.count} evenly spaced points from {num(design.low)} to "
            f"{num(design.high)}, ends included, each repeated {design.replicates} time"
            f"{'s' if design.replicates != 1 else ''} {observed}"
        )
    noise = request.noise
    if noise.kind == "sd":
        noise_sentence = (
            f"each response was the model's mean at the true parameters plus normal noise with "
            f"a standard deviation of {num(noise.value)}"
        )
        weight_sentence = "The fits were unweighted."
    else:
        noise_sentence = (
            "each response was the model's mean at the true parameters multiplied by 1 + normal "
            f"noise with a coefficient of variation of {num(noise.value)}; the fits were "
            "weighted by 1 / (cv * mean)^2 (inverse variances), matching the draw"
        )
        weight_sentence = (
            "Weights are inverse variances, so the weighted least squares is the matching fit."
        )
    truth = ", ".join(f"{name} = {num(request.truth[name])}" for name in names)
    method = spec.method
    fit_sentence = (
        f"Each simulated dataset was fitted with fit_dataset under the same specification "
        f"({_model_label(spec, model)}, method {method}, loss {spec.loss}, one deterministic "
        "start and the specification's bounds)."
    )
    interval_sentence = (
        f"The interval is the Wald interval from each fit's own standard error, at the "
        f"{num(request.level * 100)} % level"
    )
    if request.level == FIT_LEVEL:
        interval_sentence += " (the fit's reported interval). "
    else:
        interval_sentence += (
            " on the fit's residual degrees of freedom (the fit itself reports 95 %). "
        )
    interval_sentence += (
        "Coverage is the share of converged fits whose interval contains the true value."
    )
    return (
        f"Experiment feasibility simulation. {design_sentence}. With the true parameters "
        f"{truth}, {noise_sentence}. Every random draw came from one numpy.random.default_rng "
        f"generator seeded at {request.seed}. {fit_sentence} {weight_sentence} "
        f"{interval_sentence} {extrapolation}"
    )


def _model_label(spec: FitSpec, model: ModelSpec | None) -> str:
    """The model in a sentence: its name, or the polynomial's degree."""
    if model is not None:
        return model.name
    ref = spec.model
    assert isinstance(ref, PolynomialRef)  # no model only for a polynomial
    return f"a degree {ref.degree} polynomial"


def _interpretation(
    names: tuple[str, ...],
    request: SimulationRequest,
    completed: int,
    converged: int,
    means: dict[str, float | None],
    biases: dict[str, float | None],
    coverage: dict[str, float | None],
    cancelled: bool,
) -> str:
    """Coverage and bias in words, and how many fits converged."""
    if completed == 0:
        return (
            "No simulated dataset was fitted: the run was cancelled before the first one "
            "could be drawn."
        )
    share = converged / completed
    sentences = [
        f"{converged} of the {completed} simulated datasets converged ({num(share * 100)} %)."
    ]
    for name in names:
        mean = means[name]
        bias = biases[name]
        if mean is None or bias is None:
            sentences.append(f"For {name!r}, no converged fit carried an estimate to summarise.")
            continue
        covered = coverage[name]
        coverage_phrase = (
            "no interval to measure coverage from"
            if covered is None
            else f"the {num(request.level * 100)} % interval covered the true value in "
            f"{num(covered)} % of them"
        )
        sentences.append(
            f"For {name!r} the mean estimate was {num(mean)} against a true "
            f"{num(request.truth[name])}, a bias of {num(bias)}; {coverage_phrase}."
        )
    if cancelled:
        sentences.append("The run was cancelled, so these numbers describe the runs completed.")
    return " ".join(sentences)


def simulate(
    spec: FitSpec,
    request: SimulationRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> ToolResult:
    """Simulate the request's design, fit every dataset, and measure the estimates.

    Args:
        spec: the specification every simulated dataset is fitted with. Its model gives the
            parameter names and the curve; its bounds and starts are used as given. For a
            coefficient-of-variation noise model the fit is weighted (``use_weights=True``).
        request: the true values, the design, the noise, the run count, the seed, the
            interval level and the optional differences to measure power for.
        should_cancel: an optional token checked between simulated datasets. Once it turns
            true the result is built from the datasets already fitted and carries the
            warning ``cancelled after k of N runs``.

    Returns:
        One :class:`~curvestudio.tools.stats.ToolResult`: the parameter-recovery table, the
        design, the power table when asked, the run counts, a box plot of the estimates, the
        warnings and the methods text.

    Raises:
        ToolError: ``INVALID_OPTION`` for a request or design the tool cannot run, an
            out-of-range level, a truth that is not the model's parameters, or a difference
            naming an unknown parameter; ``TOO_MANY_ROWS`` above 5,000 runs.
        WorkflowError: from the model resolution or a fit, as :func:`fit_dataset` raises it.
    """
    _check_request(request)
    model = _model_for(spec)
    names = _parameter_names(spec, model)
    _check_truth(names, request.truth)
    difference = request.difference or {}
    if difference:
        _check_difference(names, difference)

    x, points = _design_points(request.design)
    mean = _curve(spec, model, x, request.truth)
    if not np.all(np.isfinite(mean)):
        bad = np.unique(x[~np.isfinite(mean)])
        listed = ", ".join(num(value) for value in bad[:8])
        raise ToolError(
            "INVALID_OPTION",
            "the model is not finite at every design point: the true curve is undefined at "
            f"x = {listed}",
        )
    if request.noise.kind == "cv":
        zeros = np.unique(x[mean == 0.0])
        if zeros.size:
            listed = ", ".join(num(value) for value in zeros[:8])
            raise ToolError(
                "INVALID_OPTION",
                "a coefficient-of-variation noise model is undefined where the mean is 0: "
                f"no weight 1 / (cv * mean)^2 exists at x = {listed}",
            )
        weights = np.asarray(1.0 / (request.noise.value * mean) ** 2, dtype=np.float64)
        fit_spec = spec.model_copy(update={"use_weights": True})
    else:
        if spec.use_weights:
            raise ToolError(
                "INVALID_OPTION",
                "the specification asks for weights, but a constant-standard-deviation "
                "simulation gives the data no weights; use a cv noise model for a weighted fit",
            )
        weights = None
        fit_spec = spec

    rng = np.random.default_rng(request.seed)
    check = cancel_checker(should_cancel)
    completed = 0
    converged = 0
    cancelled = False
    estimates: dict[str, list[float]] = {name: [] for name in names}
    errors: dict[str, list[float]] = {name: [] for name in names}
    covered: dict[str, int] = dict.fromkeys(names, 0)
    rejected: dict[str, int] = dict.fromkeys(names, 0)
    z_critical = 1.959963984540054  # the two-sided 0.05 normal quantile
    for _ in range(request.runs):
        try:
            check()
        except CancelledFitTool:
            cancelled = True
            break
        if request.noise.kind == "sd":
            y = mean + rng.normal(0.0, request.noise.value, size=x.size)
        else:
            y = mean * (1.0 + rng.normal(0.0, request.noise.value, size=x.size))
        bundle = fit_dataset(_dataset(x, y, weights), fit_spec)
        result = bundle.result
        completed += 1
        if result.status is not FitStatus.CONVERGED:
            continue
        converged += 1
        df = None if result.stats is None else int(result.stats.df)
        by_name = {parameter.name: parameter for parameter in result.parameters}
        for name in names:
            parameter = by_name.get(name)
            if parameter is None:
                continue
            estimates[name].append(float(parameter.value))
            if parameter.std_err is not None:
                errors[name].append(float(parameter.std_err))
            interval = _interval(parameter, df, request.level)
            if interval is not None and interval[0] <= request.truth[name] <= interval[1]:
                covered[name] += 1
        if difference:
            for name in names:
                delta = difference.get(name)
                if delta is None:
                    continue
                parameter = by_name.get(name)
                if parameter is None or parameter.std_err is None:
                    continue
                se = float(parameter.std_err)
                if se <= 0.0:
                    continue
                null_value = request.truth[name] - delta
                z = (float(parameter.value) - null_value) / se
                if abs(z) > z_critical:
                    rejected[name] += 1

    return _result(
        spec=spec,
        model=model,
        request=request,
        names=names,
        points=points,
        x=x,
        completed=completed,
        converged=converged,
        cancelled=cancelled,
        estimates=estimates,
        errors=errors,
        covered=covered,
        rejected=rejected,
        difference=difference,
    )


def _result(
    *,
    spec: FitSpec,
    model: ModelSpec | None,
    request: SimulationRequest,
    names: tuple[str, ...],
    points: tuple[tuple[float, int], ...],
    x: FloatArray,
    completed: int,
    converged: int,
    cancelled: bool,
    estimates: dict[str, list[float]],
    errors: dict[str, list[float]],
    covered: dict[str, int],
    rejected: dict[str, int],
    difference: dict[str, float],
) -> ToolResult:
    """Build the one result from the runs that were completed."""
    means: dict[str, float | None] = {}
    biases: dict[str, float | None] = {}
    empirical: dict[str, float | None] = {}
    reported: dict[str, float | None] = {}
    coverage: dict[str, float | None] = {}
    for name in names:
        values = estimates[name]
        mean_value = float(np.mean(values)) if values else None
        means[name] = mean_value
        biases[name] = None if mean_value is None else mean_value - request.truth[name]
        empirical[name] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        reported[name] = float(np.mean(errors[name])) if errors[name] else None
        coverage[name] = 100.0 * covered[name] / converged if converged else None

    converged_share = 100.0 * converged / completed if completed else 0.0
    recovery_rows = [
        (
            name,
            float(request.truth[name]),
            means[name],
            biases[name],
            empirical[name],
            reported[name],
            coverage[name],
            converged_share,
        )
        for name in names
    ]
    design_rows: list[tuple[float, int]] = [(float(point), count) for point, count in points]
    extrapolation = _extrapolation_note(model, points)

    warnings: list[str] = []
    if completed and (completed - converged) / completed > NON_CONVERGED_WARNING_SHARE:
        non_converged = completed - converged
        warnings.append(
            f"{num(100.0 * non_converged / completed)} % of the {completed} simulated datasets "
            f"did not converge ({non_converged} of {completed}); their estimates are not pooled, "
            f"and a non-convergence share above "
            f"{num(NON_CONVERGED_WARNING_SHARE * 100)} % means the design or the model needs "
            "attention before the experiment is run."
        )
    for name in names:
        share = coverage[name]
        if share is None:
            continue
        gap = abs(share - request.level * 100.0)
        if gap > COVERAGE_WARNING_POINTS:
            warnings.append(
                f"The {num(request.level * 100)} % interval covered the true value in "
                f"{num(share)} % of the converged runs for {name!r}, {num(gap)} points from the "
                "nominal level."
            )
    if cancelled:
        warnings.append(f"cancelled after {completed} of {request.runs} runs")

    noise_evidence = _noise_sentence(request)
    power_tables = []
    if difference:
        power_tables.append(
            table(
                "Power",
                (
                    ("parameter", "Parameter", "text"),
                    ("difference", "Difference", "number"),
                    ("power", "Power", "number"),
                ),
                [
                    (
                        name,
                        float(difference[name]),
                        (rejected[name] / completed) if completed else None,
                    )
                    for name in names
                    if name in difference
                ],
                note=(
                    "Power is the share of the simulated datasets whose two-sided test of "
                    f"H0: theta = true value - difference at alpha = {ALPHA} rejects, using the "
                    "fit's own standard error and the normal quantile. It is the simulated power "
                    "for the stated difference."
                ),
            )
        )
    return ToolResult(
        tool="feasibility",
        title=f"Experiment feasibility for {_model_label(spec, model)}",
        level=request.level,
        estimates=(
            Estimate(label="Runs", value=float(completed)),
            Estimate(label="Simulations that converged %", value=converged_share),
        ),
        assumptions=(
            assumption(
                "noise_model",
                "The simulated noise follows the model the request states",
                Verdict.PASS,
                noise_evidence,
            ),
            assumption(
                "fit_settings",
                "Every fit used the specification's starts and bounds",
                Verdict.PASS,
                "Each dataset was fitted with fit_dataset under the given specification: one "
                "deterministic start, and the bounds the specification sets.",
            ),
        ),
        warnings=tuple(warnings),
        interpretation=_interpretation(
            names, request, completed, converged, means, biases, coverage, cancelled
        ),
        methods_fragment=_methods(spec, request, model, names, x, points, extrapolation),
        counts=Counts(
            n_offered=request.runs,
            n_used=completed,
            n_excluded=request.runs - completed,
        ),
        tables=(
            table(
                "Parameter recovery",
                (
                    ("parameter", "Parameter", "text"),
                    ("true_value", "True value", "number"),
                    ("mean_estimate", "Mean estimate", "number"),
                    ("bias", "Bias", "number"),
                    ("empirical_se", "Empirical SE", "number"),
                    ("mean_reported_se", "Mean reported SE", "number"),
                    ("coverage", "Coverage %", "number"),
                    ("converged", "Converged %", "number"),
                ),
                recovery_rows,
                note=(
                    "Mean estimate, bias, empirical SE and mean reported SE use the converged "
                    f"fits only. Coverage is the share of converged fits whose "
                    f"{num(request.level * 100)} % Wald interval contains the true value; a "
                    "converged fit without an estimable interval counts as not covered."
                ),
            ),
            table(
                "Design",
                (("point", "Point", "number"), ("replicates", "Replicates", "int")),
                design_rows,
                note=extrapolation,
            ),
            *power_tables,
        ),
        plots=(
            box_plot(
                "Parameter estimates",
                "Estimate",
                [(name, np.asarray(estimates[name], dtype=np.float64)) for name in names],
            ),
        ),
    )


__all__ = [
    "DesignSpec",
    "NoiseSpec",
    "SimulationRequest",
    "simulate",
]
