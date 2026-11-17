"""One Auto-fit job: fit a candidate, score it, cross-validate it. Runs in a worker process.

Everything here is module-level and picklable. A job carries the data arrays and a
:class:`~curvestudio.discovery.types.CandidateRef`, never a ``ModelSpec`` (those hold lambdas);
the worker rebuilds the model from the registry (which loads the entry-point plugins in the
fresh interpreter) or, for a custom equation, from its definition.

:func:`run_candidate` never raises for a candidate-specific problem: a model that cannot be
built, an optimizer that raises, statistics that overflow all come back as an outcome of kind
``failed`` with the reason, so one bad model cannot end a run of hundreds.
"""

import hashlib
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.discovery.candidates import resolve_spec
from curvestudio.discovery.result import CandidateOutcome
from curvestudio.discovery.types import (
    CandidateMetrics,
    CandidateRef,
    PolynomialRef,
    RegisteredRef,
)
from curvestudio.expressions import to_sympy
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, list_models, load_plugins
from curvestudio.statistics import (
    NonlinearFitter,
    PolynomialFitter,
    StatisticsError,
    information_criteria,
    k_fold_cv,
)
from curvestudio.statistics.validation import Fitter

FloatArray = NDArray[np.float64]

DEGENERATE_CONDITION = 1e8
"""A fit whose column-normalised Jacobian is worse conditioned than this, with a parameter
at a bound, has collapsed onto a simpler model."""

DEGENERATE_EFFECT = 1e-6
"""A parameter at a bound whose ``|value| * rms(df/dparameter)`` is below this fraction of the
data range does nothing measurable (a rate of 1e-8 over an x span of 10, a zero amplitude)."""

BOUND_TOLERANCE = 1e-6
"""Relative distance (to ``max(1, |bound|)``) within which a value counts as at its bound."""

PARENT_POLL_SECONDS = 1.0
PR_SET_PDEATHSIG = 1

_NOTE_LIMIT = 3
"""How many cross-validation fold failures a note spells out."""


def candidate_seed(seed: int, candidate_id: str) -> int:
    """The seed of one candidate: ``sha256(f"{seed}:{id}")`` folded to 32 bits.

    Derived from the run seed and the id alone, so a candidate's numbers do not depend on
    which worker ran it, in which order, or which other candidates were in the run.
    """
    digest = hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest()
    return int.from_bytes(digest[:8]) % 2**32


@dataclass(frozen=True)
class CandidateJob:
    """Everything one worker needs to fit and score one candidate."""

    x: FloatArray
    y: FloatArray
    weights: FloatArray | None
    x_name: str
    y_name: str
    fingerprint: str
    ref: CandidateRef
    seed: int
    n_starts: int
    time_limit: float
    cv_folds: int | None
    stage: Literal["screen", "final"] = "final"
    start: dict[str, float] | None = None


def _watch_parent(parent_pid: int) -> None:
    """Exit this worker as soon as the process that started it is gone."""
    while True:
        time.sleep(PARENT_POLL_SECONDS)
        if os.getppid() != parent_pid:
            os._exit(1)


def init_worker(parent_pid: int, use_death_signal: bool = False) -> None:
    """Pool initialiser: die with the parent, then load the model plugins.

    A worker must not outlive a parent that was killed (``SIGKILL``, the OOM killer): it would
    keep a core busy until its candidate ends, or forever. A daemon thread polls the parent
    pid and exits when it changes; with ``use_death_signal`` (Linux, and only when the pool
    was started from the parent's main thread, because the kernel ties the signal to the
    *thread* that created the child) the kernel also sends ``SIGKILL`` at parent death, which
    works even while the worker is stuck inside native code and holds the interpreter lock.
    """
    if use_death_signal:
        _set_parent_death_signal()
    if os.getppid() != parent_pid:  # the parent died before this worker came up
        os._exit(1)
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    load_plugins()


def _set_parent_death_signal() -> None:
    """Ask the Linux kernel for ``SIGKILL`` when the parent dies; a no-op elsewhere."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except OSError, AttributeError:  # best effort: the polling thread still covers it
        return


def describe(ref: CandidateRef, spec: ModelSpec | None) -> tuple[str, str]:
    """``(name, category)`` of a candidate."""
    if isinstance(ref, PolynomialRef):
        suffix = ", through the origin" if ref.through_origin else ""
        return f"Polynomial of degree {ref.degree}{suffix}", "polynomial"
    assert spec is not None
    return spec.name, spec.category


def shape_parameter_count(spec: ModelSpec | None) -> int:
    """How many parameters are free *frequencies*: they multiply ``x`` inside ``sin``, ``cos``
    or ``tan``.

    Searching a frequency fits noise much more easily than a linear coefficient does (the
    likelihood has many local maxima, so the best of them is biased), so the ranking charges
    each one as an extra effective parameter (see
    :func:`~curvestudio.discovery.ranking.ranking_penalty`). Exponents (``x**p``) are not
    counted: a power law is a legitimate model whose exponent is its whole point, and charging
    it would push true power laws down the list. Rates (``exp(-b*x)``) and phases are not
    counted either. Polynomials have none; an expression this check cannot read counts as none.
    """
    if spec is None:
        return 0
    names = (*spec.variables, *(p.name for p in spec.parameters))
    try:
        symbols = {n: sympy.Symbol(n, real=True) for n in names}
        expr = to_sympy(spec.expression, symbols)
    except Exception:
        return 0
    x = symbols[spec.variables[0]]
    params = {symbols[p.name] for p in spec.parameters}
    found: set[sympy.Symbol] = set()
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.sin | sympy.cos | sympy.tan):
            found |= params & sympy.diff(node.args[0], x).free_symbols
    return len(found)


def is_generated(ref: CandidateRef) -> bool:
    """Whether ``ref`` is a model of a lazily generated family, not a hand-written one."""
    return isinstance(ref, RegisteredRef) and ref.model_id not in {m.id for m in list_models()}


def fit_candidate(
    data: XY,
    ref: CandidateRef,
    spec: ModelSpec | None,
    *,
    seed: int,
    n_starts: int,
    time_limit: float | None,
    start: dict[str, float] | None = None,
) -> FitResult:
    """Fit one candidate exactly as a job does (shared with the "open this row" helper).

    ``start`` overrides the model's initial guess (a warm start from an earlier fit).
    """
    if isinstance(ref, PolynomialRef):
        return fit_polynomial(data, ref.degree, through_origin=ref.through_origin)
    assert spec is not None
    config = FitConfig(n_starts=n_starts, seed=seed, time_limit=time_limit, initial=start)
    return fit(spec, data, config)


def run_candidate(job: CandidateJob) -> CandidateOutcome:
    """Fit, score and cross-validate one candidate; see the module docstring.

    Outcome kinds: ``converged`` (metrics filled), ``exact_fit`` (an exact fit: no AICc/BIC),
    ``not_converged`` (the optimizer did not reach a valid optimum), ``timed_out`` (the
    cooperative time limit ended it before it converged), ``failed`` (anything raised, or
    the optimizer failed) and ``skipped`` (the model refused the data). A fit that converged
    before the time limit is kept, with the fit's own warning. Only ``KeyboardInterrupt``
    and ``SystemExit`` pass through.
    """
    started = time.perf_counter()
    name, category = job.ref.id, ""
    try:
        spec = resolve_spec(job.ref)
        name, category = describe(job.ref, spec)
        outcome = _score(job, spec, name, category)
    except Exception as exc:  # reported as a failed candidate, never swallowed
        outcome = CandidateOutcome(
            candidate_id=job.ref.id,
            name=name,
            category=category,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    return outcome.model_copy(update={"elapsed_seconds": time.perf_counter() - started})


def degenerate_reason(spec: ModelSpec | None, data: XY, fit_result: FitResult) -> str | None:
    """Why the optimum of ``fit_result`` is degenerate, or ``None`` when it is not.

    A fit is degenerate when some parameter sits at a bound of the model (within
    ``BOUND_TOLERANCE``) and either has a negligible effect (``|value| * rms(df/dparameter)``
    below ``DEGENERATE_EFFECT`` of the data range, as a rate of 1e-8 does over an x span of
    10) or the column-normalised Jacobian is singular (condition above ``DEGENERATE_CONDITION``):
    the model has collapsed onto a simpler one, whose curve it merely copies. Polynomials are
    never degenerate. Needs a finite Jacobian at the optimum, else ``None``.
    """
    if spec is None:
        return None
    values = {p.name: p.value for p in fit_result.parameters}
    at_bound = [
        p.name
        for p in spec.parameters
        if any(
            math.isfinite(b) and abs(values[p.name] - b) <= BOUND_TOLERANCE * max(1.0, abs(b))
            for b in (p.lower, p.upper)
        )
    ]
    if not at_bound:
        return None
    active = np.ones(data.x.size, dtype=bool) if data.weights is None else data.weights > 0.0
    x, y = data.x[active], data.y[active]
    with np.errstate(all="ignore"):
        jac = np.asarray(spec.jacobian(x, **values), dtype=np.float64)
    if not bool(np.all(np.isfinite(jac))):
        return None
    names = [p.name for p in spec.parameters]
    spread = float(np.ptp(y))
    rms = np.sqrt(np.mean(jac * jac, axis=0))
    for name in at_bound:
        effect = abs(values[name]) * float(rms[names.index(name)])
        if effect <= DEGENERATE_EFFECT * spread:
            return (
                f"parameter {name} is at its bound ({values[name]:.3g}) and changes the curve "
                f"by at most {effect:.2g}, under {DEGENERATE_EFFECT:g} of the data range: the "
                "model has collapsed onto a simpler one"
            )
    norms = np.linalg.norm(jac, axis=0)
    if bool(np.all(norms > 0.0)):
        condition = float(np.linalg.cond(jac / norms))
        if condition > DEGENERATE_CONDITION:
            return (
                f"parameter {', '.join(at_bound)} is at its bound and the Jacobian is singular "
                f"(condition {condition:.2g}): the parameters are not identifiable"
            )
    return None


def _score(job: CandidateJob, spec: ModelSpec | None, name: str, category: str) -> CandidateOutcome:
    ref = job.ref
    data = XY(job.x, job.y, job.weights, job.x_name, job.y_name, job.fingerprint)
    seed = candidate_seed(job.seed, ref.id)
    result = fit_candidate(
        data,
        ref,
        spec,
        seed=seed,
        n_starts=job.n_starts,
        time_limit=job.time_limit,
        start=job.start,
    )

    def outcome(
        kind: str, reason: str, metrics: CandidateMetrics | None = None
    ) -> CandidateOutcome:
        return CandidateOutcome.model_validate(
            {
                "candidate_id": ref.id,
                "name": name,
                "category": category,
                "outcome": kind,
                "reason": reason,
                "fit": result,
                "metrics": metrics,
                "stage": job.stage,
                "generated": is_generated(ref),
                "shape_parameters": shape_parameter_count(spec),
            }
        )

    if result.status is not FitStatus.CONVERGED:
        reason = result.termination_reason
        if result.status is FitStatus.NOT_CONVERGED:
            kind = "timed_out" if "time limit" in reason else "not_converged"
        elif result.status is FitStatus.INVALID_INPUT:
            kind = "skipped"
        else:
            kind = "failed"
        return outcome(kind, reason)
    stats = result.stats
    if stats is None:
        return outcome("failed", "the converged fit carries no statistics")

    reason_degenerate = degenerate_reason(spec, data, result)
    if reason_degenerate is not None:
        result = result.model_copy(
            update={"warnings": (*result.warnings, f"DEGENERATE_FIT: {reason_degenerate}")}
        )
    exact = stats.sse == 0.0
    aicc: float | None = None
    bic: float | None = None
    if not exact:
        try:
            criteria = information_criteria(result)
        except StatisticsError as exc:
            if exc.code != "PERFECT_FIT":
                return outcome("failed", f"{exc.code}: {exc}")
            exact = True
        else:
            aicc, bic = criteria.aicc, criteria.bic
    cv_rmse, note = _cross_validate(job, data, spec, result, seed, stats.n)
    metrics = CandidateMetrics(
        k=stats.k,
        n=stats.n,
        r2=stats.r2,
        adj_r2=stats.adj_r2,
        rmse=stats.rmse,
        aicc=aicc,
        bic=bic,
        cv_rmse=cv_rmse,
    )
    if exact:
        reason = "exact fit (SSE = 0): information criteria are not defined"
        return outcome("exact_fit", "; ".join(filter(None, (reason, note))), metrics)
    return outcome("converged", note, metrics)


def _cross_validate(
    job: CandidateJob,
    data: XY,
    spec: ModelSpec | None,
    result: FitResult,
    seed: int,
    n_active: int,
) -> tuple[float | None, str]:
    """``(cv_rmse, note)``: the RMSE is ``None`` with a note when CV is incomplete or fails."""
    if job.cv_folds is None:
        return None, ""
    fitter: Fitter
    if isinstance(job.ref, PolynomialRef):
        fitter = PolynomialFitter(job.ref.degree, through_origin=job.ref.through_origin)
    else:
        assert spec is not None
        start = {p.name: p.value for p in result.parameters}
        config = FitConfig(seed=seed, time_limit=job.time_limit)
        fitter = NonlinearFitter(spec, config, start=start)
    try:
        cv = k_fold_cv(data, fitter, folds=min(job.cv_folds, n_active), seed=seed)
    except Exception as exc:  # the fit stays valid; the missing CV is explained, not hidden
        return None, f"cross-validation failed: {type(exc).__name__}: {exc}"
    if cv.rmse is not None:
        return cv.rmse, ""
    shown = "; ".join(cv.reasons[:_NOTE_LIMIT])
    more = len(cv.reasons) - _NOTE_LIMIT
    return None, f"cross-validation incomplete: {shown}" + (f" (+{more} more)" if more > 0 else "")
