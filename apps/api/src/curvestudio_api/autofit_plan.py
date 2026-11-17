"""Auto-fit on the API side: the request as an engine configuration, early validation, planning.

No fitting happens here. The handlers use :func:`check_autofit` to turn a request the engine
would refuse into a 422 before a job is queued; the job process uses :func:`plan_counts` to
know how many candidates it will fit (for progress and for the worker count).
"""

import re
from dataclasses import dataclass

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    CustomRef,
    enumerate_candidates,
    resolve_spec,
    screen,
)
from curvestudio.workflow import CustomRef as SpecCustomRef
from curvestudio.workflow import FitSpec
from curvestudio.workflow import PolynomialRef as SpecPolynomialRef
from curvestudio.workflow import RegisteredRef as SpecRegisteredRef
from curvestudio_api.schemas import AutoFitRequest
from curvestudio_api.settings import Settings

_POLYNOMIAL_ID = re.compile(r"poly(\d+)(_origin)?")

CANDIDATE_TIME_LIMIT_S = 20.0
"""Seconds one candidate may take, unless the run's whole budget is shorter."""
FAMILY_PROBE_POINTS = 10_000
"""A data size large enough that no parameter cap hides a family's models."""
FAMILY_PROBE_PARAMETERS = 10


def config_from_request(body: AutoFitRequest, settings: Settings) -> AutoFitConfig:
    """The engine configuration for ``body``, with the server's limits applied.

    The per-candidate time limit and the worker count come from the settings, not from the
    client, so one request cannot take more than its share of the machine.
    """
    return AutoFitConfig(
        rank_by=body.rank_by,
        families=None if body.families is None else tuple(body.families),
        exhaustive=body.exhaustive,
        max_params=body.max_parameters,
        n_starts=body.n_starts,
        seed=body.seed,
        time_limit=min(CANDIDATE_TIME_LIMIT_S, settings.autofit_budget_s),
        workers=settings.autofit_workers,
        cv_folds=body.cv_folds,
        custom_models=tuple(body.custom_models),
        curve_top_n=body.curve_top_n,
    )


def unknown_families(families: tuple[str, ...] | None) -> list[str]:
    """The entries of ``families`` that match no model category (nor ``polynomial``)."""
    unknown: list[str] = []
    for family in families or ():
        probe = AutoFitConfig(
            families=(family,), exhaustive=True, max_params=FAMILY_PROBE_PARAMETERS
        )
        if not enumerate_candidates(probe, FAMILY_PROBE_POINTS):
            unknown.append(family)
    return unknown


def check_autofit(xy: XY, config: AutoFitConfig) -> None:
    """Refuse, before any job exists, a run the engine would refuse or that would fit nothing.

    Cheap by design: it counts the active points and enumerates candidates (a catalogue
    lookup), but does not screen every candidate against the data, which is the job's work.

    Raises:
        AutoFitError: ``TOO_FEW_POINTS``, ``UNKNOWN_FAMILY``, ``INVALID_CUSTOM_MODEL``,
            ``DUPLICATE_MODEL_ID`` or ``NO_CANDIDATES``.
    """
    n_active = screen(xy, []).n_active
    unknown = unknown_families(config.families)
    if unknown:
        raise AutoFitError(
            "UNKNOWN_FAMILY",
            f"no model belongs to the famil{'y' if len(unknown) == 1 else 'ies'} "
            + ", ".join(repr(f) for f in unknown),
        )
    if not enumerate_candidates(config, n_active):
        raise AutoFitError(
            "NO_CANDIDATES",
            "no model fits these settings: widen the families or raise the parameter limit",
        )


@dataclass(frozen=True)
class PlanCounts:
    """How many candidates a run will fit and how many the data cannot support."""

    runnable: int
    skipped: int


def plan_counts(xy: XY, config: AutoFitConfig) -> PlanCounts:
    """Enumerate and screen the candidates exactly as the engine will.

    Raises:
        AutoFitError: as :func:`curvestudio.discovery.enumerate_candidates` and ``screen``.
    """
    n_active = screen(xy, []).n_active
    screening = screen(xy, enumerate_candidates(config, n_active))
    return PlanCounts(runnable=len(screening.runnable), skipped=len(screening.skipped))


def fit_spec_for_candidate(config: AutoFitConfig, candidate_id: str) -> FitSpec:
    """A :class:`FitSpec` naming candidate ``candidate_id`` of a run, stored on the fit opened
    from its row.

    The id alone decides the kind (``polyN``/``polyN_origin``, one of the run's custom models,
    else a registered model), so a registered model this process has not loaded is still named
    correctly. A polynomial through the origin is described by its degree alone: the
    specification has no way to say "through the origin"; the fit's own model id does.

    Raises:
        pydantic.ValidationError: the candidate does not fit a :class:`FitSpec` (a custom
            equation with more parameters than a specification holds).
    """
    polynomial = _POLYNOMIAL_ID.fullmatch(candidate_id)
    if polynomial is not None:
        return FitSpec(model=SpecPolynomialRef(degree=int(polynomial.group(1))))
    for definition in config.custom_models:
        if definition.id == candidate_id:
            model = resolve_spec(CustomRef(definition=definition))
            assert model is not None
            return FitSpec(
                model=SpecCustomRef(
                    equation=definition.expression,
                    parameters=[p.name for p in model.parameters],
                    variable=definition.variables[0],
                )
            )
    return FitSpec(model=SpecRegisteredRef(id=candidate_id))
