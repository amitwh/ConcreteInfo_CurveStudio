"""Open one Auto-fit row as a full ``FitBundle`` (curve, bands, diagnostics, summary).

An Auto-fit result carries every row's numbers and a few sampled curves, not the full
per-fit report. A UI that lets the user open any row calls :func:`bundle_for_candidate`,
which refits that single candidate the way the Auto-fit worker did (same seed, starts and
time limit, so the numbers are the row's own) and assembles the same bundle ``fit_dataset``
returns.
"""

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitResult,
    candidate_ref,
    candidate_seed,
    fit_candidate,
    resolve_spec,
)
from curvestudio.workflow.bundle import FitBundle
from curvestudio.workflow.errors import WorkflowError
from curvestudio.workflow.service import _assemble


def bundle_for_candidate(
    xy: XY, result: AutoFitResult, candidate_id: str, *, compact: bool = False
) -> FitBundle:
    """Refit candidate ``candidate_id`` of ``result`` on ``xy`` and return its full bundle.

    ``xy`` must be the data the run was made on (checked by fingerprint). The refit is
    deterministic for the stored seed, so a converged row reproduces its stored parameters.
    ``compact`` is as for :func:`~curvestudio.workflow.fit_dataset`.

    Raises:
        WorkflowError: ``DATA_MISMATCH`` (``xy`` is not the run's data), ``UNKNOWN_MODEL``
            (no such candidate: not a polynomial, not one of the run's custom models, not
            a registered model) or ``INVALID_SPEC`` (a custom model that cannot be built).
    """
    if xy.fingerprint != result.dataset_fingerprint:
        raise WorkflowError(
            "DATA_MISMATCH", "the data are not the data this Auto-fit run was made on"
        )
    config = result.config
    try:
        ref = candidate_ref(config, candidate_id)
        spec = resolve_spec(ref)
    except KeyError as exc:
        raise WorkflowError("UNKNOWN_MODEL", str(exc.args[0])) from exc
    except ValueError as exc:
        raise WorkflowError("INVALID_SPEC", str(exc)) from exc
    fitted = fit_candidate(
        xy,
        ref,
        spec,
        seed=candidate_seed(config.seed, candidate_id),
        n_starts=config.n_starts,
        time_limit=config.time_limit,
    )
    return _assemble(xy, fitted, spec, [], compact)
