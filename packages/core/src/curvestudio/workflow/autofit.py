"""Open one Auto-fit row as a full ``FitBundle`` (curve, bands, diagnostics, summary).

An Auto-fit result carries every row's numbers and a few sampled curves, not the full
per-fit report. A UI that lets the user open any row calls :func:`bundle_for_candidate`,
which refits that single candidate the way the Auto-fit worker did — same seed, time limit,
number of starts and (for a very large dataset) the same subsample, so the numbers are the
row's own — and assembles the same bundle ``fit_dataset`` returns.
"""

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitResult,
    candidate_ref,
    candidate_seed,
    fit_candidate,
    resolve_spec,
    subsample,
)
from curvestudio.workflow.bundle import FitBundle
from curvestudio.workflow.errors import WorkflowError
from curvestudio.workflow.service import _assemble


def bundle_for_candidate(
    xy: XY,
    result: AutoFitResult,
    candidate_id: str,
    *,
    compact: bool = False,
    n_starts: int | None = None,
    on_subsample: bool | None = None,
) -> FitBundle:
    """Refit candidate ``candidate_id`` of ``result`` on ``xy`` and return its full bundle.

    ``xy`` must be the data the run was made on (checked by fingerprint on the full data).
    The refit is deterministic for the stored seed, so a converged row reproduces its stored
    parameters. The row's own stage is reproduced: its number of starts comes from ``n_starts``
    when given, else from the stored outcome, else from the run's configuration; a row fitted
    on the subsample of a very large dataset is fitted on that same deterministic subsample
    when ``on_subsample`` is ``True`` (or the stored outcome says so), so it reproduces the
    row's numbers rather than a full-data refit. ``compact`` is as for
    :func:`~curvestudio.workflow.fit_dataset`.

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
    stored = next((o for o in result.outcomes if o.candidate_id == candidate_id), None)
    stored_fit = None if stored is None else stored.fit
    # A row kept from the two-start screening stage is reproduced with those two starts.
    starts = n_starts
    if starts is None:
        starts = stored_fit.n_starts if stored_fit is not None else config.n_starts
    # A row of a very large run is reproduced on the same deterministic subsample it was
    # fitted on; the full-data refit rows of that run say ``on_subsample=False``.
    use_subsample = on_subsample
    if use_subsample is None:
        use_subsample = stored.on_subsample if stored is not None else False
    data = subsample(xy) if use_subsample else xy
    fitted = fit_candidate(
        data,
        ref,
        spec,
        seed=candidate_seed(config.seed, candidate_id),
        n_starts=starts,
        time_limit=config.time_limit,
    )
    return _assemble(data, fitted, spec, [], compact)
