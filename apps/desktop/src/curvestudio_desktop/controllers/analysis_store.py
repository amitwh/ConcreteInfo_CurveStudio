"""Mapping between ``Analysis`` entries and the results the analysis controller shows.

An Auto-fit run is stored as one analysis per dataset (kind ``autofit``): the slim JSON form
of :class:`~curvestudio.discovery.AutoFitResult` (rows, metrics, reasons and sampled curves;
no per-point vectors), which :meth:`AutoFitResult.ranked_by` still re-ranks after a reopen.
A manual fit or an opened Auto-fit row is stored as kind ``fit``: the ``FitResult`` itself in
``analysis.result`` and everything else the bundle carries (summary, curve, bands, snapshot)
plus the :class:`~curvestudio.workflow.FitSpec` under ``analysis.config``, so Save/Open
round-trips a fit and the Predict tab still has its specification.
"""

import json
import re

from curvestudio.discovery import AutoFitResult
from curvestudio.project import Analysis
from curvestudio.workflow import FitBundle, FitSpec

AUTOFIT_PREFIX = "autofit-"
FIT_PREFIX = "fit-"
_ID_MAX = 200
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
"""Analysis ids are one archive path segment; anything else becomes a hyphen."""


def safe_analysis_id(prefix: str, *parts: str) -> str:
    """A valid, bounded analysis id built from ``prefix`` and ``parts``.

    Collisions are impossible between different kinds (distinct prefixes) and between
    datasets and candidates (both sanitized into the id); a re-run replaces by id.
    """
    joined = "-".join(part for part in (prefix, *parts) if part)
    slug = _SAFE.sub("-", joined).strip("-")[:_ID_MAX] or "analysis"
    if not slug[0].isalnum():
        slug = f"a{slug}"[: _ID_MAX + 1]
    return slug


def unique_fit_id(taken: set[str]) -> str:
    """The next free ``fit-N`` id among ``taken`` (manual fits get a fresh one each time)."""
    number = 1
    while f"{FIT_PREFIX}{number}" in taken:
        number += 1
    return f"{FIT_PREFIX}{number}"


def autofit_analysis_id(dataset_id: str) -> str:
    return safe_analysis_id(AUTOFIT_PREFIX, dataset_id)


def partial_autofit_analysis_id(dataset_id: str) -> str:
    """A cancelled run's own id — it never takes the canonical per-dataset slot."""
    return safe_analysis_id(AUTOFIT_PREFIX, dataset_id, "partial")


def _autofit_analysis(
    analysis_id: str, dataset_id: str, result: AutoFitResult, note: str
) -> Analysis:
    return Analysis(
        id=analysis_id,
        dataset_id=dataset_id,
        kind="autofit",
        config={"config": result.config.model_dump(mode="json")},
        result=json.loads(result.to_json(include_fits=False)),
        note=note,
    )


def autofit_analysis(dataset_id: str, result: AutoFitResult) -> Analysis:
    """The stored form of one Auto-fit run: slim JSON, re-rankable, curves included."""
    return _autofit_analysis(
        autofit_analysis_id(dataset_id),
        dataset_id,
        result,
        f"Auto-fit: {len(result.ranked)} fitted candidates, "
        f"{len(result.unconverged)} not converged, {len(result.skipped)} skipped",
    )


def partial_autofit_analysis(dataset_id: str, result: AutoFitResult) -> Analysis:
    """A cancelled run under its own id: flagged partial, shown with its notice."""
    return _autofit_analysis(
        partial_autofit_analysis_id(dataset_id),
        dataset_id,
        result,
        f"Auto-fit (partial, cancelled): {len(result.ranked)} fitted candidates, "
        f"{len(result.unconverged)} not converged, {len(result.skipped)} never attempted",
    )


def is_partial_autofit(analysis: Analysis) -> bool:
    """Whether a stored autofit analysis is a cancelled partial run."""
    return (
        analysis.kind == "autofit"
        and isinstance(analysis.result, dict)
        and bool(analysis.result.get("cancelled"))
    )


def autofit_from_analysis(analysis: Analysis) -> AutoFitResult:
    """Rebuild the stored Auto-fit result (re-ranking works; per-point vectors are gone)."""
    if not isinstance(analysis.result, dict):  # the project loader guarantees this shape
        raise ValueError(f"analysis {analysis.id!r} does not hold an Auto-fit result")
    return AutoFitResult.model_validate(analysis.result)


def fit_analysis(
    dataset_id: str,
    spec: FitSpec,
    bundle: FitBundle,
    *,
    analysis_id: str,
    source: dict[str, str] | None = None,
) -> Analysis:
    """The stored form of one fit: the ``FitResult`` plus the rest of its bundle and spec."""
    body = bundle.model_dump(mode="json")
    body.pop("result")
    config: dict[str, object] = {"spec": spec.model_dump(mode="json"), "bundle": body}
    if source is not None:
        config["source"] = source
    return Analysis(
        id=analysis_id,
        dataset_id=dataset_id,
        kind="fit",
        config=config,
        result=bundle.result,
        note=bundle.result.model_id,
    )


def bundle_from_analysis(analysis: Analysis) -> tuple[FitBundle, FitSpec]:
    """Rebuild the stored bundle and specification of a fit analysis.

    Raises:
        ValueError: the analysis is not a fit analysis or its stored parts do not validate
            (a hand-edited archive); callers treat this as "cannot restore", never fatal.
    """
    result = analysis.result
    if isinstance(result, dict):  # the project loader guarantees a FitResult for kind="fit"
        raise ValueError(f"analysis {analysis.id!r} stores JSON, not a fit result")
    try:
        spec = FitSpec.model_validate(analysis.config["spec"])
        body = dict(analysis.config["bundle"])
        body["result"] = result.model_dump(mode="json")
        return FitBundle.model_validate(body), spec
    except (KeyError, ValueError) as exc:
        raise ValueError(f"analysis {analysis.id!r} cannot be restored: {exc}") from exc


__all__ = [
    "AUTOFIT_PREFIX",
    "FIT_PREFIX",
    "autofit_analysis",
    "autofit_analysis_id",
    "autofit_from_analysis",
    "bundle_from_analysis",
    "fit_analysis",
    "is_partial_autofit",
    "partial_autofit_analysis",
    "partial_autofit_analysis_id",
    "safe_analysis_id",
    "unique_fit_id",
]
