"""Report metadata every format shares: tool identity, time, conventions, assumptions.

One source for the JSON header, the XLSX summary and the PDF front matter, so the
formats cannot disagree about who wrote a report or under which conventions.
"""

import platform
from datetime import datetime
from importlib import metadata

from curvestudio.fitting.result import IST
from curvestudio.workflow import FitBundle

TOOL_NAME = "CurveStudio"

_TRACKED_DISTRIBUTIONS = {
    "curvestudio-core": "curvestudio-core",
    "curvestudio-plotting": "curvestudio-plotting",
    "curvestudio-reporting": "curvestudio-reporting",
}

ASSUMPTION_PARAGRAPHS = (
    "The fit minimises the (weighted) sum of squared residuals; when weights are used "
    "they are the data's weight column.",
    "Parameter standard errors, confidence intervals and both band kinds are pointwise "
    "intervals at the level recorded with each band, taken from the covariance at the "
    "solution; for nonlinear models every uncertainty statement is a first-order "
    "(linearised) approximation.",
    "The prediction band adds the residual variance to the mean-response (confidence) "
    "band; the two answer different questions and must not be interchanged.",
    "Information criteria follow the convention recorded with the report; only "
    "differences between models fitted to the same response data mean anything.",
    "Every statistical test carries its own assumptions and limitations; no test, "
    "statistic or ranking certifies a model as scientifically valid.",
    "Predictions outside the observed data range are extrapolation and rely on the "
    "model form alone.",
    "This report does not re-fit or re-compute anything: every number is carried over "
    "from the recorded fit result.",
)


def _version_of(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    """Versions of the packages that produced the report (never empty strings)."""
    versions = {key: _version_of(dist) for key, dist in _TRACKED_DISTRIBUTIONS.items()}
    versions["python"] = platform.python_version()
    return versions


def generated_at() -> str:
    """When the report was written, as an ISO 8601 IST (+05:30) timestamp."""
    return datetime.now(IST).isoformat(timespec="seconds")


def convention_of(bundle: FitBundle) -> str | None:
    """The information-criteria convention of the fit, or ``None`` when undefined.

    ``None`` means the fit did not converge (or was refused), so no criterion was
    computed — the reports say so instead of quoting a convention that was not used.
    """
    return None if bundle.summary is None else bundle.summary.criteria.convention


def assumptions_text() -> str:
    """The assumptions and limitations every report carries, as one paragraph."""
    return " ".join(ASSUMPTION_PARAGRAPHS)
