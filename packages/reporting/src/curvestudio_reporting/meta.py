"""Report metadata every format shares: tool identity, time, conventions, assumptions.

One source for the JSON header, the XLSX summary and the PDF front matter, so the
formats cannot disagree about who wrote a report or under which conventions. The same
holds for what the PDF and the XLSX say of a fit's audit and methods text.
"""

import platform
from datetime import datetime
from importlib import metadata

from curvestudio.fitting.result import IST
from curvestudio.tools import AuditCheck, FitAudit, MethodsText, Verdict
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


# -- what a report says of a fit's audit and methods text ------------------------------------------

FLAGGED_VERDICTS = (Verdict.CAUTION, Verdict.FAIL)
"""The audit's checks a report lists one by one. A passed or not-applicable check prints
nothing of its own: the audit's own summary says that every check that could run passed."""


def methods_paragraph(methods: MethodsText | None, audit: FitAudit | None) -> str:
    """The methods paragraph followed by the audit's methods fragment, as one text.

    ``""`` when neither is given, so a report given neither adds no row or section. What is
    not given is left out: the paragraph alone is written without the fragment, and the
    fragment alone without the paragraph.
    """
    given = (
        None if methods is None else methods.paragraph,
        None if audit is None else audit.methods_fragment,
    )
    return " ".join(text for text in given if text is not None)


def flagged_checks(audit: FitAudit) -> tuple[AuditCheck, ...]:
    """The audit's checks that read caution or fail, in the audit's order."""
    return tuple(check for check in audit.checks if check.verdict in FLAGGED_VERDICTS)
