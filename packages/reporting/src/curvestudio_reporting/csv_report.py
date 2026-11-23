"""CSV report tables: parameters, observations and statistics.

Three plain CSV tables instead of one multi-section file, so each is directly
loadable into any tool. Floats are written with ``repr`` (shortest round trip),
undefined numbers are empty cells with a note, and any text cell that could be
read as a spreadsheet formula is neutralised.
"""

import csv
import io
import re
from collections.abc import Iterable, Sequence

from curvestudio.fitting import FitStatistics, FitStatus
from curvestudio.statistics import CrossValidation, InformationCriteria
from curvestudio.workflow import DataSnapshot, FitBundle

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")
_FORMULA_SIGNS = ("=", "+", "-", "@")
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
"""Characters XML 1.0 forbids; openpyxl refuses any cell containing one of them."""

type StatValue = float | int | str | None
type _StatisticRow = tuple[str, StatValue, str]
"""(name, value, note); a numeric value of ``None`` means "not defined"."""


def neutralise(text: str) -> str:
    """Prefix ``'`` so a text cell can never be executed as a spreadsheet formula.

    Cells starting with ``= + - @``, a tab or a newline are treated as
    formulas (or as list/paragraph commands) by Excel/LibreOffice/Sheets; a leading
    apostrophe keeps them text while staying visible in exports. Whitespace before
    the dangerous character does not help — spreadsheets skip leading blanks — so
    ``" =1+1"`` and ``"\\n=1"`` are treated as formulas too, while plain indented
    text ("  day") is left alone.
    """
    if text.startswith(_DANGEROUS_PREFIXES) or text.lstrip().startswith(_FORMULA_SIGNS):
        return f"'{text}"
    return text


def sanitise(text: str) -> str:
    """Strip XML-illegal control characters (``\\x00``, ``\\x1b``, ...).

    They carry no meaning in a report and crash the XLSX writer (openpyxl rejects
    them) and the PDF paragraph parser; the legal ``\\t``, ``\\n`` and ``\\r`` stay.
    """
    return _XML_ILLEGAL.sub("", text)


def safe_text(text: str) -> str:
    """The standard cell treatment everywhere text is written: sanitise + neutralise."""
    return neutralise(sanitise(text))


PREPARED_BY = "Prepared by "
"""The words each report puts in front of the identity. ``curvestudio_plotting`` carries the
same literal separately: the import contracts forbid plotting importing reporting."""


def byline_text(byline: str | None) -> str | None:
    """The identity a report's byline names, or ``None`` when there is none.

    ``byline`` is the identity text alone ("Alice", "Lab" or "Alice, Lab"), never the
    "Prepared by" words: each renderer adds those, here and in the web report. Unset,
    empty and whitespace-only all mean no byline. It is user-typed text like any other,
    so control characters are stripped and the surrounding whitespace trimmed.
    """
    if byline is None:
        return None
    return sanitise(byline).strip() or None


def subsampled_note(data: DataSnapshot) -> str | None:
    """The visible note when the exported rows are a subsample of what was fitted.

    ``None`` means the export shows every row. The exact wording is shared by the
    CSV, XLSX and PDF reports so no format claims completeness the others deny.
    """
    shown = len(data.x)
    if data.subsampled or data.n_total > shown:
        return f"showing {shown} of {data.n_total} observations (subsampled)"
    return None


def _number(value: float | int | None) -> str:
    return "" if value is None else repr(value)


def _cell(value: StatValue) -> str:
    if value is None:
        return ""
    if isinstance(value, float | int):
        return repr(value)
    return safe_text(value)


def aligned(values: list[float], rows: list[float]) -> list[float | None]:
    """``values`` padded with ``None`` where it does not line up with ``rows``.

    A compact bundle's data snapshot may be a subsample of the rows the fit was
    computed on; a failed fit may have no per-row values at all. Either way the
    report never pairs an x with a fitted value from another row.
    """
    if len(values) == len(rows):
        return list(values)
    return [None] * len(rows)


def _write_csv(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    """Rows joined as CSV with ``\n`` line endings (platform-independent)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(body)
    return buffer.getvalue()


def parameters_csv(bundle: FitBundle) -> str:
    """One row per fitted parameter: value, standard error and 95 % CI, or empty."""
    body = [
        [
            safe_text(estimate.name),
            _number(estimate.value),
            _number(estimate.std_err),
            _number(estimate.ci_low),
            _number(estimate.ci_high),
        ]
        for estimate in bundle.result.parameters
    ]
    return _write_csv(["parameter", "value", "std_err", "ci_low", "ci_high"], body)


def observations_csv(bundle: FitBundle) -> str:
    """The fitted data: x, y, weight, fitted value and residual per observation.

    ``fitted``/``residual`` stay empty when the bundle's data snapshot is a
    subsample (compact bundle) or the fit never produced them: no row is ever
    paired with numbers from a different length array. When the snapshot is a
    subsample, a final note row says how many of the fitted rows are shown.
    """
    data = bundle.data
    fitted = aligned(bundle.result.fitted, data.x)
    residuals = aligned(bundle.result.residuals, data.x)
    weights = [None] * len(data.x) if data.weights is None else list(data.weights)
    body = [
        [_number(x), _number(y), _number(weight), _number(fit_at_x), _number(residual)]
        for x, y, weight, fit_at_x, residual in zip(
            data.x, data.y, weights, fitted, residuals, strict=True
        )
    ]
    if (note := subsampled_note(data)) is not None:
        body.append([safe_text(note), "", "", "", ""])
    return _write_csv(
        [safe_text(data.x_name), safe_text(data.y_name), "weight", "fitted", "residual"], body
    )


def statistics_csv(bundle: FitBundle) -> str:
    """Fit statistics as (statistic, value, note) rows; undefined means empty + note."""
    body = [
        [safe_text(name), _cell(value), safe_text(note)]
        for name, value, note in statistics_rows(bundle)
    ]
    return _write_csv(["statistic", "value", "note"], body)


def statistics_rows(bundle: FitBundle) -> list[_StatisticRow]:
    """The statistics of one bundle, in a fixed order, shared by every format.

    The ``status`` row always comes first; a fit that did not converge is labelled
    with its reason there, and the statistics it cannot have (criteria,
    cross-validation) are ``None`` with a "not defined" note rather than a number.
    """
    result = bundle.result
    not_converged = result.status is not FitStatus.CONVERGED
    refusal = (
        f"not defined: the fit is {result.status.value} ({result.termination_reason})"
        if not_converged
        else None
    )
    rows: list[_StatisticRow] = [
        (
            "status",
            result.status.value,
            result.termination_reason if not_converged else "the fit converged",
        )
    ]

    stats = result.stats
    if stats is None:
        undefined = refusal or "not defined"
        for name in ("n", "k", "df", "sse", "mse", "rmse", "mae", "r2", "adj_r2", "residual_se"):
            rows.append((name, None, undefined))
    else:
        rows.extend(_fit_rows(stats))

    summary = bundle.summary
    criteria = None if summary is None else summary.criteria
    cv = None if summary is None else summary.cv
    rows.extend(_criteria_rows(criteria, refusal))
    rows.extend(_cv_rows(cv, refusal))
    return rows


_CONSTANT_RESPONSE = "not defined: the response is constant (zero total sum of squares)"
_NO_DF = "not defined: no residual degrees of freedom"


def _fit_rows(stats: FitStatistics) -> list[_StatisticRow]:
    r2_note = f"{stats.r2_definition} definition"
    rows: list[_StatisticRow] = [
        ("n", stats.n, ""),
        ("k", stats.k, ""),
        ("df", stats.df, ""),
        ("sse", stats.sse, ""),
        ("mse", stats.mse, ""),
        ("rmse", stats.rmse, ""),
        ("mae", stats.mae, ""),
    ]
    # FitStatistics leaves r2 undefined exactly for a constant response (sst = 0)
    # with a non-perfect fit; a perfect fit is reported as r2 = 1.
    rows.append(("r2", stats.r2, r2_note if stats.r2 is not None else _CONSTANT_RESPONSE))
    if stats.adj_r2 is None:
        reason = _NO_DF if stats.r2 is not None else _CONSTANT_RESPONSE
        rows.append(("adj_r2", None, reason))
    else:
        rows.append(("adj_r2", stats.adj_r2, r2_note))
    rows.append(("residual_se", stats.residual_se, "" if stats.residual_se is not None else _NO_DF))
    return rows


def _criteria_rows(
    criteria: InformationCriteria | None, refusal: str | None
) -> list[_StatisticRow]:
    if criteria is None:
        undefined = refusal or "not defined"
        return [("aic", None, undefined), ("aicc", None, undefined), ("bic", None, undefined)]
    return [
        ("aic", criteria.aic, "Gaussian likelihood convention (see report header)"),
        (
            "aicc",
            criteria.aicc,
            ""
            if criteria.aicc is not None
            else "not defined: n - k' - 1 <= 0, so the small-sample correction divides by zero",
        ),
        ("bic", criteria.bic, "Gaussian likelihood convention (see report header)"),
    ]


def _cv_rows(cv: CrossValidation | None, refusal: str | None) -> list[_StatisticRow]:
    if cv is None:
        if refusal is not None:
            return [("cv_rmse", None, refusal)]
        return [("cv_rmse", None, "not run for this fit")]
    note = ""
    if cv.rmse is None:
        note = f"not defined: {cv.n_failed} of {cv.folds} folds failed ({'; '.join(cv.reasons)})"
    elif cv.status == "incomplete":
        note = f"incomplete: {cv.n_failed} of {cv.folds} folds failed"
    return [("cv_rmse", cv.rmse, note)]
