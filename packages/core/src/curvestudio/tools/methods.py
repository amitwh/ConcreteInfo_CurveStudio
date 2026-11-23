"""The methods text: the paragraph, equation, table and citations an author pastes into a paper.

:func:`methods_text` reads a :class:`~curvestudio.workflow.FitBundle` and returns a
:class:`MethodsText` in one of three styles. It fits nothing, reads and writes no file and
touches no network. Every number in the text is one the fit result carries, rounded by one
rule, or one of a few named counts and settings: the observations used, the interval level,
the seed and the version. Nothing is guessed: what the text needed and did not get is listed
in ``missing``, each entry a code and a sentence.

The module follows its seams: :mod:`._methods_numbers` names the model that was fitted and
rounds its estimates; this module writes the paragraph, lays out the equation and the table
in each style, and holds the public names. The paragraph is the same text in every style,
escaped for LaTeX in that style. What only the author knows (the instrument, the sample
size, the cleaning) is listed in ``to_complete``, never written for them.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

import curvestudio
from curvestudio.fitting import FitResult, FitStatus
from curvestudio.tools._common import require_matching_spec, split_rows
from curvestudio.tools._errors import ToolError
from curvestudio.tools._methods_numbers import (
    VALUE_DIGITS,
    _fitted_model,
    _Model,
    _show,
    _Shown,
    _significant,
)
from curvestudio.tools._wording import joined
from curvestudio.workflow import FitBundle, FitSpec

Style = Literal["text", "markdown", "latex"]
STYLES: tuple[Style, ...] = ("text", "markdown", "latex")

MAKER = "ConcreteInfo"
ADDRESS = "https://curvestudio.concreteinfo.cloud"
TO_COMPLETE: tuple[str, ...] = (
    "Instrument and measurement method",
    "How the sample size was chosen",
    "How the data were cleaned",
)
NO_SOURCE = {
    "polynomial": "A polynomial model; no source is cited for it.",
    "custom": "A custom equation; no source is cited for it.",
}
"""What ``citation_model`` says for a model that has no reference to cite."""

NOT_DETERMINED = "not determined"
NO_STANDARD_ERROR_TEXT = "(standard error not determined)"

LOSSES = {"soft_l1": "soft L1", "huber": "Huber", "cauchy": "Cauchy", "arctan": "arctan"}
"""The core's loss names other than ``linear``, as a sentence writes them."""
METHODS = {
    "lm": "Levenberg-Marquardt",
    "trf": "trust-region reflective",
    "dogbox": "dogleg",
    "nelder": "Nelder-Mead",
    "powell": "Powell",
    "bfgs": "BFGS",
    "lbfgsb": "L-BFGS-B",
    "differential_evolution": "differential evolution",
    "dual_annealing": "dual annealing",
}
"""The names the core gives its solvers (``FitResult.method``), as a sentence writes them."""

_LATEX_TEXT = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
)
"""Escapes for the characters LaTeX reserves (a bare ``%`` comments out the rest of a
line, an underscore outside mathematics does not compile), so that a sentence, or an
equation written in the core's syntax, stands in LaTeX text."""


class MethodsText(BaseModel):
    """The methods text of one fit, in one style.

    ``paragraph`` states what was fitted and how: the same text in every style, escaped for
    LaTeX in the ``latex`` style. ``equation`` and ``table`` give the estimates with their
    standard errors and confidence intervals. ``citation_curvestudio``
    cites the software and ``citation_model`` the model's sources, one per string.
    ``to_complete`` lists what only the author can write, and ``missing`` what the text needed
    and did not get: each entry is a code, a colon and a sentence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    paragraph: str
    equation: str
    table: str
    citation_curvestudio: str
    citation_model: tuple[str, ...]
    to_complete: tuple[str, ...]
    missing: tuple[str, ...]


class _Missing:
    """What the text needed and did not get, kept in the order the paragraph comes to it."""

    def __init__(self) -> None:
        self._sentences: dict[str, str] = {}

    def add(self, code: str, sentence: str) -> None:
        self._sentences[code] = sentence

    def lines(self) -> tuple[str, ...]:
        return tuple(f"{code}: {text}" for code, text in self._sentences.items())


# -- the paragraph --------------------------------------------------------------------------------


def _weighting(
    result: FitResult, spec: FitSpec | None, carries_weights: bool
) -> Literal["ordinary", "weighted", ""]:
    """Whether the fit used weights: ``""`` when nothing says, so the text does not either.

    A polynomial's result says it itself (``wls`` or ``ols``), and that record stands over any
    specification. A nonlinear result does not record it, so the specification is asked; with
    neither, weights in the data leave it unknown.
    """
    if result.method == "wls":
        return "weighted"
    if result.method == "ols":
        return "ordinary"
    if spec is not None:
        return "weighted" if spec.use_weights and carries_weights else "ordinary"
    return "ordinary" if not carries_weights else ""


def _estimator(result: FitResult, weighting: str) -> str:
    words = [weighting, "least squares"]
    if result.loss != "linear":
        words.append(f"with the {LOSSES.get(result.loss, result.loss)} loss")
    return " ".join(word for word in words if word)


def _setting(value: float) -> str:
    """A number the specification set, shown as it was given (``0.0`` -> ``0``, ``0.5`` as is)."""
    return f"{value:.15g}"


def _bounds_sentence(spec: FitSpec | None) -> str:
    """The bounds the specification set on the parameters; ``""`` when it set none.

    An open end (``None``) is left out, so a parameter with one end free reads ``a > 0`` or
    ``b < 10``, and one with both free is not named. The bounds a registered model declares
    are the model's, not the specification's, and are not spoken of here.
    """
    if spec is None:
        return ""
    bounded: list[str] = []
    for name, (lower, upper) in spec.bounds.items():
        if lower is None:
            if upper is not None:
                bounded.append(f"{name} < {_setting(upper)}")
        elif upper is None:
            bounded.append(f"{name} > {_setting(lower)}")
        else:
            bounded.append(f"{name} in ({_setting(lower)}, {_setting(upper)})")
    if not bounded:
        return ""
    return f"The parameters were bounded: {', '.join(bounded)}."


def _left_out(bundle: FitBundle, missing: _Missing) -> str:
    """The observations that took no part, each count with its reason."""
    data = bundle.data
    if data.subsampled:
        missing.add(
            "SUBSAMPLED_DATA",
            "the stored data are a subsample of the data that were fitted, so the "
            "observations left out cannot be counted",
        )
        return (
            f"The stored data are a subsample of the {data.n_total} rows that were fitted, so "
            "the number of observations left out cannot be given here."
        )
    excluded, _ = split_rows(data)
    if not excluded:
        return ""
    counts: dict[str, int] = {}
    for row in excluded:
        counts[row.reason] = counts.get(row.reason, 0) + 1
    if len(counts) == 1:
        reasons = next(iter(counts))
    else:  # more than one reason: each with its share
        reasons = ", ".join(f"{count} {reason}" for reason, count in counts.items())
    plural = "s were" if len(excluded) != 1 else " was"
    return f"{len(excluded)} observation{plural} left out of the fit ({reasons})."


def _estimates_sentence(shown: tuple[_Shown, ...], percent: str, missing: _Missing) -> str:
    """Where the estimates are given, and what could not be determined for any of them."""
    where = "in the equation and table below"
    intervals = f"with their standard errors and {percent} % confidence intervals"
    bare = [row.name for row in shown if row.std_err is None]
    if len(bare) == len(shown):
        missing.add(
            "NO_STANDARD_ERROR",
            "the standard error could not be determined for any parameter, so no interval is given",
        )
        return (
            f"The estimates are given {where}; their standard errors and intervals could not "
            "be determined."
        )
    sentence = f"The estimates are given {intervals} {where}"
    if bare:
        one = len(bare) == 1
        missing.add(
            "NO_STANDARD_ERROR",
            f"the standard error could not be determined for {joined(bare)}, so no interval "
            f"is given for {'it' if one else 'them'}",
        )
        count = "1" if one else str(len(bare))
        sentence += (
            f", except for {count} of the {len(shown)} parameters, whose standard error and "
            "interval could not be determined"
        )
    return sentence + "."


def _version(bundle: FitBundle, missing: _Missing) -> str:
    """The CurveStudio version that made the fit, else the one in use (which is then said)."""
    recorded = bundle.result.provenance.software_versions.get("curvestudio")
    if recorded is not None and recorded != "unknown":
        return recorded
    package = curvestudio.__version__
    missing.add(
        "VERSION_FROM_PACKAGE",
        "the fit did not record which CurveStudio version made it, so the text gives the "
        f"version of the package in use, {package}",
    )
    return package


def _observations(result: FitResult) -> int:
    stats = result.stats
    if stats is None:
        raise ToolError(
            "NO_STATISTICS",
            "the fit converged but carries no statistics, so the number of observations it "
            "used is not known",
        )
    return stats.n


def _paragraph(
    bundle: FitBundle,
    result: FitResult,
    model: _Model,
    spec: FitSpec | None,
    shown: tuple[_Shown, ...],
    percent: str,
    missing: _Missing,
) -> tuple[str, str]:
    """The paragraph, and the CurveStudio version it names."""
    if spec is None:
        missing.add(
            "NO_SPEC",
            "the fit's specification was not given, so its settings (bounds, starting "
            "values, whether weights were used) are not known and the text says nothing "
            "about them",
        )
    n = _observations(result)
    weighting = _weighting(result, spec, bundle.data.weights is not None)
    if weighting == "":
        missing.add(
            "WEIGHTING_UNKNOWN",
            "the data carry weights, but it is not known whether the fit used them, so the "
            "estimator is called least squares only",
        )
    solver = (
        f", using the {METHODS.get(result.method, result.method)} algorithm"
        if model.nonlinear
        else ""
    )
    sentences = [
        f"{model.description[0].upper()}{model.description[1:]} was fitted to {n} "
        f"observations by {_estimator(result, weighting)}{solver}."
    ]
    bounded = _bounds_sentence(spec)
    if bounded:
        sentences.append(bounded)
    left_out = _left_out(bundle, missing)
    if left_out:
        sentences.append(left_out)
    sentences.append(_estimates_sentence(shown, percent, missing))
    version = _version(bundle, missing)
    sentences.append(
        f"The fit was made with CurveStudio {version} ({MAKER}, {ADDRESS}), with random "
        f"seed {result.provenance.seed}."
    )
    if model.assumptions:
        closing = model.assumptions.strip()
        closing += "" if closing.endswith(".") else "."
        sentences.append(f"The model assumes that {closing}")
    return " ".join(sentences), version


# -- the equation and the table -------------------------------------------------------------------


def _equation_lines(result: FitResult, model: _Model, shown: tuple[_Shown, ...]) -> list[str]:
    """The fitted model, then its estimates as ``name = value ± std_err``, one per line."""
    if model.registered is not None:
        with_error = [row for row in shown if row.std_err is not None]
        digits = max((_significant(row.value) for row in with_error), default=VALUE_DIGITS)
        values = {parameter.name: parameter.value for parameter in result.parameters}
        head = model.registered.equation_text(values, max(digits, 1))
    else:
        head = result.equation
    lines = [head]
    for row in shown:
        if row.std_err is None:
            lines.append(f"{row.name} = {row.value} {NO_STANDARD_ERROR_TEXT}")
        else:
            lines.append(f"{row.name} = {row.value} ± {row.std_err}")
    return lines


def _latex_text(text: str) -> str:
    """``text`` set as LaTeX text, so words keep their spaces inside mathematics."""
    return "\\text{" + text + "}"


def _term(power: int) -> str:
    """A polynomial's term in LaTeX, by the coefficient's name: ``b1 x``, ``b2 x^{2}``."""
    if power == 0:
        return "b0"
    return f"b{power} x" if power == 1 else f"b{power} x^{{{power}}}"


def _latex_head(result: FitResult, model: _Model) -> str:
    """The fitted model's first LaTeX line, with its names (not its numbers)."""
    left, right = "y", ""
    if model.registered is not None:
        left, _, right = model.registered.latex().partition("=")
    elif model.powers:
        right = " + ".join(_term(power) for power in model.powers)
    else:
        # A custom equation is shown as the author wrote it, its right-hand side in text mode:
        # the tools layer cannot typeset it, and it is not dressed up as if it had been.
        left, _, right = result.equation.partition(" = ")
        right = _latex_text(right.translate(_LATEX_TEXT))
    return f"{left.strip()} &= {right.strip()}"


def _latex_equation(result: FitResult, model: _Model, shown: tuple[_Shown, ...]) -> str:
    """The equation and its estimates as one complete display: ``\\[``, an ``aligned`` block,
    ``\\]``, which compiles as it comes."""
    lines = [_latex_head(result, model)]
    for row in shown:
        if row.std_err is None:
            lines.append(rf"{row.name} &= {row.value}\ {_latex_text(NO_STANDARD_ERROR_TEXT)}")
        else:
            lines.append(rf"{row.name} &= {row.value} \pm {row.std_err}")
    return "\n".join(
        [
            r"\[",
            r"\begin{aligned}",
            *(f"{line} \\\\" for line in lines),
            r"\end{aligned}",
            r"\]",
        ]
    )


def _equation(result: FitResult, model: _Model, shown: tuple[_Shown, ...], style: Style) -> str:
    if style == "latex":
        return _latex_equation(result, model, shown)
    lines = _equation_lines(result, model, shown)
    return "\n".join(lines) if style == "text" else "  \n".join(lines)


def _headers(percent: str) -> tuple[str, ...]:
    return ("Parameter", "Estimate", "Std. error", f"Lower ({percent} %)", f"Upper ({percent} %)")


def _cells(shown: tuple[_Shown, ...]) -> list[tuple[str, ...]]:
    return [
        (
            row.name,
            row.value,
            NOT_DETERMINED if row.std_err is None else row.std_err,
            NOT_DETERMINED if row.lower is None else row.lower,
            NOT_DETERMINED if row.upper is None else row.upper,
        )
        for row in shown
    ]


def _table(shown: tuple[_Shown, ...], percent: str, style: Style) -> str:
    headers, rows = _headers(percent), _cells(shown)
    if style == "markdown":
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * 5) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    if style == "latex":
        head = " & ".join(header.replace("%", r"\%") for header in headers)
        # A tabular sets its cells in text mode. Each one goes in math mode, so a name with an
        # underscore compiles and a negative number prints a minus sign, not a hyphen.
        body = [
            " & ".join(_latex_text(cell) if cell == NOT_DETERMINED else f"${cell}$" for cell in row)
            for row in rows
        ]
        lines = [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            f"{head} \\\\",
            r"\midrule",
            *(f"{row} \\\\" for row in body),
            r"\bottomrule",
            r"\end{tabular}",
        ]
        return "\n".join(lines)
    widths = [max(len(line[column]) for line in (headers, *rows)) for column in range(5)]
    lines = []
    for line in (headers, *rows):
        numbers = "  ".join(
            f"{cell:>{width}}" for cell, width in zip(line[1:], widths[1:], strict=True)
        )
        lines.append(f"{line[0]:<{widths[0]}}  {numbers}".rstrip())
    return "\n".join(lines)


def _citation_model(model: _Model) -> tuple[str, ...]:
    """The model's sources, verbatim; a model that has none says so in one sentence."""
    if model.registered is not None:
        return model.references or ("A registered model; no source is cited for it.",)
    kind = "custom" if model.nonlinear else "polynomial"
    return (NO_SOURCE[kind],)


# -- the tool -------------------------------------------------------------------------------------


def methods_text(
    bundle: FitBundle,
    *,
    spec: FitSpec | None = None,
    style: Style = "text",
    level: float = 0.95,
) -> MethodsText:
    """The methods text of one finished fit.

    Args:
        bundle: a fit that converged.
        spec: the specification the fit ran with. It says whether the weights were used;
            without it the text relies on what the result records and lists ``NO_SPEC`` in
            ``missing``.
        style: ``"text"`` (plain lines), ``"markdown"`` or ``"latex"`` for the equation and
            the table. The paragraph is the same text in every style, escaped for LaTeX in
            that style. In LaTeX the equation is a complete display (``\\[``, an
            ``aligned`` block, ``\\]``): an author who wants it numbered replaces the wrapper
            with an
            ``equation`` environment. The table is a ``tabular`` with ``booktabs`` rules and
            needs ``\\usepackage{booktabs}``; the display needs ``amsmath``.
        level: the confidence level of the intervals the fit's parameters carry, stated in the
            text. The workflow's fits carry 0.95; a result produced with another level must be
            given that one here, as the result itself does not record it.

    Returns:
        The paragraph, the equation, the table, the citations, what the author must complete
        and what is missing.

    Raises:
        ToolError: ``MODEL_MISMATCH`` for a specification that is not the one the fit was
            computed with (it is checked before anything else). ``FIT_NOT_CONVERGED_INPUT``
            for a fit that did not converge: there are no estimates to report.
            ``INVALID_LEVEL`` for a ``level`` outside 0 and 1, and ``NO_STATISTICS`` for a
            fit that carries none.
    """
    result = bundle.result
    require_matching_spec(result, spec)
    if result.status is not FitStatus.CONVERGED:
        raise ToolError(
            "FIT_NOT_CONVERGED_INPUT",
            f"the fit is {result.status.value} ({result.termination_reason}): a fit that did "
            "not converge has no estimates to report",
        )
    if not 0.0 < level < 1.0:
        raise ToolError("INVALID_LEVEL", f"level must lie strictly between 0 and 1, got {level!r}")
    model = _fitted_model(result)
    shown = _show(result.parameters)
    missing = _Missing()
    percent = f"{100.0 * level:.6g}"
    paragraph, version = _paragraph(bundle, result, model, spec, shown, percent, missing)
    if style == "latex":
        paragraph = paragraph.translate(_LATEX_TEXT)
    return MethodsText(
        paragraph=paragraph,
        equation=_equation(result, model, shown, style),
        table=_table(shown, percent, style),
        citation_curvestudio=f"CurveStudio {version}. {MAKER}. {ADDRESS}",
        citation_model=_citation_model(model),
        to_complete=TO_COMPLETE,
        missing=missing.lines(),
    )


def all_methods_texts(
    bundle: FitBundle, *, spec: FitSpec | None = None, level: float = 0.95
) -> dict[Style, MethodsText]:
    """The methods text of one finished fit in every style; see :func:`methods_text`.

    Raises:
        ToolError: as :func:`methods_text` does.
    """
    return {style: methods_text(bundle, spec=spec, style=style, level=level) for style in STYLES}
