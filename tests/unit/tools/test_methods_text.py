"""The methods text on real fits: every statement is read back from the fit that was made."""

import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from test_tools_cross_validation import _origin_bundle

import curvestudio
from curvestudio.data import XY
from curvestudio.models import get_model
from curvestudio.tools import MethodsText, ToolError, all_methods_texts, methods_text
from curvestudio.tools.methods import STYLES
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    fit_dataset,
)
from curvestudio.workflow.bundle import COMPACT_MAX_POINTS

QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
LINE = FitSpec(model=PolynomialRef(degree=1))
CUSTOM_LINE = FitSpec(model=CustomRef(equation="a*x + b", parameters=["a", "b"]))
HUBER_LINE = CUSTOM_LINE.model_copy(update={"loss": "huber"})
SOFT_L1_LINE = CUSTOM_LINE.model_copy(update={"loss": "soft_l1"})
GROWTH = FitSpec(model=RegisteredRef(id="exp_growth"), seed=3)
ADDRESS = "https://curvestudio.concreteinfo.cloud"
TO_COMPLETE = (
    "Instrument and measurement method",
    "How the sample size was chosen",
    "How the data were cleaned",
)
NO_SPEC = (
    "NO_SPEC: the fit's specification was not given, so its settings (bounds, starting "
    "values, whether weights were used) are not known and the text says nothing about them"
)
NO_SOURCE = {
    "polynomial": "A polynomial model; no source is cited for it.",
    "custom": "A custom equation; no source is cited for it.",
}
NUMERAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _xy(x: Sequence[float], y: Sequence[float], weights: Sequence[float] | None = None) -> XY:
    return XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="fp-methods",
    )


def quadratic_xy(n: int = 40, seed: int = 0) -> XY:
    x = np.linspace(0.0, 10.0, n)
    noise = np.random.default_rng(seed).normal(0.0, 0.1, n)
    return _xy(x, 1.0 + 0.5 * x - 0.03 * x**2 + noise)


def growth_xy(weights: Sequence[float] | None = None) -> XY:
    x = np.linspace(0.0, 3.0, 30)
    noise = np.random.default_rng(0).normal(0.0, 0.05, 30)
    return _xy(x, 2.0 * np.exp(0.4 * x) + noise, weights)


def line_xy() -> XY:
    x = np.linspace(0.0, 10.0, 30)
    return _xy(x, 2.0 * x + 1.0 + np.random.default_rng(0).normal(0.0, 0.1, 30))


def weighted_xy() -> XY:
    """A quadratic on 45 weighted rows, 5 of them with weight 0."""
    x = np.linspace(0.0, 10.0, 45)
    y = 1.0 + 0.5 * x - 0.03 * x**2 + np.random.default_rng(0).normal(0.0, 0.1, 45)
    weights = np.ones(45)
    weights[[3, 10, 17, 24, 31]] = 0.0
    return _xy(x, y, weights)


USED = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
IGNORED = FitSpec(model=PolynomialRef(degree=2), use_weights=False)


def weighted_quadratic() -> FitBundle:
    """The weighted quadratic, fitted with its weights."""
    return fit_dataset(weighted_xy(), USED)


def without_uncertainty(bundle: FitBundle, *, which: str = "first") -> FitBundle:
    """The same fit with the standard errors and intervals removed from its parameters."""
    drop = {bundle.result.parameters[0].name} if which == "first" else None
    parameters = tuple(
        p.model_copy(update={"std_err": None, "ci_low": None, "ci_high": None})
        if drop is None or p.name in drop
        else p
        for p in bundle.result.parameters
    )
    result = bundle.result.model_copy(update={"parameters": parameters})
    return bundle.model_copy(update={"result": result})


def without_recorded_version(bundle: FitBundle) -> FitBundle:
    """The same fit with the CurveStudio version left out of what it recorded."""
    versions = dict(bundle.result.provenance.software_versions)
    del versions["curvestudio"]
    provenance = bundle.result.provenance.model_copy(update={"software_versions": versions})
    result = bundle.result.model_copy(update={"provenance": provenance})
    return bundle.model_copy(update={"result": result})


def version_of(bundle: FitBundle) -> str:
    """The CurveStudio version the text is expected to name: the fit's own."""
    return bundle.result.provenance.software_versions["curvestudio"]


def codes(text: MethodsText) -> list[str]:
    return [entry.split(":", 1)[0] for entry in text.missing]


def compact_line() -> FitBundle:
    x = np.linspace(0.0, 10.0, COMPACT_MAX_POINTS + 1)
    return fit_dataset(_xy(x, 2.0 * x + 1.0), LINE, compact=True)


# -- what it states -------------------------------------------------------------------------------


def test_methods_text_states_only_what_is_true() -> None:
    plain = fit_dataset(quadratic_xy(), QUADRATIC)
    text = methods_text(plain)
    assert text.paragraph == (
        "A polynomial of degree 2 was fitted to 40 observations by ordinary least squares. "
        "The estimates are given with their standard errors and 95 % confidence intervals in "
        f"the equation and table below. The fit was made with CurveStudio {version_of(plain)} "
        f"(ConcreteInfo, {ADDRESS}), with random seed 0."
    )
    assert text.citation_curvestudio == f"CurveStudio {version_of(plain)}. ConcreteInfo. {ADDRESS}"
    assert text.citation_model == (NO_SOURCE["polynomial"],)
    assert text.missing == (NO_SPEC,)
    assert text.to_complete == TO_COMPLETE

    # weights: said when they were used, not when the data carry them but the fit dropped them
    paragraph = methods_text(weighted_quadratic(), spec=USED).paragraph
    assert "fitted to 40 observations by weighted least squares." in paragraph
    assert "5 observations were left out of the fit (weight is zero)." in paragraph
    paragraph = methods_text(fit_dataset(weighted_xy(), IGNORED), spec=IGNORED).paragraph
    assert "fitted to 45 observations by ordinary least squares." in paragraph
    assert "left out" not in paragraph
    # the fit's own record stands over a specification that disagrees with it
    paragraph = methods_text(weighted_quadratic(), spec=IGNORED).paragraph
    assert "fitted to 40 observations by weighted least squares." in paragraph

    # a robust loss is named, with the solver a nonlinear model needs
    robust = fit_dataset(line_xy(), HUBER_LINE)
    assert robust.result.status.value == "converged"
    paragraph = methods_text(robust, spec=HUBER_LINE).paragraph
    assert (
        "by ordinary least squares with the Huber loss, using the trust-region reflective "
        "algorithm."
    ) in paragraph

    # a registered model: its name, its solver, its references verbatim, its assumptions last
    model = get_model("exp_growth")
    assert NUMERAL.findall(model.name + model.assumptions + model.expression) == []
    grown = fit_dataset(growth_xy(), GROWTH)
    assert grown.result.status.value == "converged"
    text = methods_text(grown, spec=GROWTH)
    assert text.paragraph.startswith("Exponential growth was fitted to 30 observations")
    assert "using the Levenberg-Marquardt algorithm." in text.paragraph
    assert "with random seed 3." in text.paragraph
    assert text.citation_model == model.references
    assert text.paragraph.endswith(f"The model assumes that {model.assumptions}")
    assert text.missing == ()

    # a polynomial through the origin and a custom equation say what they are
    origin, _ = _origin_bundle()
    assert methods_text(origin, spec=QUADRATIC).paragraph.startswith(
        "A polynomial of degree 2 through the origin was fitted"
    )
    custom = fit_dataset(line_xy(), CUSTOM_LINE)
    text = methods_text(custom, spec=CUSTOM_LINE)
    assert text.paragraph.startswith(
        "A custom equation (y = a*x + b) was fitted to 30 observations by ordinary least "
        "squares, using the Levenberg-Marquardt algorithm."
    )
    assert text.citation_model == (NO_SOURCE["custom"],)
    latex = methods_text(custom, spec=CUSTOM_LINE, style="latex")
    assert latex.equation.splitlines()[1] == r"y &= \text{a*x + b} \\"


def test_a_parameter_without_a_standard_error_is_said_to_have_none() -> None:
    clean = fit_dataset(quadratic_xy(), QUADRATIC)
    first = clean.result.parameters[0].name
    partly = without_uncertainty(clean)
    for style in STYLES:
        text = methods_text(partly, spec=QUADRATIC, style=style)
        assert text.equation.count("(standard error not determined)") == 1
        lines = [line.strip() for line in text.equation.splitlines()]
        line = next(line for line in lines if line.split(" ", 1)[0] == first)
        assert "(standard error not determined)" in line
        if style == "text":
            value = re.split(r"\s{2,}", text.table.splitlines()[1])[1]
            assert line == f"{first} = {value} (standard error not determined)"
        assert (
            text.table.replace("\\text{not determined}", "not determined").count("not determined")
            == 3
        )
        assert codes(text) == ["NO_STANDARD_ERROR"]
        assert first in text.missing[0]
    assert (
        "except for 1 of the 3 parameters, whose standard error and interval could not be "
        "determined."
    ) in methods_text(partly, spec=QUADRATIC).paragraph

    every = without_uncertainty(clean, which="all")
    text = methods_text(every, spec=QUADRATIC)
    assert (
        "The estimates are given in the equation and table below; their standard errors and "
        "intervals could not be determined."
    ) in text.paragraph
    assert "±" not in text.equation
    for style in STYLES:
        table = methods_text(every, spec=QUADRATIC, style=style).table
        plain = table.replace("\\text{not determined}", "not determined")
        assert plain.count("not determined") == 9


def test_what_is_not_known_is_said_not_guessed() -> None:
    # weights in the data, no specification: a nonlinear fit does not record whether it used
    # them, so the text calls the estimator least squares and says so
    WEIGHTED_GROWTH = GROWTH.model_copy(update={"use_weights": True})
    grown = fit_dataset(growth_xy(np.linspace(0.5, 2.0, 30)), WEIGHTED_GROWTH)
    assert grown.data.weights is not None
    unknown = methods_text(grown)
    assert "observations by least squares, using" in unknown.paragraph
    assert "weighted" not in unknown.paragraph
    assert "ordinary" not in unknown.paragraph
    assert codes(unknown) == ["NO_SPEC", "WEIGHTING_UNKNOWN"]
    known = methods_text(grown, spec=WEIGHTED_GROWTH)
    assert "observations by weighted least squares, using" in known.paragraph
    assert codes(known) == []
    # a polynomial records it: the fit itself says whether it was weighted
    unannotated = methods_text(weighted_quadratic())
    assert "by weighted least squares." in unannotated.paragraph
    assert codes(unannotated) == ["NO_SPEC"]

    # a compact bundle: the rows left out cannot be counted from a subsample
    big = compact_line()
    text = methods_text(big, spec=LINE)
    rows = COMPACT_MAX_POINTS + 1
    assert (
        f"The stored data are a subsample of the {rows} rows that were fitted, so the number "
        "of observations left out cannot be given here."
    ) in text.paragraph
    assert codes(text) == ["SUBSAMPLED_DATA"]

    # a fit that recorded no version: the package's is used, and that is said
    bare = without_recorded_version(big)
    text = methods_text(bare, spec=LINE)
    version = curvestudio.__version__
    assert f"CurveStudio {version} (ConcreteInfo" in text.paragraph
    assert text.citation_curvestudio == f"CurveStudio {version}. ConcreteInfo. {ADDRESS}"
    assert codes(text) == ["SUBSAMPLED_DATA", "VERSION_FROM_PACKAGE"]
    assert version in text.missing[1]

    # the three styles, each what a single call gives
    every_style = all_methods_texts(grown, spec=GROWTH)
    assert tuple(every_style) == STYLES
    for style in STYLES:
        assert every_style[style] == methods_text(grown, spec=GROWTH, style=style)


def test_the_interval_level_is_the_one_asked_for() -> None:
    clean = fit_dataset(quadratic_xy(), QUADRATIC)
    for style in STYLES:
        text = methods_text(clean, spec=QUADRATIC, style=style, level=0.99)
        assert "99 % confidence intervals" in text.paragraph
        assert ("99 \\%" if style == "latex" else "99 %") in text.table
    for level in (0.0, 1.0, 1.5, -0.5, math.nan):
        with pytest.raises(ToolError) as caught:
            methods_text(clean, level=level)
        assert caught.value.code == "INVALID_LEVEL"


# -- where its numbers come from ------------------------------------------------------------------


def _quantum(numeral: str) -> float:
    """The place value of the last digit ``numeral`` shows: ``1.23`` -> 0.01, ``1.2e+03`` -> 100."""
    mantissa, _, exponent = numeral.partition("e")
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return 10.0 ** (int(exponent) - decimals) if exponent else 10.0 ** (-decimals)


def _decimals(text: str) -> int:
    mantissa = text.partition("e")[0]
    return len(mantissa.split(".", 1)[1]) if "." in mantissa else 0


def _significant(text: str) -> int:
    return len(text.partition("e")[0].lstrip("-+").replace(".", "").lstrip("0"))


def _numerals(text: str, names: Sequence[str]) -> list[str]:
    """Every numeral in ``text``; a parameter's own name (``b0``, ``x1``) is a word, not one."""
    for name in names:
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", " ", text)
    return NUMERAL.findall(text)


def _data_rows(style: str, table: str) -> list[list[str]]:
    """The table's data rows (not its header) as cells, in the style's own layout."""
    if style == "markdown":
        return [
            [cell.strip() for cell in line.strip("|").split("|")] for line in table.splitlines()[2:]
        ]
    if style == "latex":
        body = table.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
        rows = []
        for row in body.strip("\n").split("\n"):
            cells = row.removesuffix(r"\\").split(" & ")
            rows.append([cell.strip().removeprefix("$").removesuffix("$") for cell in cells])
        return rows
    return [re.split(r"\s{2,}", line.rstrip()) for line in table.splitlines()[1:]]


def test_every_number_in_the_methods_text_comes_from_the_result() -> None:
    origin, _ = _origin_bundle()
    big = compact_line()
    cases: list[tuple[FitBundle, FitSpec | None, int | None]] = [
        (fit_dataset(quadratic_xy(), QUADRATIC), None, 2),
        (fit_dataset(quadratic_xy(seed=4), QUADRATIC), QUADRATIC, 2),
        (without_uncertainty(fit_dataset(quadratic_xy(), QUADRATIC)), QUADRATIC, 2),
        (weighted_quadratic(), USED, 2),
        (fit_dataset(growth_xy(), GROWTH), GROWTH, None),
        (fit_dataset(line_xy(), CUSTOM_LINE), CUSTOM_LINE, None),
        (fit_dataset(line_xy(), SOFT_L1_LINE), SOFT_L1_LINE, None),
        (origin, QUADRATIC, 2),
        (big, LINE, 1),
    ]
    for bundle, spec, degree in cases:
        result = bundle.result
        assert result.status.value == "converged"
        stats = result.stats
        assert stats is not None
        names = [p.name for p in result.parameters]
        rows = len(bundle.data.x)
        without = sum(1 for p in result.parameters if p.std_err is None)
        # The named constants, written out:
        allowed = (
            {"95"}  # the interval level, as a percent
            | set(NUMERAL.findall(version_of(bundle)))  # the version's digits
            | {str(result.provenance.seed)}  # the seed
            | {  # counts of observations: those used, those stored, those left out, every row
                str(stats.n),
                str(rows),
                str(rows - stats.n),
                str(bundle.data.n_total),
            }
            | {str(len(names)), str(without)}  # counts of parameters
        )
        if degree is not None:  # the polynomial degree, and the powers 1 up to it
            allowed |= {str(degree), *(str(power) for power in range(1, degree + 1))}
        own = [
            number
            for p in result.parameters
            for number in (p.value, p.std_err, p.ci_low, p.ci_high)
            if number is not None
        ]
        for style in STYLES:
            text = methods_text(bundle, spec=spec, style=style)
            numerals = [
                numeral
                for block in (text.paragraph, text.equation, text.table)
                for numeral in _numerals(block, names)
            ]
            strays = [
                numeral
                for numeral in numerals
                if numeral not in allowed
                and not any(
                    abs(float(numeral) - number) <= 0.5 * _quantum(numeral) * (1.0 + 1e-9)
                    for number in own
                )
            ]
            assert strays == [], f"{style}: {result.model_id}: {strays}"

            # the rule itself: a value and its standard error share decimal places, chosen so
            # that the error shows two significant digits
            rows_by_style = _data_rows(style, text.table)
            for cells, p in zip(rows_by_style, result.parameters, strict=True):
                cells = [cell.replace("\\text{not determined}", "not determined") for cell in cells]
                if p.std_err is None:
                    assert cells[2:] == ["not determined"] * 3
                    continue
                value, error = cells[1], cells[2]
                assert _decimals(value) == _decimals(error)
                assert _significant(error) == 2, error


# -- its form -------------------------------------------------------------------------------------


def _spans(line: str) -> list[tuple[int, int]]:
    """Where each cell of a line of the text table starts and ends."""
    spans: list[tuple[int, int]] = []
    position = 0
    for part in re.split(r"(\s{2,})", line):
        if part.strip():
            spans.append((position, position + len(part)))
        position += len(part)
    return spans


def _environments_closed(text: str) -> bool:
    stack: list[str] = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]*)\}", text):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def test_each_style_is_well_formed() -> None:
    custom = fit_dataset(line_xy(), CUSTOM_LINE)
    cases = (
        (weighted_quadratic(), USED),
        (custom, CUSTOM_LINE),
        (without_uncertainty(custom), CUSTOM_LINE),
        (without_uncertainty(custom, which="all"), CUSTOM_LINE),
    )
    for bundle, spec in cases:
        k = len(bundle.result.parameters)
        with_error = any(p.std_err is not None for p in bundle.result.parameters)
        ordinary = methods_text(bundle, spec=spec).paragraph

        markdown = methods_text(bundle, spec=spec, style="markdown")
        table = markdown.table.splitlines()
        assert table[0].startswith("| Parameter | Estimate | Std. error |")
        assert table[0].count("|") == 6
        assert re.fullmatch(r"\|(?: ?-{3,} ?\|){5}", table[1])
        assert len(table) == k + 2
        assert all(line.count("|") == 6 for line in table[2:])
        lines = markdown.equation.split("  \n")
        assert len(lines) == k + 1 and "\n" not in "".join(lines)
        assert markdown.paragraph == ordinary

        latex = methods_text(bundle, spec=spec, style="latex")
        for block in (latex.table, latex.equation):
            assert block.count("{") == block.count("}")
            assert _environments_closed(block)
            assert re.findall(r"(?<!\\)%", block) == []
        assert latex.table.startswith("\\begin{tabular}{lrrrr}\n\\toprule\n")
        assert latex.table.endswith("\\bottomrule\n\\end{tabular}")
        assert latex.table.count(" \\\\\n") == k + 1
        assert latex.table.count(" & ") == 4 * (k + 1)
        assert latex.equation.startswith("\\begin{aligned}\n")
        assert latex.equation.endswith("\n\\end{aligned}")
        assert latex.equation.count(" &= ") == k + 1
        assert latex.equation.count(" \\\\\n") == k + 1
        assert ("\\pm" in latex.equation) is with_error
        assert latex.paragraph == ordinary

        text = methods_text(bundle, spec=spec, style="text")
        lines = text.table.splitlines()
        spans = [_spans(line) for line in lines]
        assert all(len(row) == 5 for row in spans), spans
        assert len({row[0][0] for row in spans}) == 1  # the names start in the first column
        for column in range(1, 5):  # every number of a column ends at the same place
            assert len({row[column][1] for row in spans}) == 1, (column, spans)
        assert ("\n" not in text.paragraph) and text.paragraph == ordinary
        assert ("±" in text.equation) is with_error


def test_the_methods_text_names_what_the_author_must_complete() -> None:
    clean = fit_dataset(quadratic_xy(), QUADRATIC)
    plain = fit_dataset(line_xy(), CUSTOM_LINE)
    for bundle, spec in ((clean, None), (clean, QUADRATIC), (plain, CUSTOM_LINE)):
        for style in STYLES:
            text = methods_text(bundle, spec=spec, style=style)
            assert text.to_complete == TO_COMPLETE
            assert ("NO_SPEC" in codes(text)) is (spec is None)
    text = methods_text(clean)
    assert set(MethodsText.model_fields) == {
        "paragraph",
        "equation",
        "table",
        "citation_curvestudio",
        "citation_model",
        "to_complete",
        "missing",
    }
    with pytest.raises(ValidationError):
        MethodsText.model_validate({**text.model_dump(), "extra": 1})
    with pytest.raises(ValidationError):
        text.paragraph = "changed"  # type: ignore[misc]


def test_methods_text_refuses_a_specification_that_is_not_the_fit_s() -> None:
    clean = fit_dataset(quadratic_xy(), QUADRATIC)
    grown = fit_dataset(growth_xy(), GROWTH)
    for bundle, foreign in (
        (clean, FitSpec(model=PolynomialRef(degree=3))),
        (grown, FitSpec(model=RegisteredRef(id="gaussian"))),
    ):
        for write in (methods_text, all_methods_texts):
            with pytest.raises(ToolError) as caught:
                write(bundle, spec=foreign)
            assert caught.value.code == "MODEL_MISMATCH"
            assert "MODEL_MISMATCH" in str(caught.value)

    # without a specification nothing is compared: the text is what it was
    unannotated = methods_text(clean)
    assert "A polynomial of degree 2 was fitted" in unannotated.paragraph
    assert codes(unannotated) == ["NO_SPEC"]

    # a polynomial through the origin is a polynomial of its degree to a specification
    origin, _ = _origin_bundle()
    assert methods_text(origin, spec=QUADRATIC).paragraph.startswith(
        "A polynomial of degree 2 through the origin was fitted"
    )


def test_a_fit_that_did_not_converge_is_refused(scenarios) -> None:
    bundle = scenarios.non_converged()
    for call in (methods_text, all_methods_texts):
        with pytest.raises(ToolError) as caught:
            call(bundle)
        assert caught.value.code == "FIT_NOT_CONVERGED_INPUT"
        assert bundle.result.status.value in caught.value.message


def test_the_same_fit_gives_the_same_text(scenarios) -> None:
    for bundle in (scenarios.clean(), scenarios.zero_weights()):
        for style in STYLES:
            first = methods_text(bundle, style=style)
            assert first == methods_text(bundle, style=style)
            assert first == all_methods_texts(bundle)[style]
            assert MethodsText.model_validate(first.model_dump()) == first


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no TeX engine is installed")
def test_the_latex_compiles(tmp_path: Path) -> None:
    """What no check on shape can say: a TeX engine accepts the equation and the table.

    A name with an underscore, a negative estimate, a value that could not be determined: each
    is set in the wrong mode by a naive table, and only the compiler says so.
    """
    x = np.linspace(0.0, 4.0, 30)
    y = -2.0 * np.exp(-0.5 * x) + np.random.default_rng(0).normal(0.0, 0.01, 30)
    underscored = FitSpec(model=CustomRef(equation="-k_1*exp(-k_2*x)", parameters=["k_1", "k_2"]))
    clean = fit_dataset(quadratic_xy(), QUADRATIC)
    cases = (
        (fit_dataset(_xy(x, y), underscored), underscored),
        (weighted_quadratic(), USED),
        (fit_dataset(growth_xy(), GROWTH), GROWTH),
        (without_uncertainty(clean), QUADRATIC),
        (without_uncertainty(clean, which="all"), QUADRATIC),
    )
    parts = [r"\documentclass{article}", r"\usepackage{amsmath,booktabs}", r"\begin{document}"]
    for bundle, spec in cases:
        text = methods_text(bundle, spec=spec, style="latex")
        parts += [r"\begin{equation*}", text.equation, r"\end{equation*}", text.table]
    parts.append(r"\end{document}")
    (tmp_path / "methods.tex").write_text("\n".join(parts), encoding="utf-8")

    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "methods.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    log = (tmp_path / "methods.log").read_text(encoding="utf-8", errors="replace")
    assert run.returncode == 0, log[-3000:]
    assert (tmp_path / "methods.pdf").exists()
    assert "! " not in log and "Missing" not in log, log[-3000:]
