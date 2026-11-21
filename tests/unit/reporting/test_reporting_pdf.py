"""``fit_to_pdf``: a report a human can read, and a machine can verify.

Text assertions decode the PDF's text layer through its ``/ToUnicode`` CMaps
(the mapping every real text extractor relies on), so the tests need no
external tool and still verify what a reader sees — including non-Latin
glyphs the raw content stream encodes as font-subset byte codes.
"""

import io
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from reportlab.platypus import Table

from curvestudio.workflow import FitBundle, format_equation
from curvestudio_reporting import fit_to_pdf
from curvestudio_reporting.pdf_report import _WIDTH, _flowables

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]


def _pdf_objects(data: bytes) -> dict[int, bytes]:
    """The body of every numbered object (xref-free split; test-only parsing)."""
    objects: dict[int, bytes] = {}
    for match in re.finditer(rb"(\d+)\s+0\s+obj\b", data):
        number, start = int(match.group(1)), match.end()
        nxt = re.search(rb"\d+\s+0\s+obj\b", data[start:])
        objects[number] = data[start : start + nxt.start() if nxt else len(data)]
    return objects


def _unescape_pdf_literal(literal: bytes) -> bytes:
    """Resolve reportlab's ``\\(``, ``\\\\``) and octal ``\\ddd`` escapes."""
    out, i = bytearray(), 0
    while i < len(literal):
        if literal[i] == 0x5C and i + 1 < len(literal):
            nxt = literal[i + 1]
            if 0x30 <= nxt <= 0x37:  # \ddd octal byte
                j, digits = i + 1, b""
                while j < len(literal) and len(digits) < 3 and 0x30 <= literal[j] <= 0x37:
                    digits += literal[j : j + 1]
                    j += 1
                out.append(int(digits, 8))
                i = j
            else:
                out.append(nxt)
                i += 2
        else:
            out.append(literal[i])
            i += 1
    return bytes(out)


def _parse_cmap(cmap: bytes) -> dict[int, str]:
    """Subset byte code -> Unicode, from ``beginbfchar`` entries."""
    mapping: dict[int, str] = {}
    for match in re.finditer(rb"<([0-9A-Fa-f]{2})>\s*<([0-9A-Fa-f]{2,8})>", cmap):
        utf16 = match.group(2)
        mapping[int(match.group(1), 16)] = "".join(
            chr(int(utf16[i : i + 4], 16)) for i in range(0, len(utf16), 4)
        )
    return mapping


def pdf_text(data: bytes) -> str:
    """The PDF's visible text, one drawn line per extracted line.

    ``fit_to_pdf`` writes uncompressed streams under the ``uncompressed`` fixture,
    so the content streams and ``/ToUnicode`` CMaps parse without decompression.
    Font-subset text (``/F2+0 12 Tf ... (\\003\\021) Tj``) decodes through the
    active font's CMap, exactly as poppler or a PDF viewer's text layer does.
    Consecutive ``Tj`` fragments belong to one line (reportlab splits at escaped
    entities); ``T*``/``Tm``/``Td``/``BT``/``ET`` move to the next line or cell.
    """
    streams, font_cmap = {}, {}
    for number, body in _pdf_objects(data).items():
        begin = body.find(b"stream\n")
        if begin != -1:
            streams[number] = body[begin + len(b"stream\n") : body.rfind(b"endstream")]
            body = body[:begin]
        if match := re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", body):
            font_cmap[number] = int(match.group(1))
    cmaps = {number: _parse_cmap(streams[cmap]) for number, cmap in font_cmap.items()}
    names = {
        match.group(1).decode(): int(match.group(2))
        for match in re.finditer(rb"/(F\d+(?:\+\d+)?)\s+(\d+)\s+0\s+R", data)
    }
    lines: list[str] = []
    for stream in streams.values():
        if b"Tj" not in stream and b"TJ" not in stream:
            continue
        cmap: dict[int, str] = {}
        current: list[str] = []
        for match in re.finditer(
            rb"/(F\d+(?:\+\d+)?)\s+[\d.]+\s+Tf"
            rb"|\(((?:[^()\\]|\\.)*)\)\s*Tj"
            rb"|\[((?:[^\[\]\\]|\\.)*)\]\s*TJ"
            rb"|(?<![\w/])(T\*|Tm|Td|TD|BT|ET)(?![\w/])",
            stream,
        ):
            if match.group(1):  # Tf: switch the active font's CMap
                cmap = cmaps.get(names.get(match.group(1).decode(), -1), {})
            elif match.group(2) is not None:  # (text) Tj: one fragment of the line
                literal = _unescape_pdf_literal(match.group(2))
                current.append("".join(cmap.get(b, "?") for b in literal))
            elif match.group(3) is not None:  # [(a) -120 (b)] TJ: kerned fragments
                fragments = re.findall(rb"\(((?:[^()\\]|\\.)*)\)", match.group(3))
                current.append(
                    "".join(
                        "".join(cmap.get(b, "?") for b in _unescape_pdf_literal(f))
                        for f in fragments
                    )
                )
            elif current:  # line/cell move: flush what was drawn so far
                lines.append("".join(current))
                current = []
        if current:
            lines.append("".join(current))
    return "\n".join(lines)


def norm(text: str) -> str:
    """Collapse wrapping/spacing so paragraph breaks do not break substring checks."""
    return " ".join(text.split())


@pytest.fixture
def uncompressed() -> Iterator[None]:
    """Write the PDF with pageCompression=0 so tests can read the text streams."""
    import reportlab.rl_config

    old = reportlab.rl_config.pageCompression
    reportlab.rl_config.pageCompression = 0
    yield
    reportlab.rl_config.pageCompression = old


def make_pdf(bundle: FitBundle, tmp_path: Path) -> bytes:
    target = tmp_path / "fit.pdf"
    fit_to_pdf(bundle, target)
    return target.read_bytes()


def test_pdf_starts_with_the_header_and_names_the_tool(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    data = make_pdf(poly_bundle(), tmp_path)
    assert data.startswith(b"%PDF")
    assert "CurveStudio" in pdf_text(data)


def test_pdf_contains_the_equation_and_identification(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    bundle = poly_bundle()
    text = norm(pdf_text(make_pdf(bundle, tmp_path)))
    assert format_equation(bundle.equation_text) in text
    assert "**" not in text
    assert bundle.result.provenance.dataset_fingerprint in text
    assert "+05:30" in text, "the generated-at timestamp must be IST"


def test_pdf_embeds_fit_and_residual_figures(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    data = make_pdf(poly_bundle(), tmp_path)
    # /ImageB /ImageC /ImageI appear in every page's ProcSet; only real embedded
    # figures are image XObjects.
    assert b"/Subtype /Image" in data, "a converged fit embeds its fit and residual figures"


def test_pdf_for_a_non_converged_fit_warns_on_page_one_and_hides_the_fit(
    non_converged_bundle: FitBundle, tmp_path: Path, uncompressed: None
) -> None:
    data = make_pdf(non_converged_bundle, tmp_path)
    text = norm(pdf_text(data))
    assert "FIT DID NOT CONVERGE" in text
    assert non_converged_bundle.result.termination_reason in text
    assert b"/Subtype /Image" not in data, "a failed fit must not show a curve or fit figure"


def test_pdf_for_a_failed_fit_warns_too(
    failed_bundle: FitBundle, tmp_path: Path, uncompressed: None
) -> None:
    assert "FIT DID NOT CONVERGE" in norm(pdf_text(make_pdf(failed_bundle, tmp_path)))


def test_markup_in_names_survives_the_pdf(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    bundle = poly_bundle(x_name="<b>&</b>", y_name="a<b>c&amp;d</b>")
    data = make_pdf(bundle, tmp_path)  # must not raise: text is XML-escaped for reportlab
    assert "<b>&</b>" in norm(pdf_text(data))


def test_undefined_statistics_are_marked_not_faked(
    constant_bundle: FitBundle, tmp_path: Path, uncompressed: None
) -> None:
    text = norm(pdf_text(make_pdf(constant_bundle, tmp_path)))
    assert "not defined" in text
    assert "NaN" not in text and "Infinity" not in text


def test_assumptions_section_carries_the_information_criteria_convention(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    bundle = poly_bundle()
    assert bundle.summary is not None
    text = norm(pdf_text(make_pdf(bundle, tmp_path)))
    assert "Gaussian, AIC = n" in text, "the convention head must be readable text"
    assert "Assumptions and limitations" in text


def test_pdf_writes_to_a_binary_stream_too(
    poly_bundle: FitBundleFactory, uncompressed: None
) -> None:
    buffer = io.BytesIO()
    fit_to_pdf(poly_bundle(), buffer)
    assert buffer.getvalue().startswith(b"%PDF")


def test_every_table_has_a_width_per_column_and_fits_the_frame(
    poly_bundle: FitBundleFactory,
    non_converged_bundle: FitBundle,
) -> None:
    for bundle in (poly_bundle(), non_converged_bundle):
        for flowable in _flowables(bundle):
            if not isinstance(flowable, Table):
                continue
            columns = len(flowable._cellvalues[0])  # type: ignore[attr-defined]
            widths = list(flowable._colWidths)  # type: ignore[attr-defined]
            assert len(widths) == columns, "one explicit width per column, no auto leftovers"
            assert sum(widths) <= _WIDTH + 1e-6, "a table must never exceed the frame width"


def test_parameter_table_keeps_five_columns_for_a_non_converged_fit(
    non_converged_bundle: FitBundle,
) -> None:
    tables = [f for f in _flowables(non_converged_bundle) if isinstance(f, Table)]
    parameter_table = next(
        t
        for t in tables
        if _row_texts(t)[0][0] == "parameter"  # type: ignore[attr-defined]
    )
    header = _row_texts(parameter_table)[0]  # type: ignore[attr-defined]
    assert header == ["parameter", "value", "std_err", "95% CI low", "95% CI high"]
    # the empty SE/CI cells of a non-converged fit are still five real columns
    rows = _row_texts(parameter_table)  # type: ignore[attr-defined]
    assert all(len(row) == 5 for row in rows), "every row keeps all five columns"


def _row_texts(table: Table) -> list[list[str]]:
    """The plain text of a Table's cells (Paragraph cells carry their own text)."""
    return [
        [cell.text if hasattr(cell, "text") else str(cell) for cell in row]
        for row in table._cellvalues  # type: ignore[attr-defined]
    ]


def test_very_long_cell_text_is_truncated_not_overflowing(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    long_name = " ".join(f"w{i}" for i in range(700))
    data = make_pdf(poly_bundle(x_name=long_name), tmp_path)  # must render, not overflow
    text = norm(pdf_text(data))
    assert "w0 w1" in text, "the first 200 characters are still shown"
    assert "w600" not in text, "nothing beyond the display maximum is shown"
    assert "…" in text, "a truncated cell ends in an ellipsis"


def test_control_characters_are_stripped_from_pdf_text(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    data = make_pdf(poly_bundle(x_name="a\x1bb\x00c"), tmp_path)  # must render, not crash
    text = pdf_text(data)
    assert "abc" in norm(text), "XML-illegal control characters must not reach the PDF"


@pytest.mark.skipif(
    sys.platform == "win32", reason="pdf text extraction of embedded TTF glyphs differs on Windows"
)
def test_non_latin_text_renders_in_the_pdf_font(
    poly_bundle: FitBundleFactory, tmp_path: Path, uncompressed: None
) -> None:
    data = make_pdf(poly_bundle(x_name="σ² ≤ −x", y_name="αβγ"), tmp_path)  # noqa: RUF001
    assert b"/FontFile2" in data, "a Unicode TrueType font must be embedded"
    text = pdf_text(data)
    assert "σ² ≤ −x" in text and "αβγ" in text  # noqa: RUF001 — U+2212 is the point
    if shutil.which("pdftotext"):  # cross-check with poppler when available
        pop = subprocess.run(
            ["pdftotext", str(tmp_path / "fit.pdf"), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "σ² ≤ −x" in pop  # noqa: RUF001 — the minus sign itself is under test


def test_subsampled_export_is_flagged_in_the_pdf(
    compact_bundle: FitBundle, tmp_path: Path, uncompressed: None
) -> None:
    assert compact_bundle.data.subsampled
    text = norm(pdf_text(make_pdf(compact_bundle, tmp_path)))
    note = (
        f"showing {len(compact_bundle.data.x)} "
        f"of {compact_bundle.data.n_total} observations (subsampled)"
    )
    assert note in text


def test_diagnostics_notes_end_in_a_single_full_stop(
    poly_bundle: FitBundleFactory,
) -> None:
    from curvestudio_reporting.pdf_report import _diagnostics_notes

    notes = _diagnostics_notes(poly_bundle())
    assert notes, "a converged bundle carries diagnostics notes"
    assert ".." not in notes, "no doubled full stop where joined limitations end in one"
    assert notes.endswith(".")
