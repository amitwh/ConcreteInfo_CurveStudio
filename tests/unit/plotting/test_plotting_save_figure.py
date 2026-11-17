"""``save_figure``: PNG/SVG/PDF export — valid, safe and byte-reproducible."""

import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import pytest

from curvestudio.workflow import FitBundle
from curvestudio_plotting import fit_figure, save_figure

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def figure(poly_bundle: FitBundleFactory) -> matplotlib.figure.Figure:
    return fit_figure(poly_bundle())


def test_png_has_magic_bytes_and_repeats_identically(figure: matplotlib.figure.Figure) -> None:
    first, second = BytesIO(), BytesIO()
    save_figure(figure, first, "png")
    save_figure(figure, second, "png")
    assert first.getvalue().startswith(PNG_MAGIC)
    assert first.getvalue() == second.getvalue(), "the same figure must save to identical bytes"


def test_save_figure_accepts_a_path(figure: matplotlib.figure.Figure, tmp_path: Path) -> None:
    target = tmp_path / "fit.png"
    save_figure(figure, target, "png")
    assert target.read_bytes().startswith(PNG_MAGIC)


def test_svg_is_valid_xml_without_scripts_or_timestamps(
    figure: matplotlib.figure.Figure,
) -> None:
    first, second = BytesIO(), BytesIO()
    save_figure(figure, first, "svg")
    save_figure(figure, second, "svg")
    svg = first.getvalue()
    root = ET.fromstring(svg)  # raises on malformed XML
    assert root.tag.endswith("svg")
    assert b"<script" not in svg.lower()
    assert b"dc:date" not in svg, "the SVG must not carry a generation date (reproducible bytes)"
    assert svg == second.getvalue()


def test_svg_fonttype_is_paths(figure: matplotlib.figure.Figure) -> None:
    buffer = BytesIO()
    save_figure(figure, buffer, "svg")
    svg = buffer.getvalue().decode("utf-8")
    ET.fromstring(svg)
    # fonttype "path" renders as glyph path data, not <text> elements referencing fonts
    assert "<text" not in svg


def test_pdf_starts_with_header_and_repeats_identically(
    figure: matplotlib.figure.Figure,
) -> None:
    first, second = BytesIO(), BytesIO()
    save_figure(figure, first, "pdf")
    save_figure(figure, second, "pdf")
    assert first.getvalue().startswith(b"%PDF")
    assert b"CreationDate" not in first.getvalue()
    assert first.getvalue() == second.getvalue()


def test_unknown_format_is_rejected(figure: matplotlib.figure.Figure) -> None:
    with pytest.raises(ValueError, match="format"):
        save_figure(figure, BytesIO(), "jpg")  # type: ignore[arg-type]


def test_no_pyplot_global_state_is_created(
    poly_bundle: FitBundleFactory, non_converged_bundle: FitBundle
) -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    buffer = BytesIO()
    save_figure(fit_figure(poly_bundle()), buffer, "png")
    save_figure(fit_figure(non_converged_bundle), buffer, "svg")
    save_figure(fit_figure(poly_bundle(), show_prediction=True), buffer, "pdf")
    assert pyplot.get_fignums() == []


def test_rc_context_change_is_guarded_by_a_lock(
    figure: matplotlib.figure.Figure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rc_context`` mutates global rcParams, so it must run under the module lock."""
    from curvestudio_plotting import figures

    assert hasattr(figures, "_RC_LOCK"), "save_figure needs a lock around its rcParams change"
    seen: list[bool] = []
    real_rc_context = matplotlib.rc_context

    def spy_rc_context(*args: object, **kwargs: object) -> object:
        seen.append(figures._RC_LOCK.locked())
        return real_rc_context(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(matplotlib, "rc_context", spy_rc_context)
    save_figure(figure, BytesIO(), "svg")
    assert seen == [True], "the rc_context must be entered while the lock is held"
