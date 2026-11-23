"""``add_byline``: the "Prepared by" line along the top edge of a publication figure."""

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from curvestudio.workflow import FitBundle
from curvestudio_plotting import add_byline, fit_figure, residual_figure, save_figure

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]

IDENTITY = "Amit Haridas, ConcreteInfo"
"""What the caller passes: the identity text alone, as the reports' byline argument does."""
BYLINE = f"Prepared by {IDENTITY}"
"""What the figure shows: add_byline adds the words itself."""


def drawn(fig: Figure) -> RendererBase:
    """Give the figure an Agg canvas, draw it, and return the renderer it was drawn with."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def byline_artist(fig: Figure):
    return next(text for text in fig.texts if "Prepared by" in text.get_text())


@pytest.mark.parametrize("titled", [False, True], ids=["untitled", "titled"])
@pytest.mark.parametrize("builder", [fit_figure, residual_figure], ids=["fit", "residual"])
def test_add_byline_puts_one_line_above_every_axes(
    poly_bundle: FitBundleFactory, builder: FitBundleFactory, titled: bool
) -> None:
    bundle = poly_bundle()
    fig = builder(bundle)
    if titled:
        if builder is fit_figure:
            fig = fit_figure(bundle, title="My fit")
        else:
            fig.suptitle("Residual diagnostics")
    add_byline(fig, IDENTITY)
    renderer = drawn(fig)

    byline = byline_artist(fig)
    assert byline.get_text() == BYLINE and "\n" not in BYLINE
    box = byline.get_window_extent(renderer)
    assert 0 < box.y0 < box.y1 <= fig.bbox.height, "the line sits inside the figure's top edge"
    for ax in fig.axes:
        # the tight box is the axes with its title, tick labels, axis labels and legend
        assert box.y0 > ax.get_tightbbox(renderer).y1, "the byline overlaps an axes or its title"
    if fig._suptitle is not None:
        assert box.y0 > fig._suptitle.get_window_extent(renderer).y1, "it overlaps the title"
    assert abs((box.x0 + box.x1) / 2 - fig.bbox.width / 2) < 1.0, "centred"


def test_add_byline_cleans_its_text_and_draws_nothing_for_a_blank_one(
    poly_bundle: FitBundleFactory,
) -> None:
    marked = fit_figure(poly_bundle())
    add_byline(marked, "Amit\x1b Haridas\x00,\n ConcreteInfo")
    assert byline_artist(marked).get_text() == BYLINE

    blank = fit_figure(poly_bundle())
    add_byline(blank, " \n\t")
    assert [text.get_text() for text in blank.texts] == ["ConcreteInfo CurveStudio"]  # watermark
    untouched, plain = BytesIO(), BytesIO()
    save_figure(blank, untouched, "png")
    save_figure(fit_figure(poly_bundle()), plain, "png")  # saves are byte-reproducible
    assert untouched.getvalue() == plain.getvalue(), "no band is reserved for nothing"


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_add_byline_survives_every_export_format(poly_bundle: FitBundleFactory, fmt: str) -> None:
    plain, marked = fit_figure(poly_bundle()), fit_figure(poly_bundle())
    add_byline(marked, IDENTITY)
    without, with_byline = BytesIO(), BytesIO()
    save_figure(plain, without, fmt)  # type: ignore[arg-type]
    save_figure(marked, with_byline, fmt)  # type: ignore[arg-type]

    data = with_byline.getvalue()
    assert data and data != without.getvalue(), f"the {fmt} export carries the byline"
    if fmt == "svg":  # text is written as glyph paths: one more text group than without it
        assert data.count(b'id="text_') == without.getvalue().count(b'id="text_') + 1


def test_add_byline_keeps_a_title_clear_on_a_compressed_layout() -> None:
    """The compressed variant positions the suptitle from the layout itself, below the band."""
    fig = Figure(figsize=(6.4, 4.2), dpi=150, layout="compressed")
    fig.subplots().plot([0.0, 1.0], [0.0, 1.0])
    fig.suptitle("Compressed")
    add_byline(fig, IDENTITY)
    renderer = drawn(fig)
    box = byline_artist(fig).get_window_extent(renderer)
    title = fig._suptitle.get_window_extent(renderer)  # type: ignore[attr-defined]
    axes_top = fig.axes[0].get_tightbbox(renderer).y1
    assert box.y0 > title.y1 and title.y0 > axes_top


def test_add_byline_needs_a_constrained_layout() -> None:
    """On another figure nothing could be reserved, so the no-overlap promise could not hold."""
    fig = Figure(figsize=(6.4, 4.2), dpi=150)
    fig.subplots()
    with pytest.raises(TypeError, match="constrained layout"):
        add_byline(fig, IDENTITY)
    assert fig.texts == []
    add_byline(fig, "   ")  # nothing to print: nothing to reserve either
    assert fig.texts == []
