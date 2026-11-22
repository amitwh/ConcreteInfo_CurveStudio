"""``fit_figure``: data, curve, bands, and the non-convergence banner."""

from typing import TYPE_CHECKING

import pytest
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from curvestudio.workflow import FitBundle
from curvestudio_plotting import WATERMARK_ALPHA, WATERMARK_TEXT, fit_figure
from curvestudio_plotting.palette import (
    BAND_CONFIDENCE_FILL,
    BAND_PREDICTION_FILL,
    DATA_COLOR,
    MODEL_COLORS,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]


def figure_texts(fig: Figure) -> list[str]:
    """Every text a reader can see: figure texts plus each axes' title and texts."""
    texts: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        title = ax.get_title()
        texts.extend([title, *(t.get_text() for t in ax.texts)])
    return [t for t in texts if t]


def line_labels(fig: Figure) -> list[str]:
    return [line.get_label() for ax in fig.axes for line in ax.get_lines()]


def watermark_artists(fig: Figure) -> list:
    return [text for text in fig.texts if text.get_text() == WATERMARK_TEXT]


def test_every_fit_figure_carries_the_brand_watermark(poly_bundle: FitBundleFactory) -> None:
    marks = watermark_artists(fit_figure(poly_bundle()))
    assert len(marks) == 1
    assert marks[0].get_alpha() == pytest.approx(WATERMARK_ALPHA)


def test_non_converged_figure_carries_the_brand_watermark(
    non_converged_bundle: FitBundle, failed_bundle: FitBundle
) -> None:
    for bundle in (non_converged_bundle, failed_bundle):
        assert watermark_artists(fit_figure(bundle)), bundle.result.status


def test_converged_fit_draws_curve_data_and_confidence_band(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    assert bundle.confidence_band is not None
    fig = fit_figure(bundle)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == bundle.equation_text  # equation as the subtitle
    curves = [line for line in ax.get_lines() if line.get_label() != "_nolegend_"]
    assert len(curves) == 1, "a converged fit draws exactly one curve line"
    assert any("fitted" in label for label in line_labels(fig))
    assert ax.collections, "the data points are drawn"
    assert len(ax.collections) >= 2, "data scatter plus at least the confidence band"


def test_band_toggles_control_what_is_drawn(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    assert bundle.prediction_band is not None
    default = fit_figure(bundle)
    assert len(default.axes[0].collections) == 2  # data + confidence band
    no_bands = fit_figure(bundle, show_confidence=False)
    assert len(no_bands.axes[0].collections) == 1  # data only
    prediction = fit_figure(bundle, show_confidence=False, show_prediction=True)
    assert len(prediction.axes[0].collections) == 2
    both = fit_figure(bundle, show_prediction=True)
    assert len(both.axes[0].collections) == 3


def test_figure_uses_the_shared_palette(poly_bundle: FitBundleFactory) -> None:
    """The exported report matches the app: rank 1 curve, neutral data and bands."""
    bundle = poly_bundle()
    fig = fit_figure(bundle, show_prediction=True)
    ax = fig.axes[0]
    curve = next(line for line in ax.get_lines() if line.get_label() != "_nolegend_")
    assert to_hex(curve.get_color()).upper() == MODEL_COLORS[0].upper()
    # draw order: data scatter, then the confidence band, then the prediction band
    data, confidence, prediction = ax.collections[:3]
    assert to_hex(data.get_facecolor()[0]).upper() == DATA_COLOR["light"].upper()
    assert to_rgba(confidence.get_facecolor()[0]) == to_rgba(BAND_CONFIDENCE_FILL["light"])
    assert to_rgba(prediction.get_facecolor()[0]) == to_rgba(BAND_PREDICTION_FILL["light"])


def test_title_goes_above_the_equation_subtitle(poly_bundle: FitBundleFactory) -> None:
    fig = fit_figure(poly_bundle(), title="My fit")
    assert fig._suptitle is not None and fig._suptitle.get_text() == "My fit"
    assert fig.axes[0].get_title() != "My fit"  # the axes title stays the equation


def test_log_model_figure_builds_and_draws_its_curve(log_bundle: FitBundle) -> None:
    assert log_bundle.curve is not None
    fig = fit_figure(log_bundle)
    assert len(fig.axes[0].get_lines()) == 1
    assert fig.axes[0].get_title() == log_bundle.equation_text


def test_constant_y_figure_builds(constant_bundle: FitBundle) -> None:
    fig = fit_figure(constant_bundle)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_lines()


def test_non_converged_fit_never_draws_a_curve(
    non_converged_bundle: FitBundle, failed_bundle: FitBundle
) -> None:
    for bundle in (non_converged_bundle, failed_bundle):
        fig = fit_figure(bundle)
        assert not any(
            isinstance(artist, Line2D) for ax in fig.axes for artist in ax.get_children()
        ), f"a {bundle.result.status.value} fit must not show a curve line"
        texts = " | ".join(figure_texts(fig))
        assert "FIT DID NOT CONVERGE" in texts
        assert bundle.result.termination_reason in texts
        assert fig.axes[0].collections, "the data points are still drawn"


def test_non_converged_banner_uses_the_actual_reason(non_converged_bundle: FitBundle) -> None:
    reason = non_converged_bundle.result.termination_reason
    fig = fit_figure(non_converged_bundle)
    texts = figure_texts(fig)
    assert any(text.startswith("FIT DID NOT CONVERGE: ") and reason in text for text in texts)


def test_axis_labels_come_from_the_data_names(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle(x_name="time (days)", y_name="strength (MPa)")
    fig = fit_figure(bundle)
    assert fig.axes[0].get_xlabel() == "time (days)"
    assert fig.axes[0].get_ylabel() == "strength (MPa)"


def test_dollar_signs_render_literally_never_as_mathtext(
    poly_bundle: FitBundleFactory,
) -> None:
    """``$...$`` in user text must not enter matplotlib's mathtext parser."""
    from io import BytesIO

    from curvestudio_plotting import residual_figure, save_figure

    bundle = poly_bundle(x_name="$\\bad$", y_name="Cost ($) vs ($)")
    fig = fit_figure(bundle, title="Cost ($) vs ($)")
    assert fig.axes[0].get_xlabel() == "$\\bad$"  # the label text is kept verbatim
    for figure in (fig, residual_figure(bundle)):
        save_figure(figure, BytesIO(), "png")  # rendering parses mathtext: must not raise
        save_figure(figure, BytesIO(), "svg")


def test_equation_text_with_dollar_signs_renders(poly_bundle: FitBundleFactory) -> None:
    from io import BytesIO

    from curvestudio_plotting import save_figure

    bundle = poly_bundle()
    marked = bundle.model_copy(update={"equation_text": "y = $a$ * x + $b$"})
    fig = fit_figure(marked)
    assert fig.axes[0].get_title() == "y = $a$ * x + $b$"
    save_figure(fig, BytesIO(), "png")  # must not raise: the title is never parsed as math
