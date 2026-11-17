"""``residual_figure``: the 2x2 diagnostics panel set, honest when undefined."""

from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from curvestudio.workflow import FitBundle
from curvestudio_plotting import residual_figure

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]

PANEL_TITLES = ("Residuals vs fitted", "Residuals vs x", "Normal Q-Q", "Histogram")


def axes_texts(ax: Axes) -> list[str]:
    return [t.get_text() for t in ax.texts]


def test_converged_fit_has_four_populated_panels(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    assert bundle.summary is not None
    fig = residual_figure(bundle)
    assert len(fig.axes) == 4
    for ax, title in zip(fig.axes, PANEL_TITLES, strict=True):
        assert ax.get_title() == title
        drawn = bool(ax.collections or ax.lines or ax.patches or ax.containers)
        assert drawn, f"panel {title!r} must plot data for a converged fit"
        assert not any(t.strip().lower() == "n/a" for t in axes_texts(ax))
    qq = fig.axes[2]
    assert qq.get_xlabel() and qq.get_ylabel()


def test_undefined_summary_shows_na_panels(non_converged_bundle: FitBundle) -> None:
    assert non_converged_bundle.summary is None
    fig: Figure = residual_figure(non_converged_bundle)
    assert len(fig.axes) == 4
    for ax, title in zip(fig.axes, PANEL_TITLES, strict=True):
        assert ax.get_title() == title
        texts = axes_texts(ax)
        assert any(t.strip().lower() == "n/a" for t in texts), (
            f"panel {title!r} must say n/a when its data is undefined"
        )


def test_failed_fit_shows_na_panels(failed_bundle: FitBundle) -> None:
    fig = residual_figure(failed_bundle)
    all_texts = " ".join(t for ax in fig.axes for t in axes_texts(ax))
    assert "n/a" in all_texts
