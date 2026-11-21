"""Rank colours: one palette across the plot, the ranking swatch and the overlay toggle."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

from curvestudio.discovery import CandidateMetrics, RankedFit
from curvestudio_desktop.views.plot_view import OverlayCurve, PlotView
from curvestudio_desktop.views.ranking_table import RankingTableModel
from curvestudio_plotting.palette import MODEL_COLORS, model_color


def _row(model_id: str, rank: int) -> RankedFit:
    return RankedFit(
        rank=rank,
        model_id=model_id,
        name=f"Model {model_id}",
        metrics=CandidateMetrics(k=2, n=10, r2=0.9),
    )


def test_a_single_curve_is_drawn_in_its_rank_colour(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    plot.set_fit([1.0, 2.0], [2.0, 4.0], rank=3, name="cubic")

    assert plot.drawn_curve_ranks() == [3]
    assert plot.drawn_curve_colors() == [model_color(3).lower()]
    assert plot.legend_names() == []  # one curve needs no legend


def test_three_overlay_curves_get_three_different_rank_colours(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    curves = [
        OverlayCurve("poly2", [1.0, 2.0], [2.0, 4.0], 1),
        OverlayCurve("exp", [1.0, 2.0], [2.0, 3.8], 2),
        OverlayCurve("log", [1.0, 2.0], [2.0, 3.6], 3),
    ]
    plot.set_fit_overlay(curves, band_x=[1.0, 2.0], confidence=([1.5, 3.5], [2.5, 4.5]))

    assert plot.drawn_curve_ranks() == [1, 2, 3]
    assert plot.drawn_curve_colors() == [model_color(rank).lower() for rank in (1, 2, 3)]
    assert len(set(plot.drawn_curve_colors())) == 3
    assert plot.legend_names() == ["poly2", "exp", "log"]


def test_bands_are_neutral_and_never_a_model_colour(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    plot.set_fit(
        [1.0, 2.0],
        [2.0, 4.0],
        ([1.5, 3.5], [2.5, 4.5]),
        ([1.0, 3.0], [3.0, 5.0]),
        rank=1,
    )
    colors = plot.band_colors()
    assert len(colors) == 2
    for colour in colors:
        red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
        assert red == green == blue, f"band {colour} has a hue"
        assert colour.upper() not in {model.upper() for model in MODEL_COLORS}


def test_the_rank_swatch_equals_the_curve_colour_for_that_rank() -> None:
    model = RankingTableModel()
    model.set_rows([_row("a", 1), _row("b", 3), _row("c", 8)])
    for row_index, rank in enumerate((1, 3, 8)):
        decoration = model.data(model.index(row_index, 0), Qt.ItemDataRole.DecorationRole)
        assert decoration is not None
        assert decoration.name().upper() == model_color(rank).upper()


def test_show_top3_overlays_three_curves_and_persists(
    make_window, qtbot, write_file, settings
) -> None:
    from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored

    window = make_window()
    qtbot.addWidget(window)
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    result = wait_stored(qtbot, window, window.selection.active_id)
    view = window.analysis_panel.autofit_view
    rows = view.table.rows()
    if sum(row.model_id in result.curves for row in rows[:3]) < 3:
        pytest.skip("this tiny run did not sample three distinct top rows")

    view.table.select_row(0)
    assert len(window.plot_view.drawn_curve_ranks()) == 1  # off: the selected curve only

    view.top3_check.setChecked(True)
    assert len(window.plot_view.drawn_curve_ranks()) == 3
    ranks = window.plot_view.drawn_curve_ranks()
    assert ranks == [1, 2, 3]
    assert window.plot_view.drawn_curve_colors() == [model_color(rank).lower() for rank in ranks]
    assert len(set(window.plot_view.drawn_curve_colors())) == 3
    assert window.plot_view.legend_names() == [row.name for row in rows[:3]]
    if window.plot_view.band_colors():
        for colour in window.plot_view.band_colors():
            red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
            assert red == green == blue
    assert str(settings.value("view/showTop3")).strip().lower() in ("true", "1")

    second = make_window()
    qtbot.addWidget(second)
    assert second.analysis_panel.autofit_view.show_top3() is True  # remembered

    view.top3_check.setChecked(False)
    assert len(window.plot_view.drawn_curve_ranks()) == 1
    assert window.plot_view.legend_names() == []
