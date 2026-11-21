"""Position colours: one palette across the plot, the ranking swatch and the overlay toggle.

Colours come from the row's 1-based position in the displayed ranking, never from its
(possibly tied) ``RankedFit.rank`` — ties must stay visibly distinct.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt

from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CandidateMetrics,
    RankedFit,
    RowCurve,
)
from curvestudio_desktop.views.main_window import MainWindow
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


def _result_with_curves(rows: list[RankedFit]) -> AutoFitResult:
    return AutoFitResult(
        config=AutoFitConfig(),
        dataset_fingerprint="f" * 64,
        n_active=10,
        ranked=tuple(rows),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
        curves={row.model_id: RowCurve(x=[0.0, 1.0], y=[1.0, 2.0]) for row in rows},
    )


def test_a_single_curve_is_drawn_in_its_position_colour(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    plot.set_fit([1.0, 2.0], [2.0, 4.0], position=3, name="cubic")

    assert plot.drawn_curve_positions() == [3]
    assert plot.drawn_curve_colors() == [model_color(3).lower()]
    assert plot.legend_names() == []  # one curve needs no legend


def test_three_overlay_curves_get_three_different_position_colours(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    curves = [
        OverlayCurve("poly2", [1.0, 2.0], [2.0, 4.0], 1),
        OverlayCurve("exp", [1.0, 2.0], [2.0, 3.8], 2),
        OverlayCurve("log", [1.0, 2.0], [2.0, 3.6], 3),
    ]
    plot.set_fit_overlay(curves, band_x=[1.0, 2.0], confidence=([1.5, 3.5], [2.5, 4.5]))

    assert plot.drawn_curve_positions() == [1, 2, 3]
    assert plot.drawn_curve_colors() == [model_color(position).lower() for position in (1, 2, 3)]
    assert len(set(plot.drawn_curve_colors())) == 3
    assert plot.legend_names() == ["poly2", "exp", "log"]


def test_tied_ranks_still_get_three_distinct_overlay_colours(qtbot) -> None:
    """A 1, 2, 2 ranking must draw three different colours (positions 1, 2, 3)."""
    rows = [_row("a", 1), _row("b", 2), _row("c", 2)]
    overlay = MainWindow._overlay_curves(_result_with_curves(rows), rows, "a")
    assert [entry.position for entry in overlay] == [1, 2, 3]

    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([0.0, 1.0]), np.array([1.0, 2.0]), "x", "y")
    plot.set_fit_overlay(overlay)
    assert plot.drawn_curve_positions() == [1, 2, 3]
    assert plot.drawn_curve_colors() == [
        model_color(1).lower(),
        model_color(2).lower(),
        model_color(3).lower(),
    ]
    assert len(set(plot.drawn_curve_colors())) == 3
    assert plot.legend_names() == ["Model a", "Model b", "Model c"]


def test_bands_are_neutral_and_never_a_model_colour(qtbot) -> None:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    plot.set_fit(
        [1.0, 2.0],
        [2.0, 4.0],
        ([1.5, 3.5], [2.5, 4.5]),
        ([1.0, 3.0], [3.0, 5.0]),
        position=1,
    )
    colors = plot.band_colors()
    assert len(colors) == 2
    for colour in colors:
        red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
        assert red == green == blue, f"band {colour} has a hue"
        assert colour.upper() not in {model.upper() for model in MODEL_COLORS}


def test_every_swatch_equals_the_curve_colour_for_its_position_including_ties(qtbot) -> None:
    rows = [_row("a", 1), _row("b", 2), _row("c", 2)]
    model = RankingTableModel()
    model.set_rows(rows)
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.set_data(np.array([0.0, 1.0]), np.array([1.0, 2.0]), "x", "y")
    for index, row in enumerate(rows):
        position = index + 1
        decoration = model.data(model.index(index, 0), Qt.ItemDataRole.DecorationRole)
        assert decoration is not None
        assert decoration.name().upper() == model_color(position).upper()
        # selecting the row with the overlay off draws its curve in the swatch colour
        plot.set_fit([0.0, 1.0], [1.0, 2.0], position=position, name=row.name)
        assert plot.drawn_curve_colors() == [model_color(position).lower()]


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
    assert len(window.plot_view.drawn_curve_positions()) == 1  # off: selected curve only

    view.top3_check.setChecked(True)
    assert len(window.plot_view.drawn_curve_positions()) == 3
    positions = window.plot_view.drawn_curve_positions()
    assert positions == [1, 2, 3]
    assert window.plot_view.drawn_curve_colors() == [
        model_color(position).lower() for position in positions
    ]
    # the first three rows may include tied ranks; the colours must still differ
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
    assert len(window.plot_view.drawn_curve_positions()) == 1
    assert window.plot_view.legend_names() == []
