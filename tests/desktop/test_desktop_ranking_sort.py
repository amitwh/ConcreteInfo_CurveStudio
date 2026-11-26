"""The ranking table's display-only column sorting (design item 12).

Clicking a header reorders the *visible* rows — ascending, then descending, then back to
the ranked order — while the table's own rows (the stored ranking) never change, and the
rank column stays visible so the true order is always readable.
"""

from PySide6.QtCore import Qt

from curvestudio.discovery import CandidateMetrics, RankedFit
from curvestudio_desktop.views.ranking_table import COLUMNS, RankingTableView


def _row(
    model_id: str, name: str, rank: int, r2: float | None, rmse: float | None = 0.1
) -> RankedFit:
    return RankedFit(
        rank=rank,
        model_id=model_id,
        name=name,
        category="test",
        metrics=CandidateMetrics(k=2, n=20, r2=r2, adj_r2=r2, rmse=rmse),
        status="converged",
    )


ROWS = (
    _row("best", "Best model", 1, 0.99),
    _row("mid", "Middle model", 2, 0.90),
    _row("worst", "Worst model", 3, 0.50),
)


def displayed_ids(view: RankingTableView) -> list[str]:
    """The model ids in visual order: proxy row 0 is the first row displayed."""
    proxy = view.model()
    assert proxy is not None
    ids: list[str] = []
    for row in range(proxy.rowCount()):
        source_row = proxy.mapToSource(proxy.index(row, 0)).row()
        ranked = view.rows()[source_row]
        ids.append(ranked.model_id)
    return ids


def click_header(view: RankingTableView, column: int) -> None:
    view.horizontalHeader().sectionClicked.emit(column)


def test_header_clicks_cycle_display_order_and_never_the_ranking(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)

    assert displayed_ids(view) == ["best", "mid", "worst"]
    r2_column = COLUMNS.index("R²")

    click_header(view, r2_column)  # ascending
    assert displayed_ids(view) == ["worst", "mid", "best"]
    assert view.rows() == ROWS  # the stored ranking is untouched

    click_header(view, r2_column)  # descending
    assert displayed_ids(view) == ["best", "mid", "worst"]
    assert view.rows() == ROWS

    click_header(view, r2_column)  # back to the ranked order
    assert displayed_ids(view) == ["best", "mid", "worst"]
    assert view.rows() == ROWS


def test_sorting_another_column_starts_ascending(qtbot) -> None:
    # a ranking whose stored order is by R², with RMSE ordered differently
    rows = (
        _row("best", "Best model", 1, 0.99, rmse=0.20),
        _row("mid", "Middle model", 2, 0.90, rmse=0.05),
        _row("worst", "Worst model", 3, 0.50, rmse=0.10),
    )
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(rows)
    rmse_column = COLUMNS.index("RMSE")

    click_header(view, rmse_column)
    assert displayed_ids(view) == ["mid", "worst", "best"]  # ascending RMSE
    click_header(view, rmse_column)
    assert displayed_ids(view) == ["best", "worst", "mid"]  # descending


def test_rank_column_sorts_by_rank_descending(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)
    rank_column = COLUMNS.index("Rank")

    click_header(view, rank_column)  # ranks ascend already: stable
    assert displayed_ids(view) == ["best", "mid", "worst"]
    click_header(view, rank_column)  # descending
    assert displayed_ids(view) == ["worst", "mid", "best"]


def test_model_column_sorts_alphabetically(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)
    model_column = COLUMNS.index("Model")

    click_header(view, model_column)
    assert displayed_ids(view) == ["best", "mid", "worst"]  # alphabetical by name
    click_header(view, model_column)
    assert displayed_ids(view) == ["worst", "mid", "best"]


def test_undefined_metrics_sort_last_in_both_directions(qtbot) -> None:
    rows = (
        _row("a", "A", 1, 0.9),
        _row("none", "Undefined", 2, None),
        _row("b", "B", 3, 0.5),
    )
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(rows)
    r2_column = COLUMNS.index("R²")

    click_header(view, r2_column)
    assert displayed_ids(view) == ["b", "a", "none"]  # ascending, n/a last
    click_header(view, r2_column)
    assert displayed_ids(view) == ["a", "b", "none"]  # descending, n/a still last


def test_selection_maps_through_the_display_order(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)
    r2_column = COLUMNS.index("R²")
    click_header(view, r2_column)  # display: worst, mid, best

    # selecting the first *displayed* row reports the row that is shown there
    view.selectRow(0)
    assert view.selected_row() is not None and view.selected_row().model_id == "worst"

    # selecting by stored index selects the same row wherever it is displayed
    row = view.select_row(0)  # "best": displayed third
    assert row is not None and row.model_id == "best"
    assert view.selected_row() is not None and view.selected_row().model_id == "best"


def test_sort_indicator_follows_the_cycle(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)
    header = view.horizontalHeader()
    r2_column = COLUMNS.index("R²")

    assert header.sortIndicatorSection() == -1
    click_header(view, r2_column)
    assert header.sortIndicatorSection() == r2_column
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    click_header(view, r2_column)
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    click_header(view, r2_column)
    assert header.sortIndicatorSection() == -1


def test_set_rows_keeps_the_display_sort(qtbot) -> None:
    view = RankingTableView()
    qtbot.addWidget(view)
    view.set_rows(ROWS)
    r2_column = COLUMNS.index("R²")
    click_header(view, r2_column)

    # a re-render (re-rank or new run) keeps the user's display order
    view.set_rows(tuple(reversed(ROWS)))
    assert displayed_ids(view) == ["worst", "mid", "best"]
    assert [row.model_id for row in view.rows()] == ["worst", "mid", "best"]
