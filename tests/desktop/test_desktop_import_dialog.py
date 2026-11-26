"""Import dialogs: preview, column pickers, issue list with row numbers, OK gating."""

from desktop_data import (
    CLIPBOARD_COMMA_TEXT,
    CLIPBOARD_TAB_TEXT,
    GOOD_CSV,
    MANY_CSV,
    MANY_ROWS,
    TEXT_COLUMN_CSV,
)

from curvestudio_desktop.views.import_dialog import ClipboardImportDialog, ImportDialog


def test_file_dialog_previews_clean_data(qtbot, write_file) -> None:
    dialog = ImportDialog(path=write_file("points.csv", GOOD_CSV))
    qtbot.addWidget(dialog)
    assert dialog.column_choices() == ["x", "y"]
    assert dialog.selected_columns() == ("x", "y", None)
    assert dialog.preview_row_count() == 5
    assert dialog.issue_texts() == []
    assert dialog.ok_enabled() is True
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None


def test_preview_is_capped_at_50_rows(qtbot, write_file) -> None:
    dialog = ImportDialog(path=write_file("many.csv", MANY_CSV))
    qtbot.addWidget(dialog)
    assert dialog.preview_row_count() == 50
    assert dialog.total_rows() == MANY_ROWS


def test_errors_are_listed_with_row_numbers_and_block_ok(qtbot, write_file) -> None:
    dialog = ImportDialog(path=write_file("bad.csv", TEXT_COLUMN_CSV))
    qtbot.addWidget(dialog)
    issues = dialog.issue_texts()
    assert any("NON_NUMERIC" in text for text in issues)
    assert any("2" in text for text in issues)  # the bad cell is on displayed row 2
    assert dialog.ok_enabled() is False
    assert dialog.outcome() is None or dialog.outcome().xy is None


def test_warnings_do_not_block_ok(qtbot, write_file) -> None:
    text = "x,y\n1,2.0\n2,4.1\n2,4.1\n3,6.2\n4,7.9\n"  # duplicate observation
    dialog = ImportDialog(path=write_file("dup.csv", text))
    qtbot.addWidget(dialog)
    assert any("DUPLICATE_ROW" in text_ for text_ in dialog.issue_texts())
    assert dialog.ok_enabled() is True


def test_column_pickers_can_retarget_columns(qtbot, write_file) -> None:
    text = "x,y,z\n1,2.0,3.5\n2,4.1,6.9\n3,6.2,10.4\n4,7.9,13.9\n5,10.3,17.4\n"
    dialog = ImportDialog(path=write_file("three.csv", text))
    qtbot.addWidget(dialog)
    assert dialog.selected_columns() == ("x", "y", None)
    dialog.select_columns(x="x", y="z", weight=None)
    assert dialog.selected_columns() == ("x", "z", None)
    assert dialog.ok_enabled() is True
    outcome = dialog.outcome()
    assert outcome is not None and outcome.y_column == "z"


def test_weight_column_can_be_chosen(qtbot, write_file) -> None:
    text = "x,y,w\n1,2.0,1.0\n2,4.1,1.5\n3,6.2,1.0\n4,7.9,2.0\n5,10.3,1.0\n"
    dialog = ImportDialog(path=write_file("w.csv", text))
    qtbot.addWidget(dialog)
    dialog.select_columns(x="x", y="y", weight="w")
    outcome = dialog.outcome()
    assert outcome is not None
    assert outcome.xy is not None and outcome.xy.weights is not None


def test_unreadable_file_shows_error_not_traceback(qtbot, write_file) -> None:
    dialog = ImportDialog(path=write_file("garbage.csv", b"\xff\x9c\x00 broken"))
    qtbot.addWidget(dialog)
    assert dialog.ok_enabled() is False
    assert any("UNREADABLE" in text for text in dialog.issue_texts())
    assert dialog.outcome() is None


def test_no_file_selected_leaves_ok_disabled(qtbot) -> None:
    dialog = ImportDialog()
    qtbot.addWidget(dialog)
    assert dialog.ok_enabled() is False


def test_clipboard_dialog_parses_tabs_and_commas(qtbot) -> None:
    tabbed = ClipboardImportDialog(initial_text=CLIPBOARD_TAB_TEXT)
    qtbot.addWidget(tabbed)
    outcome = tabbed.outcome()
    assert outcome is not None and outcome.xy is not None
    assert outcome.x_column == "time" and len(outcome.xy.x) == 5

    comma = ClipboardImportDialog(initial_text=CLIPBOARD_COMMA_TEXT)
    qtbot.addWidget(comma)
    outcome = comma.outcome()
    assert outcome is not None and outcome.xy is not None


def test_clipboard_dialog_blocks_bad_data(qtbot) -> None:
    broken = ClipboardImportDialog(initial_text="a,b\n1,2\nx,y,z\n3,4\n")
    qtbot.addWidget(broken)
    assert broken.ok_enabled() is False
    assert broken.issue_texts()


def test_clipboard_from_system_clipboard(qtbot) -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.clipboard().setText(CLIPBOARD_COMMA_TEXT)
    dialog = ClipboardImportDialog.from_clipboard()
    qtbot.addWidget(dialog)
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None


def _counting_import(monkeypatch):
    """Wrap the workflow's import_dataset with a call counter.

    The dialog resolves the importer where it lives (its module import is deferred off
    the window's start-up path), so the counter wraps it there.
    """
    import curvestudio.workflow as workflow

    calls = {"n": 0}
    real = workflow.import_dataset

    def counting(source, filename, **kwargs):
        calls["n"] += 1
        return real(source, filename, **kwargs)

    monkeypatch.setattr(workflow, "import_dataset", counting)
    return calls


THREE_COLUMN_CSV = "x,y,z\n1,2.0,3.5\n2,4.1,6.9\n3,6.2,10.4\n4,7.9,13.9\n5,10.3,17.4\n"


def test_column_changes_remap_without_reparsing(qtbot, write_file, monkeypatch) -> None:
    """Switching columns reuses the parsed dataset — the file is read exactly once."""
    calls = _counting_import(monkeypatch)
    dialog = ImportDialog(path=write_file("three.csv", THREE_COLUMN_CSV))
    qtbot.addWidget(dialog)
    assert calls["n"] == 1
    dialog.select_columns(x="x", y="z", weight=None)
    dialog.select_columns(x="z", y="y", weight="x")
    assert calls["n"] == 1, "column changes must remap in memory, not re-parse"
    outcome = dialog.outcome()
    assert outcome is not None
    assert (outcome.x_column, outcome.y_column, outcome.weight_column) == ("z", "y", "x")
    assert outcome.xy is not None


def test_choosing_a_new_file_parses_again(qtbot, write_file, monkeypatch) -> None:
    calls = _counting_import(monkeypatch)
    dialog = ImportDialog(path=write_file("a.csv", GOOD_CSV))
    qtbot.addWidget(dialog)
    assert calls["n"] == 1
    dialog._set_path(write_file("b.csv", THREE_COLUMN_CSV))
    assert calls["n"] == 2  # a new source really is parsed


def test_clipboard_column_changes_remap_without_reparsing(qtbot, monkeypatch) -> None:
    calls = _counting_import(monkeypatch)
    dialog = ClipboardImportDialog(
        initial_text="time,load,other\n0,0.5,1\n1,2.4,2\n2,4.6,3\n3,6.1,4\n4,8.5,5\n"
    )
    qtbot.addWidget(dialog)
    assert calls["n"] == 1
    dialog._x_box.setCurrentText("time")
    dialog._y_box.setCurrentText("other")
    assert calls["n"] == 1
    outcome = dialog.outcome()
    assert outcome is not None and outcome.y_column == "other"


def test_dialog_parse_can_run_async(qtbot, write_file) -> None:
    """Production passes an async runner: the dialog still ends up with an outcome."""
    from curvestudio_desktop.jobs import JobRunner

    dialog = ImportDialog(path=write_file("async.csv", GOOD_CSV), jobs=JobRunner())
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.outcome() is not None, timeout=5000)
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None
    assert dialog.ok_enabled() is True


def test_weight_remap_applies_weights(qtbot, write_file) -> None:
    text = "x,y,w\n1,2.0,1.0\n2,4.1,1.5\n3,6.2,1.0\n4,7.9,2.0\n5,10.3,1.0\n"
    dialog = ImportDialog(path=write_file("w.csv", text))
    qtbot.addWidget(dialog)
    dialog.select_columns(x="x", y="y", weight="w")
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None
    assert outcome.xy.weights is not None


# -- the embedded spreadsheet grid -------------------------------------------------------


def test_grid_takes_a_whole_pasted_spreadsheet_block(qtbot) -> None:
    from curvestudio_desktop.views.spreadsheet import SpreadsheetGrid

    grid = SpreadsheetGrid()
    qtbot.addWidget(grid)
    pasted = grid.paste_text("x\ty\n0\t1.5\n1\t2.5\n2\t4.0\n")
    assert pasted == 4
    assert (grid.rowCount(), grid.columnCount()) == (4, 2)
    assert grid.cell(0, 0) == "x" and grid.cell(0, 1) == "y"
    assert grid.cell(2, 1) == "2.5"


def test_grid_round_trip_keeps_quoted_cells(qtbot) -> None:
    """A spreadsheet cell holding a comma or a tab stays one cell, not two columns."""
    from curvestudio_desktop.views.spreadsheet import SpreadsheetGrid, parse_table

    grid = SpreadsheetGrid()
    qtbot.addWidget(grid)
    grid.paste_text('name\tvalue\n"a,b"\t1\n"x\ty"\t2\n')
    rows = grid.rows()
    assert rows[1] == ["a,b", "1"]
    assert rows[2] == ["x\ty", "2"]
    assert parse_table(grid.to_tsv()) == rows


def test_grid_paste_lands_at_the_current_cell_and_grows(qtbot) -> None:
    from curvestudio_desktop.views.spreadsheet import SpreadsheetGrid

    grid = SpreadsheetGrid()
    qtbot.addWidget(grid)
    grid.paste_text("a\tb\tc\n1\t2\t3\n", replace=True)

    grid.setCurrentCell(1, 1)  # overwrite the middle cell's row
    grid.paste_text("9\t8\n")
    assert grid.cell(1, 0) == "1"  # untouched
    assert grid.cell(1, 1) == "9" and grid.cell(1, 2) == "8"

    grid.setCurrentCell(0, 2)  # a paste wider than the grid grows a column
    grid.paste_text("p\tq\n")
    assert grid.columnCount() == 4
    assert grid.cell(0, 2) == "p" and grid.cell(0, 3) == "q"


def test_grid_ignores_an_empty_paste(qtbot) -> None:
    from curvestudio_desktop.views.spreadsheet import SpreadsheetGrid

    grid = SpreadsheetGrid()
    qtbot.addWidget(grid)
    assert grid.paste_text("   \n") == 0
    assert grid.is_blank()


def test_clipboard_dialog_imports_a_pasted_grid(qtbot) -> None:
    from PySide6.QtWidgets import QApplication

    dialog = ClipboardImportDialog()
    qtbot.addWidget(dialog)
    assert dialog.ok_enabled() is False  # nothing pasted yet

    QApplication.clipboard().setText("time\tload\n0\t1.2\n1\t2.3\n2\t4.4\n")
    dialog.paste_from_clipboard()
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None
    assert outcome.x_column == "time" and len(outcome.xy.x) == 3
    assert dialog.ok_enabled() is True


def test_grid_edits_refresh_the_preview(qtbot) -> None:
    dialog = ClipboardImportDialog(initial_text="x\ty\n1\t2.0\n2\t4.1\n3\t6.2\n")
    qtbot.addWidget(dialog)
    assert dialog.ok_enabled() is True

    dialog.grid().set_cell(1, 1, "not a number")
    qtbot.wait(500)  # the edit debounce, then the (synchronous) re-parse
    assert dialog.ok_enabled() is False
    assert any("NON_NUMERIC" in text for text in dialog.issue_texts())
