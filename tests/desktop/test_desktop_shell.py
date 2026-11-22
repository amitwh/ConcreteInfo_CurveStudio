"""The desktop shell: theme, fonts, icons, dock chrome and status bar polish."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel

from curvestudio_desktop.theme import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    apply_theme,
    mono_font,
    set_theme_preference,
    theme_preference,
)
from curvestudio_desktop.views.main_window import ANALYSIS_DOCK_WIDTH, MainWindow


@pytest.fixture()
def clean_appearance(qapp):
    """Apply themes freely in a test, then put the application back as it was."""
    style = qapp.style().objectName()
    palette = qapp.palette()
    sheet = qapp.styleSheet()
    font = qapp.font()
    yield qapp
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)
    qapp.setFont(font)


def test_bundled_fonts_load_and_mono_font_is_jetbrains(clean_appearance) -> None:
    apply_theme(clean_appearance, preference="light")
    families = QFontDatabase.families()
    assert "Inter" in families
    assert "JetBrains Mono" in families
    mono = mono_font()
    assert mono.family() == "JetBrains Mono"
    assert mono.fixedPitch()
    assert clean_appearance.font().family() == "Inter"


def test_stylesheet_and_palette_are_applied(clean_appearance) -> None:
    apply_theme(clean_appearance, preference="light")
    qss = clean_appearance.styleSheet()
    assert qss.strip()
    assert "QPushButton" in qss and "QMenu" in qss and "QScrollBar" in qss
    assert "bannerWarning" in qss and "bannerError" in qss
    window_color = clean_appearance.palette().color(QPalette.ColorRole.Window).name()
    assert window_color == LIGHT_TOKENS["bg"]


def test_toolbar_is_icon_only_movable_and_has_shortcut_tooltips(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    toolbar = window._toolbar
    assert toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
    assert toolbar.isMovable()
    for action in (
        window.action_new,
        window.action_open,
        window.action_save,
        window.action_import,
        window.action_clipboard,
        window.action_log_x,
        window.action_log_y,
        window.action_reset_view,
        window.action_export_plot,
    ):
        assert not action.icon().isNull(), f"{action.text()} has no icon"
        assert action.toolTip()
        shortcut = action.shortcut().toString()
        if shortcut:
            assert shortcut in action.toolTip()


def test_dock_title_bars_expose_collapse_float_and_close(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    for name, title in (("project", "Project"), ("analysis", "Analysis"), ("table", "Data table")):
        bar = window._title_bars[name]
        assert bar.label.text() == title
        assert bar.collapse_button.objectName() == "dockCollapseButton"
        assert bar.float_button.objectName() == "dockFloatButton"
        assert bar.close_button.objectName() == "dockCloseButton"
        for button in (bar.collapse_button, bar.float_button, bar.close_button):
            assert not button.icon().isNull()
            assert button.toolTip()
        features = window.docks[name].features()
        assert features & QDockWidget.DockWidgetFeature.DockWidgetMovable
        assert features & QDockWidget.DockWidgetFeature.DockWidgetFloatable
        assert features & QDockWidget.DockWidgetFeature.DockWidgetClosable


def test_collapsing_a_vertical_dock_hides_content_and_shrinks_to_a_strip(
    make_window, qtbot
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()
    dock = window.project_dock
    content = dock.widget()
    original_width = dock.width()
    assert original_width > 100
    assert not content.isHidden()

    window.set_dock_collapsed("project", True)
    QApplication.processEvents()
    assert content.isHidden()
    assert window._title_bars["project"].isVisible()
    assert not window._title_bars["project"].collapse_button.isHidden()
    assert dock.width() <= 44, f"collapsed dock is {dock.width()}px wide"

    window.set_dock_collapsed("project", False)
    QApplication.processEvents()
    assert not content.isHidden()
    assert dock.width() >= original_width - 10


def test_collapsing_the_bottom_dock_leaves_its_title_bar(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()
    dock = window.table_dock
    content = dock.widget()
    original_height = dock.height()

    window.set_dock_collapsed("table", True)
    QApplication.processEvents()
    assert content.isHidden()
    bar = window._title_bars["table"]
    assert bar.isVisible() and not bar.label.isHidden()
    assert dock.height() <= bar.sizeHint().height() + 4

    window.set_dock_collapsed("table", False)
    QApplication.processEvents()
    assert not content.isHidden()
    assert dock.height() > bar.sizeHint().height()
    assert original_height > 0


def test_collapsed_state_persists_across_windows(make_window, qtbot, settings, controller) -> None:
    first = make_window()
    qtbot.addWidget(first)
    first.resize(1280, 820)
    first.show()
    QApplication.processEvents()
    expanded_width = first.analysis_panel.width()
    first.set_dock_collapsed("analysis", True)
    first._save_layout()
    settings.sync()
    assert str(settings.value("main_window/collapsed/analysis")).lower() in ("true", "1")

    second = MainWindow(controller, settings=settings)
    qtbot.addWidget(second)
    second.resize(1280, 820)
    second.show()
    QApplication.processEvents()
    assert second._collapsed["analysis"] is True
    assert second.analysis_panel.widget().isHidden()
    assert second._title_bars["analysis"].collapsed is True

    # expanding after a restart restores the remembered size, not the strip
    second.set_dock_collapsed("analysis", False)
    QApplication.processEvents()
    assert second.analysis_panel.widget().isHidden() is False
    # a usable width is restored (the offscreen layout gives back slightly less)
    assert second.analysis_panel.width() >= expanded_width * 0.85


def test_theme_preference_persists_and_switching_changes_the_look(
    make_window, qtbot, settings, clean_appearance, controller
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.apply_theme_preference("dark")
    assert settings.value("appearance/theme") == "dark"
    assert theme_preference(settings) == "dark"
    assert clean_appearance.palette().color(QPalette.ColorRole.Window).name() == DARK_TOKENS["bg"]
    assert clean_appearance.styleSheet().strip()

    window.apply_theme_preference("light")
    assert settings.value("appearance/theme") == "light"
    assert clean_appearance.palette().color(QPalette.ColorRole.Window).name() == LIGHT_TOKENS["bg"]

    # a fresh window reads the stored choice into its radio actions
    set_theme_preference("dark", settings)
    fresh = MainWindow(controller, settings=settings)
    qtbot.addWidget(fresh)
    assert fresh._theme_actions["dark"].isChecked()
    assert not fresh._theme_actions["light"].isChecked()


def test_theme_switch_repaints_the_plot_surface(make_window, qtbot, clean_appearance) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()

    window.apply_theme_preference("light")
    light = window.plot_view.plot_background()
    window.apply_theme_preference("dark")
    dark = window.plot_view.plot_background()
    assert light != dark, "the plot kept its light surface inside the dark UI"
    assert light == "#ffffff"
    assert dark == "#1e1e1e"
    # and switching back restores the light surface (the repaint is honest both ways)
    window.apply_theme_preference("light")
    assert window.plot_view.plot_background() == light


def test_reset_layout_restores_defaults(make_window, qtbot, settings) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()
    window.set_dock_collapsed("table", True)
    window.table_dock.hide()
    assert settings.contains("main_window/collapsed/table")

    window._reset_layout()
    QApplication.processEvents()
    assert window._collapsed == {"project": False, "analysis": False, "table": False}
    assert not window.analysis_panel.widget().isHidden()
    assert not window.table_dock.isHidden()
    assert not settings.contains("main_window/state")
    assert not settings.contains("main_window/collapsed/table")
    assert window.analysis_panel.width() >= ANALYSIS_DOCK_WIDTH - 2


def test_export_actions_are_disabled_with_nothing_to_export(make_window, qtbot, write_file) -> None:
    from desktop_data import GOOD_CSV

    window = make_window()
    qtbot.addWidget(window)
    assert window.action_export_results.isEnabled() is False
    assert window.action_export_ranking.isEnabled() is False
    assert window.import_file(write_file("points.csv", GOOD_CSV))
    # data alone is not a result: both stay disabled until a fit or ranking exists
    assert window.action_export_results.isEnabled() is False
    assert window.action_export_ranking.isEnabled() is False


def test_window_icon_and_status_bar_polish(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    assert not window.windowIcon().isNull()
    assert window._job_progress.isHidden()
    label = window.statusBar().findChild(QLabel, "noNetworkLabel")
    assert label is not None and label.text().startswith("Offline")
    assert mono_font(9).family() == "JetBrains Mono"


# -- panels: resizable, movable, and always findable again -----------------------------


def test_the_data_table_divider_has_real_travel(make_window, qtbot) -> None:
    """The chart/data-table divider must be able to give the table real height.

    The Analysis panel's content used to set a 546 px minimum height, so the bottom dock
    had nothing to gain and the divider looked dead. The panel's floor now leaves room.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    table = window.table_dock
    before = table.height()
    window.resizeDocks([table], [400], Qt.Orientation.Vertical)
    QApplication.processEvents()
    assert table.height() >= before + 100, f"{before} -> {table.height()}"
    assert table.height() >= 280, table.height()
    # the Analysis dock shrinks and never reaches into the table dock
    assert window.analysis_panel.geometry().bottom() < table.geometry().top()


def test_a_floating_panel_can_be_made_smaller(make_window, qtbot) -> None:
    """A floated panel is a real window: it resizes down to a usable size, not a wall."""
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    dock = window.analysis_panel
    dock.setFloating(True)
    QApplication.processEvents()
    dock.resize(320, 400)
    QApplication.processEvents()
    assert dock.width() <= 340, f"the floating panel is pinned at {dock.width()} px"
    assert dock.height() <= 420, f"the floating panel is pinned at {dock.height()} px"


def test_a_floating_title_bar_starts_a_platform_window_move(
    make_window, qtbot, monkeypatch
) -> None:
    """The custom title bar hands a floating panel's drag to the window manager.

    Qt re-dispatches mouse moves after a window shifts, and a QDockWidget that receives
    a move without its press dereferences its drag state — moving the window by hand
    from a mouse handler crashed. ``startSystemMove`` is the supported path (and the
    reason ``float_button`` alone used to leave the panel unmovable).
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QWindow
    from PySide6.QtTest import QTest

    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    dock = window.project_dock
    dock.setFloating(True)
    QApplication.processEvents()

    calls: list[bool] = []
    monkeypatch.setattr(QWindow, "startSystemMove", lambda _self: calls.append(True) or True)
    bar = dock.titleBarWidget()
    QTest.mousePress(
        bar,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(bar.width() // 2, bar.height() // 2),
    )
    QTest.mouseRelease(
        bar,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(bar.width() // 2, bar.height() // 2),
    )
    QApplication.processEvents()
    assert calls == [True], "the floating title bar never asked for a platform move"


def test_double_clicking_a_floating_title_docks_it_back(make_window, qtbot) -> None:
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    dock = window.table_dock
    dock.setFloating(True)
    QApplication.processEvents()
    assert dock.isFloating()
    bar = dock.titleBarWidget()
    QTest.mouseDClick(
        bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10)
    )
    QApplication.processEvents()
    assert not dock.isFloating()
    assert window.dockWidgetArea(dock) is Qt.DockWidgetArea.BottomDockWidgetArea


def test_the_float_button_toggles_and_reflects_the_state(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    bar = window._title_bars["analysis"]
    assert bar.float_button.toolTip() == "Float panel"
    bar.float_button.click()
    QApplication.processEvents()
    assert window.analysis_panel.isFloating()
    assert bar.float_button.isChecked()
    assert bar.float_button.toolTip() == "Dock panel"

    bar.float_button.click()
    QApplication.processEvents()
    assert not window.analysis_panel.isFloating()
    assert not bar.float_button.isChecked()
    assert bar.float_button.toolTip() == "Float panel"


def test_panels_menu_rehomes_a_panel_closed_while_floating(make_window, qtbot) -> None:
    """Closing a floated panel must not leave it adrift: View → Panels brings it back."""
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    dock = window.analysis_panel
    dock.setFloating(True)
    QApplication.processEvents()
    action = window._panel_actions["analysis"]
    action.setChecked(False)  # what the X on a floating window does
    QApplication.processEvents()
    assert not dock.isVisible()

    action.setChecked(True)
    QApplication.processEvents()
    assert dock.isVisible()
    assert not dock.isFloating()
    assert window.dockWidgetArea(dock) is Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.width() >= ANALYSIS_DOCK_WIDTH * 0.9


def test_panels_menu_returns_a_panel_from_a_neighbours_area(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.table_dock)
    assert window.dockWidgetArea(window.table_dock) is Qt.DockWidgetArea.RightDockWidgetArea
    window.set_dock_visible("table", True)
    QApplication.processEvents()
    assert window.dockWidgetArea(window.table_dock) is Qt.DockWidgetArea.BottomDockWidgetArea
    assert window.table_dock.isVisible()


def test_the_editable_model_picker_shows_its_whole_line(make_window, qtbot) -> None:
    """The combo's inner QLineEdit fits the field the combo gives it.

    The generic QLineEdit padding rule also matched the combo's embedded editor, adding
    a second 5 px inset inside an 18 px field: 8 px of a 16 px line stayed visible.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    QApplication.processEvents()

    combo = window.analysis_panel.manual_view.model_combo
    editor = combo.lineEdit()
    assert editor is not None
    # before the fix the field was 18 px with 10 px of the generic padding inside it:
    # eight pixels of a sixteen-pixel line — the "half-cut model name"
    assert editor.height() >= editor.fontMetrics().height() + 2, (
        f"the editor is {editor.height()} px for {editor.fontMetrics().height()} px of text"
    )
