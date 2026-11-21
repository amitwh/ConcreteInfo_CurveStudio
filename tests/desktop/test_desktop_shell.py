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
