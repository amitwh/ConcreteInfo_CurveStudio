"""The generated application stylesheet.

``stylesheet(scheme)`` renders the brand tokens into one QSS string. The palette in
:mod:`curvestudio_desktop.theme` uses the same tokens, so native popups (combo
lists, menus) and styled widgets agree in both light and dark themes. Buttons pick
up a semantic role through a dynamic ``role`` property: ``primary`` (accent),
``destructive`` (danger) or the default secondary surface button.
"""

from string import Template

from curvestudio_desktop.theme import Scheme, tokens

_QSS = Template(
    """
QMainWindow, QDialog {
    background: $bg;
}
QMenuBar {
    background: $bg;
    color: $text;
    border-bottom: 1px solid $border;
    padding: 2px 4px;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background: $selection;
    color: $heading;
}
QMenu {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 10px;
    border-radius: 6px;
}
QMenu::item:selected {
    background: $selection;
    color: $heading;
}
QMenu::item:disabled {
    color: $muted;
}
QMenu::icon {
    padding-left: 6px;
}
QMenu::separator {
    height: 1px;
    background: $border;
    margin: 4px 8px;
}
QToolBar {
    background: $surface;
    border: none;
    border-bottom: 1px solid $border;
    spacing: 2px;
    padding: 3px 8px;
}
QToolBar::separator {
    width: 1px;
    background: $border;
    margin: 5px 6px;
}
QToolButton {
    background: transparent;
    color: $text;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px;
}
QToolButton:hover {
    background: $selection;
    border-color: $border;
}
QToolButton:pressed, QToolButton:checked {
    background: $selection;
    border-color: $accent;
}
QToolButton:disabled {
    color: $muted;
}
QDockWidget {
    background: $bg;
    color: $text;
}
QDockWidget::title {
    background: $surface;
    padding: 6px;
    border-bottom: 1px solid $border;
}
QWidget#dockTitleBar {
    background: $surface;
    border-bottom: 1px solid $border;
}
QWidget#dockTitleBar QToolButton {
    padding: 2px;
}
QLabel#dockTitleLabel {
    background: transparent;
    color: $heading;
    font-weight: 600;
}
QStatusBar {
    background: $surface;
    color: $muted;
    border-top: 1px solid $border;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: $muted;
    padding: 0 6px;
}
QLabel#observationsLabel {
    color: $heading;
    font-weight: 600;
}
QLabel#noNetworkLabel {
    color: $muted;
}
QPushButton#cancelJobButton {
    padding: 2px 8px;
}
QTabWidget::pane {
    background: $surface;
    border: 1px solid $border;
    border-radius: 6px;
    top: -1px;
}
QTabWidget::tab-bar {
    left: 4px;
}
QTabBar::tab {
    background: transparent;
    color: $muted;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 6px 10px;
}
QTabBar::tab:hover {
    background: $selection;
    color: $heading;
}
QTabBar::tab:selected {
    color: $heading;
    border-bottom-color: $accent;
}
QHeaderView::section {
    background: $bg;
    color: $muted;
    border: none;
    border-bottom: 1px solid $border;
    padding: 5px 8px;
}
QTableView, QTreeWidget, QTreeView, QListWidget {
    background: $surface;
    alternate-background-color: $bg;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    gridline-color: $border;
    selection-background-color: $selection;
    selection-color: $heading;
}
QTableView::item, QTreeWidget::item, QListWidget::item {
    padding: 3px 6px;
}
QTableView::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {
    background: $selection;
    color: $heading;
}
QTableCornerButton::section {
    background: $bg;
    border: none;
}
QPushButton {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    color: $heading;
    border-color: $accent;
}
QPushButton:pressed {
    background: $selection;
}
QPushButton:default {
    border-color: $accent;
}
QPushButton:disabled {
    background: $bg;
    color: $muted;
    border-color: $border;
}
QPushButton[role="primary"] {
    background: $accent;
    color: $on_accent;
    border-color: $accent;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background: $accent_dark;
    border-color: $accent_dark;
    color: $on_accent;
}
QPushButton[role="primary"]:pressed {
    background: $accent_dark;
}
QPushButton[role="primary"]:disabled {
    background: $bg;
    color: $muted;
    border-color: $border;
}
QPushButton[segment="true"] {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 4px 12px;
}
QPushButton[segment="true"]:hover {
    border-color: $accent;
    color: $heading;
}
QPushButton[segment="true"]:checked {
    background: $accent;
    color: $on_accent;
    border-color: $accent;
    font-weight: 600;
}
QPushButton[segment="true"]:checked:hover {
    background: $accent_dark;
    border-color: $accent_dark;
    color: $on_accent;
}
QPushButton[role="destructive"] {
    background: $danger;
    color: $on_accent;
    border-color: $danger;
    font-weight: 600;
}
QPushButton[role="destructive"]:hover {
    border-color: $heading;
    color: $on_accent;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: $accent;
    selection-color: $on_accent;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border-color: $accent;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QPlainTextEdit:disabled {
    background: $bg;
    color: $muted;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
/* An editable combo hosts a real QLineEdit child, and a spin box its own line edit.
   QComboBox hands that field the rect its own metrics allow (the combo's 30 px minus
   this rule's 5 px vertical padding: 18 px), so the generic QLineEdit padding above
   would apply a second inset inside it and clip the text to half height. The outer
   widget already draws the frame and the padding; the inner editor must not repeat
   them. */
QComboBox QLineEdit, QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    margin: 0;
}
QComboBox QAbstractItemView {
    background: $surface;
    color: $text;
    border: 1px solid $border;
    selection-background-color: $selection;
    selection-color: $heading;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: transparent;
    width: 14px;
    border: none;
}
QCheckBox {
    color: $text;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid $border;
    border-radius: 3px;
    background: $surface;
}
QCheckBox::indicator:hover {
    border-color: $accent;
}
QCheckBox::indicator:checked {
    background: $accent;
    border-color: $accent;
}
QGroupBox {
    background: transparent;
    color: $heading;
    border: 1px solid $border;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: $border;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: $accent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: $border;
    min-width: 24px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: $accent;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: transparent;
    width: 0;
    height: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}
QSplitter::handle {
    background: $border;
}
QSplitter::handle:hover {
    background: $accent;
}
QProgressBar {
    background: $bg;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background: $accent;
    border-radius: 5px;
}
QToolTip {
    background: $heading;
    color: $bg;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel#bannerWarning {
    background: $selection;
    color: $warning;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 6px 8px;
    font-weight: 600;
}
QLabel#bannerError {
    background: $selection;
    color: $danger;
    border: 1px solid $danger;
    border-radius: 6px;
    padding: 6px 8px;
    font-weight: 600;
}
QLabel#equationValidation {
    background: transparent;
    padding: 0;
}
QLabel#equationValidation[state="error"] {
    color: $danger;
}
QLabel#equationValidation[state="ok"] {
    color: $success;
}
QScrollArea {
    background: transparent;
    border: none;
}
QWidget#detailsHeader {
    background: $surface;
    border: 1px solid $border;
    border-radius: 6px;
}
QLabel#summaryModel, QLabel#detailsModel {
    color: $heading;
    font-weight: 600;
}
QLabel#summaryMeta, QLabel#detailsMeta {
    color: $muted;
}
QLabel#summaryHint {
    color: $muted;
    padding: 8px;
}
QLabel#summaryMetricCaption {
    color: $muted;
}
QLabel#summaryMetricValue {
    color: $heading;
}
QLabel#detailsMetrics {
    color: $text;
}
QLabel#statusBadge {
    background: $bg;
    color: $muted;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 1px 8px;
    font-weight: 600;
}
QLabel#statusBadge[state="ok"] {
    color: $success;
    border-color: $success;
}
QLabel#statusBadge[state="failed"] {
    color: $danger;
    border-color: $danger;
}
QTableView#rankingTable::item {
    padding: 3px 4px;
}
QHeaderView#rankingTableHeader::section {
    padding: 5px 4px;
}
"""
)


def stylesheet(scheme: Scheme) -> str:
    """Render the QSS for a resolved scheme."""
    return _QSS.substitute(tokens(scheme))


__all__ = ["stylesheet"]
