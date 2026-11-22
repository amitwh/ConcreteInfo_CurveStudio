"""A searchable dropdown: type to filter, pick from the list, never free text.

The Manual fit model picker has to be usable without knowing the catalogue: typing three
letters should narrow 200 models to a handful. ``QComboBox`` alone reads as a text box
(and, without an explicitly drawn arrow, even loses the affordance), so this subclass:

* keeps the standard combo list (grouped, with separators) as the *authority*,
* adds a case-insensitive, contains-matching completer for type-to-filter,
* refuses free text (``NoInsert``) — the value is always a catalogue entry,
* draws a chevron in the drop-down area so it reads as a dropdown in every theme.

Callers map a selection through the item *data* (the model id), never the index: the
category separators occupy item positions, so index arithmetic against the model list is
wrong by the number of separators above the row.
"""

from collections.abc import Sequence

from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

from curvestudio_desktop.views.icons import icon

_SEARCH_HINT = "type to search"
_CHEVRON_INSET = 4


class SearchableComboBox(QComboBox):
    """An editable combo used strictly as a searchable picker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(10)
        completer = QCompleter(self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setMaxVisibleItems(12)
        self.setCompleter(completer)
        editor = self.lineEdit()
        if editor is not None:
            editor.setPlaceholderText(_SEARCH_HINT)
            editor.setClearButtonEnabled(False)

    # -- search ----------------------------------------------------------------------------

    def set_search_entries(self, entries: Sequence[str]) -> None:
        """The strings the completer filters (``"Name (id) — category"``)."""
        completer = self.completer()
        if completer is None:
            return
        completer.setModel(QStringListModel(list(entries), completer))

    def select_by_id(self, model_id: str) -> bool:
        """Select the item carrying ``model_id`` in its data (never by position)."""
        index = self.findData(model_id)
        if index < 0:
            return False
        self.setCurrentIndex(index)
        return True

    def selected_id(self) -> str | None:
        """The id stored on the current item, or ``None`` (the custom-equation entry)."""
        data = self.currentData()
        return data if isinstance(data, str) else None

    def restore_text(self) -> None:
        """Put the current item's text back after an abandoned search."""
        editor = self.lineEdit()
        if editor is not None:
            editor.setText(self.itemText(self.currentIndex()))

    # -- look ------------------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        option = self._option()
        arrow = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxArrow,
            self,
        )
        painter = QStylePainter(self)
        chevron = icon("chevron-down")
        chevron.paint(
            painter,
            arrow.adjusted(_CHEVRON_INSET, 0, -_CHEVRON_INSET, 0),
            Qt.AlignmentFlag.AlignCenter,
        )

    def _option(self) -> QStyleOptionComboBox:
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        return option


__all__ = ["SearchableComboBox"]
