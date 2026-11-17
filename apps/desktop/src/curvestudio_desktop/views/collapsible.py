"""A collapsible section: a checkable tool-button header over an optional body."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Header row with an arrow; the body (a list of ``id — reason`` rows) shows on demand."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(False)
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.ArrowType.RightArrow)
        self._button.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self._list = QListWidget()
        self._list.setVisible(False)
        self._list.setWordWrap(True)
        self._list.setAlternatingRowColors(True)
        self._button.toggled.connect(self._toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._button)
        layout.addWidget(self._list)

    def _toggle(self, open: bool) -> None:
        self._list.setVisible(open)
        self._button.setArrowType(Qt.ArrowType.DownArrow if open else Qt.ArrowType.RightArrow)

    def set_items(self, pairs: list[tuple[str, str]]) -> None:
        """Fill the body with ``(label, reason)`` rows; the reason is the row's tooltip."""
        self._list.clear()
        for label, reason in pairs:
            item = QListWidgetItem(label if not reason else f"{label} — {reason}")
            if reason:
                item.setToolTip(reason)
            self._list.addItem(item)
        count = len(pairs)
        self._button.setText(f"{self._title()} ({count})")
        self.setVisible(count > 0)

    def _title(self) -> str:
        text = self._button.text()
        return text.split(" (")[0]

    def title(self) -> str:
        """The header as shown (including its live count)."""
        return self._button.text()

    def labels(self) -> list[str]:
        return [self._list.item(row).text() for row in range(self._list.count())]
