"""The Help browser: the canonical help text, section by section, with search.

``docs/help-content.md`` is copied verbatim into the package under
``resources/help/`` and read through ``importlib.resources``, so the frozen app renders
exactly the document the tests pin. The dialog is non-modal, remembers its geometry in
QSettings (``help/geometry``) and lets the user fit and read at the same time.
"""

import html
import re
from dataclasses import dataclass
from importlib import resources

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

GEOMETRY_KEY = "help/geometry"
TITLE = "Help contents"
MINIMUM_SIZE = (640, 420)
DEFAULT_SIZE = (940, 660)

_HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_UNORDERED = re.compile(r"^\s*[-*]\s+")
_ORDERED = re.compile(r"^\s*\d+\.\s+")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_INLINE = re.compile(
    r"`(?P<code>[^`]+)`"
    r"|\*\*(?P<bold>[^*]+)\*\*"
    r"|\*(?P<italic>[^*]+)\*"
    r"|\[(?P<label>[^\]]+)\]\((?P<href>[^)]+)\)"
    r"|<(?P<auto>https?://[^>]+)>"
)
_BROWSER_CSS = """
code { font-family: 'JetBrains Mono', monospace; }
table { border-collapse: collapse; }
th, td { border: 1px solid #999999; padding: 4px 8px; }
th { font-weight: 600; }
"""


@dataclass(frozen=True)
class HelpSection:
    """One ``##`` section: its heading text and its Markdown body."""

    title: str
    body: str


def help_markdown() -> str:
    """The packaged canonical help text."""
    asset = resources.files("curvestudio_desktop")
    for part in ("resources", "help", "help-content.md"):
        asset = asset / part
    return asset.read_text(encoding="utf-8")


def parse_sections(markdown: str) -> tuple[HelpSection, ...]:
    """Split the document at every ``##`` heading, preserving order."""
    headings = list(_HEADING.finditer(markdown))
    sections: list[HelpSection] = []
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections.append(
            HelpSection(title=match.group("title").strip(), body=markdown[start:end].strip())
        )
    return tuple(sections)


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _escape_attribute(text: str) -> str:
    return html.escape(text, quote=True)


def _inline(text: str) -> str:
    """Inline Markdown: code spans, bold, italic, links and angle-bracket autolinks."""
    pieces: list[str] = []
    position = 0
    for match in _INLINE.finditer(text):
        pieces.append(_escape(text[position : match.start()]))
        if match.group("code") is not None:
            pieces.append(f"<code>{_escape(match.group('code'))}</code>")
        elif match.group("bold") is not None:
            pieces.append(f"<strong>{_inline(match.group('bold'))}</strong>")
        elif match.group("italic") is not None:
            pieces.append(f"<em>{_inline(match.group('italic'))}</em>")
        elif match.group("label") is not None:
            href = _escape_attribute(match.group("href"))
            pieces.append(f'<a href="{href}">{_inline(match.group("label"))}</a>')
        else:
            href = _escape_attribute(match.group("auto"))
            pieces.append(f'<a href="{href}">{href}</a>')
        position = match.end()
    pieces.append(_escape(text[position:]))
    return "".join(pieces)


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _table(rows: list[str]) -> str:
    parsed = [_cells(row) for row in rows]
    if len(parsed) >= 2 and all(_SEPARATOR_CELL.match(cell) for cell in parsed[1]):
        header, body = parsed[0], parsed[2:]
    else:
        header, body = parsed[0], parsed[1:]
    head = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
    rendered = [f"<table><thead><tr>{head}</tr></thead><tbody>"]
    for row in body:
        cells = "".join(f"<td>{_inline(cell)}</td>" for cell in row)
        rendered.append(f"<tr>{cells}</tr>")
    rendered.append("</tbody></table>")
    return "".join(rendered)


def _starts_block(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith("---")
        or _UNORDERED.match(line) is not None
        or _ORDERED.match(line) is not None
    )


def _list_items(lines: list[str], index: int, marker: re.Pattern[str]) -> tuple[list[str], int]:
    items: list[str] = []
    while index < len(lines):
        raw = lines[index]
        if not raw.strip():
            break
        match = marker.match(raw)
        if match is not None:
            items.append(raw[match.end() :].strip())
            index += 1
            continue
        if items and not _starts_block(raw):
            items[-1] = f"{items[-1]} {raw.strip()}"
            index += 1
            continue
        break
    return items, index


def markdown_to_html(markdown: str) -> str:
    """Render the supported Markdown subset (headings, paragraphs, lists, tables, rules)."""
    lines = markdown.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{_inline(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line.startswith("---"):
            flush_paragraph()
            blocks.append("<hr/>")
            index += 1
            continue
        if line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
            index += 1
            continue
        if line.startswith("|"):
            flush_paragraph()
            rows: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            blocks.append(_table(rows))
            continue
        if _UNORDERED.match(raw) is not None:
            flush_paragraph()
            items, index = _list_items(lines, index, _UNORDERED)
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            blocks.append(f"<ul>{body}</ul>")
            continue
        if _ORDERED.match(raw) is not None:
            flush_paragraph()
            items, index = _list_items(lines, index, _ORDERED)
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            blocks.append(f"<ol>{body}</ol>")
            continue
        paragraph.append(line)
        index += 1
    flush_paragraph()
    return "\n".join(blocks)


def render_section(section: HelpSection) -> str:
    """One section as an HTML fragment, heading included."""
    return f"<h1>{_escape(section.title)}</h1>\n{markdown_to_html(section.body)}"


class HelpDialog(QDialog):
    """The canonical help content: a section list, a rendered view, and search."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
        section: str | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._settings = settings if settings is not None else QSettings()
        self._sections = parse_sections(help_markdown())
        self.setObjectName("helpDialog")
        self.setWindowTitle(TITLE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.resize(*DEFAULT_SIZE)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self._build()
        self._restore_geometry()
        if not (section is not None and self.select_section(section)) and self._sections:
            self.section_list.setCurrentRow(0)

    # -- construction ------------------------------------------------------------------------

    def _build(self) -> None:
        self.search = QLineEdit()
        self.search.setObjectName("helpSearch")
        self.search.setPlaceholderText("Search help…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_sections)
        self.section_list = QListWidget()
        self.section_list.setObjectName("helpSectionList")
        self.section_list.setAlternatingRowColors(True)
        for index, section in enumerate(self._sections):
            item = QListWidgetItem(section.title)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.section_list.addItem(item)
        self.section_list.currentRowChanged.connect(self._render_row)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.section_list, stretch=1)

        self.browser = QTextBrowser()
        self.browser.setObjectName("helpBrowser")
        self.browser.setOpenExternalLinks(True)
        self.browser.document().setDefaultStyleSheet(_BROWSER_CSS)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("helpSplitter")
        splitter.addWidget(left)
        splitter.addWidget(self.browser)
        splitter.setSizes([240, 700])
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)

        close = QPushButton("Close")
        close.setObjectName("helpCloseButton")
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(buttons)

    # -- sections ----------------------------------------------------------------------------

    def sections(self) -> tuple[str, ...]:
        """Every section title, in canonical order."""
        return tuple(section.title for section in self._sections)

    def current_section(self) -> str | None:
        row = self.section_list.currentRow()
        if 0 <= row < len(self._sections) and not self.section_list.item(row).isHidden():
            return self._sections[row].title
        return None

    def select_section(self, query: str) -> bool:
        """Show the first section whose title or body contains ``query`` (case-insensitive)."""
        needle = query.strip().casefold()
        if not needle:
            return False
        if self.search.text():
            self.search.clear()
        for index, section in enumerate(self._sections):
            haystack = f"{section.title}\n{section.body}".casefold()
            if needle in haystack:
                self.section_list.setCurrentRow(index)
                self.section_list.scrollToItem(self.section_list.item(index))
                return True
        return False

    def _render_row(self, row: int) -> None:
        if 0 <= row < len(self._sections):
            self.browser.setHtml(render_section(self._sections[row]))

    def _filter_sections(self, text: str) -> None:
        needle = text.strip().casefold()
        first_match = -1
        for index, section in enumerate(self._sections):
            matches = not needle or needle in f"{section.title}\n{section.body}".casefold()
            self.section_list.item(index).setHidden(not matches)
            if matches and first_match < 0:
                first_match = index
        if first_match < 0:
            self.section_list.setCurrentRow(-1)
            self.browser.setHtml(f"<p>No sections match “{_escape(text.strip())}”.</p>")
            return
        if needle:
            self.section_list.setCurrentRow(first_match)
        else:
            row = self.section_list.currentRow()
            self.section_list.setCurrentRow(row if row >= 0 else first_match)

    # -- geometry persistence ----------------------------------------------------------------

    def _restore_geometry(self) -> None:
        stored = self._settings.value(GEOMETRY_KEY, "")
        if isinstance(stored, str) and stored:
            self.restoreGeometry(QByteArray.fromBase64(bytes(stored, encoding="ascii")))

    def save_geometry(self) -> None:
        """Remember the window's geometry (the main window also calls this when saving)."""
        data = bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        self._settings.setValue(GEOMETRY_KEY, data)
        self._settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_geometry()
        super().closeEvent(event)


__all__ = [
    "GEOMETRY_KEY",
    "TITLE",
    "HelpDialog",
    "HelpSection",
    "help_markdown",
    "markdown_to_html",
    "parse_sections",
    "render_section",
]
