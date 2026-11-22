"""Vendored Lucide icons rendered to ``QIcon`` with the current theme colour.

The SVGs are the canonical Lucide stroke drawings (ISC; see the licence beside them)
on a 24x24 grid. :func:`icon` reads the vendored file through ``importlib.resources``,
substitutes the theme's text colour for ``currentColor``, normalises the stroke weight
to the toolbar's single value and caches the rendered result per ``(name, colour)``, so
a theme switch simply asks for a new colour and the old icons stay cached. A name with
no vendored file raises :class:`KeyError` — a missing icon fails in the tests, never in
the window. The application/window icon comes from the staged brand PNG.
"""

import re
from functools import cache
from importlib import resources
from typing import cast

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from curvestudio_desktop.theme import current_tokens

_RENDER_SIZE = 64
#: One stroke weight for the whole toolbar. Lucide ships 2; the set has always looked
#: lighter, so every file is normalised to this value as it is read.
_STROKE_WIDTH = "1.8"
_STROKE_WIDTH_PATTERN = re.compile(r'stroke-width="[^"]*"')
_LUCIDE_DIR = ("resources", "icons", "lucide")

#: Internal name -> vendored Lucide file. Only files that exist in the staged set.
_ICONS: dict[str, str] = {
    "new": "file-plus.svg",
    "open": "folder-open.svg",
    "save": "save.svg",
    "import": "download.svg",
    "clipboard": "clipboard.svg",
    "undo": "undo-2.svg",
    "log-x": "bar-chart-3.svg",
    "log-y": "chart-no-axes-column.svg",
    "reset": "rotate-ccw.svg",
    "export": "upload.svg",
    "details": "book-open.svg",
    "help": "help-circle.svg",
    "about": "info.svg",
    "collapse": "minimize-2.svg",
    "expand": "maximize-2.svg",
    "float": "external-link.svg",
    "chevron-down": "chevron-down.svg",
    "close": "x.svg",
    "panel-project": "panel-left.svg",
    "panel-analysis": "panel-right.svg",
    "panel-table": "panel-bottom.svg",
}

_ICON_CACHE: dict[tuple[str, str], QIcon] = {}


def icon_color() -> str:
    """The stroke colour: the live palette's text colour, else the theme token."""
    app = QApplication.instance()
    if app is not None:
        palette = cast("QApplication", app).palette()
        return str(palette.color(QPalette.ColorRole.WindowText).name())
    return current_tokens()["text"]


def _svg_text(name: str, color: str) -> str:
    """The vendored SVG with the theme colour and the toolbar stroke weight applied."""
    filename = _ICONS[name]  # KeyError for an unknown name: fail loudly, never blank
    asset = resources.files("curvestudio_desktop")
    for part in (*_LUCIDE_DIR, filename):
        asset = asset / part
    source = asset.read_text(encoding="utf-8")
    source = source.replace("currentColor", color)
    return _STROKE_WIDTH_PATTERN.sub(f'stroke-width="{_STROKE_WIDTH}"', source)


def icon(name: str, color: str | None = None) -> QIcon:
    """A themed icon; ``color`` defaults to the current theme's text colour."""
    stroke = color if color is not None else icon_color()
    key = (name, stroke)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached
    renderer = QSvgRenderer(QByteArray(_svg_text(name, stroke).encode("utf-8")))
    pixmap = QPixmap(QSize(_RENDER_SIZE, _RENDER_SIZE))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    rendered = QIcon(pixmap)
    _ICON_CACHE[key] = rendered
    return rendered


@cache
def app_icon() -> QIcon:
    """The staged brand application icon (empty only if the asset is missing)."""
    try:
        asset = resources.files("curvestudio_desktop") / "resources" / "brand" / "app-icon.png"
        data = asset.read_bytes()
    except OSError, ModuleNotFoundError:
        return QIcon()
    pixmap = QPixmap()
    if not pixmap.loadFromData(data, "PNG"):
        return QIcon()
    return QIcon(pixmap)


__all__ = ["app_icon", "icon", "icon_color"]
