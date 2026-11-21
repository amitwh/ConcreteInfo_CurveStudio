"""Small inline-SVG icon set rendered to ``QIcon`` with the current theme colour.

The SVGs are stroke drawings on a 24x24 grid. :func:`icon` substitutes the
theme's text colour and caches the rendered result per ``(name, colour)``, so a
theme switch simply asks for a new colour and the old icons stay cached. The
application/window icon comes from the staged brand PNG.
"""

from functools import cache
from importlib import resources
from typing import cast

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from curvestudio_desktop.theme import current_tokens

_RENDER_SIZE = 64

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    "{body}</svg>"
)

_ICONS: dict[str, str] = {
    "new": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5"/><path d="M12 12v6M9 15h6"/>',
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h8V3"/><path d="M8 14h8v7H8z"/>',
    "import": '<path d="M12 3v10"/><path d="m8 9 4 4 4-4"/>'
    '<path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "clipboard": '<rect x="6" y="5" width="12" height="16" rx="2"/>'
    '<path d="M9 5V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1"/>',
    "undo": '<path d="M9 7 4 12l5 5"/><path d="M4 12h10a5 5 0 0 1 5 5v1"/>',
    "log-x": '<path d="M4 4v16h16"/><path d="M7 20v-3M11 20v-5M15 20v-2M19 20v-4"/>',
    "log-y": '<path d="M4 20V4h16"/><path d="M4 8h3M4 12h5M4 16h3"/>',
    "reset": '<path d="M4 10a8 8 0 1 1 1.7 8.7"/><path d="M4 4v6h6"/>',
    "export": '<path d="M12 15V3"/><path d="m8 7 4-4 4 4"/>'
    '<path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
    "details": '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>',
    "help": '<circle cx="12" cy="12" r="9"/>'
    '<path d="M9.5 9a2.5 2.5 0 1 1 4 2c-.9.6-1.5 1.2-1.5 2.5"/>'
    '<circle cx="12" cy="17" r="0.6" fill="{color}" stroke="none"/>',
    "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/>'
    '<circle cx="12" cy="7.5" r="0.6" fill="{color}" stroke="none"/>',
    "collapse": '<path d="m4 12 5-5M4 12l5 5M20 12l-5-5M20 12l-5 5"/>',
    "expand": '<path d="m9 12-5-5M9 12 4 17M15 12l5-5M15 12l5 5"/>',
    "float": '<rect x="4" y="7" width="12" height="12" rx="2"/>'
    '<path d="M14 3h7v7"/><path d="m21 3-8 8"/>',
    "close": '<path d="m6 6 12 12M18 6 6 18"/>',
    "panel-project": '<rect x="3" y="4" width="7" height="16" rx="1"/>'
    '<rect x="12" y="4" width="9" height="16" rx="1"/>',
    "panel-analysis": '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M13 4v16"/>',
    "panel-table": '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 12h18"/>',
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
    body = _ICONS[name].replace("{color}", color)
    return _SVG.format(color=color, body=body)


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
