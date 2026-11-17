"""Light/dark theme: follow the system palette, mirrored into pyqtgraph.

Qt 6 widgets pick up the platform palette on their own; pyqtgraph draws its own
background, so it is configured explicitly from the system colour scheme before
the first plot is created. The app never forces a palette.
"""

from typing import Literal

from PySide6.QtGui import QGuiApplication

Scheme = Literal["light", "dark", "unknown"]

_DARK_BACKGROUND = "#1e1e1e"
_DARK_FOREGROUND = "#d4d4d4"


def color_scheme(app: QGuiApplication) -> Scheme:
    """The system colour scheme as Qt reports it ("unknown" on older platforms)."""
    hints = app.styleHints()
    if hints is None:
        return "unknown"
    from PySide6.QtCore import Qt

    scheme = hints.colorScheme()
    if scheme is Qt.ColorScheme.Dark:
        return "dark"
    if scheme is Qt.ColorScheme.Light:
        return "light"
    return "unknown"


def apply_theme(app: QGuiApplication, scheme: Scheme | None = None) -> Scheme:
    """Match pyqtgraph's background to ``scheme`` (or the system scheme)."""
    import pyqtgraph as pg

    resolved = scheme if scheme is not None else color_scheme(app)
    if resolved == "dark":
        pg.setConfigOptions(background=_DARK_BACKGROUND, foreground=_DARK_FOREGROUND)
    else:  # light, and the safe default when the platform does not say
        pg.setConfigOptions(background="w", foreground="k")
    return resolved
