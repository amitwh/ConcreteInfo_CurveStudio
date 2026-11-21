"""ConcreteInfo theme: bundled fonts, brand tokens, palette, QSS and pyqtgraph.

One place owns the look: the brand tokens below feed both the Qt palette and the
generated stylesheet, so native popups (combo lists, menus) and styled widgets agree.
:func:`apply_theme` is idempotent and safe to call again at runtime (View → Theme):
it applies Fusion, swaps the palette and stylesheet, and reconfigures pyqtgraph.

The preference is stored in QSettings under ``appearance/theme`` as ``system``,
``light`` or ``dark``. ``system`` follows the platform colour scheme Qt reports.
"""

from importlib import resources
from typing import Literal

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

Scheme = Literal["light", "dark", "unknown"]
Preference = Literal["system", "light", "dark"]

THEME_SETTINGS_KEY = "appearance/theme"
DEFAULT_PREFERENCE: Preference = "system"

_INTER = "Inter"
_MONO = "JetBrains Mono"
_UI_POINT_SIZE = 10
_FONT_POINT_SIZE = 10
_FONT_FILES = (
    "Inter-Regular.ttf",
    "Inter-Medium.ttf",
    "Inter-SemiBold.ttf",
    "JetBrainsMono-Regular.ttf",
    "JetBrainsMono-Medium.ttf",
)

# pyqtgraph's historical colours: existing callers and tests rely on these values.
_DARK_BACKGROUND = "#1e1e1e"
_DARK_FOREGROUND = "#d4d4d4"

LIGHT_TOKENS: dict[str, str] = {
    "bg": "#fafbfc",
    "surface": "#ffffff",
    "text": "#464646",
    "heading": "#0d0b09",
    "muted": "#6e6c6c",
    "border": "#e3e3e3",
    "accent": "#e5461f",
    "accent_dark": "#c93a18",
    "success": "#1a7a56",
    "warning": "#a86b00",
    "danger": "#b42318",
    "selection": "#fdf3f1",
    "on_accent": "#ffffff",
}

DARK_TOKENS: dict[str, str] = {
    "bg": "#0d0b09",
    "surface": "#1c1917",
    "text": "#f1f1f1",
    "heading": "#ffffff",
    "muted": "#9a9696",
    "border": "#33302e",
    "accent": "#e5461f",
    "accent_dark": "#ff6b47",
    "success": "#4ecb8d",
    "warning": "#e8c56a",
    "danger": "#f2b8b5",
    "selection": "#2a1a15",
    "on_accent": "#ffffff",
}

_SCHEMES: tuple[Preference, ...] = ("system", "light", "dark")
_current_scheme: Scheme = "light"
_fonts_loaded = False


# -- fonts -----------------------------------------------------------------------------------


def ensure_fonts_loaded() -> None:
    """Load the bundled Inter and JetBrains Mono faces once (missing files are skipped)."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    base = resources.files("curvestudio_desktop") / "resources" / "fonts"
    for name in _FONT_FILES:
        try:
            with resources.as_file(base / name) as path:
                QFontDatabase.addApplicationFont(str(path))
        except OSError, ModuleNotFoundError:
            continue  # a partial install must never stop the app from starting


def application_font() -> QFont:
    """The UI font: Inter 10 pt (Qt falls back to the system sans if it is missing)."""
    font = QFont(_INTER, _UI_POINT_SIZE)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    return font


def mono_font(point_size: float | None = None) -> QFont:
    """JetBrains Mono for code, equations and numeric cells."""
    font = QFont(_MONO, int(point_size if point_size is not None else _FONT_POINT_SIZE))
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


# -- tokens and palette ----------------------------------------------------------------------


def tokens(scheme: Scheme) -> dict[str, str]:
    """The brand tokens for a resolved scheme (``unknown`` resolves to the safe light set)."""
    return dict(DARK_TOKENS if scheme == "dark" else LIGHT_TOKENS)


def current_scheme() -> Scheme:
    """The scheme :func:`apply_theme` last applied (light before the first call)."""
    return _current_scheme


def current_tokens() -> dict[str, str]:
    return tokens(_current_scheme)


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


def resolve_scheme(app: QGuiApplication, preference: Preference) -> Scheme:
    """The concrete light/dark scheme a preference resolves to (system fallback: light)."""
    if preference == "dark":
        return "dark"
    if preference == "light":
        return "light"
    reported = color_scheme(app)
    return "light" if reported == "unknown" else reported


def palette_for(scheme: Scheme) -> QPalette:
    """The Fusion palette matching the brand tokens (so native popups agree with the QSS)."""
    palette_tokens = DARK_TOKENS if scheme == "dark" else LIGHT_TOKENS
    palette = QPalette()
    roles = (
        (QPalette.ColorRole.Window, palette_tokens["bg"]),
        (QPalette.ColorRole.WindowText, palette_tokens["text"]),
        (QPalette.ColorRole.Base, palette_tokens["surface"]),
        (QPalette.ColorRole.AlternateBase, palette_tokens["bg"]),
        (QPalette.ColorRole.Text, palette_tokens["text"]),
        (QPalette.ColorRole.Button, palette_tokens["surface"]),
        (QPalette.ColorRole.ButtonText, palette_tokens["text"]),
        (QPalette.ColorRole.BrightText, palette_tokens["heading"]),
        (QPalette.ColorRole.Highlight, palette_tokens["accent"]),
        (QPalette.ColorRole.HighlightedText, palette_tokens["on_accent"]),
        (QPalette.ColorRole.ToolTipBase, palette_tokens["heading"]),
        (QPalette.ColorRole.ToolTipText, palette_tokens["bg"]),
        (QPalette.ColorRole.PlaceholderText, palette_tokens["muted"]),
        (QPalette.ColorRole.Link, palette_tokens["accent"]),
    )
    for role, value in roles:
        palette.setColor(role, QColor(value))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.HighlightedText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(palette_tokens["muted"]))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(palette_tokens["border"])
    )
    return palette


# -- preference persistence ------------------------------------------------------------------


def _normalise_preference(value: object) -> Preference:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in _SCHEMES else DEFAULT_PREFERENCE


def theme_preference(settings: QSettings | None = None) -> Preference:
    """The stored preference (``system`` when unset or invalid)."""
    store = settings if settings is not None else QSettings()
    return _normalise_preference(store.value(THEME_SETTINGS_KEY, DEFAULT_PREFERENCE))


def set_theme_preference(preference: Preference, settings: QSettings | None = None) -> None:
    """Persist the preference; callers that switch interactively also call apply_theme."""
    store = settings if settings is not None else QSettings()
    store.setValue(THEME_SETTINGS_KEY, _normalise_preference(preference))
    store.sync()


# -- application -----------------------------------------------------------------------------


def apply_theme(
    app: QApplication,
    scheme: Scheme | None = None,
    *,
    preference: Preference | None = None,
) -> Scheme:
    """Apply Fusion, palette and QSS, and point pyqtgraph at the resolved background.

    ``scheme`` keeps the original call shape (adding it also keeps the historical
    pyqtgraph colours). ``preference`` is the persisted choice; when neither is given
    the stored QSettings preference is used. Safe to call again at runtime.
    """
    import pyqtgraph as pg

    from curvestudio_desktop.views.styles import stylesheet

    ensure_fonts_loaded()
    if preference is None:
        preference = scheme if scheme in ("light", "dark") else theme_preference()
    resolved = resolve_scheme(app, preference)

    app.setStyle("Fusion")
    app.setFont(application_font())
    app.setPalette(palette_for(resolved))
    app.setStyleSheet(stylesheet(resolved))
    app.setProperty("curvestudioTheme", resolved)

    global _current_scheme
    _current_scheme = resolved
    if resolved == "dark":
        pg.setConfigOptions(background=_DARK_BACKGROUND, foreground=_DARK_FOREGROUND)
    else:  # light, and the safe default when the platform does not say
        pg.setConfigOptions(background="w", foreground="k")
    return resolved


__all__ = [
    "DARK_TOKENS",
    "DEFAULT_PREFERENCE",
    "LIGHT_TOKENS",
    "THEME_SETTINGS_KEY",
    "Preference",
    "Scheme",
    "application_font",
    "apply_theme",
    "color_scheme",
    "current_scheme",
    "current_tokens",
    "ensure_fonts_loaded",
    "mono_font",
    "palette_for",
    "resolve_scheme",
    "set_theme_preference",
    "theme_preference",
    "tokens",
]
