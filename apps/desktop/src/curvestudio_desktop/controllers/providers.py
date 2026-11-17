"""Injectable services the controllers need: prompts, file dialogs, message boxes.

Everything here is a :class:`typing.Protocol` so tests (and the API-facing future
code paths) can supply their own answers; the widget-backed implementations live
in :mod:`curvestudio_desktop.views.dialogs`. Controllers depend on these
interfaces only, which keeps them free of any Qt widget import.
"""

from enum import Enum
from pathlib import Path
from typing import Protocol


class UnsavedAnswer(Enum):
    """What the user chose to do with unsaved changes."""

    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


class ConfirmationProvider(Protocol):
    """Asks the user what to do with unsaved changes before a destructive step."""

    def confirm_unsaved_changes(self, action: str) -> UnsavedAnswer:
        """Return SAVE, DISCARD or CANCEL for the pending ``action`` (human-readable)."""
        ...  # pragma: no cover - protocol


class PathProvider(Protocol):
    """The file dialogs: open a project, choose where to save, choose an export path."""

    def choose_open_path(self) -> Path | None:
        """A chosen ``.cstudio`` path, or ``None`` when the user cancelled."""
        ...  # pragma: no cover - protocol

    def choose_save_path(self, suggested_name: str) -> Path | None:
        """A chosen save target (starting from ``suggested_name``), or ``None``."""
        ...  # pragma: no cover - protocol

    def choose_export_path(self, title: str, filters: str) -> Path | None:
        """A chosen export target (PNG, SVG, … per ``filters``), or ``None``."""
        ...  # pragma: no cover - protocol


class Messenger(Protocol):
    """Shows messages to the user. Errors here are user-caused problems, never tracebacks."""

    def show_error(self, title: str, message: str) -> None: ...  # pragma: no cover - protocol

    def show_information(self, title: str, message: str) -> None: ...  # pragma: no cover - protocol
