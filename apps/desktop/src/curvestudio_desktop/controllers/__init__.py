"""Controllers: state and logic between Qt widgets and the CurveStudio core.

``ProjectController`` owns the project, dirty flag, undo stack and recent files;
``SelectionController`` owns which dataset is active and the columns it is shown
by. Both use QtCore only — no widget class — so they run headless in tests.

The re-exported names resolve lazily (PEP 562): importing this package imports no
controller, so a start-up path that needs only the constants does not pay for the
analysis controller's import of the whole fitting stack.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from curvestudio_desktop.controllers.analysis import (
        MODE_AUTOFIT,
        MODE_MANUAL,
        MODE_STATS,
        AnalysisController,
    )
    from curvestudio_desktop.controllers.analysis_jobs import AutoFitLive
    from curvestudio_desktop.controllers.project import (
        ProjectController,
        ist_now,
        new_empty_project,
    )
    from curvestudio_desktop.controllers.providers import (
        ConfirmationProvider,
        Messenger,
        PathProvider,
        UnsavedAnswer,
    )
    from curvestudio_desktop.controllers.selection import (
        SelectionController,
        default_columns,
    )

_EXPORTS = {
    "MODE_AUTOFIT": ".analysis",
    "MODE_MANUAL": ".analysis",
    "MODE_STATS": ".analysis",
    "AnalysisController": ".analysis",
    "AutoFitLive": ".analysis_jobs",
    "ConfirmationProvider": ".providers",
    "Messenger": ".providers",
    "PathProvider": ".providers",
    "ProjectController": ".project",
    "SelectionController": ".selection",
    "UnsavedAnswer": ".providers",
    "default_columns": ".selection",
    "ist_now": ".project",
    "new_empty_project": ".project",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(__name__ + module, package=__name__), name)
    globals()[name] = value  # later lookups skip this hook
    return value
