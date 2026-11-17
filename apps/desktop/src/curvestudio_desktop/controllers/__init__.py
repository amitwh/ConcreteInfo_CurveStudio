"""Controllers: state and logic between Qt widgets and the CurveStudio core.

``ProjectController`` owns the project, dirty flag, undo stack and recent files;
``SelectionController`` owns which dataset is active and the columns it is shown
by. Both use QtCore only — no widget class — so they run headless in tests.
"""

from curvestudio_desktop.controllers.analysis import (
    MODE_AUTOFIT,
    MODE_MANUAL,
    AnalysisController,
)
from curvestudio_desktop.controllers.analysis_jobs import AutoFitLive
from curvestudio_desktop.controllers.project import ProjectController, ist_now, new_empty_project
from curvestudio_desktop.controllers.providers import (
    ConfirmationProvider,
    Messenger,
    PathProvider,
    UnsavedAnswer,
)
from curvestudio_desktop.controllers.selection import SelectionController, default_columns

__all__ = [
    "MODE_AUTOFIT",
    "MODE_MANUAL",
    "AnalysisController",
    "AutoFitLive",
    "ConfirmationProvider",
    "Messenger",
    "PathProvider",
    "ProjectController",
    "SelectionController",
    "UnsavedAnswer",
    "default_columns",
    "ist_now",
    "new_empty_project",
]
