"""SelectionController: which dataset is active, and the x/y/weight columns shown.

UI state, reconciled against the current project every time the controller is told
the project changed. Per-dataset column choices and import issues persist in the
project archive under ``plot_settings["desktop"]["datasets"][<id>]`` (the format's
extension point for app-private JSON), so a weighted dataset reopened from a
``.cstudio`` file keeps — and fits with — its weights.
"""

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from curvestudio.data import XY, Dataset
    from curvestudio.project import Project
    from curvestudio.workflow import ImportIssue

DESKTOP_STATE_KEY = "desktop"
"""Where desktop UI state lives inside ``Project.plot_settings``."""


def default_columns(dataset: Dataset) -> tuple[str, str] | None:
    """The first two numeric columns of ``dataset`` (the workflow import heuristic)."""
    frame = dataset.table()
    numeric = [str(name) for name in frame.columns if str(frame[name].dtype.kind) in "ifu"]
    if len(numeric) < 2:
        return None
    return numeric[0], numeric[1]


class SelectionController(QObject):
    """The active dataset id plus per-dataset column choices and import issues."""

    active_changed = Signal()
    """The active dataset, or the columns/issues used to show it, changed."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._active_id: str | None = None
        self._columns: dict[str, tuple[str, str, str | None]] = {}
        self._issues: dict[str, tuple[ImportIssue, ...]] = {}
        self.last_error: str | None = None

    def set_project(self, project: Project) -> bool:
        """Reconcile with a new/replaced project, keeping the active dataset if possible.

        Returns whether :attr:`active_changed` was emitted — the emit happens only
        when reconciliation actually moved the active dataset, so refresh wiring can
        avoid redundant table/plot rebuilds (and callers must refresh themselves when
        ``False`` comes back for a project they know is new).
        """
        self._project = project
        known = {entry.id for entry in project.datasets}
        changed = False
        if self._active_id not in known:
            new_active = project.datasets[0].id if project.datasets else None
            changed = new_active != self._active_id
            self._active_id = new_active
        self._prune_state()
        self._restore_persisted(project)
        if changed:
            self.active_changed.emit()
        return changed

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def set_active(self, dataset_id: str | None) -> None:
        """Make ``dataset_id`` the active dataset (ignored when it does not exist)."""
        known = {entry.id for entry in self._project.datasets} if self._project else set()
        if dataset_id is None or dataset_id not in known:
            return
        if dataset_id == self._active_id:
            return
        self._active_id = dataset_id
        self.active_changed.emit()

    @property
    def project(self) -> Project | None:
        return self._project

    def active_dataset(self) -> Dataset | None:
        if self._project is None or self._active_id is None:
            return None
        for entry in self._project.datasets:
            if entry.id == self._active_id:
                return entry.dataset
        return None

    def columns(self, dataset_id: str) -> tuple[str, str, str | None] | None:
        """Stored choice for ``dataset_id`` or the numeric-column default."""
        if dataset_id in self._columns:
            return self._columns[dataset_id]
        dataset = self._dataset_of(dataset_id)
        if dataset is None:
            return None
        default = default_columns(dataset)
        if default is None:
            return None
        return (default[0], default[1], None)

    def set_columns(self, dataset_id: str, x: str, y: str, weight: str | None) -> None:
        self._columns[dataset_id] = (x, y, weight)
        if dataset_id == self._active_id:
            self.active_changed.emit()

    def issues(self, dataset_id: str) -> tuple[ImportIssue, ...]:
        return self._issues.get(dataset_id, ())

    def set_issues(
        self, dataset_id: str, issues: list[ImportIssue] | tuple[ImportIssue, ...]
    ) -> None:
        self._issues[dataset_id] = tuple(issues)
        if dataset_id == self._active_id:
            self.active_changed.emit()

    def xy(self) -> XY | None:
        """The active dataset's validated fit input, or ``None`` when there is none.

        A :class:`~curvestudio.data.DataError` here (for example after future data
        cleaning) leaves the plot empty and is surfaced in :attr:`last_error` for the
        status bar — it is never swallowed.
        """
        dataset = self.active_dataset()
        if dataset is None:
            return None
        columns = self.columns(self._active_id or "")
        if columns is None:
            self.last_error = "the dataset has no two numeric columns to plot"
            return None
        from curvestudio.data import DataError

        x, y, weight = columns
        try:
            result = dataset.xy(x, y, weight)
        except DataError as exc:
            self.last_error = f"{exc.code}: {exc.args[0]}"
            return None
        self.last_error = None
        return result

    def _dataset_of(self, dataset_id: str) -> Dataset | None:
        if self._project is None:
            return None
        for entry in self._project.datasets:
            if entry.id == dataset_id:
                return entry.dataset
        return None

    # -- persistence (plot_settings["desktop"]) ---------------------------------------------

    def state_for_project(self) -> dict[str, Any]:
        """The ``{"datasets": {...}}`` blob stored under ``plot_settings["desktop"]``.

        Only datasets of the *current* project appear, so removed datasets cannot
        linger in a saved archive. Plain JSON only — ``Project`` validates that.
        """
        datasets: dict[str, dict[str, object]] = {}
        if self._project is not None:
            for entry in self._project.datasets:
                columns = self._columns.get(entry.id)
                issues = self._issues.get(entry.id, ())
                if columns is None and not issues:
                    continue
                x, y, weight = columns if columns is not None else (None, None, None)
                datasets[entry.id] = {
                    "columns": [x, y, weight],
                    "issues": [
                        {
                            "code": issue.code,
                            "severity": issue.severity,
                            "message": issue.message,
                            "rows": list(issue.rows),
                        }
                        for issue in issues
                    ],
                }
        return {"datasets": datasets}

    def _restore_persisted(self, project: Project) -> None:
        """Re-apply column choices and issues stored in the project archive.

        Best effort by design: a hand-edited or older archive may hold junk in the
        desktop blob — anything that does not validate is ignored in favour of the
        computed defaults, never fatal to opening the project.
        """
        blob = project.plot_settings.get(DESKTOP_STATE_KEY)
        entries = blob.get("datasets") if isinstance(blob, dict) else None
        if not isinstance(entries, dict):
            return
        for entry in project.datasets:
            stored = entries.get(entry.id)
            if not isinstance(stored, dict):
                continue
            columns = self._valid_columns(entry.dataset, stored.get("columns"))
            if columns is not None:
                self._columns[entry.id] = columns
            issues = self._valid_issues(stored.get("issues"))
            if issues is not None:
                self._issues[entry.id] = issues

    def _valid_columns(
        self, dataset: Dataset, stored: object
    ) -> tuple[str, str, str | None] | None:
        columns = stored if isinstance(stored, list) else None
        if not columns or len(columns) != 3:
            return None
        x, y, weight = (columns[0], columns[1], columns[2])
        if not isinstance(x, str) or not isinstance(y, str):
            return None
        if weight is not None and not isinstance(weight, str):
            return None
        names = {str(name) for name in dataset.table().columns}
        if x not in names or y not in names or (weight is not None and weight not in names):
            return None
        return (x, y, weight)

    def _valid_issues(self, stored: object) -> tuple[ImportIssue, ...] | None:
        from curvestudio.workflow import ImportIssue

        if not isinstance(stored, list):
            return None
        parsed: list[ImportIssue] = []
        for item in stored:
            if not isinstance(item, dict):
                return None
            try:
                parsed.append(ImportIssue(**item))
            except ValidationError:
                return None
        return tuple(parsed)

    def _prune_state(self) -> None:
        known = {entry.id for entry in self._project.datasets} if self._project else set()
        self._columns = {key: value for key, value in self._columns.items() if key in known}
        self._issues = {key: value for key, value in self._issues.items() if key in known}
