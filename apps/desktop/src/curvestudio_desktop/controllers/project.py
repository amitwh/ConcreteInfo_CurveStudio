"""ProjectController: the one owner of project state between the core and the widgets.

Holds the current :class:`~curvestudio.project.Project`, the dirty flag, a bounded
undo stack and the recent-files list (persisted via ``QSettings`` — configuration
storage, never telemetry). Uses QtCore only (signals, settings): no widget class
is imported here, so every operation is testable without a display. All engine
work goes through :mod:`curvestudio.workflow` and :mod:`curvestudio.project`;
every user-caused failure becomes a :class:`~curvestudio_desktop.controllers.providers.Messenger`
call, never a traceback.
"""

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal

from curvestudio.project import (
    Analysis,
    Project,
    ProjectDataset,
    ProjectError,
    load_project,
    save_project,
)
from curvestudio.workflow import ImportOutcome, WorkflowError, import_dataset
from curvestudio_desktop.controllers.providers import (
    ConfirmationProvider,
    Messenger,
    PathProvider,
    UnsavedAnswer,
)

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
"""Every timestamp the app records carries the +05:30 offset (repo convention)."""

_UNDO_LIMIT = 100
_RECENT_KEY = "recent_projects"
_RECENT_SEPARATOR = "\n"  # path separators (``/``, ``\``) can never appear inside a path
_RECENT_LIMIT = 10
_SLUG_STRIP = re.compile(r"[^A-Za-z0-9_.-]+")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def ist_now() -> str:
    """The current moment as an ISO 8601 string with the +05:30 (IST) offset."""
    return datetime.now(IST).isoformat(timespec="seconds")


def new_empty_project(now: Callable[[], str] = ist_now) -> Project:
    """A clean untitled project, stamped with ``now()``."""
    return Project(name="Untitled project", created_at=now())


def unique_id(name: str, taken: set[str]) -> str:
    """A safe archive id derived from a dataset ``name``, unique among ``taken``."""
    slug = _SLUG_STRIP.sub("-", name.strip()).strip("-._")
    if not slug or not _ID_PATTERN.match(slug):
        slug = "dataset"
    candidate, suffix = slug, 2
    while candidate in taken:
        candidate = f"{slug}_{suffix}"
        suffix += 1
    return candidate


class ProjectController(QObject):
    """Owns the current project and every operation that changes it.

    Mutations (import, remove, rename) rebuild the frozen ``Project`` and push the
    previous state onto a bounded undo stack. Open/new/save never touch the stack:
    undo is for edits, not for file operations.
    """

    project_replaced = Signal()
    """A different project is current (new, open, undo of a structural change)."""

    datasets_changed = Signal()
    """Datasets were added or removed (the tree rebuilds; the active dataset may move)."""

    dataset_renamed = Signal(str)
    """A dataset was renamed: its id. A label update, not a structural change."""

    dataset_imported = Signal(str, object)
    """A dataset was imported: its id and the :class:`ImportOutcome` that produced it.

    Emitted before :attr:`datasets_changed` so listeners can store per-dataset state
    (column choices, issues) before the tree rebuild highlights the new dataset.
    """

    dirty_changed = Signal(bool)
    recent_files_changed = Signal()
    undo_available_changed = Signal(bool)
    status_message = Signal(str)

    def __init__(
        self,
        *,
        confirmation: ConfirmationProvider,
        paths: PathProvider,
        messenger: Messenger,
        settings: QSettings | None = None,
        now: Callable[[], str] = ist_now,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._confirmation = confirmation
        self._paths = paths
        self._messenger = messenger
        self._settings = settings if settings is not None else QSettings()
        self._now = now
        self._project = new_empty_project(now)
        self._path: Path | None = None
        self._dirty = False
        self._undo_stack: list[Project] = []
        self._state_provider: Callable[[], dict[str, Any]] | None = None

    def set_state_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        """Register the desktop UI state merged into ``plot_settings`` on every save.

        The provider returns the ``{"datasets": ...}`` blob stored under
        ``plot_settings["desktop"]`` (see :class:`SelectionController`); ``None``
        disables persistence. Called once per save, on the saving thread.
        """
        self._state_provider = provider

    # -- state ------------------------------------------------------------------------------

    @property
    def project(self) -> Project:
        return self._project

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def confirmation(self) -> ConfirmationProvider:
        return self._confirmation

    @property
    def paths(self) -> PathProvider:
        return self._paths

    @property
    def messenger(self) -> Messenger:
        return self._messenger

    # -- project lifecycle ------------------------------------------------------------------

    def new_project(self) -> bool:
        """Start a clean project. Returns ``False`` when the user cancelled."""
        if not self._confirm_discard("starting a new project"):
            return False
        self._project = new_empty_project(self._now)
        self._path = None
        self._undo_stack.clear()
        self._emit_undo_available()
        self._set_dirty(False)
        self.project_replaced.emit()
        self.status_message.emit("New project created")
        return True

    def open_project(self, path: Path | None = None) -> bool:
        """Open a ``.cstudio`` file. Returns ``False`` on cancel or failure.

        The file is loaded *before* anything is discarded, so a broken archive can
        never destroy unsaved work. The synchronous path; the UI wraps
        :meth:`load_only` / :meth:`adopt_opened` around a background thread.
        """
        if path is None:
            path = self._paths.choose_open_path()
            if path is None:
                return False
        try:
            project = self.load_only(path)
        except ProjectError as exc:
            self._messenger.show_error("Cannot open project", str(exc))
            return False
        return self.adopt_opened(project, path)

    def load_only(self, path: Path) -> Project:
        """Read and validate a ``.cstudio`` file without touching any state.

        The heavy half of opening (zip + JSON + dataset parsing) — pure, so it can
        run off the GUI thread. Raises :class:`~curvestudio.project.ProjectError`.
        """
        return load_project(path)

    def adopt_opened(self, project: Project, path: Path) -> bool:
        """Make an already-loaded project current. Returns ``False`` on cancel.

        The GUI half of opening: confirm discarding unsaved work (the loaded
        project is ready, so a refusal costs nothing), then swap state and emit.
        """
        if not self._confirm_discard(f"opening {path.name}"):
            return False
        self._project = project
        self._path = path
        self._undo_stack.clear()
        self._emit_undo_available()
        self._set_dirty(False)
        self._touch_recent(path)
        self.project_replaced.emit()
        self.status_message.emit(f"Opened {path.name}")
        return True

    def save(self) -> bool:
        """Save to the current path (asking for one when there is none yet)."""
        if self._path is None:
            return self.save_as()
        return self._save_to(self._path)

    def save_as(self) -> bool:
        """Choose a target and save there. Returns ``False`` when cancelled or failed."""
        suggested = self._path.name if self._path else f"{self._project.name}.cstudio"
        target = self._paths.choose_save_path(suggested)
        if target is None:
            return False
        if target.suffix.lower() != ".cstudio":
            target = target.with_suffix(".cstudio")
        if not self._save_to(target):
            return False
        self._path = target
        return True

    def _save_to(self, target: Path) -> bool:
        try:
            project = self.write_only(target)
        except ProjectError as exc:
            self._messenger.show_error("Cannot save project", str(exc))
            return False
        self.commit_save(project, target)
        return True

    def write_only(self, target: Path) -> Project:
        """Serialize and write the project (with UI state) to ``target``.

        The heavy half of saving — pure, so it can run off the GUI thread. Raises
        :class:`~curvestudio.project.ProjectError`; the project stays dirty until
        :meth:`commit_save` runs.
        """
        project = self._project_with_ui_state()
        save_project(project, target)  # atomic write inside the core
        return project

    def commit_save(self, project: Project, target: Path) -> None:
        """The GUI half of saving: adopt the written state and path, clear dirty."""
        self._project = project  # the merged state is the new in-memory truth
        self._path = target
        self._set_dirty(False)
        self._touch_recent(target)
        self.status_message.emit(f"Saved {target.name}")

    def _project_with_ui_state(self) -> Project:
        """The current project with the desktop blob refreshed in ``plot_settings``."""
        if self._state_provider is None:
            return self._project
        merged = dict(self._project.plot_settings)
        merged["desktop"] = self._state_provider()
        return self._rebuilt(plot_settings=merged)

    def maybe_quit(self) -> bool:
        """The close-event gate: True when the window may close."""
        return self._confirm_discard("quitting")

    # -- datasets ---------------------------------------------------------------------------

    def import_dataset_file(
        self,
        path: Path,
        *,
        x_column: str | None = None,
        y_column: str | None = None,
        weight_column: str | None = None,
    ) -> bool:
        """Import a data file and add its dataset to the project."""
        try:
            outcome = import_dataset(
                path, path.name, x_column=x_column, y_column=y_column, weight_column=weight_column
            )
        except WorkflowError as exc:
            self._messenger.show_error("Cannot import data", str(exc))
            return False
        return self.add_imported(outcome) is not None

    def import_dataset_bytes(
        self,
        raw: bytes,
        filename: str,
        *,
        x_column: str | None = None,
        y_column: str | None = None,
        weight_column: str | None = None,
    ) -> bool:
        """Import pasted text (named ``filename``) and add its dataset to the project."""
        try:
            outcome = import_dataset(
                raw, filename, x_column=x_column, y_column=y_column, weight_column=weight_column
            )
        except WorkflowError as exc:
            self._messenger.show_error("Cannot import data", str(exc))
            return False
        return self.add_imported(outcome) is not None

    def add_imported(self, outcome: ImportOutcome) -> str | None:
        """Add an import outcome's dataset; returns its id, or ``None`` when refused.

        Refused (and reported via the messenger) when the data still has error-level
        issues or no usable x/y columns — the import dialog normally prevents that,
        but the controller must stay safe on its own.
        """
        errors = [issue for issue in outcome.issues if issue.severity == "error"]
        if errors or outcome.xy is None:
            lines = [f"{issue.code}: {issue.message}" for issue in errors]
            if not lines:
                lines = ["the data has no usable x and y columns"]
            self._messenger.show_error("Cannot import data", "\n".join(lines))
            return None
        dataset_id = unique_id(outcome.dataset.name, {entry.id for entry in self._project.datasets})
        self._push_undo()
        self._project = self._rebuilt(
            datasets=(
                *self._project.datasets,
                ProjectDataset(id=dataset_id, name=outcome.dataset.name, dataset=outcome.dataset),
            )
        )
        self._set_dirty(True)
        self.dataset_imported.emit(dataset_id, outcome)
        self.datasets_changed.emit()
        self.status_message.emit(
            f"Imported {len(outcome.xy.x):,} points into {outcome.dataset.name}"
        )
        return dataset_id

    def remove_dataset(self, dataset_id: str) -> bool:
        entry = self._find(dataset_id)
        if entry is None:
            return False
        self._push_undo()
        kept_analyses = tuple(
            analysis for analysis in self._project.analyses if analysis.dataset_id != dataset_id
        )
        self._project = self._rebuilt(
            datasets=tuple(e for e in self._project.datasets if e.id != dataset_id),
            analyses=kept_analyses,
        )
        self._set_dirty(True)
        self.datasets_changed.emit()
        self.status_message.emit(f"Removed dataset {entry.name}")
        return True

    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        entry = self._find(dataset_id)
        if entry is None:
            return False
        name = new_name.strip()
        if not name:
            self._messenger.show_error(
                "Cannot rename dataset", "The name cannot be empty or only spaces"
            )
            return False
        self._push_undo()
        datasets = tuple(
            entry.model_copy(update={"name": name}) if e.id == dataset_id else e
            for e in self._project.datasets
        )
        self._project = self._rebuilt(datasets=datasets)
        self._set_dirty(True)
        self.dataset_renamed.emit(dataset_id)
        self.status_message.emit(f"Renamed dataset to {name}")
        return True

    # -- undo -------------------------------------------------------------------------------

    def undo(self) -> bool:
        """Restore the state before the last edit. Returns ``False`` at the stack bottom."""
        if not self._undo_stack:
            return False
        self._project = self._undo_stack.pop()
        self._emit_undo_available()
        self._set_dirty(True)  # conservative: an undone edit still differs from the file
        self.datasets_changed.emit()
        self.status_message.emit("Undid the last change")
        return True

    # -- recent files -----------------------------------------------------------------------

    def recent_files(self) -> list[Path]:
        """Up to ten most-recent project paths that still exist on disk."""
        stored = self._settings.value(_RECENT_KEY, "")
        if not isinstance(stored, str):
            stored = str(stored)
        paths = [Path(part) for part in stored.split(_RECENT_SEPARATOR) if part]
        return [path for path in paths if path.is_file()][:_RECENT_LIMIT]

    def _touch_recent(self, path: Path) -> None:
        stored = self._settings.value(_RECENT_KEY, "")
        if not isinstance(stored, str):
            stored = str(stored)
        existing = [part for part in stored.split(_RECENT_SEPARATOR) if part]
        entries = [str(path), *(part for part in existing if part != str(path))]
        self._settings.setValue(_RECENT_KEY, _RECENT_SEPARATOR.join(entries[:_RECENT_LIMIT]))
        self._settings.sync()
        self.recent_files_changed.emit()

    # -- internals --------------------------------------------------------------------------

    def _find(self, dataset_id: str) -> ProjectDataset | None:
        for entry in self._project.datasets:
            if entry.id == dataset_id:
                return entry
        return None

    def _rebuilt(
        self,
        *,
        datasets: tuple[ProjectDataset, ...] | None = None,
        analyses: tuple[Analysis, ...] | None = None,
        plot_settings: dict[str, Any] | None = None,
    ) -> Project:
        """A fresh validated Project (model_copy would skip the cross-field validators)."""
        return Project(
            name=self._project.name,
            created_at=self._project.created_at,
            datasets=self._project.datasets if datasets is None else datasets,
            analyses=self._project.analyses if analyses is None else analyses,
            plot_settings=self._project.plot_settings if plot_settings is None else plot_settings,
        )

    def _push_undo(self) -> None:
        self._undo_stack.append(self._project)
        if len(self._undo_stack) > _UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._emit_undo_available()

    def _emit_undo_available(self) -> None:
        self.undo_available_changed.emit(bool(self._undo_stack))

    def _set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.dirty_changed.emit(value)

    def _confirm_discard(self, action: str) -> bool:
        if not self._dirty:
            return True
        answer = self._confirmation.confirm_unsaved_changes(action)
        if answer is UnsavedAnswer.CANCEL:
            return False
        if answer is UnsavedAnswer.SAVE:
            return self.save()
        return True
