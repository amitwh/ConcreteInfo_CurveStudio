"""AnalysisController: owns Auto-fit runs, manual fits and their persistence.

Auto-fit is the primary workflow: importing or selecting a dataset without a stored run
starts one automatically (in Manual mode nothing starts on its own). Everything heavy runs
through a :class:`~curvestudio_desktop.jobs.JobRunner` on this controller's own runner —
deliberately not the window's, so a minutes-long Auto-fit never triggers the window's busy
cursor: the UI stays interactive, with progress, live ranking and Cancel. Runs stream their
outcomes as :class:`~curvestudio_desktop.controllers.analysis_jobs.AutoFitLive` snapshots
delivered on the GUI thread. Every finished run and every fit is stored in the project as an
:class:`~curvestudio.project.Analysis`, so Save/Open round-trips them and a click in the
project tree restores a stored result without refitting. The controller uses QtCore only.
"""

from collections.abc import Callable
from typing import Any, Literal, cast

from PySide6.QtCore import QObject, QSettings, Signal

from curvestudio.data import XY, DataError
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CancelToken,
    RankBy,
    candidate_ref,
)
from curvestudio.discovery import (
    CustomRef as DiscoveryCustomRef,
)
from curvestudio.discovery import (
    PolynomialRef as DiscoveryPolynomialRef,
)
from curvestudio.discovery import (
    RegisteredRef as DiscoveryRegisteredRef,
)
from curvestudio.project import Analysis
from curvestudio.workflow import (
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    Prediction,
    RegisteredRef,
    WorkflowError,
)
from curvestudio.workflow import (
    predict as workflow_predict,
)
from curvestudio_desktop.controllers import analysis_jobs, analysis_store
from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.jobs import JobHandle, JobRunner

MODE_AUTOFIT = "autofit"
MODE_MANUAL = "manual"
MODE_STATS = "stats"
_MODE_KEY = "analysis/mode"
_RANK_KEY = "analysis/rank_by"

_MODE_MESSAGES = {
    MODE_AUTOFIT: "Auto-fit mode: analyses run automatically on import",
    MODE_MANUAL: "Manual fit mode: nothing runs until you fit",
    MODE_STATS: "Statistics mode: pick a tool and its columns; nothing runs until you do",
}
"""What each mode says when it is chosen: the rule it follows, in one line."""

FitSubmit = Callable[[Callable[[FitBundle], None], Callable[[BaseException], None]], object]
"""How ``_start_fit`` reaches the runner: the two callbacks it wants delivered."""


def _reason_of(error: BaseException) -> str:
    """A stable, readable reason for an exception the workflow layers define."""
    code = getattr(error, "code", None)
    return f"{code}: {error}" if isinstance(code, str) else str(error)


class _RunSignals(QObject):
    """Emitted on the worker thread, delivered queued on the GUI thread (JobRunner pattern)."""

    live = Signal(object, int)


class AnalysisController(QObject):
    """Runs Auto-fit and manual fits off the GUI thread and keeps their results current."""

    autofit_started = Signal(str)
    """An Auto-fit run began: the dataset id."""

    autofit_live = Signal(object)
    """A streaming snapshot (:class:`AutoFitLive`) of the running Auto-fit."""

    autofit_finished = Signal(str, object)
    stat_finished = Signal(str, object)
    """An Auto-fit run ended (complete, cancelled or restored): dataset id and result."""

    autofit_failed = Signal(str)
    """An Auto-fit run could not run at all: the human-readable reason."""

    fit_started = Signal(str)
    """A fit job was submitted: the dataset id."""

    fit_finished = Signal(object, object, object)
    """A fit is ready to show: bundle, spec and meta (dict with dataset_id, source, ...)."""

    fit_failed = Signal(str)
    """A fit job raised: the reason (a non-converged fit is *not* a failure)."""

    validation_ready = Signal(int, object)
    """A spec dry-run ended: its token and the ``WorkflowError`` (``None`` when the spec runs)."""

    mode_changed = Signal(str)
    rank_by_changed = Signal(object)
    status_message = Signal(str)

    def __init__(
        self,
        project_controller: ProjectController,
        selection: SelectionController,
        *,
        jobs: JobRunner | None = None,
        fit_jobs: JobRunner | None = None,
        settings: QSettings | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project_controller
        self._selection = selection
        self._jobs = jobs if jobs is not None else JobRunner(self)
        # fits get their own runner: ``is_fit_running`` must report fit jobs only,
        # never an Auto-fit run or a spec dry-run sharing the queue
        self._fit_jobs = fit_jobs if fit_jobs is not None else JobRunner(self)
        self._settings = settings
        self._mode = self._stored_mode()
        self._rank_by = self._stored_rank_by()
        self._options = AutoFitConfig()
        self._autofit_cache: dict[str, AutoFitResult] = {}
        self._latest_runs: dict[str, AutoFitResult] = {}
        self._cancel: CancelToken | None = None
        self._running_dataset: str | None = None
        self._run_serial = 0
        self._fit_handle: JobHandle | None = None
        self._bundle: FitBundle | None = None
        self._spec: FitSpec | None = None
        self._meta: dict[str, Any] = {}
        self._validation_serial = 0
        self.run_count = 0
        self.fit_count = 0
        self._signals = _RunSignals()
        self._signals.live.connect(self._on_live)

    # -- mode and preferences ---------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_AUTOFIT, MODE_MANUAL, MODE_STATS) or mode == self._mode:
            return
        self._mode = mode
        self._store(_MODE_KEY, mode)
        self.mode_changed.emit(mode)
        self.status_message.emit(_MODE_MESSAGES[mode])

    @property
    def runner(self) -> JobRunner:
        """The runner long statistical jobs are submitted to (off the GUI thread).

        It is the controller's own runner — deliberately not the window's, so a minutes-long
        job never raises the window's busy cursor, exactly as the Auto-fit runs work.
        """
        return self._jobs

    @property
    def rank_by(self) -> RankBy:
        return self._rank_by

    def set_rank_by(self, rank_by: RankBy) -> None:
        if rank_by is self._rank_by:
            return
        self._rank_by = rank_by
        self._store(_RANK_KEY, rank_by.value)
        self.rank_by_changed.emit(rank_by)

    def autofit_options(self) -> AutoFitConfig:
        return self._options

    def selection_active_id(self) -> str | None:
        """The selection controller's active dataset (views reach it through the controller)."""
        return self._selection.active_id

    def set_autofit_options(self, config: AutoFitConfig) -> None:
        """The options the next run uses (also what an auto-started run uses)."""
        self._options = config

    def _stored_mode(self) -> str:
        value = self._settings.value(_MODE_KEY, MODE_AUTOFIT) if self._settings else MODE_AUTOFIT
        if value in (MODE_AUTOFIT, MODE_MANUAL, MODE_STATS):
            return str(value)
        return MODE_AUTOFIT

    def _stored_rank_by(self) -> RankBy:
        if self._settings is None:
            return RankBy.BEST_OVERALL
        value = self._settings.value(_RANK_KEY, RankBy.BEST_OVERALL.value)
        try:
            return RankBy(str(value))
        except ValueError:
            return RankBy.BEST_OVERALL

    def _store(self, key: str, value: str) -> None:
        if self._settings is not None:
            self._settings.setValue(key, value)
            self._settings.sync()

    # -- stored results --------------------------------------------------------------------

    def stored_autofit(self, dataset_id: str) -> AutoFitResult | None:
        """The stored Auto-fit result for ``dataset_id``, parsed once and cached.

        Only the canonical (complete) run lives here — a cancelled partial run is
        stored under its own analysis id and is never passed off as the stored one.
        """
        if dataset_id in self._autofit_cache:
            return self._autofit_cache[dataset_id]
        analysis = self._find_analysis(analysis_store.autofit_analysis_id(dataset_id))
        if analysis is None:
            return None
        try:
            result = analysis_store.autofit_from_analysis(analysis)
        except ValueError:
            return None  # a hand-edited archive: not fatal, there is just nothing stored
        self._autofit_cache[dataset_id] = result
        return result

    def autofit_for_display(self, dataset_id: str) -> AutoFitResult | None:
        """The result the Auto-fit view is showing for ``dataset_id``.

        A just-finished or restored run (partial runs included) is what the ranking
        table holds, so row lookups — a selected row's sampled curve, opening a row —
        must read it rather than the canonical stored run.
        """
        return self._latest_runs.get(dataset_id) or self.stored_autofit(dataset_id)

    def showing_stored_autofit(self, dataset_id: str, result: AutoFitResult | None) -> None:
        """The views switched back to the stored canonical run (dataset change)."""
        if result is None:
            self._latest_runs.pop(dataset_id, None)
        else:
            self._latest_runs[dataset_id] = result

    def has_current_autofit(self, dataset_id: str) -> bool:
        """Whether a stored run exists *for the data as currently selected*.

        A run made on other columns does not count: its rows cannot be opened and its
        ranking does not describe what is plotted, so Auto-fit starts again.
        """
        stored = self.stored_autofit(dataset_id)
        if stored is None:
            return False
        xy = self._xy_for(dataset_id)
        return xy is not None and stored.dataset_fingerprint == xy.fingerprint

    def maybe_autostart(self, dataset_id: str | None) -> bool:
        """Start Auto-fit for ``dataset_id`` unless a current stored run exists.

        The product rule: Auto-fit is primary, so importing or selecting a dataset runs it;
        Manual mode never starts anything by itself.
        """
        if dataset_id is None or self._mode != MODE_AUTOFIT:
            return False
        if self.is_autofit_running():
            if self._running_dataset == dataset_id:
                return False
            self.cancel_autofit()  # the target moved: the old run's partial result is dropped
        if self.has_current_autofit(dataset_id):
            return False
        if self._xy_for(dataset_id) is None:
            self.status_message.emit("Auto-fit needs two numeric columns; this dataset has none")
            return False
        self.run_autofit(dataset_id)
        return True

    # -- running Auto-fit -------------------------------------------------------------------

    def is_autofit_running(self) -> bool:
        return self._cancel is not None

    @property
    def running_dataset(self) -> str | None:
        return self._running_dataset

    def run_autofit(self, dataset_id: str) -> None:
        """Run Auto-fit on ``dataset_id`` with the current options (also "Run again")."""
        xy = self._xy_for(dataset_id)
        if xy is None:
            self.autofit_failed.emit("the dataset has no usable x and y columns")
            return
        self._cancel = CancelToken()
        self._running_dataset = dataset_id
        self._run_serial += 1
        self.run_count += 1
        serial = self._run_serial
        self.autofit_started.emit(dataset_id)

        def emit_live(live: object, s: int = serial) -> None:
            self._signals.live.emit(live, s)

        def finished(result: AutoFitResult, ds: str = dataset_id, s: int = serial) -> None:
            self._finish_autofit(ds, s, result)

        def failed(exc: BaseException, ds: str = dataset_id) -> None:
            self._fail_autofit(ds, exc)

        self._jobs.submit(
            analysis_jobs.run_autofit,
            xy,
            self._options,
            self._cancel,
            emit_live,
            on_result=finished,
            on_error=failed,
        )

    def cancel_autofit(self) -> None:
        """Ask the running Auto-fit to stop; its partial result is still delivered."""
        if self._cancel is not None:
            self._cancel.cancel()
            self.status_message.emit("Cancelling Auto-fit…")

    def _on_live(self, payload: object, serial: int) -> None:
        # a superseded run's late snapshot is dropped; so is anything after the run ended
        if serial == self._run_serial and self._cancel is not None:
            self.autofit_live.emit(payload)

    def _finish_autofit(self, dataset_id: str, serial: int, result: AutoFitResult) -> None:
        if serial != self._run_serial:
            return  # superseded (a newer run started); the partial result is not stored
        self._cancel = None
        self._running_dataset = None
        self._latest_runs[dataset_id] = result  # what the views are showing now
        if result.cancelled:
            # a partial run is its own analysis: the canonical slot (and cache) keep
            # the earlier complete run, and the partial is always shown flagged
            self._project.add_or_replace_analysis(
                analysis_store.partial_autofit_analysis(dataset_id, result)
            )
        else:
            self._autofit_cache[dataset_id] = result
            self._project.add_or_replace_analysis(
                analysis_store.autofit_analysis(dataset_id, result)
            )
        self.autofit_finished.emit(dataset_id, result)
        if result.cancelled:
            self.status_message.emit("Auto-fit cancelled: showing the partial ranking")
        else:
            self.status_message.emit(
                f"Auto-fit finished: {len(result.ranked)} models fitted "
                f"({len(result.unconverged)} not converged, {len(result.skipped)} skipped) "
                f"in {result.elapsed_seconds:.1f} s"
            )

    def _fail_autofit(self, dataset_id: str, error: BaseException) -> None:
        self._cancel = None
        self._running_dataset = None
        self.autofit_failed.emit(_reason_of(error))
        self.status_message.emit(f"Auto-fit could not run on {dataset_id}: {_reason_of(error)}")

    # -- fits (manual and opened rows) -------------------------------------------------------

    def submit_fit(
        self,
        dataset_id: str,
        xy: XY,
        spec: FitSpec,
        *,
        source: dict[str, str] | None = None,
    ) -> None:
        """Fit ``spec`` on ``xy`` off-thread; the result fills the tabs and is stored."""
        self._start_fit(
            dataset_id,
            lambda on_result, on_error: self._fit_jobs.submit(
                analysis_jobs.fit_job, xy, spec, on_result=on_result, on_error=on_error
            ),
            spec,
            analysis_store.unique_fit_id(self._taken_analysis_ids()),
            source,
        )

    def open_candidate(self, dataset_id: str, result: AutoFitResult, candidate_id: str) -> None:
        """Refit one Auto-fit row exactly as the run fitted it and show its full bundle."""
        xy = self._xy_for(dataset_id)
        if xy is None:
            self.fit_failed.emit("the dataset has no usable x and y columns")
            return
        try:
            spec = self._spec_for_candidate(result, candidate_id)
        except KeyError as exc:
            self.fit_failed.emit(str(exc.args[0]))
            return
        except (ValueError, WorkflowError) as exc:
            self.fit_failed.emit(_reason_of(exc) if isinstance(exc, WorkflowError) else str(exc))
            return
        analysis_id = analysis_store.safe_analysis_id(
            analysis_store.FIT_PREFIX, dataset_id, candidate_id
        )
        self._start_fit(
            dataset_id,
            lambda on_result, on_error: self._fit_jobs.submit(
                analysis_jobs.open_candidate_job,
                xy,
                result,
                candidate_id,
                on_result=on_result,
                on_error=on_error,
            ),
            spec,
            analysis_id,
            {"kind": "autofit", "candidate_id": candidate_id},
        )

    def _start_fit(
        self,
        dataset_id: str,
        submit: FitSubmit,
        spec: FitSpec,
        analysis_id: str,
        source: dict[str, str] | None,
    ) -> None:
        self.fit_count += 1
        meta: dict[str, Any] = {"dataset_id": dataset_id, "analysis_id": analysis_id}
        if source is not None:
            meta["source"] = source
        self.fit_started.emit(dataset_id)
        self._fit_handle = cast(
            "JobHandle",
            submit(
                lambda bundle: self._finish_fit(dataset_id, spec, meta, bundle),
                self._fail_fit,
            ),
        )

    def is_fit_running(self) -> bool:
        """Whether a fit job is in flight (a cancelled one until its worker settles).

        Fit jobs only — an Auto-fit run or a spec dry-run never counts, so the manual
        panel's Cancel and busy state track exactly what the user started there.
        """
        return self._fit_jobs.is_busy()

    def cancel_fit(self) -> None:
        """Drop the running fit's callbacks (cooperative: its worker finishes unnoticed)."""
        if self._fit_handle is not None:
            self._fit_handle.cancel()

    def _finish_fit(
        self, dataset_id: str, spec: FitSpec, meta: dict[str, Any], bundle: FitBundle
    ) -> None:
        try:
            analysis = analysis_store.fit_analysis(
                dataset_id,
                spec,
                bundle,
                analysis_id=meta["analysis_id"],
                source=meta.get("source"),
            )
        except ValueError as exc:
            self.fit_failed.emit(f"the fit cannot be stored in the project: {exc}")
            return
        self._project.add_or_replace_analysis(analysis)
        self._bundle, self._spec, self._meta = bundle, spec, meta
        self.fit_finished.emit(bundle, spec, meta)
        status = bundle.result.status.value
        reason = f" ({bundle.result.termination_reason})" if status != "converged" else ""
        self.status_message.emit(f"Fit {bundle.result.model_id}: {status}{reason}")

    def _fail_fit(self, error: BaseException) -> None:
        self.fit_failed.emit(_reason_of(error))

    def validate_spec(self, spec: FitSpec, weighted: bool) -> int:
        """Dry-run ``spec`` off-thread; :attr:`validation_ready` reports its token and verdict."""
        self._validation_serial += 1
        token = self._validation_serial

        def validated(_ok: object, t: int = token) -> None:
            self.validation_ready.emit(t, None)

        def invalid(exc: BaseException, t: int = token) -> None:
            self.validation_ready.emit(t, exc)

        self._jobs.submit(
            analysis_jobs.check_spec_job,
            spec,
            weighted,
            on_result=validated,
            on_error=invalid,
        )
        return token

    def predict(
        self,
        bundle: FitBundle,
        spec: FitSpec,
        x: list[float],
        band: Literal["confidence", "prediction"] | None,
    ) -> Prediction:
        """Predicted values at ``x`` (see :func:`curvestudio.workflow.predict`).

        Raises:
            WorkflowError: exactly what the workflow layer reports (``FIT_NOT_CONVERGED``,
                ``TOO_LARGE``, ...); the caller shows it inline.
        """
        return workflow_predict(bundle, spec, x, band)

    # -- restoring stored analyses -----------------------------------------------------------

    def restore_analysis(self, analysis_id: str) -> bool:
        """Show a stored analysis again, without running anything.

        Returns ``False`` (with a status message) when the entry cannot be rebuilt.
        """
        analysis = self._find_analysis(analysis_id)
        if analysis is None:
            return False
        if analysis.kind == "stat":
            try:
                result = analysis_store.stat_from_analysis(analysis)
            except ValueError as exc:
                self.status_message.emit(str(exc))
                return False
            self.set_mode(MODE_STATS)
            self.stat_finished.emit(analysis.dataset_id, result)
            self.status_message.emit("Restored the stored statistics result (nothing was run)")
            return True
        if analysis.kind == "autofit":
            try:
                result = analysis_store.autofit_from_analysis(analysis)
            except ValueError as exc:
                self.status_message.emit(str(exc))
                return False
            self._latest_runs[analysis.dataset_id] = result
            if not result.cancelled:  # a partial never takes the canonical slot
                self._autofit_cache[analysis.dataset_id] = result
            self.autofit_finished.emit(analysis.dataset_id, result)
            if result.cancelled:
                self.status_message.emit("Restored the partial Auto-fit run (nothing was refitted)")
            else:
                self.status_message.emit("Restored the stored Auto-fit run (nothing was refitted)")
            return True
        try:
            bundle, spec = analysis_store.bundle_from_analysis(analysis)
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return False
        self._bundle, self._spec = bundle, spec
        self._meta = {
            "dataset_id": analysis.dataset_id,
            "analysis_id": analysis.id,
            "restored": True,
        }
        self.fit_finished.emit(bundle, spec, self._meta)
        self.status_message.emit(f"Restored the stored fit {bundle.result.model_id}")
        return True

    def dataset_of_analysis(self, analysis_id: str) -> str | None:
        analysis = self._find_analysis(analysis_id)
        return None if analysis is None else analysis.dataset_id

    def analysis_label(self, analysis_id: str) -> str:
        """The tree label of an analysis: what it is, not its internal id."""
        analysis = self._find_analysis(analysis_id)
        if analysis is None:
            return analysis_id
        if analysis.kind == "autofit":
            if analysis_store.is_partial_autofit(analysis):
                return "Auto-fit (partial)"
            return "Auto-fit"
        result = analysis.result
        return result.model_id if not isinstance(result, dict) else analysis.id

    # -- current display state ---------------------------------------------------------------

    @property
    def current_bundle(self) -> FitBundle | None:
        return self._bundle

    @property
    def current_spec(self) -> FitSpec | None:
        return self._spec

    def clear_current_fit(self) -> None:
        """Drop the displayed fit (the active dataset moved away from it)."""
        self._bundle = None
        self._spec = None
        self._meta = {}

    # -- internals ---------------------------------------------------------------------------

    def _xy_for(self, dataset_id: str) -> XY | None:
        if self._selection.active_id == dataset_id:
            return self._selection.xy()
        dataset = self._selection.active_dataset()
        if dataset is None:
            return None
        columns = self._selection.columns(dataset_id)
        if columns is None:
            return None
        x, y, weight = columns
        try:
            return dataset.xy(x, y, weight)
        except DataError:
            return None

    def _find_analysis(self, analysis_id: str) -> Analysis | None:
        return next((a for a in self._project.project.analyses if a.id == analysis_id), None)

    def _taken_analysis_ids(self) -> set[str]:
        return {a.id for a in self._project.project.analyses}

    def _spec_for_candidate(self, result: AutoFitResult, candidate_id: str) -> FitSpec:
        """The workflow spec of one Auto-fit row, so its bundle can predict after a restore."""
        ref = candidate_ref(result.config, candidate_id)
        if isinstance(ref, DiscoveryPolynomialRef):
            # A through-the-origin polynomial predicts through the plain family: the spec
            # check counts parameters, and the origin variant has exactly ``degree`` of them.
            model: Any = PolynomialRef(degree=ref.degree)
        elif isinstance(ref, DiscoveryRegisteredRef):
            model = RegisteredRef(id=ref.model_id)
        elif isinstance(ref, DiscoveryCustomRef):
            parameters = ref.definition.parameters
            if parameters is None:
                raise ValueError(
                    f"candidate {candidate_id!r} has no explicit parameters; it cannot predict"
                )
            definition = ref.definition
            model = CustomRef(
                equation=definition.expression,
                parameters=[p.name for p in parameters],
                variable=definition.variables[0] if definition.variables else "x",
            )
        return FitSpec(model=model)

    def reset_for_project(self) -> None:
        """The project was replaced or opened: cached results and the shown fit are stale."""
        self._autofit_cache.clear()
        self._latest_runs.clear()
        self.clear_current_fit()
