/**
 * The workspace: datasets sidebar | chart | analysis area. The product rule — the curve
 * auto-fit should work, not manual curve fit — makes Auto-fit the default analysis mode:
 * it starts by itself after a dataset is uploaded or selected (the latest stored run is
 * shown when there is one), and Manual fit (the unchanged FitPanel/tabs) appears only
 * when the user selects it; the choice is remembered per user. The workspace owns the
 * cross-panel state: datasets, the model catalogue, the active fit bundle (a manual fit
 * or an opened Auto-fit row), the Auto-fit state machine and the fit-job poller.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from "react";
import { Upload } from "lucide-react";
import { useAuth } from "../api/auth";
import {
  api,
  type DatasetDetailOut,
  type DatasetOut,
  type FitBundle,
  type FitSummaryOut,
  type ModelInfo,
  type OpenRowOut,
} from "../api/client";
import { ErrorBox, Loading, errorText } from "../components/States";
import { useTheme } from "../components/ThemeToggle";
import { useJobPolling } from "../hooks/useJobPolling";
import { ModeSwitch } from "../workspace/ModeSwitch";
import { ChartPanel } from "../workspace/ChartPanel";
import { DatasetsSidebar } from "../workspace/DatasetsSidebar";
import { FitPanel } from "../workspace/FitPanel";
import { JobStatus } from "../workspace/JobStatus";
import { ResultsTabs } from "../workspace/results/ResultsTabs";
import { AutofitOptions } from "../workspace/autofit/AutofitOptions";
import { AutofitPanel } from "../workspace/autofit/AutofitPanel";
import { DEFAULT_AUTOFIT_OPTIONS, useAutofit, type AutoFitOptions } from "../workspace/autofit/useAutofit";
import { rowCurveBundle } from "../workspace/autofit/rowCurve";
import { loadMode, saveMode, type WorkspaceMode } from "../workspace/mode";

/** The chart requests the full 20 000-point sample the API allows. */
const CHART_POINTS = 20000;

/** Where the active fit bundle came from — the chart obeys the mode, the tabs don't. */
type BundleSource = "manual" | "opened-row";

/** Auto-fit candidate ids for polynomials (poly3, poly7_origin, …). */
const POLYNOMIAL_CANDIDATE = /^poly(\d+)(?:_origin)?$/;

/** What the manual panel should be prefilled with ("Use this model manually"). */
type ManualPreset = { kind: "registered"; id: string } | { kind: "polynomial"; degree: number };

export function WorkspacePage(): ReactElement {
  const { user } = useAuth();
  const { theme } = useTheme();
  const dark = theme === "dark";

  const [datasets, setDatasets] = useState<DatasetOut[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<unknown>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** The selection a long request was started for; late responses compare against it. */
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;
  const [detail, setDetail] = useState<DatasetDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<unknown>(null);

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<unknown>(null);

  const [mode, setMode] = useState<WorkspaceMode>(() => loadMode(user?.id ?? null));
  const [autofitOptions, setAutofitOptions] = useState<AutoFitOptions>(DEFAULT_AUTOFIT_OPTIONS);
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null);
  const [openRowBusy, setOpenRowBusy] = useState<string | null>(null);
  const [openRowError, setOpenRowError] = useState<string | null>(null);
  const [manualPreset, setManualPreset] = useState<ManualPreset | null>(null);

  const [fits, setFits] = useState<FitSummaryOut[]>([]);
  const [fitsError, setFitsError] = useState<unknown>(null);
  const [activeFitId, setActiveFitId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<FitBundle | null>(null);
  const [bundleSource, setBundleSource] = useState<BundleSource>("manual");
  const [bundleError, setBundleError] = useState<unknown>(null);
  const [fitError, setFitError] = useState<string | null>(null);

  const autofit = useAutofit();
  const autoState = autofit.stateFor(selectedId);

  const loadBundle = useCallback(async (fitId: string, source: BundleSource): Promise<void> => {
    setBundleError(null);
    setBundle(null);
    try {
      setBundle(await api.request<FitBundle>(`/fits/${fitId}`));
      setActiveFitId(fitId);
      setBundleSource(source);
    } catch (error) {
      setBundleError(error);
    }
  }, []);

  // The job poller serves manual fits and opened Auto-fit rows alike; whichever flow
  // started the job says what the finished bundle is for.
  const pendingSourceRef = useRef<BundleSource>("manual");
  const bundleSourceRef = useRef(bundleSource);
  bundleSourceRef.current = bundleSource;
  const poller = useJobPolling((fitId) => {
    setFitError(null);
    setOpenRowBusy(null);
    void loadBundle(fitId, pendingSourceRef.current);
    if (selectedId !== null) void loadFits(selectedId);
  });

  // An opened-row job that ends without a fit (failed, cancelled, lost contact) must
  // unlock the ranking table again; the inline error (if any) stays visible.
  useEffect(() => {
    const phase = poller.state.phase;
    if (phase === "failed" || phase === "error" || phase === "cancelled") {
      setOpenRowBusy(null);
    }
  }, [poller.state.phase]);

  const loadDatasets = useCallback(async (): Promise<void> => {
    setDatasetsLoading(true);
    setDatasetsError(null);
    try {
      setDatasets(await api.request<DatasetOut[]>("/datasets"));
    } catch (error) {
      setDatasetsError(error);
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  const loadFits = useCallback(async (datasetId: string): Promise<void> => {
    setFitsError(null);
    try {
      setFits(await api.request<FitSummaryOut[]>(`/datasets/${datasetId}/fits`));
    } catch (error) {
      // The fits list is a sidebar of history; a failure here keeps the workspace usable,
      // but it must not look like an empty history.
      setFits([]);
      setFitsError(error);
    }
  }, []);

  useEffect(() => {
    void loadDatasets();
    api
      .request<ModelInfo[]>("/models")
      .then(setModels)
      .catch((error: unknown) => setModelsError(error));
  }, [loadDatasets]);

  const selectDataset = useCallback(
    async (id: string): Promise<void> => {
      setSelectedId(id);
      setDetail(null);
      setDetailError(null);
      setBundle(null);
      setActiveFitId(null);
      setBundleSource("manual");
      setFitError(null);
      setFitsError(null);
      setSelectedRowId(null);
      setOpenRowBusy(null);
      setOpenRowError(null);
      setManualPreset(null);
      poller.reset();
      // Auto-fit leads: ensure a run (or the latest stored one) right after selection —
      // in either mode, so switching to Auto-fit later has the result ready.
      autofit.ensure(id);
      setDetailLoading(true);
      try {
        setDetail(
          await api.request<DatasetDetailOut>(`/datasets/${id}?points=${CHART_POINTS}`),
        );
        await loadFits(id);
      } catch (error) {
        setDetailError(error);
      } finally {
        setDetailLoading(false);
      }
    },
    [autofit, loadFits, poller],
  );

  const onFit = useCallback(
    async (spec: Record<string, unknown>): Promise<void> => {
      if (selectedId === null) return;
      setFitError(null);
      try {
        const created = await api.request<{ job_id: string }>(
          `/datasets/${selectedId}/fits`,
          { method: "POST", body: JSON.stringify(spec) },
        );
        pendingSourceRef.current = "manual";
        poller.start(created.job_id);
      } catch (error) {
        // A spec the engine rejects (bad equation, unknown model) lands here, inline.
        setFitError(errorText(error, "the fit could not be started"));
      }
    },
    [selectedId, poller],
  );

  const onUploaded = useCallback(
    (dataset: DatasetOut): void => {
      setDatasets((current) => [dataset, ...current]);
      void selectDataset(dataset.id);
    },
    [selectDataset],
  );

  const onDeletedDataset = useCallback(
    (id: string): void => {
      setDatasets((current) => current.filter((d) => d.id !== id));
      autofit.clear(id);
      if (selectedId === id) {
        setSelectedId(null);
        setDetail(null);
        setBundle(null);
        setActiveFitId(null);
        setFits([]);
        setFitsError(null);
        setSelectedRowId(null);
        poller.reset();
      }
    },
    [autofit, selectedId, poller],
  );

  const onDeleteFit = useCallback(
    async (fitId: string): Promise<void> => {
      try {
        await api.request(`/fits/${fitId}`, { method: "DELETE" });
        setFits((current) => current.filter((f) => f.id !== fitId));
        if (activeFitId === fitId) {
          setActiveFitId(null);
          setBundle(null);
        }
      } catch (error) {
        setFitError(errorText(error, "the fit could not be deleted"));
      }
    },
    [activeFitId],
  );

  const onModeChange = useCallback(
    (next: WorkspaceMode): void => {
      setMode(next);
      if (user !== null) saveMode(user.id, next);
    },
    [user],
  );

  // Keep a valid row selected: default to the best row whenever the loaded ranking
  // changes (a new run, a re-ranking) and the current pick is no longer in it.
  useEffect(() => {
    const rows = autoState.stored?.result.ranked ?? null;
    if (rows === null || rows.length === 0) {
      if (selectedRowId !== null) setSelectedRowId(null);
      return;
    }
    if (selectedRowId === null || !rows.some((row) => row.model_id === selectedRowId)) {
      setSelectedRowId(rows[0]?.model_id ?? null);
    }
  }, [autoState.stored, selectedRowId]);

  const onSelectRow = useCallback((modelId: string): void => {
    setSelectedRowId(modelId);
    // A clicked row takes the chart back from an opened row's full fit to the row's own
    // curve; a manual fit's bundle (hidden in Auto-fit mode) is left untouched.
    if (bundleSourceRef.current === "opened-row") {
      setBundle(null);
      setActiveFitId(null);
      setBundleSource("manual");
    }
  }, []);

  const onOpenRow = useCallback(
    async (candidateId: string): Promise<void> => {
      const storedId = autoState.stored?.id;
      if (selectedId === null || storedId == null) return;
      const requestedFor = selectedId;
      setOpenRowBusy(candidateId);
      setOpenRowError(null);
      try {
        const created = await api.request<OpenRowOut>(
          `/autofits/${storedId}/rows/${candidateId}/open`,
          { method: "POST" },
        );
        // The user selected another dataset while the POST was in flight: the result
        // belongs to a selection that is no longer on screen.
        if (selectedIdRef.current !== requestedFor) {
          setOpenRowBusy(null);
          return;
        }
        if (created.fit_id != null) {
          // Opened before: the fit already exists.
          await loadBundle(created.fit_id, "opened-row");
          setOpenRowBusy(null);
          return;
        }
        pendingSourceRef.current = "opened-row";
        poller.start(created.job_id);
      } catch (error) {
        setOpenRowError(errorText(error, "the row could not be opened"));
        setOpenRowBusy(null);
      }
    },
    [autoState.stored?.id, loadBundle, poller, selectedId],
  );

  const onUseManually = useCallback(
    (row: { modelId: string; name: string }): void => {
      const poly = POLYNOMIAL_CANDIDATE.exec(row.modelId);
      if (poly !== null) {
        const degree = Number.parseInt(poly[1] ?? "1", 10);
        setManualPreset({ kind: "polynomial", degree: Number.isNaN(degree) ? 1 : degree });
      } else {
        setManualPreset({ kind: "registered", id: row.modelId });
      }
      setMode("manual");
      if (user !== null) saveMode(user.id, "manual");
    },
    [user],
  );

  const onRunAutofit = useCallback((): void => {
    if (selectedId !== null) autofit.run(selectedId, autofitOptions);
  }, [autofit, autofitOptions, selectedId]);

  // In Auto-fit mode the chart shows the opened row's full bundle, else the selected
  // row's stored curve, else the data alone; Manual fit keeps its own bundle.
  const selectedCurve =
    selectedRowId !== null && autoState.stored !== null
      ? ((autoState.stored.result.curves ?? {})[selectedRowId] ?? null)
      : null;
  const chartBundle =
    mode === "auto"
      ? bundleSource === "opened-row" && bundle !== null
        ? bundle
        : selectedCurve !== null
          ? rowCurveBundle(selectedCurve)
          : null
      : bundle;

  const autofitStatus = (() => {
    switch (autoState.phase) {
      case "checking":
        return "checking…";
      case "starting":
        return "starting…";
      case "running":
        return autoState.progress === null
          ? "running…"
          : `running ${autoState.progress.done}/${autoState.progress.total}`;
      case "loading":
        return "loading…";
      case "ready":
        if (autoState.stored !== null) return autoState.stored.partial ? "done (partial)" : "done";
        return autoState.cancelled ? "cancelled (partial)" : "not run";
      case "error":
        return "failed";
      default:
        return "not run";
    }
  })();

  return (
    <div className="workspace">
      <DatasetsSidebar
        datasets={datasets}
        selectedId={selectedId}
        loading={datasetsLoading}
        loadError={datasetsError}
        onSelect={(id) => void selectDataset(id)}
        onUploaded={onUploaded}
        onDeleted={onDeletedDataset}
      />

      <div className="workspace__main">
        <ModeSwitch mode={mode} onModeChange={onModeChange} autofitStatus={autofitStatus} />
        <JobStatus
          state={poller.state}
          onCancel={() => void poller.cancel()}
          succeededText={
            // Only speak for the fit this job produced — a fit loaded from history is
            // not this job's outcome.
            poller.state.phase === "succeeded" &&
            poller.state.fitId === activeFitId &&
            bundle !== null &&
            bundle.result.status !== "converged"
              ? "Fit did not converge."
              : undefined
          }
        />
        <ChartPanel dataset={detail} bundle={chartBundle} dark={dark} />
        {detailLoading && <Loading label="Loading dataset…" />}
        {detailError !== null && <ErrorBox error={detailError} />}
        {detail === null && !detailLoading && detailError === null && (
          <p className="state state--empty">
            <Upload className="icon icon--lg" aria-hidden="true" focusable="false" />
            Upload a dataset and every model is tried and ranked automatically — or switch
            to Manual fit to choose one yourself.
          </p>
        )}
        {modelsError !== null && (
          <p className="alert alert--warning">
            The model catalogue could not be loaded ({errorText(modelsError)}). Fitting
            stays available for polynomials and custom equations.
          </p>
        )}
        {bundleError !== null && <ErrorBox error={bundleError} />}
      </div>

      <div className="workspace__side">
        {mode === "auto" ? (
          <section className="panel" aria-label="Auto-fit options">
            <h2 className="panel-title">Auto-fit options</h2>
            <AutofitOptions
              models={models}
              options={autofitOptions}
              onChange={setAutofitOptions}
              disabled={autoState.phase === "running"}
            />
          </section>
        ) : (
          <FitPanel
            dataset={detail}
            models={models}
            busy={poller.state.phase === "polling"}
            onFit={(spec) => void onFit(spec)}
            fitError={fitError}
            preset={manualPreset}
          />
        )}
      </div>

      {/* The primary results get the full width: the ranking table needs ~64 rem. */}
      <div className="workspace__results">
        {mode === "auto" && (
          <AutofitPanel
            datasetId={selectedId}
            datasetName={detail?.name ?? ""}
            state={autoState}
            models={models}
            selectedRowId={selectedRowId}
            onSelectRow={onSelectRow}
            onOpenRow={(candidateId) => void onOpenRow(candidateId)}
            onUseManually={onUseManually}
            onRun={onRunAutofit}
            onCancel={() => void autofit.cancel(selectedId ?? "")}
            onRerank={(rankBy) => {
              if (selectedId !== null) void autofit.rerank(selectedId, rankBy);
            }}
            openRowBusy={openRowBusy}
            openRowError={openRowError}
          />
        )}

        {/* In Auto-fit mode (which hosts the opened-row results) the manual FitPanel is
            not on screen, so surface the fit error (e.g. a failed delete) here instead. */}
        {fitError !== null && mode === "auto" && (
          <p className="alert alert--error" role="alert">
            {fitError}
          </p>
        )}

        {bundle !== null && activeFitId !== null && (mode === "manual" || bundleSource === "opened-row") && (
          <ResultsTabs
            fitId={activeFitId}
            bundle={bundle}
            fits={fits}
            fitsError={fitsError}
            onLoadFit={(id) => void loadBundle(id, "manual")}
            onDeleteFit={(id) => void onDeleteFit(id)}
          />
        )}
      </div>
    </div>
  );
}
