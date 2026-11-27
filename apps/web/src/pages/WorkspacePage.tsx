/**
 * The workspace: datasets sidebar | chart | analysis area. The product rule — the curve
 * auto-fit should work, not manual curve fit — makes Auto-fit the default analysis mode:
 * it starts by itself after a dataset is uploaded or selected (the latest stored run is
 * shown when there is one), and Manual fit (the unchanged FitPanel/tabs) and Statistics
 * (the statistical tools and their stored results) appear only when the user selects
 * them; the choice is remembered per user. The workspace owns the cross-panel state:
 * datasets, the model catalogue, the active fit bundle (a manual fit or an opened Auto-fit
 * row), the Auto-fit state machine, the fit-job poller and the statistics tool, result and
 * history state. Statistics mode starts no fit and no Auto-fit.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from "react";
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
  type StatResultOut,
  type StatResultSummaryOut,
} from "../api/client";
import { ErrorBox, Loading, errorText } from "../components/States";
import { exportPlotPngDataUrl } from "../components/PlotChart";
import { useTheme } from "../components/ThemeToggle";
import { useJobPolling } from "../hooks/useJobPolling";
import { ModeSwitch } from "../workspace/ModeSwitch";
import { ChartPanel } from "../workspace/ChartPanel";
import { DatasetsSidebar, type DatasetsSidebarHandle } from "../workspace/DatasetsSidebar";
import { EmptyWorkspace } from "../workspace/EmptyWorkspace";
import { FitPanel } from "../workspace/FitPanel";
import { JobStatus } from "../workspace/JobStatus";
import { ResultsTabs } from "../workspace/results/ResultsTabs";
import { OpenResultModal } from "../workspace/results/OpenResultModal";
import { AutofitOptions } from "../workspace/autofit/AutofitOptions";
import { AutofitPanel } from "../workspace/autofit/AutofitPanel";
import { DEFAULT_AUTOFIT_OPTIONS, useAutofit, type AutoFitOptions } from "../workspace/autofit/useAutofit";
import { rowCurveBundle } from "../workspace/autofit/rowCurve";
import { StatsHistory } from "../workspace/stats/StatsHistory";
import { StatsPanel } from "../workspace/stats/StatsPanel";
import { StatsPlots } from "../workspace/stats/StatsPlots";
import { StatsResult } from "../workspace/stats/StatsResult";
import { useStatsTools } from "../workspace/stats/useStatsTools";
import { loadMode, saveMode, type WorkspaceMode } from "../workspace/mode";
import { chartSamplePoints } from "../workspace/traces";
import {
  dropDataset,
  fetchDetail,
  fetchDatasets,
  fetchFits,
  fetchModels,
  getDatasets,
  getDetail,
  getModels,
  invalidateDatasets,
  invalidateFits,
} from "../workspace/workspaceCache";

/** Where the active fit bundle came from — the chart obeys the mode, the tabs don't. */
type BundleSource = "manual" | "opened-row";

/** Auto-fit candidate ids for polynomials (poly3, poly7_origin, …). */
const POLYNOMIAL_CANDIDATE = /^poly(\d+)(?:_origin)?$/;

/** What the manual panel should be prefilled with ("Use this model manually"). */
type ManualPreset = { kind: "registered"; id: string } | { kind: "polynomial"; degree: number };

export interface WorkspacePageProps {
  /** The dataset this workspace remembers (null: nothing selected until the user picks). */
  initialDatasetId?: string | null;
  /** The workspace's analysis mode; falls back to the account's remembered mode. */
  initialMode?: WorkspaceMode | null;
  /** Report a dataset selection so the owning workspace can remember it. */
  onSelectionChange?: (datasetId: string | null) => void;
  /** Report a mode change so the owning workspace can remember it. */
  onModeChange?: (mode: WorkspaceMode) => void;
}

export function WorkspacePage({
  initialDatasetId = null,
  initialMode = null,
  onSelectionChange,
  onModeChange,
}: WorkspacePageProps = {}): ReactElement {
  const { user } = useAuth();
  const { theme } = useTheme();
  const dark = theme === "dark";

  // The workspace remounts on every tab switch (AuthenticatedApp keys it by workspace):
  // whatever the cache holds renders at once, so a return visit never flashes empty.
  const [datasets, setDatasets] = useState<DatasetOut[]>(() => getDatasets()?.data ?? []);
  const [datasetsLoading, setDatasetsLoading] = useState(() => getDatasets() === null);
  const [datasetsError, setDatasetsError] = useState<unknown>(null);
  /** The sidebar's existing upload and paste actions, for the empty state's buttons. */
  const sidebarActions = useRef<DatasetsSidebarHandle>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** The selection a long request was started for; late responses compare against it. */
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;
  /** The dataset list as of the last render, read by callbacks without re-creating them. */
  const datasetsRef = useRef(datasets);
  datasetsRef.current = datasets;
  const [detail, setDetail] = useState<DatasetDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<unknown>(null);

  const [models, setModels] = useState<ModelInfo[]>(() => getModels()?.data ?? []);
  const [modelsError, setModelsError] = useState<unknown>(null);

  const [mode, setMode] = useState<WorkspaceMode>(
    () => initialMode ?? loadMode(user?.id ?? null),
  );
  /** The mode as a callback started now reads it, without re-creating the callback. */
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const [autofitOptions, setAutofitOptions] = useState<AutoFitOptions>(DEFAULT_AUTOFIT_OPTIONS);
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null);
  const [openRowBusy, setOpenRowBusy] = useState<string | null>(null);
  const [openRowError, setOpenRowError] = useState<string | null>(null);
  /** The row whose opened result the modal presents, with the fit it is waiting for
   *  (null until the open answers with one); the results panel below the ranking stays
   *  as it is — the modal is where the user is looking. */
  const [openResult, setOpenResult] = useState<{
    modelId: string;
    name: string;
    fitId: string | null;
  } | null>(null);
  const [manualPreset, setManualPreset] = useState<ManualPreset | null>(null);
  const [datasetsCollapsed, setDatasetsCollapsed] = useState(
    () => localStorage.getItem("curvestudio:datasetsCollapsed") === "1",
  );

  const [fits, setFits] = useState<FitSummaryOut[]>([]);
  const [fitsError, setFitsError] = useState<unknown>(null);
  const [activeFitId, setActiveFitId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<FitBundle | null>(null);
  const [bundleSource, setBundleSource] = useState<BundleSource>("manual");
  const [bundleError, setBundleError] = useState<unknown>(null);
  const [fitError, setFitError] = useState<string | null>(null);
  /** The live Plotly root, captured for the PDF report. */
  const chartRootRef = useRef<HTMLDivElement | null>(null);
  const captureChart = useCallback(
    () => exportPlotPngDataUrl(chartRootRef.current),
    [],
  );

  const autofit = useAutofit();
  const autoState = autofit.stateFor(selectedId);

  const stats = useStatsTools();
  const [statResults, setStatResults] = useState<StatResultSummaryOut[]>([]);
  const [statResultsError, setStatResultsError] = useState<unknown>(null);
  const [statShown, setStatShown] = useState<StatResultOut | null>(null);
  const [statBusy, setStatBusy] = useState(false);
  const [statError, setStatError] = useState<unknown>(null);
  const [statPlotIndex, setStatPlotIndex] = useState(0);

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
  // Which model the shown bundle is, so selecting that very row keeps it instead of
  // dropping the opened fit the user is looking at.
  const bundleModelRef = useRef<string | null>(null);
  bundleModelRef.current = bundle?.result.model_id ?? null;
  const poller = useJobPolling((fitId) => {
    setFitError(null);
    setOpenRowBusy(null);
    void loadBundle(fitId, pendingSourceRef.current);
    // An opened-row job that produced a fit is the result the modal is waiting for.
    if (pendingSourceRef.current === "opened-row") {
      setOpenResult((current) =>
        current !== null && current.fitId === null ? { ...current, fitId } : current,
      );
    }
    if (selectedId !== null) {
      // The job stored a fit: the dataset's cached history is out of date from here on.
      // Drop it, and refresh the live list straight into the panel — not through the
      // cache, so the invalidation survives until the next mount revalidates it.
      invalidateFits(selectedId);
      void refreshFits(selectedId);
    }
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
    const cached = getDatasets();
    if (cached !== null) {
      // The remount already rendered this list (the state above started from the cache);
      // only a list older than the TTL revalidates, in the background.
      setDatasets(cached.data);
      setDatasetsError(null);
      setDatasetsLoading(false);
      if (!cached.stale) return;
      try {
        const fetched = await fetchDatasets();
        // An upload or a delete while the revalidation was away invalidated the list:
        // the mutation's own answer owns the sidebar now.
        if (getDatasets() !== null) setDatasets(fetched);
      } catch {
        // The cached list is still on screen; the next mount revalidates again.
      }
      return;
    }
    setDatasetsLoading(true);
    setDatasetsError(null);
    try {
      setDatasets(await fetchDatasets());
    } catch (error) {
      setDatasetsError(error);
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  const loadModels = useCallback(async (): Promise<void> => {
    const cached = getModels();
    if (cached !== null && !cached.stale) {
      setModels(cached.data);
      setModelsError(null);
      return;
    }
    try {
      setModels(await fetchModels());
      setModelsError(null);
    } catch (error) {
      // With a cached catalogue on screen a failed revalidation is noise, not breakage;
      // without one the workspace must say why fitting is limited.
      if (cached === null) setModelsError(error);
    }
  }, []);

  const toggleDatasets = useCallback(() => {
    setDatasetsCollapsed((current) => {
      const next = !current;
      localStorage.setItem("curvestudio:datasetsCollapsed", next ? "1" : "0");
      return next;
    });
  }, []);

  /** The fits list on a mount or a selection: the cache serves it when it can, and only
   *  a cache miss (or an invalidation) reaches the API. */
  const loadFits = useCallback(async (datasetId: string): Promise<void> => {
    setFitsError(null);
    try {
      setFits(await fetchFits(datasetId));
    } catch (error) {
      // The fits list is a sidebar of history; a failure here keeps the workspace usable,
      // but it must not look like an empty history.
      setFits([]);
      setFitsError(error);
    }
  }, []);

  /** The fits list after a stored fit, fetched straight into the panel: the cache for
   *  this dataset stays invalidated, so the next mount revalidates it too. */
  const refreshFits = useCallback(async (datasetId: string): Promise<void> => {
    setFitsError(null);
    try {
      setFits(await api.request<FitSummaryOut[]>(`/datasets/${datasetId}/fits`));
    } catch (error) {
      setFits([]);
      setFitsError(error);
    }
  }, []);

  const loadStatHistory = useCallback(async (datasetId: string | null): Promise<void> => {
    const requestedFor = datasetId;
    setStatResultsError(null);
    try {
      const results = await api.statResults(datasetId);
      // A selection that changed while the list was away owns the panel now.
      if (selectedIdRef.current !== requestedFor) return;
      setStatResults(results);
    } catch (error) {
      if (selectedIdRef.current !== requestedFor) return;
      // A failed history must not look like an empty one.
      setStatResults([]);
      setStatResultsError(error);
    }
  }, []);

  useEffect(() => {
    void loadDatasets();
    void loadModels();
  }, [loadDatasets, loadModels]);

  // Statistics mode lists the selected dataset's stored results; with no dataset it lists
  // all of them, which is where the no-data tools' results live.
  useEffect(() => {
    if (mode !== "stats") return;
    void loadStatHistory(selectedId);
  }, [mode, selectedId, loadStatHistory]);

  const selectDataset = useCallback(
    async (id: string, nRows?: number): Promise<void> => {
      // The ref still holds the previous selection: React has not re-rendered yet.
      const reselecting = id === selectedIdRef.current;
      setSelectedId(id);
      onSelectionChange?.(id);
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
      setStatShown(null);
      setStatPlotIndex(0);
      setStatError(null);
      setStatResults([]);
      // Clicking the dataset already selected changes no dependency of the history effect,
      // yet the clear above emptied the list; load it again so the results stay on screen.
      if (modeRef.current === "stats" && reselecting) void loadStatHistory(id);
      poller.reset();
      // Auto-fit leads: ensure a run (or the latest stored one) right after selection —
      // but only in Auto-fit mode, so Statistics and Manual fit spend no run.
      autofit.ensure(id, modeRef.current);
      // The chart's sample size: above the measured threshold and with WebGL the chart
      // asks for the whole dataset (capped — traces.ts); otherwise exactly the old
      // request. An upload passes its own n_rows because the list (and the ref above)
      // has not re-rendered yet when the selection starts.
      const chartPoints = chartSamplePoints(
        nRows ?? datasetsRef.current.find((dataset) => dataset.id === id)?.n_rows,
      );
      const cachedDetail = getDetail(id, chartPoints);
      setDetailLoading(cachedDetail === null);
      try {
        if (cachedDetail !== null) {
          // A return visit: the cached detail renders at once — the chart asks for
          // nothing, and the fits list comes from the cache too unless invalidated.
          setDetail(cachedDetail);
          await loadFits(id);
        } else {
          // The chart's points and the fits history are independent requests: start them
          // together instead of holding the (small) list behind the (larger) detail fetch.
          const [detail] = await Promise.all([fetchDetail(id, chartPoints), loadFits(id)]);
          setDetail(detail);
        }
      } catch (error) {
        setDetailError(error);
      } finally {
        setDetailLoading(false);
      }
    },
    [autofit, loadFits, onSelectionChange, poller],
  );

  // A workspace remembers its dataset: select it once the list is in (a remembered
  // dataset that no longer exists falls back to the most recent one, so the workspace
  // is never a dead end).
  const initialSelectionApplied = useRef(false);
  useEffect(() => {
    if (initialSelectionApplied.current || datasetsLoading || datasets.length === 0) return;
    initialSelectionApplied.current = true;
    if (initialDatasetId === null) return;
    const remembered = datasets.some((dataset) => dataset.id === initialDatasetId);
    const target = remembered ? initialDatasetId : (datasets[0]?.id ?? null);
    if (target !== null) void selectDataset(target);
  }, [datasets, datasetsLoading, initialDatasetId, selectDataset]);

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
      // The cached list no longer holds the new dataset: the next mount refetches it
      // (the live list is updated in place below).
      invalidateDatasets();
      setDatasets((current) => [dataset, ...current]);
      void selectDataset(dataset.id, dataset.n_rows);
    },
    [selectDataset],
  );

  const onDeletedDataset = useCallback(
    (id: string): void => {
      // The dataset is gone: the cached list, detail and fits must not outlive it.
      invalidateDatasets();
      dropDataset(id);
      setDatasets((current) => current.filter((d) => d.id !== id));
      autofit.clear(id);
      if (selectedId === id) {
        setSelectedId(null);
        onSelectionChange?.(null);
        setDetail(null);
        setBundle(null);
        setActiveFitId(null);
        setFits([]);
        setFitsError(null);
        setSelectedRowId(null);
        setStatShown(null);
        setStatPlotIndex(0);
        setStatResults([]);
        poller.reset();
      }
    },
    [autofit, onSelectionChange, selectedId, poller],
  );

  const onDeleteFit = useCallback(
    async (fitId: string): Promise<void> => {
      try {
        await api.request(`/fits/${fitId}`, { method: "DELETE" });
        // The cached history holds the deleted fit; the next mount refetches it (the
        // live list is filtered below).
        if (selectedId !== null) invalidateFits(selectedId);
        setFits((current) => current.filter((f) => f.id !== fitId));
        if (activeFitId === fitId) {
          setActiveFitId(null);
          setBundle(null);
        }
      } catch (error) {
        setFitError(errorText(error, "the fit could not be deleted"));
      }
    },
    [activeFitId, selectedId],
  );

  const onRunStat = useCallback(
    async (tool: string, choices: Record<string, unknown>): Promise<void> => {
      const requestedFor = selectedId;
      setStatBusy(true);
      setStatError(null);
      try {
        const result = await api.runStat(tool, choices, requestedFor);
        // The user selected another dataset while the run was away: it belongs elsewhere.
        if (selectedIdRef.current !== requestedFor) return;
        setStatShown(result);
        setStatPlotIndex(0);
        await loadStatHistory(requestedFor);
      } catch (error) {
        if (selectedIdRef.current !== requestedFor) return;
        // A refused or failed run keeps the previous result on screen.
        setStatError(error);
      } finally {
        setStatBusy(false);
      }
    },
    [loadStatHistory, selectedId],
  );

  const onSelectStatResult = useCallback(async (id: string): Promise<void> => {
    const requestedFor = selectedIdRef.current;
    setStatError(null);
    try {
      const result = await api.statResult(id);
      if (selectedIdRef.current !== requestedFor) return;
      setStatShown(result);
      setStatPlotIndex(0);
    } catch (error) {
      if (selectedIdRef.current !== requestedFor) return;
      setStatError(error);
    }
  }, []);

  const onDeleteStatResult = useCallback(
    async (id: string): Promise<void> => {
      setStatError(null);
      try {
        await api.deleteStatResult(id);
        setStatResults((current) => current.filter((result) => result.id !== id));
        if (statShown?.id === id) {
          setStatShown(null);
          setStatPlotIndex(0);
        }
      } catch (error) {
        setStatError(error);
      }
    },
    [statShown?.id],
  );

  const changeMode = useCallback(
    (next: WorkspaceMode): void => {
      setMode(next);
      onModeChange?.(next);
      if (user !== null) saveMode(user.id, next);
    },
    [onModeChange, user],
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
    // curve — unless it is the row that fit belongs to, which would discard the very
    // result the user just opened. A manual fit's bundle is left untouched.
    if (bundleSourceRef.current === "opened-row" && bundleModelRef.current !== modelId) {
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
      // Opening a row is also choosing it. Without this the chart keeps the previously
      // selected row's colour and position, so an opened model is drawn in another
      // model's colour at nearly the same place — which reads as "nothing happened".
      setSelectedRowId(candidateId);
      setOpenRowBusy(candidateId);
      setOpenRowError(null);
      // The result opens in a modal, where the user is, instead of the panel below
      // the ranking — which renders under the fold.
      setOpenResult({
        modelId: candidateId,
        name:
          autoState.stored?.result.ranked.find((row) => row.model_id === candidateId)?.name ??
          candidateId,
        fitId: null,
      });
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
          const fitId = created.fit_id;
          setOpenResult((current) =>
            current?.modelId === candidateId ? { ...current, fitId } : current,
          );
          await loadBundle(fitId, "opened-row");
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
      onModeChange?.("manual");
      if (user !== null) saveMode(user.id, "manual");
    },
    [onModeChange, user],
  );

  const onRunAutofit = useCallback((): void => {
    if (selectedId !== null) autofit.run(selectedId, autofitOptions);
  }, [autofit, autofitOptions, selectedId]);

  // In Auto-fit mode the chart shows the opened row's full bundle, else the selected
  // row's stored curve, else the data alone; Manual fit keeps its own bundle. Memoised:
  // while a run polls, its progress patches re-render this page several times a second,
  // and a fresh rowCurveBundle on every one of those renders would re-copy the curve's
  // point arrays and re-feed Plotly.react with identical data each time.
  const chartBundle = useMemo(() => {
    if (mode !== "auto") return bundle;
    if (bundleSource === "opened-row" && bundle !== null) return bundle;
    const selectedCurve =
      selectedRowId !== null && autoState.stored !== null
        ? ((autoState.stored.result.curves ?? {})[selectedRowId] ?? null)
        : null;
    return selectedCurve !== null ? rowCurveBundle(selectedCurve) : null;
  }, [mode, bundleSource, bundle, selectedRowId, autoState.stored]);

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

  // The whole-workspace empty state (design item 7 of the 2026-11-23 UX fixes): an
  // account whose list has loaded, without error and without a single dataset gets the
  // three next steps instead of the everyday panels — a loading or failed list must not
  // read as an empty account.
  const emptyWorkspace = !datasetsLoading && datasetsError === null && datasets.length === 0;

  // The opened-result modal's props, from what the fit request is doing: the request
  // itself, then the fit's bundle load, then the reused tabs' rendering — and the
  // honest error or fallback when one of those does not arrive.
  const openResultProps = (() => {
    if (openResult === null) return null;
    const ready =
      openResult.fitId !== null &&
      bundle !== null &&
      activeFitId === openResult.fitId &&
      bundleSource === "opened-row";
    const waitingForFit = openResult.fitId !== null && !ready && bundleError === null;
    return {
      name: openResult.name,
      bundle: ready ? bundle : null,
      busy:
        (openRowBusy !== null && openRowBusy === openResult.modelId) || waitingForFit,
      error:
        openRowBusy === null && openResult.fitId === null
          ? openRowError
          : openResult.fitId !== null && bundleError !== null
            ? errorText(bundleError, "the fit could not be loaded")
            : null,
    };
  })();

  return (
    <div className={datasetsCollapsed ? "workspace workspace--rail-collapsed" : "workspace"}>
      <DatasetsSidebar
        ref={sidebarActions}
        datasets={datasets}
        selectedId={selectedId}
        loading={datasetsLoading}
        loadError={datasetsError}
        onSelect={(id) => void selectDataset(id)}
        onUploaded={onUploaded}
        onDeleted={onDeletedDataset}
        collapsed={datasetsCollapsed}
        onToggleCollapse={toggleDatasets}
      />

      {emptyWorkspace ? (
        <EmptyWorkspace
          onUpload={() => sidebarActions.current?.openUpload()}
          onPaste={() => sidebarActions.current?.openPaste()}
        />
      ) : (
        <>
          <div className="workspace__main">
            <ModeSwitch
              mode={mode}
              onModeChange={changeMode}
              autofitStatus={autofitStatus}
              statsStatus={statBusy ? "running…" : "idle"}
            />
            {mode !== "stats" && (
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
            )}
            {mode === "stats" ? (
              statShown !== null && statShown.result.plots.length > 0 ? (
                <section className="panel" aria-label="Statistical plots">
                  <StatsPlots
                    plots={statShown.result.plots}
                    index={statPlotIndex}
                    onSelect={setStatPlotIndex}
                  />
                </section>
              ) : (
                <p className="state state--empty">A chart appears here once a result has one</p>
              )
            ) : (
              <>
                {/* Rank colours and the top-3 overlay belong to the Auto-fit ranking; a manual
                    fit is not a ranked row, so it keeps the brand colour (rank 1). */}
                <ChartPanel
                  dataset={detail}
                  bundle={chartBundle}
                  dark={dark}
                  rankingRows={mode === "auto" ? (autoState.stored?.result.ranked ?? null) : null}
                  storedCurves={mode === "auto" ? (autoState.stored?.result.curves ?? null) : null}
                  selectedRowId={mode === "auto" ? selectedRowId : null}
                  rootRef={chartRootRef}
                />
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
              </>
            )}
            {detailLoading && <Loading label="Loading dataset…" />}
            {detailError !== null && <ErrorBox error={detailError} />}
          </div>
    
          <div className="workspace__side">
            {mode === "stats" ? (
              <StatsPanel
                dataset={detail}
                tools={
                  stats.loading || stats.error != null
                    ? null
                    : { families: stats.families, tools: stats.tools }
                }
                onRun={(tool, choices) => void onRunStat(tool, choices)}
                busy={statBusy}
                error={statError ?? stats.error}
              />
            ) : mode === "auto" ? (
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
            {mode === "stats" ? (
              <>
                <StatsHistory
                  results={statResults}
                  onSelect={(id) => void onSelectStatResult(id)}
                  onDelete={(id) => void onDeleteStatResult(id)}
                />
                {statResultsError !== null && <ErrorBox error={statResultsError} />}
                {statShown !== null ? (
                  <section className="panel" aria-label="Statistical result">
                    <StatsResult result={statShown.result} />
                  </section>
                ) : (
                  <p className="state state--empty">Run a tool, or open a stored result</p>
                )}
              </>
            ) : (
              <>
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
    
                {bundle !== null &&
                  activeFitId !== null &&
                  (mode === "manual" || bundleSource === "opened-row") && (
                    <ResultsTabs
                      fitId={activeFitId}
                      bundle={bundle}
                      fits={fits}
                      fitsError={fitsError}
                      onLoadFit={(id) => void loadBundle(id, "manual")}
                      onDeleteFit={(id) => void onDeleteFit(id)}
                      captureChart={captureChart}
                      datasetName={detail?.name ?? null}
                    />
                  )}
              </>
            )}
          </div>
        </>
      )}

      {openResultProps !== null && (
        <OpenResultModal
          {...openResultProps}
          onClose={() => setOpenResult(null)}
        />
      )}
    </div>
  );
}
