/**
 * The workspace: datasets sidebar | chart | fit panel + results. Owns the cross-panel
 * state — which dataset is selected, the model catalogue, the active fit bundle, the
 * job poller — and wires the flows between them.
 */

import { useCallback, useEffect, useState, type ReactElement } from "react";
import {
  api,
  type DatasetDetailOut,
  type DatasetOut,
  type FitBundle,
  type FitSummaryOut,
  type ModelInfo,
} from "../api/client";
import { ErrorBox, Loading } from "../components/States";
import { errorText } from "../components/States";
import { useTheme } from "../components/ThemeToggle";
import { useJobPolling } from "../hooks/useJobPolling";
import { ChartPanel } from "../workspace/ChartPanel";
import { DatasetsSidebar } from "../workspace/DatasetsSidebar";
import { FitPanel } from "../workspace/FitPanel";
import { JobStatus } from "../workspace/JobStatus";
import { ResultsTabs } from "../workspace/results/ResultsTabs";

/** The chart requests the full 20 000-point sample the API allows. */
const CHART_POINTS = 20000;

export function WorkspacePage(): ReactElement {
  const { theme } = useTheme();
  const dark = theme === "dark";

  const [datasets, setDatasets] = useState<DatasetOut[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<unknown>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DatasetDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<unknown>(null);

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<unknown>(null);

  const [fits, setFits] = useState<FitSummaryOut[]>([]);
  const [activeFitId, setActiveFitId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<FitBundle | null>(null);
  const [bundleError, setBundleError] = useState<unknown>(null);
  const [fitError, setFitError] = useState<string | null>(null);

  const loadBundle = useCallback(async (fitId: string): Promise<void> => {
    setBundleError(null);
    setBundle(null);
    try {
      setBundle(await api.request<FitBundle>(`/fits/${fitId}`));
      setActiveFitId(fitId);
    } catch (error) {
      setBundleError(error);
    }
  }, []);

  const poller = useJobPolling((fitId) => {
    setFitError(null);
    void loadBundle(fitId);
    if (selectedId !== null) void loadFits(selectedId);
  });

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
    try {
      setFits(await api.request<FitSummaryOut[]>(`/datasets/${datasetId}/fits`));
    } catch {
      // The fits list is a sidebar of history; a failure here keeps the workspace usable.
      setFits([]);
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
      setFitError(null);
      poller.reset();
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
    [loadFits, poller],
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
      if (selectedId === id) {
        setSelectedId(null);
        setDetail(null);
        setBundle(null);
        setActiveFitId(null);
        setFits([]);
        poller.reset();
      }
    },
    [selectedId, poller],
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
        <ChartPanel dataset={detail} bundle={bundle} dark={dark} />
        {detailLoading && <Loading label="Loading dataset…" />}
        {detailError !== null && <ErrorBox error={detailError} />}
        {detail === null && !detailLoading && detailError === null && (
          <p className="state state--empty">
            Upload a dataset and pick a model to fit. Everything runs on the same engine
            as the desktop app.
          </p>
        )}
        {modelsError !== null && (
          <p className="alert alert--warning">
            The model catalogue could not be loaded ({errorText(modelsError)}). Fitting
            stays available for polynomials and custom equations.
          </p>
        )}
        {bundleError !== null && <ErrorBox error={bundleError} />}
        {bundle !== null && activeFitId !== null && (
          <ResultsTabs
            fitId={activeFitId}
            bundle={bundle}
            fits={fits}
            onLoadFit={(id) => void loadBundle(id)}
            onDeleteFit={(id) => void onDeleteFit(id)}
          />
        )}
      </div>

      <div className="workspace__side">
        <FitPanel
          dataset={detail}
          models={models}
          busy={poller.state.phase === "polling"}
          onFit={(spec) => void onFit(spec)}
          fitError={fitError}
        />
      </div>
    </div>
  );
}
