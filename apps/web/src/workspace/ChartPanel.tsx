/**
 * The centre column: the Plotly chart with its toggles. Band and log toggles are
 * disabled when they cannot apply (no band in the bundle / non-positive data), and a
 * non-converged fit contributes no curve — the toggles then show why.
 */

import { useMemo, useRef, useState, type ReactElement } from "react";
import { PlotChart, downloadPlotPng } from "../components/PlotChart";
import type { DatasetDetailOut, FitBundle } from "../api/client";
import { buildTraces, isConverged, logAxesAllowed, mergeLayout, themeLayout, type ChartToggles } from "./traces";

export interface ChartPanelProps {
  dataset: DatasetDetailOut | null;
  bundle: FitBundle | null;
  dark: boolean;
}

export function ChartPanel({ dataset, bundle, dark }: ChartPanelProps): ReactElement {
  const [toggles, setToggles] = useState<ChartToggles>({
    confidence: true,
    prediction: false,
    logX: false,
    logY: false,
  });
  const chartRootRef = useRef<HTMLDivElement>(null);

  const points = dataset?.points ?? null;
  const logAllowed = useMemo(() => logAxesAllowed(points, bundle), [points, bundle]);
  const converged = isConverged(bundle);
  const hasConfidence = converged && bundle?.confidence_band != null;
  const hasPrediction = converged && bundle?.prediction_band != null;

  const { traces, layout } = useMemo(
    () =>
      buildTraces(points, bundle, toggles, {
        x: dataset?.x_column ?? "x",
        y: dataset?.y_column ?? "y",
      }),
    [points, bundle, toggles, dataset?.x_column, dataset?.y_column],
  );

  const mergedLayout = useMemo(
    () => mergeLayout(themeLayout(dark), layout),
    [dark, layout],
  );

  return (
    <section className="panel chart-panel" aria-label="Chart">
      <div className="chart-toolbar">
        <h2 className="panel-title">
          {dataset !== null ? (
            <>
              <span className="truncate" title={dataset.name}>
                {dataset.name}
              </span>{" "}
              <span className="muted">
                ({dataset.x_column} → {dataset.y_column})
              </span>
            </>
          ) : (
            "Chart"
          )}
        </h2>
        <div className="chart-toolbar__controls" role="group" aria-label="Chart options">
          <label className="check">
            <input
              type="checkbox"
              checked={toggles.confidence}
              disabled={!hasConfidence}
              onChange={(e) => setToggles((t) => ({ ...t, confidence: e.target.checked }))}
              aria-label="Show 95% confidence band"
            />
            Confidence band
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={toggles.prediction}
              disabled={!hasPrediction}
              onChange={(e) => setToggles((t) => ({ ...t, prediction: e.target.checked }))}
              aria-label="Show 95% prediction band"
            />
            Prediction band
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={toggles.logX}
              disabled={!logAllowed.x}
              onChange={(e) => setToggles((t) => ({ ...t, logX: e.target.checked }))}
              aria-label="Logarithmic x axis"
            />
            log x
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={toggles.logY}
              disabled={!logAllowed.y}
              onChange={(e) => setToggles((t) => ({ ...t, logY: e.target.checked }))}
              aria-label="Logarithmic y axis"
            />
            log y
          </label>
          <button
            type="button"
            className="btn btn--small"
            disabled={points === null}
            onClick={() => downloadPlotPng(chartRootRef.current, dataset?.name ?? "chart")}
          >
            Download PNG
          </button>
        </div>
      </div>

      {converged && bundle?.curve === null && (
        <p className="alert alert--warning">
          The fit converged but no curve could be sampled{bundle?.curve_reason ? `: ${bundle.curve_reason}` : "."}
        </p>
      )}

      {points !== null ? (
        <PlotChart
          traces={traces}
          layout={mergedLayout}
          dark={dark}
          outerRef={chartRootRef}
          ariaLabel={`Scatter plot of ${dataset?.x_column ?? "x"} against ${dataset?.y_column ?? "y"}${converged ? " with fitted curve" : ""}`}
        />
      ) : (
        <p className="state state--empty">Select a dataset to see its points.</p>
      )}
    </section>
  );
}
