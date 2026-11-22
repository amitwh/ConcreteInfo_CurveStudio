/**
 * The centre column: the Plotly chart with its toggles. Band and log toggles are
 * disabled when they cannot apply (no band in the bundle / non-positive data), and a
 * non-converged fit contributes no curve — the toggles then show why.
 */

import { useMemo, useRef, useState, type ReactElement } from "react";
import { Database, Download } from "lucide-react";
import { PlotChart, downloadPlotPng } from "../components/PlotChart";
import type { DatasetDetailOut, FitBundle, RankedFit, RowCurve } from "../api/client";
import {
  buildTraces,
  equationLayout,
  isConverged,
  logAxesAllowed,
  mergeLayout,
  selectedEquation,
  themeLayout,
  type ChartToggles,
  type OverlayCurve,
} from "./traces";

export interface ChartPanelProps {
  dataset: DatasetDetailOut | null;
  bundle: FitBundle | null;
  dark: boolean;
  /** The stored ranking under the current ordering (for the "Show top 3" overlay). */
  rankingRows?: readonly RankedFit[] | null;
  /** The stored run's sampled curves, keyed by row id. */
  storedCurves?: Readonly<Record<string, RowCurve>> | null;
  /** The ranking row the chart follows; its rank colours the single curve. */
  selectedRowId?: string | null;
}

export function ChartPanel({
  dataset,
  bundle,
  dark,
  rankingRows = null,
  storedCurves = null,
  selectedRowId = null,
}: ChartPanelProps): ReactElement {
  const [toggles, setToggles] = useState<ChartToggles>({
    confidence: true,
    prediction: false,
    logX: false,
    logY: false,
    top3: false,
  });
  const chartRootRef = useRef<HTMLDivElement>(null);

  const points = dataset?.points ?? null;
  const logAllowed = useMemo(() => logAxesAllowed(points, bundle), [points, bundle]);
  const converged = isConverged(bundle);
  const hasConfidence = converged && bundle?.confidence_band != null;
  const hasPrediction = converged && bundle?.prediction_band != null;

  const rows = useMemo(() => rankingRows ?? [], [rankingRows]);
  const curves = useMemo<Readonly<Record<string, RowCurve>>>(
    () => storedCurves ?? {},
    [storedCurves],
  );
  // Colour by 1-based display position, not the tied rank value: ties are common and
  // must stay distinguishable.
  const selectedIndex = rows.findIndex((row) => row.model_id === selectedRowId);
  const selectedPosition = selectedIndex >= 0 ? selectedIndex + 1 : 1;
  // The three best-ranked rows that actually have a stored curve; the toggle is only
  // meaningful when at least two can be overlaid.
  const overlay = useMemo<OverlayCurve[]>(() => {
    const entries: OverlayCurve[] = [];
    rows.slice(0, 3).forEach((row, index) => {
      const curve = curves[row.model_id];
      if (curve !== undefined) {
        entries.push({
          name: row.name,
          position: index + 1,
          curve: { x: curve.x, y: curve.y },
        });
      }
    });
    return entries;
  }, [rows, curves]);
  const hasOverlay = overlay.length >= 2;

  const { traces, layout } = useMemo(
    () =>
      buildTraces(
        points,
        bundle,
        toggles,
        {
          x: dataset?.x_column ?? "x",
          y: dataset?.y_column ?? "y",
        },
        { dark, selectedPosition, overlay },
      ),
    [points, bundle, toggles, dataset?.x_column, dataset?.y_column, dark, selectedPosition, overlay],
  );

  // The selected model's fitted equation is captioned on the chart itself: it is the
  // thing the curve *is*, and reading it should not need a trip to the ranking's details.
  const equation = useMemo(
    () => selectedEquation(rows, selectedRowId, bundle),
    [rows, selectedRowId, bundle],
  );

  const mergedLayout = useMemo(
    () => mergeLayout(themeLayout(dark), layout, equationLayout(equation, dark)),
    [dark, layout, equation],
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
              checked={toggles.top3}
              disabled={!hasOverlay}
              onChange={(e) => setToggles((t) => ({ ...t, top3: e.target.checked }))}
              aria-label="Show the three best-ranked models"
            />
            Show top 3
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
            <Download className="icon" aria-hidden="true" focusable="false" />
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
        <p className="state state--empty">
          <Database className="icon icon--lg" aria-hidden="true" focusable="false" />
          Select a dataset to see its points.
        </p>
      )}
    </section>
  );
}
