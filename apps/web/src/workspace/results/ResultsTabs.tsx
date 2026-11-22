/**
 * The tabbed results panel. A non-converged or failed fit gets exactly one red banner
 * here — "Fit did not converge: <reason>" (or the matching heading) — and the parameter
 * estimates stay hidden behind a "Show anyway" toggle: the honest-UI contract.
 */

import { useState, type ReactElement } from "react";
import type { FitBundle, FitSummaryOut } from "../../api/client";
import { exportFitPdf } from "../../lib/report/export";
import { isConverged } from "../traces";
import { EquationTab } from "./EquationTab";
import { HistoryTab } from "./HistoryTab";
import { ParametersTab } from "./ParametersTab";
import { PredictTab } from "./PredictTab";
import { ResidualsTab } from "./ResidualsTab";
import { StatisticsTab } from "./StatisticsTab";

export interface ResultsTabsProps {
  fitId: string;
  bundle: FitBundle;
  fits: FitSummaryOut[];
  /** Why the fit history could not be loaded (HistoryTab shows this instead). */
  fitsError?: unknown;
  onLoadFit: (id: string) => void;
  onDeleteFit: (id: string) => void;
  /** Capture the live chart as a PNG data URL; null (or a failure) omits it. */
  captureChart?: () => Promise<string | null>;
  /** Dataset name shown in the report. */
  datasetName?: string | null;
}

const TABS = ["Parameters", "Statistics", "Residuals", "Equation", "Predict", "History"] as const;
type TabName = (typeof TABS)[number];

const FAILURE_HEADING: Record<string, string> = {
  not_converged: "Fit did not converge",
  failed: "Fit failed",
  invalid_input: "The fit could not run",
};

export function ResultsTabs({
  fitId,
  bundle,
  fits,
  fitsError = null,
  onLoadFit,
  onDeleteFit,
  captureChart,
  datasetName = null,
}: ResultsTabsProps): ReactElement {
  const [tab, setTab] = useState<TabName>("Parameters");
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const status = bundle.result.status;
  const failed = !isConverged(bundle);
  const heading = FAILURE_HEADING[status] ?? `The fit ${status}`;

  async function onExportPdf(): Promise<void> {
    setExporting(true);
    setExportError(null);
    let chartImage: string | null = null;
    if (captureChart !== undefined) {
      // A failed capture must not lose the report: it is generated without the chart.
      try {
        chartImage = await captureChart();
      } catch {
        chartImage = null;
      }
    }
    try {
      await exportFitPdf(bundle, { datasetName, chartImage, version: __APP_VERSION__ });
    } catch (error) {
      setExportError(
        error instanceof Error && error.message !== ""
          ? error.message
          : "The PDF report could not be generated.",
      );
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className="panel results" aria-label="Fit results">
      {failed && (
        <p className="alert alert--error" role="alert">
          {heading}: {bundle.result.termination_reason}.
        </p>
      )}
      <div className="tabs" role="tablist" aria-label="Result views">
        {TABS.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            id={`tab-${name.toLowerCase()}`}
            aria-selected={tab === name}
            aria-controls={`panel-${name.toLowerCase()}`}
            className={tab === name ? "tab tab--active" : "tab"}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
        <button
          type="button"
          className="btn btn--small tabs__export"
          disabled={exporting}
          aria-busy={exporting}
          onClick={() => void onExportPdf()}
        >
          PDF report
        </button>
      </div>
      {exportError !== null && (
        <p className="alert alert--error" role="alert">
          {exportError}
        </p>
      )}
      <div
        role="tabpanel"
        id={`panel-${tab.toLowerCase()}`}
        aria-labelledby={`tab-${tab.toLowerCase()}`}
      >
        {tab === "Parameters" && <ParametersTab bundle={bundle} />}
        {tab === "Statistics" && <StatisticsTab bundle={bundle} />}
        {tab === "Residuals" && <ResidualsTab bundle={bundle} />}
        {tab === "Equation" && <EquationTab bundle={bundle} />}
        {tab === "Predict" && <PredictTab fitId={fitId} bundle={bundle} />}
        {tab === "History" && (
          <HistoryTab
            fits={fits}
            activeFitId={fitId}
            onLoad={onLoadFit}
            onDelete={onDeleteFit}
            loadError={fitsError}
          />
        )}
      </div>
    </section>
  );
}
