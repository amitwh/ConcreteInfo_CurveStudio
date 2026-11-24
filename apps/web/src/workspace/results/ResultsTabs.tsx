/**
 * The tabbed results panel. A non-converged or failed fit gets exactly one red banner
 * here — "Fit did not converge: <reason>" (or the matching heading) — and the parameter
 * estimates stay hidden behind a "Show anyway" toggle: the honest-UI contract.
 *
 * The strip above the tabs carries two actions: the PDF report, and Copy methods, which puts
 * the fit's methods text on the clipboard as plain text, Markdown or LaTeX.
 */

import { useCallback, useEffect, useId, useRef, useState, type ReactElement } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../../api/auth";
import {
  api,
  type FitBundle,
  type FitMethodsOut,
  type FitSummaryOut,
  type MethodsTextOut,
} from "../../api/client";
import { apiErrorText } from "../../components/States";
import { exportFitPdf } from "../../lib/report/export";
import { isConverged } from "../traces";
import { EquationTab } from "./EquationTab";
import { FitAuditTab } from "./FitAuditTab";
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

const TABS = [
  "Parameters",
  "Statistics",
  "Fit audit",
  "Residuals",
  "Equation",
  "Predict",
  "History",
] as const;
type TabName = (typeof TABS)[number];

/** A tab's button and panel share an id made from its name: "Fit audit" → "tab-fit-audit". */
function tabId(name: TabName, part: "tab" | "panel"): string {
  return `${part}-${name.toLowerCase().replace(/\s+/g, "-")}`;
}

const FAILURE_HEADING: Record<string, string> = {
  not_converged: "Fit did not converge",
  failed: "Fit failed",
  invalid_input: "The fit could not run",
};

/* ---- Copy methods ---- */

const NOT_CONVERGED_HINT = "No methods text for a fit that did not converge.";

/** The styles the server writes the methods text in, as the menu names them. */
const METHOD_STYLES: ReadonlyArray<{ id: MethodsStyle; label: string }> = [
  { id: "text", label: "Text" },
  { id: "markdown", label: "Markdown" },
  { id: "latex", label: "LaTeX" },
];
type MethodsStyle = keyof FitMethodsOut;

/** How a style is named in a sentence. */
const STYLE_NAME: Record<MethodsStyle, string> = {
  text: "plain text",
  markdown: "Markdown",
  latex: "LaTeX",
};

/** How the last copy ended. "Copied" is said only when the clipboard took the text. */
type MethodsOutcome =
  | { state: "working"; style: MethodsStyle }
  | { state: "copied"; style: MethodsStyle }
  | { state: "shown"; style: MethodsStyle; text: string }
  | { state: "failed"; reason: string };

/** The methods text of one style as one document: paragraph, equation, table, a blank line between. */
function methodsDocument(text: MethodsTextOut): string {
  return [text.paragraph, text.equation, text.table].join("\n\n");
}

/** The Copy methods button and the menu it opens: one entry per style. */
function CopyMethodsMenu({
  converged,
  busy,
  onChoose,
}: {
  converged: boolean;
  busy: boolean;
  onChoose: (style: MethodsStyle) => void;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();

  const close = useCallback((restoreFocus: boolean): void => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  // A press anywhere else closes the menu.
  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent): void => {
      if (!wrapRef.current?.contains(event.target as Node)) close(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, close]);

  return (
    <div
      className="tabs__menu"
      ref={wrapRef}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          close(true);
        }
      }}
    >
      <button
        ref={triggerRef}
        type="button"
        className="btn btn--small"
        disabled={!converged || busy}
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => (open ? close(false) : setOpen(true))}
      >
        Copy methods
      </button>
      {open && (
        <div
          className="tabs__menu-list"
          id={menuId}
          onBlur={(event) => {
            // Focus has left the entries, by Tab or any other way: the menu goes with it.
            if (!event.currentTarget.contains(event.relatedTarget)) close(false);
          }}
        >
          {METHOD_STYLES.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className="tabs__menu-item"
              onClick={() => {
                close(true);
                onChoose(id);
              }}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** What the last copy came to: said under the strip, or the document shown when the clipboard refused it. */
function MethodsOutcomeView({ outcome }: { outcome: MethodsOutcome }): ReactElement {
  const fieldRef = useRef<HTMLTextAreaElement>(null);
  const name = outcome.state === "failed" ? "" : STYLE_NAME[outcome.style];
  switch (outcome.state) {
    case "working":
      return (
        <p className="hint" role="status">
          {`Copying the ${name} methods text…`}
        </p>
      );
    case "copied":
      return (
        <p className="alert alert--success" role="status">
          {`Copied. The ${name} methods text is on the clipboard.`}
        </p>
      );
    case "failed":
      return (
        <p className="alert alert--error" role="alert">
          {outcome.reason}
        </p>
      );
    case "shown":
      return (
        <div className="methods-shown">
          <p role="status">
            {`The clipboard is not available here, so the ${name} methods text is shown below. Use Select all, then copy.`}
          </p>
          <textarea
            ref={fieldRef}
            aria-label={`Methods text as ${name}`}
            readOnly
            wrap="off"
            rows={12}
            value={outcome.text}
          />
          <button
            type="button"
            className="btn btn--small"
            onClick={() => fieldRef.current?.select()}
          >
            Select all
          </button>
        </div>
      );
  }
}

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
  const { user } = useAuth();
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
    // "Prepared by …" only when the account asked for it and has a name to print.
    const byline =
      user !== null && user.preferences.report_byline && user.name.trim() !== ""
        ? user.name
        : undefined;
    try {
      await exportFitPdf(bundle, { datasetName, chartImage, version: __APP_VERSION__, byline });
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

  const queryClient = useQueryClient();
  // Tagged with the fit it is about: what was said or shown for one fit is never left on
  // screen under another.
  const [lastCopy, setLastCopy] = useState<{ fitId: string; outcome: MethodsOutcome } | null>(
    null,
  );
  const methodsOutcome = lastCopy !== null && lastCopy.fitId === fitId ? lastCopy.outcome : null;

  async function copyMethods(style: MethodsStyle): Promise<void> {
    const say = (outcome: MethodsOutcome): void => setLastCopy({ fitId, outcome });
    say({ state: "working", style });
    let texts: FitMethodsOut;
    try {
      // A stored fit's methods text does not change: once read, a second copy, of any
      // style, comes from the cache instead of the server.
      texts = await queryClient.fetchQuery({
        queryKey: ["fit-methods", fitId],
        queryFn: () => api.fitMethods(fitId),
        staleTime: Infinity,
      });
    } catch (error) {
      say({ state: "failed", reason: apiErrorText(error, "the methods text could not be read") });
      return;
    }
    const text = methodsDocument(texts[style]);
    // The clipboard is missing in some browsers and on insecure origins, and a write can
    // still be refused. In both cases the document is shown, and that is said.
    const clipboard = navigator.clipboard as Clipboard | undefined;
    if (clipboard !== undefined) {
      try {
        await clipboard.writeText(text);
        say({ state: "copied", style });
        return;
      } catch {
        // shown below
      }
    }
    say({ state: "shown", style, text });
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
            id={tabId(name, "tab")}
            aria-selected={tab === name}
            aria-controls={tabId(name, "panel")}
            className={tab === name ? "tab tab--active" : "tab"}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
        <div className="tabs__actions">
          <button
            type="button"
            className="btn btn--small"
            disabled={exporting}
            aria-busy={exporting}
            onClick={() => void onExportPdf()}
          >
            PDF report
          </button>
          <CopyMethodsMenu
            converged={!failed}
            busy={methodsOutcome?.state === "working"}
            onChoose={(style) => void copyMethods(style)}
          />
        </div>
      </div>
      {failed && <p className="hint">{NOT_CONVERGED_HINT}</p>}
      {exportError !== null && (
        <p className="alert alert--error" role="alert">
          {exportError}
        </p>
      )}
      {methodsOutcome !== null && <MethodsOutcomeView outcome={methodsOutcome} />}
      <div
        role="tabpanel"
        id={tabId(tab, "panel")}
        aria-labelledby={tabId(tab, "tab")}
      >
        {tab === "Parameters" && <ParametersTab bundle={bundle} />}
        {tab === "Statistics" && <StatisticsTab bundle={bundle} />}
        {/* Keyed on the fit: another fit is another audit, with no job of the last one followed. */}
        {tab === "Fit audit" && <FitAuditTab key={fitId} fitId={fitId} converged={!failed} />}
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
