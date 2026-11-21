/**
 * The Auto-fit analysis area: progress while a run is going (determinate bar, counts,
 * elapsed, the live top-5, cancel), then the ranked result — a ranking table with an
 * ordering selector that re-ranks server-side without refitting — plus the exact-fit /
 * not-converged / skipped sections, warnings, and CSV/JSON export. A cancelled or
 * time-budget-stopped run shows what was ranked with a clear partial notice.
 */

import { useEffect, useState, type ReactElement } from "react";
import { ChevronDown, Download } from "lucide-react";
import {
  api,
  saveBlob,
  type CandidateOutcome,
  type ModelInfo,
  type RankBy,
} from "../../api/client";
import { Loading } from "../../components/States";
import { fmtInt, fmtNumber } from "../../format";
import { RankingTable } from "./RankingTable";
import type { AutofitState } from "./useAutofit";

export interface AutofitPanelProps {
  datasetId: string | null;
  datasetName: string;
  state: AutofitState;
  models: ModelInfo[];
  selectedRowId: string | null;
  onSelectRow: (modelId: string) => void;
  onOpenRow: (candidateId: string) => void;
  onUseManually: (row: { modelId: string; name: string }) => void;
  onRun: () => void;
  onCancel: () => void;
  onRerank: (rankBy: RankBy) => void;
  openRowBusy: string | null;
  openRowError: string | null;
}

/** Labels of the ordering selector; "best_overall" carries its recommendation. */
export const RANK_BY_OPTIONS: Array<{ value: RankBy; label: string }> = [
  { value: "best_overall", label: "Best overall (recommended: guards against overfitting)" },
  { value: "r2", label: "R²" },
  { value: "adj_r2", label: "adj. R²" },
  { value: "rmse", label: "RMSE" },
  { value: "aicc", label: "AICc" },
  { value: "bic", label: "BIC" },
  { value: "cv_rmse", label: "CV-RMSE" },
];

/** Page sizes offered for the stored ranking table (the live table never paginates). */
const PAGE_SIZES = [10, 25, 50, 100] as const;
const DEFAULT_PAGE_SIZE = 10;

/** Clear text for the codes a start can be refused with (the server message otherwise). */
const CODE_MESSAGES: Record<string, string> = {
  TOO_FEW_POINTS:
    "Too few points for Auto-fit — the engine refuses a dataset with fewer than four.",
  UNKNOWN_FAMILY: "One of the selected families does not exist; uncheck it in Options.",
  NO_CANDIDATES: "No model in the selected families can be fitted to this dataset.",
  INVALID_CUSTOM_MODEL: "One of the custom equations cannot be fitted.",
  DUPLICATE_MODEL_ID: "A custom equation collides with a catalogue model id.",
  QUOTA_EXCEEDED: "The daily Auto-fit quota is used up — try again tomorrow.",
  RATE_LIMITED: "Too many jobs are already waiting; wait for one to finish and try again.",
};

function errorNotice(state: AutofitState): string {
  const specific = state.errorCode !== null ? CODE_MESSAGES[state.errorCode] : undefined;
  return specific ?? (state.error ?? "the run could not be shown");
}

/** id → display name, from the run's outcomes (covers skipped/unconverged too). */
function nameMap(outcomes: CandidateOutcome[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const outcome of outcomes) {
    if (outcome.name !== "") names.set(outcome.candidate_id, outcome.name);
  }
  return names;
}

const POLYNOMIAL_ID = /^poly\d+(_origin)?$/;

export function AutofitPanel({
  datasetId,
  datasetName,
  state,
  models,
  selectedRowId,
  onSelectRow,
  onOpenRow,
  onUseManually,
  onRun,
  onCancel,
  onRerank,
  openRowBusy,
  openRowError,
}: AutofitPanelProps): ReactElement {
  const [exportError, setExportError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);

  const storedId = state.stored?.id ?? null;
  const storedPartial = state.stored?.partial ?? false;
  // A different run, ordering, or partial/cancelled view starts again from page 1.
  useEffect(() => {
    setPage(0);
  }, [storedId, state.rankBy, state.cancelled, storedPartial]);

  if (datasetId === null) {
    return (
      <section className="panel" aria-label="Auto-fit">
        <h2 className="panel-title">Auto-fit</h2>
        <p className="state state--empty">
          Select or upload a dataset — Auto-fit starts by itself and ranks every model.
        </p>
      </section>
    );
  }

  const running =
    state.phase === "running" || state.phase === "starting" || state.phase === "checking";
  const stored = state.stored;
  const rows = stored !== null ? stored.result.ranked : (state.partialRows ?? []);
  const totalRows = rows.length;
  const pageCount = Math.max(1, Math.ceil(totalRows / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const pageRows = rows.slice(currentPage * pageSize, currentPage * pageSize + pageSize);
  const rangeStart = totalRows === 0 ? 0 : currentPage * pageSize + 1;
  const rangeEnd = Math.min(totalRows, (currentPage + 1) * pageSize);
  const curvesAvailable = new Set(Object.keys(stored?.result.curves ?? {}));
  // Rows that can prefill the manual panel: catalogue models and polynomials
  // (polyN / polyN_origin); custom-equation rows cannot be named by id there.
  const manuallySelectable = new Set<string>(models.map((model) => model.id));
  for (const row of rows) {
    if (POLYNOMIAL_ID.test(row.model_id)) manuallySelectable.add(row.model_id);
  }

  const onExport = (format: "csv" | "json"): void => {
    if (stored === null) return;
    setExportError(null);
    void api
      .download(`/autofits/${stored.id}/export?format=${format}`)
      .then(({ blob, filename }) => saveBlob(blob, filename ?? `autofit-${stored.id}.${format}`))
      .catch(() => setExportError("the export could not be downloaded; try again"));
  };

  return (
    <section className="panel autofit-panel" aria-label="Auto-fit">
      <div className="chart-toolbar">
        <h2 className="panel-title">
          Auto-fit
          {datasetName !== "" && (
            <>
              {" — "}
              <span className="truncate" title={datasetName}>
                {datasetName}
              </span>
            </>
          )}
        </h2>
        <div className="btn-row">
          {!running && state.phase !== "loading" && (
            <button type="button" className="btn btn--small btn--primary" onClick={onRun}>
              {stored !== null || state.cancelled ? "Run again" : "Run Auto-fit"}
            </button>
          )}
          {stored !== null && (
            <>
              <button type="button" className="btn btn--small" onClick={() => onExport("csv")}>
                <Download className="icon" aria-hidden="true" focusable="false" />
                Export CSV
              </button>
              <button type="button" className="btn btn--small" onClick={() => onExport("json")}>
                <Download className="icon" aria-hidden="true" focusable="false" />
                Export JSON
              </button>
            </>
          )}
        </div>
      </div>

      {state.phase === "error" && state.error !== null && (
        <p className="alert alert--error" role="alert">
          {errorNotice(state)}
        </p>
      )}

      {running && <ProgressView state={state} onCancel={onCancel} />}
      {state.phase === "loading" && <Loading label="Loading the stored run…" />}

      {state.phase === "ready" && state.cancelled && (
        <p className="alert alert--warning" role="status">
          Run cancelled — showing what was ranked before it stopped (partial). Nothing was
          stored for this run.
        </p>
      )}
      {state.phase === "ready" && stored !== null && stored.partial && (
        <p className="alert alert--warning" role="status">
          The run stopped at its time budget, so some candidates were never fitted — the
          ranking covers {fmtInt(stored.result.ranked.length)} of {fmtInt(stored.n_candidates)}{" "}
          candidates (partial).
        </p>
      )}

      {state.phase === "ready" && rows.length === 0 && !state.cancelled && stored !== null && (
        <p className="state state--empty">
          No model converged on this dataset. Try more families in Options, or fit manually.
        </p>
      )}

      {state.phase === "ready" && rows.length > 0 && (
        <>
          {stored?.result.overfitting_note != null && (
            <p className="alert alert--warning">{stored.result.overfitting_note}</p>
          )}
          <div className="field autofit-panel__order">
            <label htmlFor="autofit-rankby">Order by</label>
            <select
              id="autofit-rankby"
              value={state.rankBy}
              onChange={(e) => onRerank(e.target.value as RankBy)}
            >
              {RANK_BY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          {state.error !== null && (
            <p className="alert alert--error" role="alert">
              {state.error}
            </p>
          )}
          <RankingTable
            rows={pageRows}
            selectedId={selectedRowId}
            curvesAvailable={curvesAvailable}
            onSelect={onSelectRow}
            onOpen={onOpenRow}
            // A cancelled run stored nothing, so its partial rows cannot be opened.
            actions={stored !== null}
            onUseManually={(row) => onUseManually({ modelId: row.model_id, name: row.name })}
            manuallySelectable={manuallySelectable}
            openRowBusy={openRowBusy}
            positionOffset={currentPage * pageSize}
          />
          <Pagination
            page={currentPage}
            pageCount={pageCount}
            pageSize={pageSize}
            rangeStart={rangeStart}
            rangeEnd={rangeEnd}
            total={totalRows}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size);
              setPage(0);
            }}
          />
          {stored !== null && (
            <OutcomeSections outcomes={stored.result.outcomes} result={stored.result} />
          )}
        </>
      )}

      {openRowError !== null && (
        <p className="alert alert--error" role="alert">
          {openRowError}
        </p>
      )}
      {exportError !== null && (
        <p className="alert alert--error" role="alert">
          {exportError}
        </p>
      )}
    </section>
  );
}

/** Page controls for the stored ranking table: range summary, Previous/Next, page size. */
function Pagination({
  page,
  pageCount,
  pageSize,
  rangeStart,
  rangeEnd,
  total,
  onPageChange,
  onPageSizeChange,
}: {
  page: number;
  pageCount: number;
  pageSize: number;
  rangeStart: number;
  rangeEnd: number;
  total: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}): ReactElement {
  return (
    <div className="pagination" role="group" aria-label="Ranking pages">
      <span className="pagination__summary">
        {fmtInt(rangeStart)}–{fmtInt(rangeEnd)} of {fmtInt(total)}
      </span>
      <button
        type="button"
        className="btn btn--small"
        onClick={() => onPageChange(page - 1)}
        disabled={page <= 0}
      >
        Previous
      </button>
      <button
        type="button"
        className="btn btn--small"
        onClick={() => onPageChange(page + 1)}
        disabled={page >= pageCount - 1}
      >
        Next
      </button>
      <span className="pagination__size">
        <label htmlFor="ranking-page-size">Rows per page</label>
        <select
          id="ranking-page-size"
          value={pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
        >
          {PAGE_SIZES.map((size) => (
            <option key={size} value={size}>
              {size}
            </option>
          ))}
        </select>
      </span>
    </div>
  );
}

function ProgressView({
  state,
  onCancel,
}: {
  state: AutofitState;
  onCancel: () => void;
}): ReactElement {
  const progress = state.progress;
  const done = progress?.done ?? 0;
  const total = progress?.total ?? 0;
  const statusText =
    state.phase === "checking"
      ? "Looking for an existing Auto-fit…"
      : state.phase === "starting"
        ? "Starting the run…"
        : progress === null
          ? "Waiting for the first results…"
          : `Fitting ${fmtInt(done)} of ${fmtInt(total)} candidates`;

  return (
    <div className="autofit-progress">
      <progress
        // A missing value makes the bar indeterminate until the first progress arrives.
        {...(total > 0 ? { value: done, max: total } : {})}
        aria-label="Auto-fit progress"
      />
      <div className="autofit-progress__row" role="status" aria-live="polite">
        <span className="autofit-progress__text">
          {progress !== null && (
            <>
              <span className="num">
                {fmtInt(progress.done)}/{fmtInt(progress.total)}
              </span>{" "}
              candidates · converged {fmtInt(progress.converged)} · not converged{" "}
              {fmtInt(progress.failed)} · skipped {fmtInt(progress.skipped)}
              <span> · {progress.elapsed_s.toFixed(1)} s</span>
            </>
          )}
          {progress === null && statusText}
        </span>
        <button
          type="button"
          className="btn btn--small"
          onClick={onCancel}
          disabled={state.jobId === null}
          title={
            state.jobId === null
              ? "Cancel becomes available once the run has started."
              : undefined
          }
        >
          Cancel
        </button>
      </div>
      {progress !== null && progress.top.length > 0 && (
        <>
          <h3 className="autofit-progress__heading">Best so far</h3>
          <RankingTable
            rows={progress.top}
            selectedId={null}
            curvesAvailable={new Set()}
            onSelect={() => {}}
            onOpen={() => {}}
            actions={false}
          />
        </>
      )}
    </div>
  );
}

function OutcomeSections({
  outcomes,
  result,
}: {
  outcomes: CandidateOutcome[];
  result: {
    exact_fits: string[];
    unconverged: Record<string, string>;
    skipped: Record<string, string[]>;
    warnings: string[];
    elapsed_seconds?: number;
  };
}): ReactElement {
  const names = nameMap(outcomes);
  const exactFits = result.exact_fits;
  const unconverged = Object.entries(result.unconverged);
  const skipped = Object.entries(result.skipped);

  return (
    <div className="autofit-outcomes">
      {exactFits.length > 0 && (
        <section className="autofit-section" aria-label="Exact fits">
          <h3>Exact fits</h3>
          <p className="hint">
            These reproduce the data exactly (SSE = 0), so information criteria are not
            defined for them — they are listed separately from the ranking.
          </p>
          <ul className="autofit-outcomes__list">
            {exactFits.map((id) => (
              <li key={id}>
                <code>{id}</code>
                {names.get(id) !== undefined && <> — {names.get(id)}</>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {unconverged.length > 0 && (
        <details className="details">
          <summary>
            <ChevronDown
              className="icon details__chevron"
              aria-hidden="true"
              focusable="false"
            />
            Not converged ({fmtInt(unconverged.length)}) — the run took{" "}
            {fmtNumber(result.elapsed_seconds ?? 0)} s
          </summary>
          <ul className="autofit-outcomes__list">
            {unconverged.map(([id, reason]) => (
              <li key={id}>
                <strong>{names.get(id) ?? id}</strong>
                <span className="muted"> — {reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {skipped.length > 0 && (
        <details className="details">
          <summary>
            <ChevronDown
              className="icon details__chevron"
              aria-hidden="true"
              focusable="false"
            />
            Skipped ({fmtInt(skipped.length)})
          </summary>
          <ul className="autofit-outcomes__list">
            {skipped.map(([id, reasons]) => (
              <li key={id}>
                <strong>{names.get(id) ?? id}</strong>
                {reasons.map((reason) => (
                  <span key={reason} className="muted">
                    {" "}
                    — {reason}
                  </span>
                ))}
              </li>
            ))}
          </ul>
        </details>
      )}

      {result.warnings.length > 0 && (
        <section className="autofit-section" aria-label="Warnings">
          <h3>Warnings</h3>
          <ul className="autofit-outcomes__list">
            {result.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
