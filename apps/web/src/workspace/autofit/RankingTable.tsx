/**
 * The Auto-fit ranking table. Rows come pre-ordered from the server (the ordering is
 * the server's job — the client never re-sorts); a metric that is not defined renders
 * as "n/a", never 0 or NaN. Rows with the same fitted curve share their rank, and
 * `same_curve_as` names the others. Keyboard: arrow keys move the selection, Enter
 * opens the selected row's full results.
 */

import { useRef, type KeyboardEvent, type ReactElement } from "react";
import { CopyButton } from "../../components/CopyButton";
import { fmtEquation, fmtInt, fmtNumber } from "../../format";
import type { RankedFit } from "../../api/client";

/** Human label + semantic badge for a candidate's fit status. */
const STATUS_BADGE: Record<string, { label: string; tone: "ok" | "bad" }> = {
  converged: { label: "Converged", tone: "ok" },
  not_converged: { label: "Not converged", tone: "bad" },
};

function StatusBadge({ status }: { status: string }): ReactElement {
  const known = STATUS_BADGE[status];
  if (known !== undefined) {
    return (
      <span className={`badge badge--${known.tone}`}>{known.label}</span>
    );
  }
  return <span className="badge badge--muted">{status}</span>;
}

export interface RankingTableProps {
  rows: RankedFit[];
  selectedId: string | null;
  /** Model ids whose curve was sampled into the stored result. */
  curvesAvailable: ReadonlySet<string>;
  onSelect: (modelId: string) => void;
  onOpen: (modelId: string) => void;
  /** Shows per-row action buttons (the progress view's live table has none). */
  actions?: boolean;
  onUseManually?: (row: RankedFit) => void;
  /** Model ids that can prefill the manual panel (registered models, polynomials). */
  manuallySelectable?: ReadonlySet<string>;
  openRowBusy?: string | null;
}

export function RankingTable({
  rows,
  selectedId,
  curvesAvailable,
  onSelect,
  onOpen,
  actions = true,
  onUseManually,
  manuallySelectable,
  openRowBusy = null,
}: RankingTableProps): ReactElement {
  const rowRefs = useRef(new Map<string, HTMLTableRowElement>());

  const focusRow = (modelId: string): void => {
    rowRefs.current.get(modelId)?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTableSectionElement>): void => {
    if (rows.length === 0) return;
    const index = rows.findIndex((row) => row.model_id === selectedId);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      const next = rows[Math.min(index + 1, rows.length - 1)] ?? rows[rows.length - 1];
      if (next !== undefined) {
        onSelect(next.model_id);
        focusRow(next.model_id);
      }
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      const previous = rows[Math.max(index - 1, 0)] ?? rows[0];
      if (previous !== undefined) {
        onSelect(previous.model_id);
        focusRow(previous.model_id);
      }
    } else if (event.key === "Enter" && selectedId !== null) {
      event.preventDefault();
      onOpen(selectedId);
    }
  };

  return (
    <div className="table-wrap">
      <table className="table table--compact ranking-table" aria-label="Ranked candidates">
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Model</th>
            <th scope="col">Equation (fitted)</th>
            <th scope="col" className="num">
              R²
            </th>
            <th scope="col" className="num">
              adj. R²
            </th>
            <th scope="col" className="num">
              RMSE
            </th>
            <th scope="col" className="num">
              AICc
            </th>
            <th scope="col" className="num">
              BIC
            </th>
            <th scope="col" className="num">
              CV-RMSE
            </th>
            <th scope="col" className="num">
              k
            </th>
            <th scope="col">Status</th>
            {actions && (
              <th scope="col" className="ranking-row__actions">
                Actions
              </th>
            )}
          </tr>
        </thead>
        <tbody onKeyDown={onKeyDown}>
          {rows.map((row) => {
            const selected = row.model_id === selectedId;
            const hasCurve = curvesAvailable.has(row.model_id);
            const equation =
              row.equation_fitted === null ? null : fmtEquation(row.equation_fitted);
            return (
              <tr
                key={row.model_id}
                ref={(element) => {
                  if (element === null) rowRefs.current.delete(row.model_id);
                  else rowRefs.current.set(row.model_id, element);
                }}
                tabIndex={selected ? 0 : -1}
                aria-selected={selected}
                className={selected ? "ranking-row ranking-row--selected" : "ranking-row"}
                onClick={() => onSelect(row.model_id)}
              >
                <td>
                  <span className="num">{fmtInt(row.rank)}</span>
                  {(row.same_curve_as ?? []).length > 0 && (
                    <span className="hint ranking-row__same">
                      same curve as: {(row.same_curve_as ?? []).join(", ")}
                    </span>
                  )}
                </td>
                <td>
                  <span className="ranking-row__name" title={row.name}>
                    {row.name}
                  </span>
                  {row.category !== "" && <span className="chip">{row.category}</span>}
                  {!hasCurve && <span className="hint ranking-row__nocurve">no stored curve</span>}
                </td>
                <td className="ranking-row__equation-cell">
                  {equation !== null && equation !== "" ? (
                    <span className="ranking-row__equation-line">
                      <code className="equation ranking-row__equation">{equation}</code>
                      <CopyButton text={equation} label="equation" compact />
                    </span>
                  ) : (
                    <span className="muted">n/a</span>
                  )}
                </td>
                <td className="num">{fmtNumber(row.metrics.r2)}</td>
                <td className="num">{fmtNumber(row.metrics.adj_r2)}</td>
                <td className="num">{fmtNumber(row.metrics.rmse)}</td>
                <td className="num">{fmtNumber(row.metrics.aicc)}</td>
                <td className="num">{fmtNumber(row.metrics.bic)}</td>
                <td className="num">{fmtNumber(row.metrics.cv_rmse)}</td>
                <td className="num">{fmtInt(row.metrics.k)}</td>
                <td>
                  <StatusBadge status={row.status} />
                </td>
                {actions && (
                  <td className="ranking-row__actions">
                    <div className="btn-row">
                      <button
                        type="button"
                        className="btn btn--small"
                        aria-label="Open full results"
                        disabled={openRowBusy !== null}
                        onClick={(event) => {
                          event.stopPropagation();
                          onOpen(row.model_id);
                        }}
                      >
                        {openRowBusy === row.model_id ? "Opening…" : "Open"}
                      </button>
                      {onUseManually !== undefined &&
                        (manuallySelectable?.has(row.model_id) ?? false) && (
                          <button
                            type="button"
                            className="btn btn--small"
                            aria-label="Use this model manually"
                            onClick={(event) => {
                              event.stopPropagation();
                              onUseManually(row);
                            }}
                          >
                            Use manually
                          </button>
                        )}
                    </div>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
