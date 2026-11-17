/** Earlier fits of this dataset: click to load one, delete with an inline confirm. */

import { useState, type ReactElement } from "react";
import type { FitSummaryOut } from "../../api/client";
import { fmtDate } from "../../format";

function modelText(spec: Record<string, unknown>): string {
  const model = spec.model as Record<string, unknown> | undefined;
  if (model === undefined) return "fit";
  if (model.kind === "polynomial") return `Polynomial (degree ${String(model.degree ?? "?")})`;
  if (model.kind === "registered") return String(model.id ?? "model");
  if (model.kind === "custom") return `Custom: ${String(model.equation ?? "")}`;
  return "fit";
}

export function HistoryTab({
  fits,
  activeFitId,
  onLoad,
  onDelete,
}: {
  fits: FitSummaryOut[];
  activeFitId: string;
  onLoad: (id: string) => void;
  onDelete: (id: string) => void;
}): ReactElement {
  const [confirmId, setConfirmId] = useState<string | null>(null);

  if (fits.length === 0) {
    return <p className="state state--empty">No fits recorded for this dataset yet.</p>;
  }

  return (
    <ul className="history-list">
      {fits.map((fit) => (
        <li
          key={fit.id}
          className={fit.id === activeFitId ? "history history--active" : "history"}
        >
          <button
            type="button"
            className="history__load"
            onClick={() => onLoad(fit.id)}
            title={`${modelText(fit.spec)} · ${fmtDate(fit.created_at)}`}
          >
            <span className="truncate">{modelText(fit.spec)}</span>
            <span className="history__meta">
              <span className={`badge badge--${fit.status === "converged" ? "ok" : "bad"}`}>
                {fit.status}
              </span>{" "}
              {fmtDate(fit.created_at)}
            </span>
          </button>
          {confirmId === fit.id ? (
            <span className="history__confirm">
              <button type="button" className="btn btn--small btn--danger" onClick={() => onDelete(fit.id)}>
                Delete
              </button>
              <button type="button" className="btn btn--small" onClick={() => setConfirmId(null)}>
                Keep
              </button>
            </span>
          ) : (
            <button
              type="button"
              className="btn btn--small history__delete"
              aria-label={`Delete fit ${fit.id}`}
              onClick={() => setConfirmId(fit.id)}
            >
              Delete
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}
