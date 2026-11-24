/**
 * The stored statistical results, newest first: the tool's title and its IST time, a click
 * to open one, and a delete that asks before it removes anything.
 */

import { useState, type ReactElement } from "react";
import { Trash2 } from "lucide-react";
import type { StatResultSummaryOut } from "../../api/client";
import { fmtDateTime } from "../../format";

export interface StatsHistoryProps {
  results: StatResultSummaryOut[];
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

/** `DD/MM/YYYY, HH:mm IST` from the app's own date-time formatter. */
function rowTime(iso: string): string {
  const shown = fmtDateTime(iso);
  const space = shown.indexOf(" ");
  if (space === -1) return shown;
  return `${shown.slice(0, space)},${shown.slice(space)} IST`;
}

function timeOf(iso: string): number {
  const at = new Date(iso).getTime();
  return Number.isNaN(at) ? 0 : at;
}

export function StatsHistory({
  results,
  onSelect,
  onDelete,
}: StatsHistoryProps): ReactElement | null {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  if (results.length === 0) return null;
  const ordered = [...results].sort(
    (left, right) => timeOf(right.created_at) - timeOf(left.created_at),
  );

  return (
    <section aria-label="Statistics history">
      <h2 className="panel-title">Stored results</h2>
      <ul className="history-list">
        {ordered.map((result) => (
          <li key={result.id} className="history">
            <button
              type="button"
              className="history__load"
              onClick={() => onSelect(result.id)}
              title={`${result.title} · ${rowTime(result.created_at)}`}
            >
              <span className="truncate">{result.title}</span>
              <span className="history__meta">{` — ${rowTime(result.created_at)}`}</span>
            </button>
            {confirmId === result.id ? (
              <span className="history__confirm">
                <button
                  type="button"
                  className="btn btn--small btn--danger"
                  onClick={() => {
                    setConfirmId(null);
                    onDelete(result.id);
                  }}
                >
                  <Trash2 className="icon" aria-hidden="true" focusable="false" />
                  Delete
                </button>
                <button type="button" className="btn btn--small" onClick={() => setConfirmId(null)}>
                  Cancel
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="btn btn--small"
                aria-label={`Delete result ${result.id}`}
                onClick={() => setConfirmId(result.id)}
              >
                <Trash2 className="icon" aria-hidden="true" focusable="false" />
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
