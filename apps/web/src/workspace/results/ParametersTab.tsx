/**
 * Fitted parameters: estimate, standard error, 95% CI. The engine does not report t/p
 * values, so those columns do not exist. For a fit that did not converge the estimates
 * are hidden behind a "Show anyway" toggle — they are diagnostics, not results.
 * Copy button yields tab-separated values for pasting into a spreadsheet. The column
 * headers reorder the visible rows only (ascending → descending → the engine's order);
 * the TSV copy always carries the engine's order.
 */

import { useMemo, useState, type ReactElement } from "react";
import { CopyButton } from "../../components/CopyButton";
import type { FitBundle } from "../../api/client";
import { fmtNumber } from "../../format";
import { SortableTh, cycleSort, sortRows, type SortKey, type SortState } from "../sorting";

type Row = {
  name: string;
  value: number;
  std_err?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
};

const SORT_KEYS: Record<string, SortKey<Row>> = {
  name: (row) => row.name,
  estimate: (row) => row.value,
  std_err: (row) => row.std_err ?? null,
  ci: (row) => row.ci_low ?? null,
};

export function ParametersTab({ bundle }: { bundle: FitBundle }): ReactElement {
  const converged = bundle.result.status === "converged";
  const [showAnyway, setShowAnyway] = useState(false);
  const [sort, setSort] = useState<SortState | null>(null);

  const rows: Row[] = bundle.result.parameters.map((parameter) => ({ ...parameter }));
  const ordered = useMemo(() => sortRows(rows, sort, SORT_KEYS), [rows, sort]);

  const tsv = [
    "name\testimate\tstd_err\tci_low\tci_high",
    ...rows.map(
      (row) =>
        `${row.name}\t${row.value}\t${row.std_err ?? ""}\t${row.ci_low ?? ""}\t${row.ci_high ?? ""}`,
    ),
  ].join("\n");

  const onSort = (column: string): void => {
    setSort((current) => cycleSort(current, column));
  };

  return (
    <div className="table-wrap">
      {!converged && (
        <p className="alert alert--warning">
          Estimates from a non-converged fit — do not use.{" "}
          {!showAnyway && (
            <button type="button" className="btn btn--small" onClick={() => setShowAnyway(true)}>
              Show anyway
            </button>
          )}
        </p>
      )}
      {(converged || showAnyway) && (
        <>
          <table className="table">
            <caption className="sr-only">Fitted parameter estimates with uncertainties</caption>
            <thead>
              <tr>
                <SortableTh column="name" state={sort} onSort={onSort}>
                  Parameter
                </SortableTh>
                <SortableTh column="estimate" state={sort} onSort={onSort} className="num">
                  Estimate
                </SortableTh>
                <SortableTh column="std_err" state={sort} onSort={onSort} className="num">
                  Std. error
                </SortableTh>
                <SortableTh column="ci" state={sort} onSort={onSort} className="num">
                  95% CI
                </SortableTh>
              </tr>
            </thead>
            <tbody>
              {ordered.map((row) => (
                <tr key={row.name}>
                  <th scope="row">
                    <code>{row.name}</code>
                  </th>
                  <td className="num">{fmtNumber(row.value)}</td>
                  <td className="num">{fmtNumber(row.std_err)}</td>
                  <td className="num">
                    {row.ci_low != null && row.ci_high != null
                      ? `${fmtNumber(row.ci_low, { digits: 4 })} – ${fmtNumber(row.ci_high, { digits: 4 })}`
                      : "n/a"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            <CopyButton text={tsv} label="parameters" />
          </p>
        </>
      )}
    </div>
  );
}
