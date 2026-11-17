/**
 * Fitted parameters: estimate, standard error, 95% CI. The engine does not report t/p
 * values, so those columns do not exist. For a fit that did not converge the estimates
 * are hidden behind a "Show anyway" toggle — they are diagnostics, not results.
 * Copy button yields tab-separated values for pasting into a spreadsheet.
 */

import { useState, type ReactElement } from "react";
import { CopyButton } from "../../components/CopyButton";
import type { FitBundle } from "../../api/client";
import { fmtNumber } from "../../format";

type Row = {
  name: string;
  value: number;
  std_err?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
};

export function ParametersTab({ bundle }: { bundle: FitBundle }): ReactElement {
  const converged = bundle.result.status === "converged";
  const [showAnyway, setShowAnyway] = useState(false);

  const rows: Row[] = bundle.result.parameters.map((parameter) => ({ ...parameter }));

  const tsv = [
    "name\testimate\tstd_err\tci_low\tci_high",
    ...rows.map(
      (row) =>
        `${row.name}\t${row.value}\t${row.std_err ?? ""}\t${row.ci_low ?? ""}\t${row.ci_high ?? ""}`,
    ),
  ].join("\n");

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
                <th scope="col">Parameter</th>
                <th scope="col">Estimate</th>
                <th scope="col">Std. error</th>
                <th scope="col">95% CI</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
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
