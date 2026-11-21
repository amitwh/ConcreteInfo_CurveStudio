/**
 * Goodness-of-fit statistics, information criteria (with the convention text as a
 * tooltip), cross-validation, and the residual diagnostic tests with their assumption
 * strings. Undefined statistics render "n/a" — never 0, never NaN.
 */

import type { ReactElement } from "react";
import { ChevronDown } from "lucide-react";
import { CopyButton } from "../../components/CopyButton";
import type { FitBundle } from "../../api/client";
import { fmtInt, fmtNumber, fmtPValue } from "../../format";

function Stat({ label, value, title }: { label: string; value: string; title?: string }): ReactElement {
  return (
    <div className="stat" title={title}>
      <dt>{label}</dt>
      <dd className="num">{value}</dd>
    </div>
  );
}

function Diagnostics({ bundle }: { bundle: FitBundle }): ReactElement | null {
  const tests = bundle.summary?.diagnostics.tests ?? [];
  if (tests.length === 0) return null;
  return (
    <details className="details" open>
      <summary>
        <ChevronDown
          className="icon details__chevron"
          aria-hidden="true"
          focusable="false"
        />
        Diagnostic tests ({tests.length})
      </summary>
      <ul className="diagnostics">
        {tests.map((test) => (
          <li key={test.name} className="diagnostic">
            <p>
              <strong>{test.name}</strong>{" "}
              {test.status === "ok" ? (
                <>
                  <span className="num">stat {fmtNumber(test.statistic, { digits: 4 })}</span>
                  {test.p_value !== null && (
                    <>
                      {" · "}
                      <span className="num">p {fmtPValue(test.p_value)}</span>
                    </>
                  )}
                  {test.df !== null && (
                    <>
                      {" · "}
                      <span className="num">df {fmtNumber(test.df, { digits: 2 })}</span>
                    </>
                  )}
                </>
              ) : (
                <span className="muted"> not applicable{test.reason ? ` — ${test.reason}` : ""}</span>
              )}
            </p>
            <p className="hint">H₀: {test.null_hypothesis}</p>
            {test.assumptions.length > 0 && (
              <p className="hint">Assumes: {test.assumptions.join("; ")}</p>
            )}
            {test.limitations.length > 0 && (
              <p className="hint">Limitations: {test.limitations.join("; ")}</p>
            )}
          </li>
        ))}
      </ul>
      <p className="hint">{tests[0]?.note}</p>
    </details>
  );
}

export function StatisticsTab({ bundle }: { bundle: FitBundle }): ReactElement {
  const stats = bundle.result.stats;
  const criteria = bundle.summary?.criteria;
  const cv = bundle.summary?.cv ?? null;
  const rs = bundle.summary?.result_summary ?? {};

  const rows: Array<[string, string]> = [
    ["Model", String(rs.model_id ?? bundle.result.model_id)],
    ["Status", bundle.result.status],
    ["Method", `${bundle.result.method} · loss ${bundle.result.loss}`],
    ["n", fmtInt(stats?.n)],
    ["k (parameters)", fmtInt(stats?.k)],
    ["df (residual)", fmtInt(stats?.df)],
    ["R²", fmtNumber(stats?.r2)],
    ["Adjusted R²", fmtNumber(stats?.adj_r2)],
    ["RMSE", fmtNumber(stats?.rmse)],
    ["SSE", fmtNumber(stats?.sse)],
    ["MAE", fmtNumber(stats?.mae)],
    ["Residual SE", fmtNumber(stats?.residual_se)],
    ["AIC", fmtNumber(criteria?.aic)],
    ["AICc", fmtNumber(criteria?.aicc)],
    ["BIC", fmtNumber(criteria?.bic)],
    [
      "CV RMSE",
      cv === null ? "n/a" : fmtNumber(cv.rmse),
    ],
  ];
  const tsv = rows.map(([label, value]) => `${label}\t${value}`).join("\n");

  return (
    <div>
      {stats === null && (
        <p className="alert alert--warning">
          Statistics are not defined for this fit
          {bundle.summary_reason !== null ? `: ${bundle.summary_reason}` : "."}
        </p>
      )}
      <dl className="stats">
        <Stat label="R²" value={fmtNumber(stats?.r2)} />
        <Stat label="Adjusted R²" value={fmtNumber(stats?.adj_r2)} />
        <Stat label="RMSE" value={fmtNumber(stats?.rmse)} />
        <Stat label="SSE" value={fmtNumber(stats?.sse)} />
        <Stat label="n" value={fmtInt(stats?.n)} />
        <Stat label="k" value={fmtInt(stats?.k)} />
        <Stat label="df" value={fmtInt(stats?.df)} />
        <Stat
          label="AIC"
          value={fmtNumber(criteria?.aic)}
          title={criteria?.convention}
        />
        <Stat
          label="AICc"
          value={fmtNumber(criteria?.aicc)}
          title={
            criteria?.convention ??
            "the small-sample correction is undefined when n − k′ − 1 ≤ 0"
          }
        />
        <Stat label="BIC" value={fmtNumber(criteria?.bic)} title={criteria?.convention} />
      </dl>
      {criteria?.convention !== undefined && (
        <p className="hint">Information criteria convention: {criteria.convention}</p>
      )}
      {cv !== null && (
        <p className="hint">
          Cross-validation: {cv.folds} folds, seed {fmtInt(cv.seed)}
          {cv.rmse === null
            ? ` — pooled RMSE not defined (${cv.n_failed} fold(s) failed)`
            : `, pooled RMSE ${fmtNumber(cv.rmse)}`}
          {cv.reasons.length > 0 && `; ${cv.reasons.join("; ")}`}
        </p>
      )}
      <Diagnostics bundle={bundle} />
      {bundle.summary?.warnings.length ? (
        <ul className="issues">
          {bundle.summary.warnings.map((warning, index) => (
            <li key={index} className="issue issue--warning">
              {warning}
            </li>
          ))}
        </ul>
      ) : null}
      <p>
        <CopyButton text={tsv} label="statistics" />
      </p>
    </div>
  );
}
