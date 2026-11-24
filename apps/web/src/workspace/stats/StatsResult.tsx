/**
 * The one view of a statistical result: the title and the level, the estimates and the
 * hypothesis tests, every assumption as a card, the warnings, every table with its note, the
 * plots, the interpretation, the methods fragment to copy, and the rows the tool left out.
 *
 * Nothing is hidden and nothing is invented: a number the tool could not give reads as an em
 * dash, a p-value is read through `pText`, and a verdict states evidence, never proof.
 */

import { Fragment, useState, type ReactElement } from "react";
import type {
  HypothesisTest,
  StatCheck,
  StatTable,
  StatTableColumn,
  TableCell,
  ToolResult,
} from "../../api/client";
import { CopyButton } from "../../components/CopyButton";
import { fmtInt, fmtNumber } from "../../format";
import { pText } from "./pText";
import { StatsPlots } from "./StatsPlots";

export interface StatsResultProps {
  result: ToolResult;
}

const DASH = "—";

/** How many of the excluded rows are listed before the rest are only counted. */
const EXCLUDED_LISTED_MAX = 200;

const VERDICT_LABEL: Record<StatCheck["verdict"], string> = {
  pass: "Pass",
  caution: "Caution",
  fail: "Fail",
  not_applicable: "Not applicable",
};

/** A finite number as the app formats it; a tool's `None` is an em dash, never a zero. */
function shownNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? fmtNumber(value) : DASH;
}

/** The confidence level as a reader says it: 0.95 is "95%". */
function confidenceText(level: number): string {
  return `${fmtNumber(level * 100, { digits: 2 })}%`;
}

/** A test's degrees of freedom; a numerator and a denominator are both shown when both exist. */
function dfText(test: HypothesisTest): string {
  const { df, df_denominator: denominator } = test;
  if (df != null && denominator != null) return `${fmtNumber(df)} / ${fmtNumber(denominator)}`;
  if (df != null) return fmtNumber(df);
  if (denominator != null) return fmtNumber(denominator);
  return DASH;
}

/**
 * One table cell as its column reads. A p-value goes through `pText` (a missing one is the
 * core's "not computed"); any other missing number is an em dash, never a zero.
 */
function cellText(cell: TableCell | undefined, column: StatTableColumn): string {
  if (column.kind === "text") return cell == null ? "" : String(cell);
  if (column.kind === "p_value") {
    if (cell == null) return pText(null);
    return typeof cell === "number" ? pText(cell) : String(cell);
  }
  if (cell == null) return DASH;
  if (typeof cell !== "number") return String(cell);
  return column.kind === "int" ? fmtInt(cell) : fmtNumber(cell);
}

/** A verdict as a pill in the audit tab's tones; the word carries the meaning. */
function VerdictPill({ verdict }: { verdict: StatCheck["verdict"] }): ReactElement {
  return <span className={`verdict verdict--${verdict}`}>{VERDICT_LABEL[verdict]}</span>;
}

/** A statistic or threshold a verdict turned on, labelled. */
function Figure({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <p className="audit-check__figure">
      <span className="audit-check__lead">{label}:</span>{" "}
      <span className="audit-check__value">{value}</span>
    </p>
  );
}

function AssumptionCard({ check }: { check: StatCheck }): ReactElement {
  const quiet = check.verdict === "not_applicable";
  return (
    <li
      className={`audit-check audit-check--${check.verdict}${quiet ? " audit-check--quiet" : ""}`}
      data-check={check.id}
    >
      <h4 className="audit-check__title">
        <VerdictPill verdict={check.verdict} />
        <span>{check.title}</span>
      </h4>
      <p className="audit-check__evidence">{check.evidence}</p>
      {check.statistic !== null && <Figure label="Statistic" value={check.statistic} />}
      {check.threshold !== null && <Figure label="Threshold" value={check.threshold} />}
      <p className="muted">{check.explanation}</p>
      {check.next_step !== null && (
        <p className="audit-check__next">
          <span className="audit-check__lead">Next step:</span> {check.next_step}
        </p>
      )}
    </li>
  );
}

function TestsTable({ tests }: { tests: HypothesisTest[] }): ReactElement {
  return (
    <section aria-label="Tests">
      <h3>Tests</h3>
      <div className="table-wrap">
        <table className="table" aria-label="Tests">
          <thead>
            <tr>
              <th scope="col">Test</th>
              <th scope="col">Method</th>
              <th scope="col" className="num">
                Statistic
              </th>
              <th scope="col" className="num">
                df
              </th>
              <th scope="col" className="num">
                p
              </th>
              <th scope="col" className="num">
                Adjusted p
              </th>
              <th scope="col">Adjustment</th>
            </tr>
          </thead>
          <tbody>
            {tests.map((test, index) => (
              <Fragment key={`${test.name}-${index}`}>
                <tr>
                  <td>{test.name}</td>
                  <td>{test.method}</td>
                  <td className="num">{shownNumber(test.statistic)}</td>
                  <td className="num">{dfText(test)}</td>
                  <td className="num">{test.p_value_text}</td>
                  <td className="num">
                    {test.adjusted_p != null ? pText(test.adjusted_p) : DASH}
                  </td>
                  <td>{test.adjustment ?? DASH}</td>
                </tr>
                <tr>
                  <td colSpan={7}>
                    <p className="hint">{`H₀: ${test.null_hypothesis}`}</p>
                    {test.reason != null && (
                      <p className="hint">{`Could not run: ${test.reason}`}</p>
                    )}
                  </td>
                </tr>
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ResultTable({ table }: { table: StatTable }): ReactElement {
  const numeric = (column: StatTableColumn): string | undefined =>
    column.kind === "text" ? undefined : "num";
  return (
    <section aria-label={table.title}>
      <h4>{table.title}</h4>
      <div className="table-wrap">
        <table className="table" aria-label={table.title}>
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column.key} scope="col" className={numeric(column)}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {table.columns.map((column, columnIndex) => (
                  <td key={column.key} className={numeric(column)}>
                    {cellText(row[columnIndex], column)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.note != null && <p className="hint">{table.note}</p>}
    </section>
  );
}

export function StatsResult({ result }: StatsResultProps): ReactElement {
  const [plotIndex, setPlotIndex] = useState(0);
  const listed = result.excluded.slice(0, EXCLUDED_LISTED_MAX);
  const remaining = result.excluded.length - listed.length + result.excluded_unlisted;

  return (
    <div>
      <h2>{result.title}</h2>
      <p className="hint">{`Confidence level: ${confidenceText(result.level)}`}</p>

      <section aria-label="Estimates">
        <h3>Estimates</h3>
        <div className="table-wrap">
          <table className="table" aria-label="Estimates">
            <thead>
              <tr>
                <th scope="col">Label</th>
                <th scope="col" className="num">
                  Value
                </th>
                <th scope="col" className="num">
                  SE
                </th>
                <th scope="col" className="num">
                  Lower
                </th>
                <th scope="col" className="num">
                  Upper
                </th>
              </tr>
            </thead>
            <tbody>
              {result.estimates.map((estimate, index) => (
                <tr key={`${estimate.label}-${index}`}>
                  <td>{estimate.label}</td>
                  <td className="num">{fmtNumber(estimate.value)}</td>
                  <td className="num">{shownNumber(estimate.se)}</td>
                  <td className="num">{shownNumber(estimate.lower)}</td>
                  <td className="num">{shownNumber(estimate.upper)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <TestsTable tests={result.tests} />

      <section aria-label="Assumptions">
        <h3>Assumptions</h3>
        <ol className="audit-checks">
          {result.assumptions.map((check) => (
            <AssumptionCard key={check.id} check={check} />
          ))}
        </ol>
      </section>

      {result.warnings.length > 0 && (
        <section aria-label="Warnings">
          <h3>Warnings</h3>
          <ul className="issues">
            {result.warnings.map((warning, index) => (
              <li key={index} className="issue issue--warning">
                {warning}
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.tables.map((table, index) => (
        <ResultTable key={`${table.title}-${index}`} table={table} />
      ))}

      {result.plots.length > 0 && (
        <section aria-label="Plots">
          <h3>Plots</h3>
          <StatsPlots plots={result.plots} index={plotIndex} onSelect={setPlotIndex} />
        </section>
      )}

      <section aria-label="Interpretation">
        <h3>Interpretation</h3>
        <p>{result.interpretation}</p>
      </section>

      <section aria-label="Methods">
        <h3>Methods</h3>
        <p>{result.methods_fragment}</p>
        <p>
          <CopyButton text={result.methods_fragment} label="methods text" />
        </p>
      </section>

      <section aria-label="Rows">
        <h3>Rows</h3>
        <p>
          {`${fmtInt(result.counts.n_offered)} offered, `}
          {`${fmtInt(result.counts.n_used)} used, `}
          {`${fmtInt(result.counts.n_excluded)} excluded.`}
        </p>
        {listed.length > 0 && (
          <details className="details">
            <summary>{`Excluded rows (${fmtInt(listed.length)})`}</summary>
            <div className="table-wrap">
              <table className="table" aria-label="Excluded rows">
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      Row
                    </th>
                    <th scope="col">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {listed.map((row) => (
                    <tr key={row.index}>
                      <td className="num">{fmtInt(row.index + 1)}</td>
                      <td>{row.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {remaining > 0 && <p className="hint">{`and ${fmtInt(remaining)} more`}</p>}
          </details>
        )}
      </section>
    </div>
  );
}
