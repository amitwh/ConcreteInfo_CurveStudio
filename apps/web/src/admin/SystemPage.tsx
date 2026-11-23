/**
 * How the API is running: its version, how long it has been up, the job workers and
 * their queue, the database, and how long each route's requests took.
 */

import { useQuery } from "@tanstack/react-query";
import { useState, type ReactElement } from "react";
import { DataTable, type DataTableColumn } from "../components/DataTable";
import { Empty } from "../components/States";
import { StatTile } from "../components/StatTile";
import { fmtBytes, fmtDateTime, fmtInt, fmtUptime } from "../format";
import {
  system,
  systemLatency,
  type LatencyPeriod,
  type RouteLatencyOut,
  type SystemOut,
} from "./adminApi";
import { QueryState } from "./QueryState";

const SYSTEM_KEY = ["admin", "system"] as const;
const LATENCY_KEY = ["admin", "latency"] as const;

const PERIODS: ReadonlyArray<{ value: LatencyPeriod; label: string }> = [
  { value: 1, label: "1 day" },
  { value: 7, label: "7 days" },
  { value: 30, label: "30 days" },
];

/** The figures are whole milliseconds. */
function ms(value: number): string {
  return `${fmtInt(value)} ms`;
}

const LATENCY_COLUMNS: DataTableColumn<RouteLatencyOut>[] = [
  { header: "Route", cell: (cell) => cell.row.original.route_template },
  { header: "Requests", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.count) },
  { header: "Errors", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.errors) },
  { header: "Median", meta: { num: true }, cell: (cell) => ms(cell.row.original.p50_ms) },
  { header: "95th percentile", meta: { num: true }, cell: (cell) => ms(cell.row.original.p95_ms) },
  { header: "Slowest", meta: { num: true }, cell: (cell) => ms(cell.row.original.max_ms) },
];

export function SystemPage(): ReactElement {
  const systemQuery = useQuery({ queryKey: SYSTEM_KEY, queryFn: system });
  const [days, setDays] = useState<LatencyPeriod>(1);
  const latencyQuery = useQuery({
    queryKey: [...LATENCY_KEY, days],
    queryFn: () => systemLatency(days),
  });

  return (
    <>
      <h2>System</h2>
      <QueryState label="Loading the system figures…" query={systemQuery}>
        {(figures) => <SystemTiles figures={figures} />}
      </QueryState>

      <section className="card">
        <h3>Request latency</h3>
        <fieldset className="fieldset admin-period">
          <legend>Period</legend>
          {PERIODS.map((period) => (
            <label className="check" key={period.value}>
              <input
                type="radio"
                name="latency-period"
                value={period.value}
                checked={days === period.value}
                onChange={() => setDays(period.value)}
              />
              {period.label}
            </label>
          ))}
        </fieldset>
        <QueryState label="Loading the request latency…" query={latencyQuery}>
          {(rows) => (
            <DataTable
              columns={LATENCY_COLUMNS}
              rows={rows}
              caption="Request latency by route"
              empty={<Empty>No requests were recorded in this period.</Empty>}
            />
          )}
        </QueryState>
        <p className="hint">
          Route templates only, never raw paths. A restart loses at most the last hour of figures.
        </p>
      </section>
    </>
  );
}

function SystemTiles({ figures }: { figures: SystemOut }): ReactElement {
  const own = figures.current_release === figures.version;
  const size = figures.database.size_bytes;
  const revision =
    figures.database.migration_head === ""
      ? "The schema revision is not known."
      : `Schema revision ${figures.database.migration_head}.`;
  return (
    <div className="admin-tiles">
      <StatTile
        label="API version"
        value={figures.version}
        explanation="The version of the API serving this console."
      />
      <StatTile
        label="Current release"
        value={figures.current_release}
        explanation={
          own ? "The API's own version: no release is set." : "The release the console counts as the latest."
        }
      />
      <StatTile
        label="Uptime"
        value={fmtUptime(figures.uptime_s)}
        explanation={`Since the API started on ${fmtDateTime(figures.started_at)}.`}
      />
      <StatTile
        label="Jobs"
        value={figures.jobs.running ? "Running" : "Stopped"}
        explanation={`${fmtInt(figures.jobs.workers)} ${figures.jobs.workers === 1 ? "worker" : "workers"}; ${fmtInt(figures.jobs.queue_depth)} of ${fmtInt(figures.jobs.queue_max)} queue ${figures.jobs.queue_max === 1 ? "slot" : "slots"} used.`}
      />
      <StatTile
        label="Database"
        value={size === null ? "Not reported" : fmtBytes(size)}
        explanation={`${size === null ? "The size is not reported on this database" : "Size of the database"}. ${revision}`}
      />
    </div>
  );
}