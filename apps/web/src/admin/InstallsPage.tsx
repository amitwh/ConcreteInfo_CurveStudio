/**
 * The desktop install figures: how many installations have been seen, how many are on the
 * latest version, and how they break down by version, operating system, architecture,
 * package and session, with the new ones per week.
 */

import { useQuery } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { DataTable, type DataTableColumn } from "../components/DataTable";
import { Empty } from "../components/States";
import { StatTile } from "../components/StatTile";
import { fmtInt, fmtPercent, fromIstDateInput } from "../format";
import { installsSummary, type InstallSummaryOut } from "./adminApi";
import { QueryState } from "./QueryState";

/** The install figures depend on nothing the page chooses. */
const INSTALLS_KEY = ["admin", "installs"] as const;

type NameCount = InstallSummaryOut["by_version"][number];
type WeekCount = InstallSummaryOut["new_per_week"][number];

export function InstallsPage(): ReactElement {
  const query = useQuery({ queryKey: INSTALLS_KEY, queryFn: installsSummary });
  return (
    <>
      <h2>Installs</h2>
      <QueryState label="Loading the install figures…" query={query}>
        {(summary) => (
          <>
            <InstallTiles installs={summary} />
            <div className="admin-grid">
              {BREAKDOWNS.map((breakdown) => (
                <section className="card" key={breakdown.title}>
                  <h3>{breakdown.title}</h3>
                  <DataTable
                    columns={breakdown.columns}
                    rows={breakdown.rows(summary)}
                    caption={breakdown.title}
                    empty={<Empty>No installation has been seen yet.</Empty>}
                  />
                </section>
              ))}
            </div>
            <section className="card">
              <h3>New installations per week</h3>
              <p className="hint">Each week starts on a Monday, by the IST calendar day an installation was first seen. The last week is the one in progress.</p>
              <DataTable
                columns={WEEK_COLUMNS}
                rows={summary.new_per_week}
                caption="New installations per week"
                empty={<Empty>No installation has been seen yet.</Empty>}
              />
            </section>
          </>
        )}
      </QueryState>
    </>
  );
}

/**
 * The install figures, each beside what it counts. The overview shows them too, which is
 * why they are a component of their own.
 */
export function InstallTiles({ installs }: { installs: InstallSummaryOut }): ReactElement {
  const version = installs.latest_release;
  const onLatest =
    installs.latest_share === 0
      ? `No installation has reported this version (${version}). The console cannot tell how many are still in use: a desktop installation reports only on its first launch and after an upgrade.`
      : `Installations whose last reported version is ${version}. The console cannot tell how many are still in use: a desktop installation reports only on its first launch and after an upgrade.`;
  return (
    <div className="admin-tiles">
      <StatTile
        label="Desktop installs"
        value={fmtInt(installs.total)}
        explanation="Distinct installations ever seen."
      />
      <StatTile
        label="New in 30 days"
        value={fmtInt(installs.new_30d)}
        explanation="Installations first seen in the last 30 days."
      />
      <StatTile
        label="Latest version"
        value={version}
        explanation="The release the console counts as the latest. It is the API's own version when no release is set."
      />
      <StatTile
        label="On the latest version"
        value={fmtPercent(installs.latest_share)}
        explanation={onLatest}
      />
    </div>
  );
}

/** The API folds a value it does not know into "other". */
function shown(name: string): string {
  return name === "other" ? "Other or unknown" : name;
}

function breakdownColumns(first: string): DataTableColumn<NameCount>[] {
  return [
    { header: first, cell: (cell) => shown(cell.row.original.name) },
    { header: "Installations", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.count) },
  ];
}

interface Breakdown {
  title: string;
  columns: DataTableColumn<NameCount>[];
  rows: (summary: InstallSummaryOut) => NameCount[];
}

const BREAKDOWNS: Breakdown[] = [
  { title: "By version", columns: breakdownColumns("Version"), rows: (summary) => summary.by_version },
  {
    title: "By operating system",
    columns: breakdownColumns("Operating system"),
    rows: (summary) => summary.by_os,
  },
  {
    title: "By architecture",
    columns: breakdownColumns("Architecture"),
    rows: (summary) => summary.by_arch,
  },
  { title: "By package", columns: breakdownColumns("Package"), rows: (summary) => summary.by_package },
  { title: "By session", columns: breakdownColumns("Session"), rows: (summary) => summary.by_session },
];

const IST_WEEKDAY = new Intl.DateTimeFormat("en-IN", {
  weekday: "long",
  timeZone: "Asia/Kolkata",
});

/** "2026-11-23" → "Monday 23/11/2026": the weekday follows from the IST day. */
function weekLabel(weekStart: string): string {
  const start = fromIstDateInput(weekStart);
  if (start === null) return weekStart;
  const [year, month, day] = weekStart.split("-");
  return `${IST_WEEKDAY.format(start)} ${day}/${month}/${year}`;
}

const WEEK_COLUMNS: DataTableColumn<WeekCount>[] = [
  { header: "Week starting", cell: (cell) => weekLabel(cell.row.original.week_start) },
  {
    header: "New installations",
    meta: { num: true },
    cell: (cell) => fmtInt(cell.row.original.count),
  },
];
