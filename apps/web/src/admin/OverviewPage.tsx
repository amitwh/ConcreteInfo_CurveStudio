/**
 * The console's front page: every figure the overview carries, each beside what it
 * counts, and the ninety-day series of sign-ups and active accounts as one chart.
 */

import { useQuery } from "@tanstack/react-query";
import { useMemo, type ReactElement } from "react";
import type { PlotlyLayout, PlotlyTrace } from "plotly.js-dist-min";
import { Empty } from "../components/States";
import { PlotChart } from "../components/PlotChart";
import { StatTile } from "../components/StatTile";
import { useTheme } from "../components/ThemeToggle";
import { fmtDate, fmtInt } from "../format";
import { themeLayout } from "../workspace/traces";
import { overview, type OverviewOut } from "./adminApi";
import { InstallTiles } from "./InstallsPage";
import { QueryState } from "./QueryState";

/** The overview depends on nothing the page chooses. */
const OVERVIEW_KEY = ["admin", "overview"] as const;

export function OverviewPage(): ReactElement {
  const query = useQuery({ queryKey: OVERVIEW_KEY, queryFn: overview });
  return (
    <>
      <h2>Overview</h2>
      <QueryState label="Loading the overview…" query={query}>
        {(data) => <Overview data={data} />}
      </QueryState>
    </>
  );
}

function Overview({ data }: { data: OverviewOut }): ReactElement {
  const { theme } = useTheme();
  const { accounts, active_users, installs, open_errors, autofit_runs_24h, series } = data;
  const chart = useMemo(() => seriesChart(series, theme === "dark"), [series, theme]);

  return (
    <>
      <div className="admin-tiles">
        <StatTile
          label="Accounts"
          value={fmtInt(accounts.total)}
          explanation="Registered accounts."
        />
        <StatTile
          label="Verified"
          value={fmtInt(accounts.verified)}
          explanation="Accounts whose e-mail address was confirmed."
        />
        <StatTile
          label="New in 30 days"
          value={fmtInt(accounts.new_30d)}
          explanation="Accounts created in the last 30 days."
        />
        <StatTile
          label="Active users"
          value={
            <span className="stat-tile__figures">
              <span className="stat-tile__figure">
                <b>{fmtInt(active_users.d1)}</b>
                <small>1 day</small>
              </span>
              <span className="stat-tile__figure">
                <b>{fmtInt(active_users.d7)}</b>
                <small>7 days</small>
              </span>
              <span className="stat-tile__figure">
                <b>{fmtInt(active_users.d30)}</b>
                <small>30 days</small>
              </span>
            </span>
          }
          explanation="Accounts with at least one signed-in request in the last 1, 7 and 30 days (today counts)."
        />
      </div>

      <InstallTiles installs={installs} />

      <div className="admin-tiles">
        <StatTile
          label="Open errors"
          value={fmtInt(open_errors)}
          explanation="Error events not yet acknowledged, resolved or ignored."
        />
        <StatTile
          label="Auto-fit runs in 24 hours"
          value={fmtInt(autofit_runs_24h)}
          explanation="Auto-fit runs started in the last 24 hours, all accounts together."
        />
      </div>

      <section className="card">
        <h3>Sign-ups and active accounts per day</h3>
        {series.length === 0 ? (
          <Empty>There is nothing to chart yet.</Empty>
        ) : (
          <>
            <p className="hint">{seriesSummary(series)}</p>
            <PlotChart
              traces={chart.traces}
              layout={chart.layout}
              ariaLabel="Line chart of accounts created per day and accounts active per day"
              dark={theme === "dark"}
            />
          </>
        )}
      </section>
    </>
  );
}

/** The series in words, for anyone who cannot read the chart. */
function seriesSummary(series: OverviewOut["series"]): string {
  const first = series[0];
  const last = series.at(-1);
  if (first === undefined || last === undefined) return "";
  const signups = series.reduce((sum, point) => sum + point.signups, 0);
  const lastDay = fmtDate(last.day);
  return (
    `Sign-ups and active accounts per day, for the ${fmtInt(series.length)} days from ` +
    `${fmtDate(first.day)} to ${lastDay}. ${fmtInt(signups)} accounts were created in that time; ` +
    `${fmtInt(last.active)} were active on ${lastDay}. ` +
    "An account is active on a day when it made at least one signed-in request that day."
  );
}

/** Two lines over the IST days: accounts created, accounts active. */
function seriesChart(
  series: OverviewOut["series"],
  dark: boolean,
): { traces: PlotlyTrace[]; layout: Partial<PlotlyLayout> } {
  const days = series.map((point) => point.day);
  const theme = themeLayout(dark);
  return {
    traces: [
      {
        type: "scatter",
        mode: "lines",
        name: "Sign-ups",
        x: days,
        y: series.map((point) => point.signups),
        line: { color: "#e5461f", width: 2 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Active accounts",
        x: days,
        y: series.map((point) => point.active),
        line: { color: "#1a7a56", width: 2 },
      },
    ],
    layout: {
      ...theme,
      margin: { t: 40, r: 8, b: 48, l: 48 },
      // A date-only string is the same calendar day in UTC and IST, so the ticks read true.
      xaxis: { ...theme.xaxis, type: "date", tickformat: "%d/%m/%Y" },
      yaxis: { ...theme.yaxis, title: { text: "Accounts" }, rangemode: "tozero" },
      legend: { orientation: "h", y: 1.12, yanchor: "bottom" },
      hovermode: "x unified",
    },
  };
}
