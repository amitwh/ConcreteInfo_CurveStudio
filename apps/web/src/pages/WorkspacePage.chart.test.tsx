/**
 * The workspace chart's replot discipline, end to end through the real
 * PlotChart → ChartPanel chain (only the plotly bundle is mocked, the pattern of
 * PlotChart.test.tsx): `Plotly.react` must fire only when the chart's data or visual
 * inputs change. An Auto-fit poll tick patches progress state several times a second;
 * those re-renders must not re-feed the chart with identical data. A genuine data
 * change — another ranking row selected — must still replot. Also here: the dataset
 * detail and the fits history are independent requests and go out together.
 */

import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { AuthProvider } from "../api/auth";
import { api } from "../api/client";
import type { components } from "../api/schema";
import { ThemeProvider } from "../components/ThemeToggle";
import { clear as clearWorkspaceCache } from "../workspace/workspaceCache";
import { WorkspacePage } from "./WorkspacePage";

const plotly = vi.hoisted(() => ({
  react: vi.fn((..._args: unknown[]) => Promise.resolve()),
  purge: vi.fn(),
  resize: vi.fn(),
}));

vi.mock("../lib/plotly-bundle", () => ({
  default: {
    react: plotly.react,
    purge: plotly.purge,
    Plots: { resize: plotly.resize },
    downloadImage: vi.fn(),
  },
}));

type DatasetOut = components["schemas"]["DatasetOut"];
type DatasetDetailOut = components["schemas"]["DatasetDetailOut"];
type AutofitSummaryOut = components["schemas"]["AutofitSummaryOut"];
type AutofitOut = components["schemas"]["AutofitOut"];
type RankedFit = components["schemas"]["RankedFit"];
type RowCurve = components["schemas"]["RowCurve"];
type JobOut = components["schemas"]["JobOut"];

/** The page under its providers: theme, auth, and the query client the app has at its root. */
function renderPage() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ThemeProvider>
        <AuthProvider>
          <MemoryRouter>
            <WorkspacePage />
          </MemoryRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

const DATASET: DatasetOut = {
  id: "ds-1",
  name: "growth.csv",
  n_rows: 2,
  columns: ["x", "y"],
  x_column: "x",
  y_column: "y",
  weight_column: null,
  issues: [],
  created_at: "2026-11-17T05:00:00Z",
};

const DETAIL: DatasetDetailOut = {
  ...DATASET,
  points: { n_total: 2, x: [1, 2], y: [2, 4], w: null },
  table_stored: true,
  table_columns: [],
};

const SUMMARY: AutofitSummaryOut = {
  id: "af-1",
  dataset_id: "ds-1",
  created_at: "2026-11-17T09:00:00Z",
  best_model_id: "exp_decay",
  n_candidates: 2,
  rank_by: "best_overall",
};

const CURVE_A: RowCurve = {
  x: [0, 1, 2],
  y: [1, 2, 3],
  confidence_lower: null,
  confidence_upper: null,
  prediction_lower: null,
  prediction_upper: null,
  notes: [],
};

const CURVE_B: RowCurve = {
  x: [0, 1, 2],
  y: [4, 5, 6],
  confidence_lower: null,
  confidence_upper: null,
  prediction_lower: null,
  prediction_upper: null,
  notes: [],
};

function rankedRow(rank: number, modelId: string, name: string): RankedFit {
  return {
    rank,
    model_id: modelId,
    name,
    category: "Exponential",
    equation_fitted: `y = ${rank} * exp(-0.3 * x)`,
    metrics: { k: 2, n: 40, r2: 0.99, adj_r2: 0.98, rmse: 0.05, aicc: -120, bic: -117, cv_rmse: 0.06 },
    status: "converged",
  } as RankedFit;
}

const STORED: AutofitOut = {
  id: "af-1",
  dataset_id: "ds-1",
  created_at: "2026-11-17T09:00:00Z",
  best_model_id: "exp_decay",
  n_candidates: 2,
  rank_by: "best_overall",
  partial: false,
  result: {
    config: { rank_by: "best_overall" } as unknown as AutofitOut["result"]["config"],
    dataset_fingerprint: "fp",
    n_active: 40,
    ranked: [rankedRow(1, "exp_decay", "Exponential decay"), rankedRow(2, "quadratic", "Quadratic")],
    outcomes: [],
    skipped: {},
    unconverged: {},
    exact_fits: [],
    warnings: [],
    cancelled: false,
    elapsed_seconds: 1.2,
    curves: { exp_decay: CURVE_A, quadratic: CURVE_B },
    overfitting_note: null,
  },
} as unknown as AutofitOut;

function runningJob(done: number): JobOut {
  return {
    id: "job-9",
    kind: "autofit",
    status: "running",
    dataset_id: "ds-1",
    spec: {},
    result_ref: null,
    error_code: null,
    error_message: null,
    created_at: "2026-11-17T10:00:00Z",
    started_at: "2026-11-17T10:00:01Z",
    finished_at: null,
    progress: {
      done,
      total: 12,
      converged: done,
      failed: 0,
      skipped: 0,
      elapsed_s: 0.9,
      partial: false,
      top: [],
    },
  } as unknown as JobOut;
}

/** Every route the page hits on selection and while re-running Auto-fit. */
function route(path: string, options?: RequestInit): Promise<unknown> {
  if (path === "/datasets") return Promise.resolve([DATASET]);
  if (path === "/models") return Promise.resolve([]);
  if (path === "/datasets/ds-1?points=20000") return Promise.resolve(DETAIL);
  if (path === "/datasets/ds-1/fits") return Promise.resolve([]);
  if (path === "/datasets/ds-1/autofits") return Promise.resolve([SUMMARY]);
  if (path === "/autofits/af-1") return Promise.resolve(STORED);
  if (path === "/datasets/ds-1/autofit" && options?.method === "POST") {
    return Promise.resolve({ job_id: "job-9" });
  }
  return Promise.reject(new Error(`unexpected request: ${path}`));
}

let request: MockInstance;
/** How many times the re-run's job has been polled, so its progress can advance. */
let pollCount = 0;

beforeEach(() => {
  pollCount = 0;
  // The workspace's fetch cache is module-level: a test must not meet another's data.
  clearWorkspaceCache();
  request = vi.spyOn(api, "request").mockImplementation(
    (async (path: string, options?: RequestInit) => {
      const target = String(path);
      if (target === "/jobs/job-9") {
        pollCount += 1;
        return runningJob(pollCount === 1 ? 3 : 5);
      }
      return route(target, options);
    }) as unknown as typeof api.request,
  );
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

/** Let the mocked promises resolve and React apply the resulting state. */
async function flush(): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

/** Select the dataset and wait until the stored run, its row selection and the chart settle. */
async function selectDataset(): Promise<void> {
  renderPage();
  await flush();
  await flush();
  fireEvent.click(screen.getByTitle("growth.csv"));
  await flush();
  await flush();
  await flush();
  expect(screen.getByText("Exponential decay")).toBeTruthy();
}

/** The number of `Plotly.react` calls so far. */
function reactCalls(): number {
  return plotly.react.mock.calls.length;
}

/** The y values of the chart's fitted-curve trace in the last `Plotly.react` call. */
function lastFittedCurveY(): unknown {
  const traces = (plotly.react.mock.calls.at(-1)?.[1] ?? []) as Array<{
    name?: string;
    y?: number[];
  }>;
  return traces.find((trace) => trace.name === "Fitted curve")?.y;
}

describe("WorkspacePage chart replot discipline", () => {
  it("does not re-feed Plotly when an Auto-fit progress tick re-renders the workspace", async () => {
    await selectDataset();
    const callsBefore = reactCalls();
    // The chart was actually fed during the settle, so the assertion below cannot pass vacuously.
    expect(callsBefore).toBeGreaterThanOrEqual(1);

    // Re-run Auto-fit ("Run again"): the stored run stays on screen while the new job polls.
    fireEvent.click(screen.getByRole("button", { name: "Run again" }));
    await flush();
    await flush();
    expect(request).toHaveBeenCalledWith("/datasets/ds-1/autofit", expect.anything());

    // Two poll ticks, each patching progress state and re-rendering the workspace.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(request).toHaveBeenCalledWith("/jobs/job-9");
    // The patch really rendered: the progress text the panel shows moved.
    expect(screen.getByText("3/12")).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });
    expect(request).toHaveBeenCalledWith("/jobs/job-9");
    expect(screen.getByText("5/12")).toBeTruthy();
    expect(request.mock.calls.filter(([path]) => String(path) === "/jobs/job-9")).toHaveLength(2);

    // The data behind the chart is unchanged, so not one further `Plotly.react` may fire.
    expect(reactCalls()).toBe(callsBefore);
  });

  it("still replots when the selected ranking row — the chart's data — changes", async () => {
    const { container } = renderPage();
    await flush();
    await flush();
    fireEvent.click(screen.getByTitle("growth.csv"));
    await flush();
    await flush();
    await flush();

    const callsBefore = reactCalls();
    const rows = () => Array.from(container.querySelectorAll("tr.ranking-row"));
    expect(rows()).toHaveLength(2);
    fireEvent.click(rows()[1]?.querySelector(".ranking-row__name") as HTMLElement);
    await flush();

    expect(reactCalls()).toBe(callsBefore + 1);
    expect(lastFittedCurveY()).toEqual(CURVE_B.y);
  });

  it("asks for the chart's points and the fits history together, not one after the other", async () => {
    let resolveDetail: (value: DatasetDetailOut) => void = () => {};
    request.mockImplementation(
      (async (path: string, options?: RequestInit) => {
        const target = String(path);
        if (target === "/datasets/ds-1?points=20000") {
          return new Promise<DatasetDetailOut>((resolve) => {
            resolveDetail = resolve;
          });
        }
        return route(target, options);
      }) as unknown as typeof api.request,
    );

    renderPage();
    await flush();
    await flush();
    fireEvent.click(screen.getByTitle("growth.csv"));
    await flush();

    // The still-pending detail request must not hold the fits list back.
    expect(request).toHaveBeenCalledWith("/datasets/ds-1/fits");

    resolveDetail(DETAIL);
    await flush();
    await flush();
    expect(request).toHaveBeenCalledWith("/datasets/ds-1?points=20000");
    // The points reached the chart once the deferred detail resolved.
    const traces = (plotly.react.mock.calls.at(-1)?.[1] ?? []) as Array<{
      name?: string;
      y?: number[];
    }>;
    expect(traces.find((trace) => trace.name === "Data")?.y).toEqual(DETAIL.points?.y);
  });
});
