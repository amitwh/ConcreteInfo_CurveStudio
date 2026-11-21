import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { AuthProvider } from "../api/auth";
import { api } from "../api/client";
import type { components } from "../api/schema";
import { ThemeProvider } from "../components/ThemeToggle";
import { WorkspacePage } from "./WorkspacePage";

// jsdom cannot host a real Plotly graph; this test is about the workspace's UI state.
vi.mock("../components/PlotChart", () => ({
  PlotChart: () => <div data-testid="plot" />,
  downloadPlotPng: () => {},
}));

type DatasetOut = components["schemas"]["DatasetOut"];
type DatasetDetailOut = components["schemas"]["DatasetDetailOut"];
type FitSummaryOut = components["schemas"]["FitSummaryOut"];
type AutofitSummaryOut = components["schemas"]["AutofitSummaryOut"];
type AutofitOut = components["schemas"]["AutofitOut"];
type RankedFit = components["schemas"]["RankedFit"];
type JobOut = components["schemas"]["JobOut"];

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
};

const SUMMARY: AutofitSummaryOut = {
  id: "af-1",
  dataset_id: "ds-1",
  created_at: "2026-11-17T09:00:00Z",
  best_model_id: "exp_decay",
  n_candidates: 1,
  rank_by: "best_overall",
};

const ROW = {
  rank: 1,
  model_id: "exp_decay",
  name: "Exponential decay",
  category: "Exponential",
  equation_fitted: "y = 2 * exp(-0.3 * x)",
  metrics: { k: 2, n: 40, r2: 0.99, adj_r2: 0.98, rmse: 0.05, aicc: -120, bic: -117, cv_rmse: 0.06 },
  status: "converged",
} as RankedFit;

const STORED: AutofitOut = {
  id: "af-1",
  dataset_id: "ds-1",
  created_at: "2026-11-17T09:00:00Z",
  best_model_id: "exp_decay",
  n_candidates: 1,
  rank_by: "best_overall",
  partial: false,
  result: {
    config: { rank_by: "best_overall" } as unknown as AutofitOut["result"]["config"],
    dataset_fingerprint: "fp",
    n_active: 40,
    ranked: [ROW],
    outcomes: [],
    skipped: {},
    unconverged: {},
    exact_fits: [],
    warnings: [],
    cancelled: false,
    elapsed_seconds: 1.2,
    curves: {},
    overfitting_note: null,
  },
} as unknown as AutofitOut;

const FAILED_JOB: JobOut = {
  id: "job-1",
  kind: "fit",
  status: "failed",
  dataset_id: "ds-1",
  spec: {},
  result_ref: null,
  error_code: "FIT_ERROR",
  error_message: "the fit failed",
  created_at: "2026-11-17T10:00:00Z",
  started_at: null,
  finished_at: null,
} as JobOut;

/** Every route WorkspacePage hits on mount, on selection and while opening a row. */
function route(path: string): Promise<unknown> {
  if (path === "/datasets") return Promise.resolve([DATASET]);
  if (path === "/models") return Promise.resolve([]);
  if (path === "/datasets/ds-1?points=20000") return Promise.resolve(DETAIL);
  if (path === "/datasets/ds-1/autofits") return Promise.resolve([SUMMARY]);
  if (path === "/autofits/af-1") return Promise.resolve(STORED);
  if (path === "/datasets/ds-1/fits") return Promise.resolve([] as FitSummaryOut[]);
  if (path === "/autofits/af-1/rows/exp_decay/open") {
    return Promise.resolve({ fit_id: null, job_id: "job-1" });
  }
  if (path === "/jobs/job-1") return Promise.resolve(FAILED_JOB);
  return Promise.reject(new Error(`unexpected request: ${path}`));
}

let request: MockInstance;

beforeEach(() => {
  request = vi.spyOn(api, "request").mockImplementation(
    (async (path: string) => route(String(path))) as unknown as typeof api.request,
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

describe("WorkspacePage — opened Auto-fit rows", () => {
  it("re-enables the ranking table's Open buttons when the opened-row job fails", async () => {
    render(
      <ThemeProvider>
        <AuthProvider>
          <WorkspacePage />
        </AuthProvider>
      </ThemeProvider>,
    );
    await flush();
    await flush();

    fireEvent.click(screen.getByTitle("growth.csv"));
    await flush();
    await flush();

    const open = screen.getByRole("button", { name: "Open full results" }) as HTMLButtonElement;
    expect(open.disabled).toBe(false);
    fireEvent.click(open);
    await flush();
    expect((screen.getByRole("button", { name: "Opening…" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(request).toHaveBeenCalledWith("/autofits/af-1/rows/exp_decay/open", {
      method: "POST",
    });

    // The poller sees the job fail; the table must not stay locked on "Opening…".
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    const reopened = screen.getByRole("button", { name: "Open full results" }) as HTMLButtonElement;
    expect(reopened.disabled).toBe(false);
    expect(screen.queryByRole("button", { name: "Opening…" })).toBeNull();
  });
});
