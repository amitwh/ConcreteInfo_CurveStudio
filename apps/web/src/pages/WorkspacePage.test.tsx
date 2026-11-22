import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { AuthProvider } from "../api/auth";
import { ApiError, api } from "../api/client";
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
type FitBundle = components["schemas"]["FitBundle"];
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

const FIT_SUMMARY: FitSummaryOut = {
  id: "fit-1",
  dataset_id: "ds-1",
  created_at: "2026-11-17T08:00:00Z",
  status: "converged",
  spec: { model: { kind: "registered", id: "exp_decay" } },
};

/** Only the fields the chart, default Parameters tab and History tab touch. */
const FIT_BUNDLE = {
  result: {
    status: "converged",
    termination_reason: "converged",
    model_id: "quadratic",
    parameters: [],
  },
  curve: null,
  confidence_band: null,
  prediction_band: null,
} as unknown as FitBundle;

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

/** When true the open-row call answers with an existing fit instead of a polled job. */
let openAsExistingFit = false;

/** Every route WorkspacePage hits on mount, on selection and while opening a row. */
function route(path: string, options?: RequestInit): Promise<unknown> {
  if (path === "/datasets") return Promise.resolve([DATASET]);
  if (path === "/models") return Promise.resolve([]);
  if (path === "/datasets/ds-1?points=20000") return Promise.resolve(DETAIL);
  if (path === "/datasets/ds-1/autofits") return Promise.resolve([SUMMARY]);
  if (path === "/autofits/af-1") return Promise.resolve(STORED);
  if (path === "/datasets/ds-1/fits") return Promise.resolve([FIT_SUMMARY]);
  if (path === "/fits/fit-1") {
    // The GET loads an opened row's bundle; the DELETE fails to surface a fit error.
    return options?.method === "DELETE"
      ? Promise.reject(new ApiError(404, "NOT_FOUND", "the fit does not exist", null))
      : Promise.resolve(FIT_BUNDLE);
  }
  if (path === "/autofits/af-1/rows/exp_decay/open") {
    return Promise.resolve(
      openAsExistingFit
        ? { fit_id: "fit-1", job_id: null }
        : { fit_id: null, job_id: "job-1" },
    );
  }
  if (path === "/jobs/job-1") return Promise.resolve(FAILED_JOB);
  return Promise.reject(new Error(`unexpected request: ${path}`));
}

let request: MockInstance;

beforeEach(() => {
  openAsExistingFit = false;
  request = vi.spyOn(api, "request").mockImplementation(
    (async (path: string, options?: RequestInit) =>
      route(String(path), options)) as unknown as typeof api.request,
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
    const busy = screen.getByRole("button", { name: "Open full results" }) as HTMLButtonElement;
    expect(busy.disabled).toBe(true);
    expect(busy.textContent).toBe("Opening…");
    expect(request).toHaveBeenCalledWith("/autofits/af-1/rows/exp_decay/open", {
      method: "POST",
    });

    // The poller sees the job fail; the table must not stay locked on "Opening…".
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    const reopened = screen.getByRole("button", { name: "Open full results" }) as HTMLButtonElement;
    expect(reopened.disabled).toBe(false);
    expect(reopened.textContent).toBe("Open");
  });
});

describe("WorkspacePage — layout and errors", () => {
  async function selectDataset(): Promise<void> {
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
  }

  it("renders the results in the full-width results row, not the middle column", async () => {
    await selectDataset();
    const results = document.querySelector(".workspace__results");
    const main = document.querySelector(".workspace__main");
    expect(results).not.toBeNull();
    expect(results?.querySelector('[aria-label="Auto-fit"]')).not.toBeNull();
    expect(main?.querySelector('[aria-label="Auto-fit"]')).toBeNull();
  });

  it("shows a fit error (e.g. a failed delete) above the results in opened-row mode", async () => {
    openAsExistingFit = true;
    await selectDataset();

    // Open the row: the route answers with an existing fit, so no job is polled.
    fireEvent.click(screen.getByRole("button", { name: "Open full results" }));
    await flush();
    await flush();

    // Delete the opened fit from History; the route rejects.
    fireEvent.click(screen.getByRole("tab", { name: "History" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete fit fit-1" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await flush();

    const results = document.querySelector(".workspace__results");
    expect(results?.textContent).toContain("the fit does not exist");
    expect(screen.getByRole("tab", { name: "History" })).toBeTruthy();
  });
});

describe("WorkspacePage — opening a row selects it", () => {
  it("moves the selection to the opened row so the chart, colour and ranking agree", async () => {
    const second = {
      ...ROW,
      rank: 2,
      model_id: "quadratic",
      name: "Quadratic",
      category: "Polynomial",
      equation_fitted: "y = 1 + 0.5*x + 0.1*x^2",
    } as RankedFit;
    const twoRows = {
      ...STORED,
      n_candidates: 2,
      result: { ...STORED.result, ranked: [ROW, second] },
    } as AutofitOut;
    request.mockImplementation((async (path: string, options?: RequestInit) =>
      String(path) === "/autofits/af-1" ? twoRows : route(String(path), options)) as unknown as typeof api.request);

    const { container } = render(
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

    const rows = Array.from(container.querySelectorAll("tr.ranking-row"));
    expect(rows).toHaveLength(2);
    expect(rows[0]?.getAttribute("aria-selected")).toBe("true");
    expect(rows[1]?.getAttribute("aria-selected")).toBe("false");

    // Open the second row: the selection must follow it, otherwise the chart draws the
    // opened fit in the first row's colour and nothing looks like it changed.
    const openButtons = screen.getAllByRole("button", { name: "Open full results" });
    fireEvent.click(openButtons[1] as HTMLButtonElement);
    await flush();

    const after = Array.from(container.querySelectorAll("tr.ranking-row"));
    expect(after[1]?.getAttribute("aria-selected")).toBe("true");
    expect(after[0]?.getAttribute("aria-selected")).toBe("false");
  });
});

describe("WorkspacePage — selecting a row after opening it", () => {
  it("keeps the opened result when its own row is clicked, and drops it for another row", async () => {
    const second = {
      ...ROW,
      rank: 2,
      model_id: "quadratic",
      name: "Quadratic",
      category: "Polynomial",
      equation_fitted: "y = 1 + 0.5*x + 0.1*x^2",
    } as RankedFit;
    const twoRows = {
      ...STORED,
      n_candidates: 2,
      result: { ...STORED.result, ranked: [ROW, second] },
    } as AutofitOut;
    openAsExistingFit = true; // the open answers with a ready fit, so no job polling
    request.mockImplementation((async (path: string, options?: RequestInit) => {
      const target = String(path);
      if (target === "/autofits/af-1") return twoRows;
      // Any row's Open answers with the ready fit, so no job polling is involved.
      if (target.endsWith("/open")) return { fit_id: "fit-1", job_id: null };
      return route(target, options);
    }) as unknown as typeof api.request);

    const { container } = render(
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

    const openButtons = screen.getAllByRole("button", { name: "Open full results" });
    fireEvent.click(openButtons[1] as HTMLButtonElement);
    await flush();
    expect(screen.queryByText("Parameters")).not.toBeNull();

    // Clicking the row whose fit is open must not throw that fit away.
    const rows = () => Array.from(container.querySelectorAll("tr.ranking-row"));
    fireEvent.click(rows()[1]?.querySelector(".ranking-row__name") as HTMLElement);
    await flush();
    expect(screen.queryByText("Parameters")).not.toBeNull();

    // Selecting a different row goes back to that row's own curve.
    fireEvent.click(rows()[0]?.querySelector(".ranking-row__name") as HTMLElement);
    await flush();
    expect(screen.queryByText("Parameters")).toBeNull();
  });
});
