/**
 * The More tools tab: each fit tool posts its own body, its job is followed to the stored
 * result, and that result is drawn by `StatsResult`; a refusal or a failed job says why in
 * the tab; Cancel posts the cancel route; and the tab strip still lists every view.
 */

import { act, fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import {
  ApiError,
  api,
  type FitBundle,
  type JobOut,
  type StatResultOut,
  type ToolResult,
} from "../../api/client";
import { renderWithQuery } from "../../test/render";
import { FitToolsTab } from "./FitToolsTab";
import { ResultsTabs } from "./ResultsTabs";

// jsdom cannot host a real Plotly graph; every chart is replaced by a stand-in here.
vi.mock("../../components/PlotChart", () => ({
  PlotChart: () => <div data-testid="plot" />,
}));
vi.mock("../../lib/report/export", () => ({ exportFitPdf: vi.fn() }));
vi.mock("../../api/auth", () => ({ useAuth: () => ({ user: null }) }));

const FIT = "fit-1";
const DISCERN_PATH = `/fits/${FIT}/discernibility`;
const SIMULATE_PATH = `/fits/${FIT}/simulate`;
const ADVISOR_PATH = `/fits/${FIT}/advisor`;

/** One answer for `api.request`: a body to return, or an error to throw. */
type Reply = { ok: unknown } | { refused: ApiError };

/** The answers per path, in the order they are asked for; the last one repeats. */
const script = new Map<string, Reply[]>();
/** How many times each path was requested. */
const asked = new Map<string, number>();

function answer(path: string, ...replies: Reply[]): void {
  script.set(path, replies);
}

function times(path: string): number {
  return asked.get(path) ?? 0;
}

/** The body sent with the one POST that reached `path`, parsed. */
function posted(path: string): unknown {
  const calls = request.mock.calls.filter(
    ([p, init]) => String(p) === path && init?.method === "POST",
  );
  expect(calls).toHaveLength(1);
  return JSON.parse(String(calls[0]?.[1]?.body));
}

let request: MockInstance;

beforeEach(() => {
  script.clear();
  asked.clear();
  request = vi.spyOn(api, "request").mockImplementation(
    (async (path: string) => {
      asked.set(path, (asked.get(path) ?? 0) + 1);
      const replies = script.get(path);
      if (replies === undefined || replies.length === 0) {
        throw new Error(`no answer scripted for ${path}`);
      }
      const reply = (replies.length > 1 ? replies.shift() : replies[0]) as Reply;
      if ("refused" in reply) throw reply.refused;
      return reply.ok;
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

/** A request and the state it sets: a few empty turns of the event loop. */
async function settle(): Promise<void> {
  for (let turn = 0; turn < 5; turn += 1) await flush();
}

/** Move the poller's clock forward, then let what it started settle. */
async function pass(milliseconds: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
  await settle();
}

/* ---- The fit the tab reads, and the server's answers ---- */

function bundle(): FitBundle {
  return {
    result: {
      model_id: "exp_decay",
      equation: "y = a * exp(-b * x)",
      equation_fitted: null,
      parameters: [
        { name: "a", value: 2, std_err: 0.1, ci_low: 1.8, ci_high: 2.2 },
        { name: "b", value: 0.5, std_err: 0.02, ci_low: 0.46, ci_high: 0.54 },
      ],
      covariance: null,
      correlation: null,
      stats: null,
      fitted: [],
      residuals: [],
      status: "converged",
      termination_reason: "converged: rtol",
      method: "lm",
      loss: "linear",
      initial_params: {},
      n_starts: 1,
      warnings: [],
      provenance: {
        software_versions: {},
        dataset_fingerprint: "fp",
        created_at: "2026-11-24T05:00:00+05:30",
        seed: 0,
      },
    },
    summary: null,
    summary_reason: "no summary",
    curve: null,
    curve_reason: "not drawn",
    confidence_band: null,
    prediction_band: null,
    bands_reason: "not drawn",
    equation_text: "y = a * exp(-b * x)",
    warnings: [],
    data: {
      x: [0, 1],
      y: [2, 1.3],
      weights: null,
      x_name: "x",
      y_name: "y",
      n_total: 2,
      subsampled: false,
    },
  } as unknown as FitBundle;
}

/** One job as the server reports it; once it has succeeded, `result_ref` is the result row. */
function job(
  status: JobOut["status"],
  kind: JobOut["kind"],
  resultId: string | null,
  overrides: Partial<JobOut> = {},
): JobOut {
  const succeeded = status === "succeeded";
  return {
    id: "job-1",
    kind,
    status,
    dataset_id: "ds-1",
    spec: { fit_id: FIT },
    result_ref: succeeded ? resultId : null,
    result_id: succeeded ? resultId : null,
    progress: null,
    error_code: null,
    error_message: null,
    created_at: "2026-11-24T05:00:00+05:30",
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

function toolResult(title: string): ToolResult {
  return {
    tool: "discernibility",
    title,
    level: 0.95,
    estimates: [
      { label: "Model exp_decay", value: 4.2, se: null, lower: null, upper: null },
    ],
    tests: [],
    assumptions: [],
    warnings: [],
    tables: [],
    plots: [
      {
        kind: "scatter",
        title: "The leading models on the data's range",
        x: [0, 1],
        y: [2, 1],
        lines: [],
        x_label: "x",
        y_label: "y",
      },
    ],
    interpretation: "The leading models cannot be told apart on these data.",
    methods_fragment: "Models were compared by their AICc distance.",
    counts: { n_offered: 2, n_used: 2, n_excluded: 0 },
    excluded: [],
    excluded_unlisted: 0,
  };
}

function stored(id: string, title: string): StatResultOut {
  return {
    id,
    dataset_id: "ds-1",
    tool: "discernibility",
    title,
    request: {},
    result: toolResult(title),
    created_at: "2026-11-24T05:05:00+05:30",
  };
}

/** A run that succeeds on its first poll and stores `resultId`. */
function scriptRun(
  path: string,
  kind: JobOut["kind"],
  resultId: string,
  result: StatResultOut,
): void {
  answer(path, { ok: { job_id: "job-1" } });
  answer("/jobs/job-1", { ok: job("succeeded", kind, resultId) });
  answer(`/stats/results/${resultId}`, { ok: result });
}

function renderTab(): ReturnType<typeof renderWithQuery> {
  return renderWithQuery(<FitToolsTab fitId={FIT} bundle={bundle()} />);
}

function click(name: string): void {
  fireEvent.click(screen.getByRole("button", { name }));
}

/** Every tool's run button, disabled or not. */
function runButtons(): HTMLButtonElement[] {
  return [
    screen.getByRole("button", { name: /Run the discernibility report|Running…/ }),
    screen.getByRole("button", { name: /Run the simulation|Running…/ }),
    screen.getByRole("button", { name: /Run the advisor|Running…/ }),
  ] as HTMLButtonElement[];
}

describe("FitToolsTab — the three tools", () => {
  it("posts the discernibility body and renders the stored result's title and a table", async () => {
    scriptRun(DISCERN_PATH, "discern", "result-1", stored("result-1", "Model discernibility report"));
    renderTab();

    fireEvent.change(screen.getByLabelText("Confidence level"), { target: { value: "0.9" } });
    fireEvent.change(screen.getByLabelText("Grid points"), { target: { value: "120" } });
    click("Run the discernibility report");
    await settle();

    expect(posted(DISCERN_PATH)).toEqual({ level: 0.9, grid_points: 120 });
    expect(times("/jobs/job-1")).toBe(0);

    await pass(300);
    expect(times("/jobs/job-1")).toBe(1);
    expect(times("/stats/results/result-1")).toBe(1);

    expect(
      screen.getByRole("heading", { name: "Model discernibility report" }),
    ).toBeTruthy();
    const estimates = screen.getByRole("table", { name: "Estimates" });
    expect(within(estimates).getByText("Model exp_decay")).toBeTruthy();
    expect(screen.getByTestId("plot")).toBeTruthy();
  });

  it("posts the simulation body built from the fit's own parameter values", async () => {
    scriptRun(SIMULATE_PATH, "simulate", "result-2", stored("result-2", "Feasibility simulation"));
    renderTab();

    click("Run the simulation");
    await settle();

    expect(posted(SIMULATE_PATH)).toEqual({
      truth: { a: 2, b: 0.5 },
      design: { low: 0, high: 1, count: 5, replicates: 1 },
      noise: { kind: "sd", value: 0.1 },
      runs: 200,
      seed: 0,
      level: 0.95,
      difference: null,
    });

    await pass(300);
    expect(screen.getByRole("heading", { name: "Feasibility simulation" })).toBeTruthy();
    expect(screen.getByRole("table", { name: "Estimates" })).toBeTruthy();
  });

  it("posts the advisor body and narrows to a named parameter when that objective is chosen", async () => {
    scriptRun(ADVISOR_PATH, "advise", "result-3", stored("result-3", "Where to measure next"));
    renderTab();

    fireEvent.change(screen.getByLabelText("Objective"), {
      target: { value: "parameter:b" },
    });
    click("Run the advisor");
    await settle();

    expect(posted(ADVISOR_PATH)).toEqual({
      points: 3,
      candidates: 400,
      replicates: 1,
      objective: "parameter",
      parameter: "b",
      level: 0.95,
      low: null,
      high: null,
    });

    await pass(300);
    expect(screen.getByRole("heading", { name: "Where to measure next" })).toBeTruthy();
    expect(screen.getByRole("table", { name: "Estimates" })).toBeTruthy();
  });

  it("keeps the last result while a new run is under way", async () => {
    scriptRun(DISCERN_PATH, "discern", "result-1", stored("result-1", "First report"));
    renderTab();

    click("Run the discernibility report");
    await pass(300);
    expect(screen.getByRole("heading", { name: "First report" })).toBeTruthy();

    // A second run: the first result stays until the second replaces it.
    answer(DISCERN_PATH, { ok: { job_id: "job-2" } });
    answer("/jobs/job-2", { ok: job("running", "discern", null, { id: "job-2" }) });
    click("Run the discernibility report");
    await settle();

    expect(screen.getByRole("heading", { name: "First report" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
  });
});

describe("FitToolsTab — errors", () => {
  it("shows a failed job's NO_RANKING code and message, and re-enables the buttons", async () => {
    answer(DISCERN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", {
      ok: job("failed", "discern", null, {
        error_code: "NO_RANKING",
        error_message: "this fit was not opened from an Auto-fit ranking",
      }),
    });
    renderTab();

    click("Run the discernibility report");
    await pass(300);

    expect(screen.getByRole("alert").textContent).toBe(
      "The tool failed: NO_RANKING: this fit was not opened from an Auto-fit ranking.",
    );
    for (const button of runButtons()) expect(button.disabled).toBe(false);
  });

  it("shows a 409's detail.code and message, and re-enables the buttons", async () => {
    answer(DISCERN_PATH, {
      refused: new ApiError(
        409,
        "CONFLICT",
        "a discernibility report of this fit is already queued or running",
        { code: "DISCERNIBILITY_RUNNING", job_id: "job-9" },
      ),
    });
    renderTab();

    click("Run the discernibility report");
    await settle();

    expect(screen.getByRole("alert").textContent).toBe(
      "DISCERNIBILITY_RUNNING: a discernibility report of this fit is already queued or running",
    );
    for (const button of runButtons()) expect(button.disabled).toBe(false);
  });

  it("refuses a malformed body in the tab, before any request", async () => {
    renderTab();

    fireEvent.change(screen.getByLabelText("Design low"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Design high"), { target: { value: "1" } });
    click("Run the simulation");
    await settle();

    expect(screen.getByRole("alert").textContent).toBe("the design range needs low below high");
    expect(times(SIMULATE_PATH)).toBe(0);
    for (const button of runButtons()) expect(button.disabled).toBe(false);
  });
});

describe("FitToolsTab — cancel", () => {
  it("posts the cancel route and returns the tab to idle", async () => {
    answer(DISCERN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", { ok: job("running", "discern", null) });
    answer("/jobs/job-1/cancel", { ok: job("cancelled", "discern", null) });
    renderTab();

    click("Run the discernibility report");
    await pass(300);
    expect(screen.getByRole("status").textContent).toMatch(/running/i);

    click("Cancel");
    await settle();

    expect(times("/jobs/job-1/cancel")).toBe(1);
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.getByRole("status").textContent).toMatch(/cancelled/i);
    for (const button of runButtons()) expect(button.disabled).toBe(false);

    // Polling has stopped: no further reads however long the wait.
    await pass(10_000);
    expect(times("/jobs/job-1")).toBe(1);
  });
});

describe("ResultsTabs — the More tools tab", () => {
  it("lists More tools among the results tabs and renders it beside the others", () => {
    renderWithQuery(
      <ResultsTabs
        fitId={FIT}
        bundle={bundle()}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Parameters",
      "Statistics",
      "Fit audit",
      "More tools",
      "Residuals",
      "Equation",
      "Predict",
      "History",
    ]);

    // The default tab still renders its own view.
    expect(screen.getByRole("table")).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: "More tools" }));
    expect(screen.getByRole("heading", { level: 3, name: "More tools" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run the discernibility report" })).toBeTruthy();

    // And the first tab still renders when it is chosen again.
    fireEvent.click(screen.getByRole("tab", { name: "Parameters" }));
    expect(screen.queryByRole("heading", { level: 3, name: "More tools" })).toBeNull();
    expect(screen.getByRole("table")).toBeTruthy();
  });

  it("keeps a finished result across a tab switch", async () => {
    scriptRun(DISCERN_PATH, "discern", "result-9", stored("result-9", "Kept report"));
    renderWithQuery(
      <ResultsTabs
        fitId={FIT}
        bundle={bundle()}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "More tools" }));
    click("Run the discernibility report");
    await pass(300);
    expect(screen.getByRole("heading", { name: "Kept report" })).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: "Parameters" }));
    fireEvent.click(screen.getByRole("tab", { name: "More tools" }));

    expect(screen.getByRole("heading", { name: "Kept report" })).toBeTruthy();
    expect(times(DISCERN_PATH)).toBe(1);
  });
});
