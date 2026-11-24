import { act, fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import {
  ApiError,
  api,
  type AuditCheckOut,
  type AuditVerdict,
  type FitAuditOut,
  type JobOut,
} from "../../api/client";
import { renderWithQuery } from "../../test/render";
import { FitAuditTab } from "./FitAuditTab";

const FIT = "fit-1";
const AUDIT_PATH = `/fits/${FIT}/audit`;
const RUN_PATH = `/fits/${FIT}/audit/cross-validation`;

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
  const calls = request.mock.calls.filter(([p, init]) => String(p) === path && init?.method === "POST");
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
  for (let turn = 0; turn < 4; turn += 1) await flush();
}

/** Move the poller's clock forward, then let what it started settle. */
async function pass(milliseconds: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
  await settle();
}

/* ---- A representative audit of a converged fit, in the server's wording ---- */

const RESTART = "Check the starting values and the bounds, or try a model with fewer parameters.";
const NORMALITY_NEXT_STEP =
  "If a few points cause it, try a robust loss (Huber or soft-L1). Otherwise say in your " +
  "write-up that the intervals rely on an approximation.";
const VARIANCE_NEXT_STEP =
  "If the spread grows with the signal, use weights from replicates or a robust loss, and say so.";
const TREND_NEXT_STEP =
  "The residuals show a curved trend, so the model is missing a shape the data contain. " +
  "Try a model with one more parameter.";
const SIMPLER_NEXT_STEP =
  "Report the simpler model unless the extra parameter answers a question of your own.";
/** The rule on a p-value, as the server words it for a residual check. */
const P_THRESHOLD = "p >= 0.1 pass; 0.01 <= p < 0.1 caution; p < 0.01 fail";

function check(
  id: string,
  verdict: AuditVerdict,
  title: string,
  evidence: string,
  extra: Partial<AuditCheckOut> = {},
): AuditCheckOut {
  return {
    id,
    title,
    verdict,
    statistic: null,
    threshold: null,
    evidence,
    explanation: `${id}: what this check is about.`,
    next_step: null,
    ...extra,
  };
}

function convergedAudit(overrides: Partial<FitAuditOut> = {}): FitAuditOut {
  return {
    checks: [
      check("converged", "pass", "The fit converged", "Status converged: converged: rtol", {
        threshold: "status converged",
      }),
      check("perfect_fit", "pass", "The fit is not exact", "RMSE 0.0412 against a y range of 2.6"),
      check("residual_normality", "caution", "Residuals may depart from a normal distribution", "Shapiro-Wilk p = 0.081", {
        statistic: "0.081",
        threshold: P_THRESHOLD,
        next_step: NORMALITY_NEXT_STEP,
      }),
      check(
        "residual_independence",
        "pass",
        "Residuals show no autocorrelation in x order",
        "Ljung-Box p = 0.78, residuals taken in ascending x.",
        { statistic: "Q(10) = 6.4, p = 0.78", threshold: P_THRESHOLD },
      ),
      check("constant_variance", "fail", "Residual variance changes with the fitted value", "Breusch-Pagan p = 0.0004", {
        statistic: "0.0004",
        next_step: VARIANCE_NEXT_STEP,
      }),
      check("influential_points", "pass", "No point has a large influence", "Largest Cook's distance 0.21 is below the 50% region"),
      check("residual_trend", "fail", "Residuals show a trend in x", "F test of cubic terms in the residuals: p = 0.0006", {
        next_step: TREND_NEXT_STEP,
      }),
      check("parameters_identifiable", "pass", "The parameters are identifiable", "No parameter sits at a bound and no interval covers more than half its range"),
      check("points_per_parameter", "pass", "There are enough points per parameter", "24 points for 3 parameters gives 8 per parameter"),
      check("cross_validation", "not_applicable", "Predictive error was not assessed", "cross-validation has not been run", {
        next_step: "Run cross-validation to check this.",
      }),
      check("simpler_model", "caution", "A simpler model fits as well", "Exponential decay (2 parameters) (exp_decay_2) reaches the same fit with 2 parameters instead of 3 and lies 0.6 AICc from the best.", {
        statistic: "3 parameters here; the simplest within 2 AICc of the best has 2 (exp_decay_2)",
        threshold: "delta AICc <= 2",
        next_step: SIMPLER_NEXT_STEP,
      }),
    ],
    overall: "fail",
    summary: `2 checks failed, 2 need a look. Residual variance changes with the fitted value. ${VARIANCE_NEXT_STEP}`,
    methods_fragment:
      "Residuals showed a trend in x (p = 0.0006) and variance that changes with the fitted value (Breusch-Pagan p = 0.0004); Shapiro-Wilk p = 0.081, a possible departure from normality. Cross-validation was not run.",
    simpler_model: {
      model_id: "exp_decay_2",
      name: "Exponential decay (2 parameters)",
      n_parameters: 2,
      delta_aicc: 0.6,
      cv_rmse: null,
    },
    n_observations: 24,
    n_parameters: 3,
    excluded: [],
    ...overrides,
  };
}

const CROSS_VALIDATION_DONE = check(
  "cross_validation",
  "pass",
  "Held-out error is close to what a correct model shows",
  "Cross-validation RMSE 0.0452 against an in-sample RMSE of 0.0412 (ratio 1.1) in 5 folds",
  { statistic: "1.1", threshold: "ratio <= 1.25 pass" },
);

function notConvergedAudit(): FitAuditOut {
  const reason = "maximum number of iterations exceeded";
  const rest: Array<[string, string]> = [
    ["perfect_fit", "Whether the fit is exact was not assessed"],
    ["residual_normality", "Normality of the residuals was not tested"],
    ["residual_independence", "Independence of the residuals was not tested"],
    ["constant_variance", "Constancy of the residual variance was not tested"],
    ["influential_points", "Influence of single points was not assessed"],
    ["residual_trend", "A trend of the residuals in x was not tested"],
    ["parameters_identifiable", "The parameters were not checked"],
    ["points_per_parameter", "Points per parameter were not counted"],
    ["cross_validation", "Predictive error was not assessed"],
    ["simpler_model", "Simpler models were not compared"],
  ];
  return convergedAudit({
    checks: [
      check("converged", "fail", "The fit did not converge", `Status not_converged: ${reason}`, {
        threshold: "status converged",
        next_step: RESTART,
      }),
      ...rest.map(([id, title]) => check(id, "not_applicable", title, "the fit did not converge")),
    ],
    overall: "fail",
    summary: `The fit did not converge (${reason}), so there is nothing to audit yet. ${RESTART}`,
    simpler_model: null,
  });
}

/** A cross-validation job as the server reports it: once it has succeeded, `result_ref` is
 *  the DATASET it ran on and `result_id` the fit its result is stored on. */
function job(status: JobOut["status"], overrides: Partial<JobOut> = {}): JobOut {
  const succeeded = status === "succeeded";
  return {
    id: "job-1",
    kind: "audit",
    status,
    dataset_id: "ds-1",
    spec: { fit_id: FIT, folds: 5 },
    result_ref: succeeded ? "ds-1" : null,
    result_id: succeeded ? FIT : null,
    progress: null,
    error_code: null,
    error_message: null,
    created_at: "2026-11-24T05:00:00+05:30",
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

const PILLS: Record<AuditVerdict, string> = {
  pass: "Pass",
  caution: "Caution",
  fail: "Fail",
  not_applicable: "Not applicable",
};

function cards(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>(".audit-check"));
}

function card(id: string): HTMLElement {
  const found = cards().find((element) => element.getAttribute("data-check") === id);
  expect(found, `no card for the check ${id}`).toBeDefined();
  return found as HTMLElement;
}

function renderTab(converged = true): ReturnType<typeof renderWithQuery> {
  return renderWithQuery(<FitAuditTab fitId={FIT} converged={converged} />);
}

function runButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: /Run cross-validation check|Run again/ }) as HTMLButtonElement;
}

function control(): HTMLElement {
  return screen.getByRole("region", { name: "Cross-validation check" });
}

function statusText(): string | null {
  return control().querySelector('[role="status"]')?.textContent ?? null;
}

describe("FitAuditTab — the checks", () => {
  it("shows one card per check, in the server's order, each with the pill for its verdict", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    expect(cards()).toHaveLength(audit.checks.length);
    cards().forEach((element, index) => {
      const expected = audit.checks[index] as AuditCheckOut;
      const pill = element.querySelector(".verdict");
      expect(element.getAttribute("data-check")).toBe(expected.id);
      expect(pill?.className).toContain(`verdict--${expected.verdict}`);
      expect(pill?.textContent).toBe(PILLS[expected.verdict]);
      expect(within(element).getByText(expected.title)).toBeTruthy();
      expect(within(element).getByText(expected.evidence)).toBeTruthy();
    });
  });

  it("gives a decided card its evidence, its explanation and its next step", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    const failing = audit.checks.find((item) => item.id === "constant_variance") as AuditCheckOut;
    const element = card("constant_variance");
    expect(element.className).not.toContain("audit-check--quiet");
    expect(within(element).getByText(failing.evidence)).toBeTruthy();
    expect(within(element).getByText(failing.explanation)).toBeTruthy();
    expect(within(element).getByText(VARIANCE_NEXT_STEP)).toBeTruthy();

    // A check that passed needs no next step, and the server sends none.
    const passing = audit.checks.find((item) => item.id === "converged") as AuditCheckOut;
    expect(within(card("converged")).getByText(passing.explanation)).toBeTruthy();
    expect(card("converged").textContent).not.toContain("Next step");
  });

  it("keeps a check that did not apply quiet: its title and the evidence that says why", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    const pending = audit.checks.find((item) => item.id === "cross_validation") as AuditCheckOut;
    const element = card("cross_validation");
    expect(element.className).toContain("audit-check--quiet");
    expect(within(element).getByText(pending.title)).toBeTruthy();
    expect(within(element).getByText("cross-validation has not been run")).toBeTruthy();
    expect(within(element).queryByText(pending.explanation)).toBeNull();
  });

  it("shows the overall verdict as a pill, and the audit's summary", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    const overall = document.querySelector(".audit-summary .verdict") as HTMLElement;
    expect(overall.className).toContain("verdict--fail");
    expect(overall.textContent).toBe("Fail");
    expect(within(document.querySelector(".audit-summary") as HTMLElement).getByText(audit.summary)).toBeTruthy();
  });

  it("names the simpler model the audit points to, with n/a where it has no figure", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    const line = document.querySelector(".audit-simpler") as HTMLElement;
    expect(line).not.toBeNull();
    expect(line.textContent).toContain("Exponential decay (2 parameters)");
    expect(line.textContent).toContain("exp_decay_2");
    expect(line.textContent).toContain("2 parameters");
    expect(line.textContent).toContain("0.6");
    expect(line.textContent).toContain("n/a");
  });

  it("says nothing about a simpler model when the audit names none", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit({ simpler_model: null }) });
    renderTab();
    await settle();

    expect(document.querySelector(".audit-simpler")).toBeNull();
  });

  it("says it is loading until the audit arrives", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    renderTab();

    expect(screen.getByRole("status").textContent).toBe("Loading the fit audit…");

    await settle();
    expect(screen.queryByText("Loading the fit audit…")).toBeNull();
  });

  it("shows why the audit could not be read, with a Retry that reads it again", async () => {
    answer(
      AUDIT_PATH,
      { refused: new ApiError(500, "INTERNAL_ERROR", "the server tripped", {}) },
      { ok: convergedAudit() },
    );
    renderTab();
    await settle();

    expect(screen.getByRole("alert").textContent).toBe("the server tripped");
    expect(cards()).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await settle();

    expect(screen.queryByRole("alert")).toBeNull();
    expect(times(AUDIT_PATH)).toBe(2);
    expect(cards()).toHaveLength(11);
  });

  it("says honestly that an audit with no checks should not happen", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit({ checks: [] }) });
    renderTab();
    await settle();

    expect(
      screen.getByText(/came back with no checks, which should not happen: every fit has its convergence check/i),
    ).toBeTruthy();
    expect(cards()).toHaveLength(0);
  });

  it("for a fit that did not converge: the red converged card with its reason and next step, the rest quiet", async () => {
    const audit = notConvergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab(false);
    await settle();

    const converged = card("converged");
    const first = audit.checks[0] as AuditCheckOut;
    expect(converged.querySelector(".verdict")?.className).toContain("verdict--fail");
    expect(converged.className).not.toContain("audit-check--quiet");
    expect(within(converged).getByText(first.title)).toBeTruthy();
    expect(within(converged).getByText("Status not_converged: maximum number of iterations exceeded")).toBeTruthy();
    expect(within(converged).getByText(RESTART)).toBeTruthy();

    const rest = cards().slice(1);
    expect(rest).toHaveLength(10);
    for (const element of rest) {
      expect(element.className).toContain("audit-check--quiet");
      expect(element.querySelector(".verdict")?.textContent).toBe("Not applicable");
      expect(within(element).getByText("the fit did not converge")).toBeTruthy();
    }

    const overall = document.querySelector(".audit-summary .verdict") as HTMLElement;
    expect(overall.textContent).toBe("Fail");
    expect(document.querySelector(".audit-summary")?.textContent).toContain(audit.summary);
  });
});

describe("FitAuditTab — the numbers behind the checks", () => {
  /** The labelled lines of a card, in order: "Statistic: …", "Threshold: …". */
  function figures(id: string): Array<string | null> {
    return Array.from(card(id).querySelectorAll(".audit-check__figure")).map(
      (line) => line.textContent,
    );
  }

  it("shows a check's statistic and threshold as labelled lines under its evidence", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    expect(figures("residual_independence")).toEqual([
      `Statistic: Q(10) = 6.4, p = 0.78`,
      `Threshold: ${P_THRESHOLD}`,
    ]);
    expect(figures("constant_variance")).toEqual(["Statistic: 0.0004"]);

    // The evidence leads; the explanation and the next step keep their places after the numbers.
    const order = Array.from(card("constant_variance").children).map((child) => child.className);
    expect(order.slice(0, 3)).toEqual([
      "audit-check__title",
      "audit-check__evidence",
      "audit-check__figure",
    ]);
    expect(order[order.length - 1]).toBe("audit-check__next");
  });

  it("leaves no line, and no label, for a check with neither a statistic nor a threshold", async () => {
    const audit = convergedAudit();
    const bare = audit.checks.find((item) => item.id === "perfect_fit") as AuditCheckOut;
    expect(bare.statistic).toBeNull();
    expect(bare.threshold).toBeNull();
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    expect(figures("perfect_fit")).toEqual([]);
    expect(card("perfect_fit").textContent).not.toMatch(/Statistic|Threshold/);
    // A check that did not apply has neither either.
    expect(figures("cross_validation")).toEqual([]);

    // Every field the server set has its one line, and nothing else has.
    const set = audit.checks.reduce(
      (count, item) => count + Number(item.statistic !== null) + Number(item.threshold !== null),
      0,
    );
    expect(set).toBeGreaterThan(0);
    expect(document.querySelectorAll(".audit-check__figure")).toHaveLength(set);
  });

  it("counts the observations and parameters the audit used", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit({ n_observations: 40, n_parameters: 3 }) });
    renderTab();
    await settle();

    const counts = document.querySelector(".audit-summary__counts");
    expect(counts?.textContent).toBe("40 observations, 3 parameters");
  });

  it("counts in en-IN digits, and says observation and parameter when there is one", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit({ n_observations: 40000, n_parameters: 1 }) });
    const first = renderTab();
    await settle();

    expect(document.querySelector(".audit-summary__counts")?.textContent).toBe(
      "40,000 observations, 1 parameter",
    );

    answer(AUDIT_PATH, { ok: convergedAudit({ n_observations: 1, n_parameters: 2 }) });
    first.unmount();
    renderTab();
    await settle();

    expect(document.querySelector(".audit-summary__counts")?.textContent).toBe(
      "1 observation, 2 parameters",
    );
  });

  it("names the rows that took no part, with their reasons, numbered from one and in row order", async () => {
    // Sent out of order: the line still reads lowest row first.
    answer(AUDIT_PATH, {
      ok: convergedAudit({
        excluded: [
          { index: 30, reason: "weight is missing" },
          { index: 4, reason: "weight is zero" },
        ],
      }),
    });
    const first = renderTab();
    await settle();

    expect(document.querySelector(".audit-summary__excluded")?.textContent).toBe(
      "2 rows took no part: row 5 (weight is zero), row 31 (weight is missing).",
    );

    answer(AUDIT_PATH, {
      ok: convergedAudit({ excluded: [{ index: 4, reason: "weight is zero" }] }),
    });
    first.unmount();
    renderTab();
    await settle();

    expect(document.querySelector(".audit-summary__excluded")?.textContent).toBe(
      "1 row took no part: row 5 (weight is zero).",
    );
  });

  it("names the first eight rows that took no part, then how many more", async () => {
    const excluded = Array.from({ length: 12 }, (_, index) => ({
      index: index * 3,
      reason: index % 2 === 0 ? "weight is zero" : "weight is missing",
    }));
    answer(AUDIT_PATH, { ok: convergedAudit({ excluded }) });
    renderTab();
    await settle();

    const line = document.querySelector(".audit-summary__excluded")?.textContent ?? "";
    expect(line).toBe(
      "12 rows took no part: " +
        ["row 1 (weight is zero)", "row 4 (weight is missing)", "row 7 (weight is zero)", "row 10 (weight is missing)", "row 13 (weight is zero)", "row 16 (weight is missing)", "row 19 (weight is zero)", "row 22 (weight is missing)"].join(", ") +
        ", and 4 more.",
    );
  });

  it("says nothing about rows when none were excluded", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit({ excluded: [] }) });
    renderTab();
    await settle();

    expect(document.querySelector(".audit-summary__excluded")).toBeNull();
    expect(document.querySelector(".audit-summary")?.textContent).not.toMatch(/took no part/);
  });

  it("shows the methods fragment nowhere: the reports carry it", async () => {
    const audit = convergedAudit({ methods_fragment: "Residuals showed no trend in x (p = 0.5)." });
    answer(AUDIT_PATH, { ok: audit });
    renderTab();
    await settle();

    expect(screen.queryByText(audit.methods_fragment)).toBeNull();
    expect(document.body.textContent).not.toContain(audit.methods_fragment);
  });
});

describe("FitAuditTab — the cross-validation control", () => {
  it("posts {folds: 5}; when the job succeeds the audit is read again and the card shows its verdict", async () => {
    const before = convergedAudit();
    const after = convergedAudit({
      checks: before.checks.map((item) =>
        item.id === "cross_validation" ? CROSS_VALIDATION_DONE : item,
      ),
      overall: "fail",
    });
    answer(AUDIT_PATH, { ok: before }, { ok: after });
    answer(RUN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", { ok: job("queued") }, { ok: job("running") }, { ok: job("succeeded") });
    renderTab();
    await settle();

    expect(within(card("cross_validation")).getByText("cross-validation has not been run")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();

    expect(posted(RUN_PATH)).toEqual({ folds: 5 });

    // While it runs: disabled, saying so, and the job can be cancelled.
    const busy = screen.getByRole("button", { name: "Running cross-validation…" }) as HTMLButtonElement;
    expect(busy.disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(statusText()).toMatch(/queued/);

    await pass(300);
    expect(times("/jobs/job-1")).toBe(1);
    expect(statusText()).toMatch(/queued/);

    await pass(600);
    expect(statusText()).toMatch(/running/);

    await pass(1200);
    expect(times("/jobs/job-1")).toBe(3);
    // The audit is read again for the fit shown: not for the job's result_ref, which for
    // this job is the dataset.
    expect(times(AUDIT_PATH)).toBe(2);
    expect([...asked.keys()]).not.toContain("/fits/ds-1/audit");

    const done = card("cross_validation");
    expect(done.className).not.toContain("audit-check--quiet");
    expect(done.querySelector(".verdict")?.textContent).toBe("Pass");
    expect(within(done).getByText(CROSS_VALIDATION_DONE.title)).toBeTruthy();
    expect(within(done).getByText(CROSS_VALIDATION_DONE.evidence)).toBeTruthy();
    expect(statusText()).toMatch(/finished/i);

    // The check is decided now, so a further run is offered as "Run again".
    const again = screen.getByRole("button", { name: "Run again" }) as HTMLButtonElement;
    expect(again.disabled).toBe(false);
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("follows the job a 409 names instead of showing an error", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(
      RUN_PATH,
      {
        refused: new ApiError(
          409,
          "CONFLICT",
          "a cross-validation of this fit is already queued or running",
          { code: "CROSS_VALIDATION_RUNNING", job_id: "job-7" },
        ),
      },
    );
    answer("/jobs/job-7", { ok: job("succeeded", { id: "job-7" }) });
    renderTab();
    await settle();

    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "Running cross-validation…" })).toBeTruthy();
    expect(times("/jobs/job-7")).toBe(0);

    await pass(300);
    expect(times("/jobs/job-7")).toBe(1);
    expect(times(AUDIT_PATH)).toBe(2);
  });

  it("shows a 409 that names no job as the error it is", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(
      RUN_PATH,
      {
        refused: new ApiError(409, "CONFLICT", "something else holds this fit", { code: "OTHER" }),
      },
    );
    renderTab();
    await settle();

    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();

    expect(screen.getByRole("alert").textContent).toBe("something else holds this fit");
    expect(runButton().disabled).toBe(false);
    expect(times(AUDIT_PATH)).toBe(1);
  });

  it("shows a refused request's reason and offers the run again", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(
      RUN_PATH,
      {
        refused: new ApiError(422, "VALIDATION_ERROR", 'folds must be at least 2, or "loo"', {
          code: "INVALID_FOLDS",
        }),
      },
    );
    renderTab();
    await settle();

    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();

    expect(screen.getByRole("alert").textContent).toBe('folds must be at least 2, or "loo"');
    const retry = runButton();
    expect(retry.disabled).toBe(false);
    expect(retry.textContent).toBe("Run cross-validation check");
  });

  it("shows a failed job's reason and keeps the audit as it was", async () => {
    const audit = convergedAudit();
    answer(AUDIT_PATH, { ok: audit });
    answer(RUN_PATH, { ok: { job_id: "job-1" } });
    answer(
      "/jobs/job-1",
      {
        ok: job("failed", {
          error_code: "INVALID_FOLDS",
          error_message: "10 folds cannot share 24 active points",
        }),
      },
    );
    renderTab();
    await settle();

    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await pass(300);

    expect(screen.getByRole("alert").textContent).toContain("10 folds cannot share 24 active points");
    expect(times(AUDIT_PATH)).toBe(1);
    expect(cards()).toHaveLength(audit.checks.length);
    expect(within(card("cross_validation")).getByText("cross-validation has not been run")).toBeTruthy();
    expect(runButton().disabled).toBe(false);
  });

  it("cancels the job through the hook, and the audit stays as it was", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(RUN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", { ok: job("running") });
    answer("/jobs/job-1/cancel", { ok: job("cancelled") });
    renderTab();
    await settle();

    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await pass(300);
    expect(statusText()).toMatch(/running/);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await settle();

    expect(times("/jobs/job-1/cancel")).toBe(1);
    expect(statusText()).toMatch(/cancelled/i);
    expect(runButton().disabled).toBe(false);

    // Polling has stopped: no further reads however long the wait.
    await pass(10_000);
    expect(times("/jobs/job-1")).toBe(1);
    expect(times(AUDIT_PATH)).toBe(1);
  });

  it("sends \"loo\" when leave one out is chosen", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(RUN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", { ok: job("succeeded") });
    renderTab();
    await settle();

    fireEvent.change(screen.getByLabelText("Folds"), { target: { value: "loo" } });
    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();
    expect(posted(RUN_PATH)).toEqual({ folds: "loo" });
  });

  it("sends {folds: 10} when ten folds are chosen", async () => {
    answer(AUDIT_PATH, { ok: convergedAudit() });
    answer(RUN_PATH, { ok: { job_id: "job-1" } });
    answer("/jobs/job-1", { ok: job("succeeded") });
    renderTab();
    await settle();

    fireEvent.change(screen.getByLabelText("Folds"), { target: { value: "10" } });
    fireEvent.click(screen.getByRole("button", { name: "Run cross-validation check" }));
    await settle();
    expect(posted(RUN_PATH)).toEqual({ folds: 10 });
  });

  it("does not offer cross-validation for a fit that did not converge, and says why", async () => {
    answer(AUDIT_PATH, { ok: notConvergedAudit() });
    renderTab(false);
    await settle();

    expect(runButton().disabled).toBe(true);
    expect(control().textContent).toMatch(
      /cross-validation refits the model, and a fit that did not converge has none to refit/i,
    );
    fireEvent.click(runButton());
    await settle();
    expect(times(RUN_PATH)).toBe(0);
  });
});
