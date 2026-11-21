import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { ApiError, api } from "../../api/client";
import type { components } from "../../api/schema";
import { useAutofit } from "./useAutofit";

type JobOut = components["schemas"]["JobOut"];
type ProgressOut = components["schemas"]["ProgressOut"];
type AutofitSummaryOut = components["schemas"]["AutofitSummaryOut"];
type AutofitOut = components["schemas"]["AutofitOut"];
type RankedFit = components["schemas"]["RankedFit"];

function progress(overrides: Record<string, unknown> = {}): ProgressOut {
  return {
    done: 2,
    total: 5,
    converged: 1,
    failed: 1,
    skipped: 0,
    elapsed_s: 1.5,
    partial: false,
    top: [],
    ...overrides,
  } as ProgressOut;
}

function row(modelId: string, rank: number, overrides: Record<string, unknown> = {}): RankedFit {
  return {
    rank,
    model_id: modelId,
    name: modelId,
    metrics: { k: 2, n: 40, r2: 0.99, adj_r2: null, rmse: 0.1, aicc: null, bic: null, cv_rmse: null },
    ...overrides,
  } as RankedFit;
}

function autofitOut(overrides: Record<string, unknown> = {}): AutofitOut {
  return {
    id: "af-1",
    dataset_id: "ds-1",
    created_at: "2026-11-17T10:00:00Z",
    best_model_id: "exp_decay",
    n_candidates: 5,
    rank_by: "best_overall",
    partial: false,
    result: {
      config: { rank_by: "best_overall" } as unknown as AutofitOut["result"]["config"],
      dataset_fingerprint: "fp",
      n_active: 40,
      ranked: [row("exp_decay", 1)],
      outcomes: [],
      skipped: {},
      unconverged: {},
      exact_fits: [],
      warnings: [],
      cancelled: false,
      elapsed_seconds: 3.2,
      curves: {},
      overfitting_note: null,
    },
    ...overrides,
  } as unknown as AutofitOut;
}

function job(status: string, overrides: Record<string, unknown> = {}): JobOut {
  return {
    id: "job-1",
    kind: "autofit",
    status,
    dataset_id: "ds-1",
    spec: {},
    result_ref: null,
    error_code: null,
    error_message: null,
    created_at: "2026-11-17T10:00:00Z",
    started_at: null,
    finished_at: null,
    ...overrides,
  } as JobOut;
}

function apiError(status: number, code: string, message: string, detail: unknown = null): ApiError {
  return new ApiError(status, code, message, detail);
}

let request: MockInstance;

beforeEach(() => {
  request = vi.spyOn(api, "request");
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useAutofit — the auto-fit state machine", () => {
  it("auto-starts a run when the dataset has none, polls, and loads the stored result", async () => {
    request
      // ensure → no stored runs
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      // POST autofit
      .mockResolvedValueOnce({ job_id: "job-1" })
      // polls
      .mockResolvedValueOnce(job("queued"))
      .mockResolvedValueOnce(job("running", { progress: progress() }))
      .mockResolvedValueOnce(job("succeeded", { result_id: "af-1", result_ref: "af-1" }))
      // load stored
      .mockResolvedValueOnce(autofitOut());
    const { result } = renderHook(() => useAutofit());

    act(() => result.current.ensure("ds-1"));
    // POST happened right after the empty list
    await act(async () => void 0);
    expect(request).toHaveBeenNthCalledWith(2, "/datasets/ds-1/autofit", {
      method: "POST",
      body: expect.any(String),
    });
    expect(result.current.stateFor("ds-1").phase).toBe("running");

    await act(async () => vi.advanceTimersByTimeAsync(300)); // queued
    await act(async () => vi.advanceTimersByTimeAsync(600)); // running + progress
    const state = result.current.stateFor("ds-1");
    expect(state.phase).toBe("running");
    expect(state.progress?.done).toBe(2);
    expect(state.progress?.total).toBe(5);

    await act(async () => vi.advanceTimersByTimeAsync(1200)); // succeeded
    expect(request).toHaveBeenLastCalledWith("/autofits/af-1");
    expect(result.current.stateFor("ds-1").phase).toBe("ready");
    expect(result.current.stateFor("ds-1").stored?.id).toBe("af-1");
    // polling stopped
    const calls = request.mock.calls.length;
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(request.mock.calls.length).toBe(calls);
  });

  it("does not start a run when one is already stored; shows it instead", async () => {
    request
      .mockResolvedValueOnce([
        { id: "af-existing", dataset_id: "ds-1", created_at: "2026-11-17T09:00:00Z", best_model_id: "exp_decay", n_candidates: 9, rank_by: "best_overall" },
      ] as AutofitSummaryOut[])
      .mockResolvedValueOnce(autofitOut({ id: "af-existing" }));
    const { result } = renderHook(() => useAutofit());

    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(result.current.stateFor("ds-1").phase).toBe("ready");
    expect(result.current.stateFor("ds-1").stored?.id).toBe("af-existing");
    const post = request.mock.calls.some(
      (call) => typeof call[0] === "string" && call[0] === "/datasets/ds-1/autofit",
    );
    expect(post).toBe(false);
  });

  it("shows a failed job with its message", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("failed", { error_message: "the worker crashed" }));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));
    const state = result.current.stateFor("ds-1");
    expect(state.phase).toBe("error");
    expect(state.error).toContain("the worker crashed");
  });

  it("surfaces TOO_FEW_POINTS from the start request's detail.code", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockRejectedValueOnce(
        apiError(422, "VALIDATION_ERROR", "the dataset cannot be auto-fitted", {
          code: "TOO_FEW_POINTS",
        }),
      );
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    const state = result.current.stateFor("ds-1");
    expect(state.phase).toBe("error");
    expect(state.errorCode).toBe("TOO_FEW_POINTS");
  });

  it("surfaces a quota rejection (429 QUOTA_EXCEEDED)", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockRejectedValueOnce(
        apiError(429, "QUOTA_EXCEEDED", "too many auto-fits today; try again tomorrow"),
      );
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    const state = result.current.stateFor("ds-1");
    expect(state.phase).toBe("error");
    expect(state.errorCode).toBe("QUOTA_EXCEEDED");
  });

  it("cancel keeps the ranked rows seen so far as a partial view", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("running", { progress: progress({ top: [row("exp_decay", 1)] }) }))
      // cancel response: cancelled job that keeps its progress snapshot
      .mockResolvedValueOnce(
        job("cancelled", { progress: progress({ top: [row("exp_decay", 1)] }) }),
      );
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300)); // running

    await act(async () => result.current.cancel("ds-1"));
    const state = result.current.stateFor("ds-1");
    expect(state.phase).toBe("ready");
    expect(state.cancelled).toBe(true);
    expect(state.partialRows?.[0]?.model_id).toBe("exp_decay");
    expect(state.stored).toBeNull();
  });

  it("re-ranks through the API without refitting when rank_by changes", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("succeeded", { result_id: "af-1", result_ref: "af-1" }))
      .mockResolvedValueOnce(autofitOut())
      .mockResolvedValueOnce(
        autofitOut({ rank_by: "aicc", result: { ...autofitOut().result, ranked: [row("poly3", 1)] } }),
      );
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));

    await act(async () => result.current.rerank("ds-1", "aicc"));
    expect(request).toHaveBeenLastCalledWith("/autofits/af-1?rank_by=aicc");
    const state = result.current.stateFor("ds-1");
    expect(state.rankBy).toBe("aicc");
    expect(state.stored?.result.ranked[0]?.model_id).toBe("poly3");
  });

  it("run() always starts a new run with the given options (Run again)", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("succeeded", { result_id: "af-1", result_ref: "af-1" }))
      .mockResolvedValueOnce(autofitOut())
      .mockResolvedValueOnce({ job_id: "job-2" })
      .mockResolvedValueOnce(job("queued"));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));

    act(() =>
      result.current.run("ds-1", {
        families: ["Exponential"],
        exhaustive: true,
        maxParameters: "6",
        nStarts: 5,
        cvFolds: 10,
        seed: "42",
      }),
    );
    await act(async () => void 0);
    const body = JSON.parse(
      String(
        [...request.mock.calls].reverse().find((c) => c[0] === "/datasets/ds-1/autofit")?.[1]?.body,
      ),
    ) as Record<string, unknown>;
    expect(body).toEqual({
      rank_by: "best_overall",
      families: ["Exponential"],
      exhaustive: true,
      max_parameters: 6,
      n_starts: 5,
      cv_folds: 10,
      seed: 42,
    });
    expect(result.current.stateFor("ds-1").phase).toBe("running");
  });

  it("keeps datasets independent: a run on one does not disturb another", async () => {
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[]) // ds-1 list
      .mockResolvedValueOnce({ job_id: "job-1" }) // ds-1 POST
      .mockResolvedValueOnce(job("running", { progress: progress() }))
      .mockResolvedValueOnce([
        { id: "af-2", dataset_id: "ds-2", created_at: "2026-11-17T09:00:00Z", best_model_id: "m", n_candidates: 3, rank_by: "best_overall" },
      ] as AutofitSummaryOut[]) // ds-2 list
      .mockResolvedValueOnce(autofitOut({ id: "af-2", dataset_id: "ds-2" }));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300)); // ds-1 running

    act(() => result.current.ensure("ds-2"));
    await act(async () => vi.advanceTimersByTimeAsync(0));
    expect(result.current.stateFor("ds-2").phase).toBe("ready");
    expect(result.current.stateFor("ds-1").phase).toBe("running");
  });

  it("an out-of-date ensure response never overwrites a newer run", async () => {
    // ds-1 has a stored run (slow response); the user hits Run again meanwhile.
    let resolveList: (value: unknown) => void = () => {};
    request
      .mockImplementationOnce(
        () => new Promise((resolve) => void (resolveList = resolve)),
      )
      .mockResolvedValueOnce({ job_id: "job-2" })
      .mockResolvedValueOnce(job("queued"));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    act(() => result.current.run("ds-1"));
    await act(async () => void 0);
    expect(result.current.stateFor("ds-1").phase).toBe("running");
    // The stale list (with an old stored run) arrives now; it must be ignored.
    await act(async () => {
      resolveList([]);
    });
    expect(result.current.stateFor("ds-1").phase).toBe("running");
  });

  it("does not start a second run when ensure is called again while the first is starting", async () => {
    let resolveStart: (value: unknown) => void = () => {};
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockImplementationOnce(() => new Promise((resolve) => void (resolveStart = resolve)));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => void 0);
    expect(request).toHaveBeenCalledTimes(2); // the list, then the start POST

    act(() => result.current.ensure("ds-1"));
    await act(async () => void 0);
    // The second ensure makes no request at all — the first flow owns the dataset.
    expect(request).toHaveBeenCalledTimes(2);
    expect(result.current.stateFor("ds-1").phase).toBe("starting");
    const posts = request.mock.calls.filter(
      (call) => typeof call[0] === "string" && call[0] === "/datasets/ds-1/autofit",
    );
    expect(posts).toHaveLength(1);
    await act(async () => {
      resolveStart({ job_id: "job-1" });
    });
  });

  it("a poll from a replaced run never overwrites the newer run's state", async () => {
    let resolvePoll: (value: unknown) => void = () => {};
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("queued", { progress: null }))
      .mockImplementationOnce(() => new Promise((resolve) => void (resolvePoll = resolve)))
      .mockResolvedValueOnce({ job_id: "job-2" });
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => void 0); // list + POST → job-1 in flight
    await act(async () => vi.advanceTimersByTimeAsync(300)); // poll 1 (queued)
    await act(async () => vi.advanceTimersByTimeAsync(600)); // poll 2 is now in flight
    expect(result.current.stateFor("ds-1").jobId).toBe("job-1");

    act(() => result.current.run("ds-1"));
    await act(async () => void 0); // POST job-2 resolves
    expect(result.current.stateFor("ds-1").jobId).toBe("job-2");

    // The old poll finally answers — it must not resurrect run 1.
    await act(async () => {
      resolvePoll(job("running", { progress: progress({ done: 4 }) }));
    });
    await act(async () => void 0);
    const state = result.current.stateFor("ds-1");
    expect(state.jobId).toBe("job-2");
    expect(state.phase).toBe("running");
    expect(state.progress).toBeNull();
  });

  it("does not cancel a job when a new run is still starting (no job id yet)", async () => {
    let resolveStart: (value: unknown) => void = () => {};
    request
      .mockResolvedValueOnce([] as AutofitSummaryOut[])
      .mockResolvedValueOnce({ job_id: "job-1" })
      .mockResolvedValueOnce(job("succeeded", { result_id: "af-1", result_ref: "af-1" }))
      .mockResolvedValueOnce(autofitOut())
      // the new run's start POST is still pending when the user hits Cancel
      .mockImplementationOnce(() => new Promise((resolve) => void (resolveStart = resolve)));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    await act(async () => void 0);
    await act(async () => vi.advanceTimersByTimeAsync(300)); // job-1 succeeded, stored loaded
    expect(result.current.stateFor("ds-1").phase).toBe("ready");

    act(() => result.current.run("ds-1"));
    await act(async () => void 0);
    expect(result.current.stateFor("ds-1").phase).toBe("starting");
    const callsBefore = request.mock.calls.length;
    await act(async () => result.current.cancel("ds-1"));
    const cancels = request.mock.calls
      .slice(callsBefore)
      .filter((call) => typeof call[0] === "string" && call[0].endsWith("/cancel"));
    expect(cancels).toHaveLength(0);
    await act(async () => {
      resolveStart({ job_id: "job-2" });
    });
  });

  it("clearing a dataset while ensure is in flight makes the response harmless", async () => {
    let resolveList: (value: unknown) => void = () => {};
    request.mockImplementationOnce(() => new Promise((resolve) => void (resolveList = resolve)));
    const { result } = renderHook(() => useAutofit());
    act(() => result.current.ensure("ds-1"));
    act(() => result.current.clear("ds-1"));
    expect(result.current.stateFor("ds-1").phase).toBe("idle");

    await act(async () => {
      resolveList([]);
    });
    expect(result.current.stateFor("ds-1").phase).toBe("idle");
    const posts = request.mock.calls.filter(
      (call) => typeof call[0] === "string" && call[0] === "/datasets/ds-1/autofit",
    );
    expect(posts).toHaveLength(0);
  });
});
