import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { api } from "../api/client";
import { useJobPolling } from "./useJobPolling";

function job(status: string, overrides: Record<string, unknown> = {}) {
  return {
    id: "job-1",
    kind: "fit",
    status,
    dataset_id: "ds-1",
    spec: {},
    result_ref: null,
    error_code: null,
    error_message: null,
    created_at: "2026-11-17T05:00:00Z",
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

// The global setup restores all spies after each test, so re-spy per test.
let getJob: MockInstance;

beforeEach(() => {
  getJob = vi.spyOn(api, "request");
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useJobPolling", () => {
  it("polls with 300ms→2s backoff until the job succeeds, then stops", async () => {
    getJob
      .mockResolvedValueOnce(job("queued"))
      .mockResolvedValueOnce(job("running"))
      .mockResolvedValueOnce(job("succeeded", { result_ref: "fit-9" }));
    const onFinished = vi.fn();
    const { result } = renderHook(() => useJobPolling(onFinished));

    act(() => result.current.start("job-1"));
    expect(result.current.state.phase).toBe("polling");

    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(getJob.mock.calls[0]?.[0]).toBe("/jobs/job-1");
    expect(result.current.state.status).toBe("queued");

    await act(async () => vi.advanceTimersByTimeAsync(600));
    expect(result.current.state.status).toBe("running");

    await act(async () => vi.advanceTimersByTimeAsync(1200));
    expect(result.current.state.phase).toBe("succeeded");
    expect(result.current.state.fitId).toBe("fit-9");
    expect(onFinished).toHaveBeenCalledWith("fit-9");

    // polling has stopped: no further GETs even after a long wait
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(getJob).toHaveBeenCalledTimes(3);
  });

  it("caps the backoff at 2 seconds", async () => {
    getJob.mockResolvedValue(job("running"));
    const { result } = renderHook(() => useJobPolling(vi.fn()));
    act(() => result.current.start("job-1"));

    await act(async () => vi.advanceTimersByTimeAsync(300)); // poll 1 at 300ms
    await act(async () => vi.advanceTimersByTimeAsync(600)); // poll 2 at 900ms
    await act(async () => vi.advanceTimersByTimeAsync(1200)); // poll 3 at 2100ms
    await act(async () => vi.advanceTimersByTimeAsync(2000)); // poll 4 at 4100ms
    await act(async () => vi.advanceTimersByTimeAsync(2000)); // poll 5 at 6100ms
    expect(getJob).toHaveBeenCalledTimes(5);
    // no poll before the 2s cap elapses
    await act(async () => vi.advanceTimersByTimeAsync(1999));
    expect(getJob).toHaveBeenCalledTimes(5);
  });

  it("shows a failed job with its error message and stops", async () => {
    getJob.mockResolvedValue(
      job("failed", { error_code: "FIT_ERROR", error_message: "worker died" }),
    );
    const { result } = renderHook(() => useJobPolling(vi.fn()));
    act(() => result.current.start("job-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(result.current.state.phase).toBe("failed");
    expect(result.current.state.message).toContain("worker died");
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(getJob).toHaveBeenCalledTimes(1);
  });

  it("cancels a running job and stops polling", async () => {
    getJob.mockResolvedValueOnce(job("running")).mockResolvedValueOnce(job("cancelled"));
    const { result } = renderHook(() => useJobPolling(vi.fn()));
    act(() => result.current.start("job-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));

    await act(async () => result.current.cancel());
    expect(result.current.state.phase).toBe("cancelled");
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(getJob).toHaveBeenCalledTimes(2); // one poll + one cancel POST
  });

  it("keeps polling through a transient network failure", async () => {
    getJob
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce(job("succeeded", { result_ref: "fit-2" }));
    const onFinished = vi.fn();
    const { result } = renderHook(() => useJobPolling(onFinished));
    act(() => result.current.start("job-1"));
    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(result.current.state.phase).toBe("polling");
    await act(async () => vi.advanceTimersByTimeAsync(600));
    expect(result.current.state.phase).toBe("succeeded");
    expect(onFinished).toHaveBeenCalledWith("fit-2");
  });

  it("gives up to an error state after repeated network failures", async () => {
    getJob.mockRejectedValue(new Error("network down"));
    const { result } = renderHook(() => useJobPolling(vi.fn()));
    act(() => result.current.start("job-1"));
    for (let i = 0; i < 5; i += 1) {
      await act(async () => vi.advanceTimersByTimeAsync(5000));
    }
    expect(result.current.state.phase).toBe("error");
    expect(result.current.state.message).toContain("server");
  });
});
