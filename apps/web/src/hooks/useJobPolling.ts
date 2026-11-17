/**
 * The fit-job state machine: POST /datasets/{id}/fits returns a job id; this hook polls
 * GET /jobs/{id} with exponential backoff (300 ms doubling to a 2 s cap) until the job
 * reaches a terminal status, and can cancel it. Transient network errors keep polling;
 * five in a row (or any API error) surface as an error state instead of hanging.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type JobOut } from "../api/client";

const MIN_DELAY_MS = 300;
const MAX_DELAY_MS = 2000;
const MAX_CONSECUTIVE_NETWORK_FAILURES = 5;

export type JobPhase = "idle" | "polling" | "succeeded" | "failed" | "cancelled" | "error";

export interface JobState {
  phase: JobPhase;
  /** Last status seen from the server (queued / running / …). */
  status?: JobOut["status"];
  /** The fit id, once the job succeeded. */
  fitId?: string;
  /** Why the job failed or polling gave up. */
  message?: string;
}

export function useJobPolling(onFinished: (fitId: string) => void): {
  state: JobState;
  start: (jobId: string) => void;
  cancel: () => Promise<void>;
  reset: () => void;
} {
  const [state, setState] = useState<JobState>({ phase: "idle" });
  const timerRef = useRef<number | null>(null);
  const jobIdRef = useRef<string | null>(null);
  const delayRef = useRef(MIN_DELAY_MS);
  const failuresRef = useRef(0);
  const stoppedRef = useRef(false);
  const finishedRef = useRef(onFinished);
  finishedRef.current = onFinished;

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    stoppedRef.current = true;
    clearTimer();
  }, [clearTimer]);

  // Latest-ref indirection: `schedule` fires `poll` from a timer, and `poll` itself
  // depends on `schedule` — a direct closure here would be a use-before-declaration.
  const pollRef = useRef<() => Promise<void>>(async () => {});

  const schedule = useCallback(
    (delay: number) => {
      delayRef.current = delay;
      clearTimer();
      timerRef.current = window.setTimeout(() => void pollRef.current(), delay);
    },
    [clearTimer],
  );

  const poll = useCallback(async () => {
    const jobId = jobIdRef.current;
    if (jobId === null || stoppedRef.current) return;
    try {
      const job = await api.request<JobOut>(`/jobs/${jobId}`);
      if (stoppedRef.current || jobIdRef.current !== jobId) return;
      failuresRef.current = 0;
      switch (job.status) {
        case "queued":
        case "running":
          setState({ phase: "polling", status: job.status });
          schedule(Math.min(delayRef.current * 2, MAX_DELAY_MS));
          return;
        case "succeeded":
          stop();
          if (job.result_ref !== null) {
            setState({ phase: "succeeded", status: job.status, fitId: job.result_ref });
            finishedRef.current(job.result_ref);
          } else {
            setState({ phase: "error", message: "the job finished but returned no fit" });
          }
          return;
        case "failed":
          stop();
          setState({
            phase: "failed",
            status: job.status,
            message: job.error_message ?? "the fit failed",
          });
          return;
        case "cancelled":
          stop();
          setState({ phase: "cancelled", status: job.status });
          return;
      }
    } catch (error) {
      if (stoppedRef.current) return;
      if (error instanceof ApiError && error.status !== 0) {
        stop();
        setState({ phase: "error", message: error.message });
        return;
      }
      failuresRef.current += 1;
      if (failuresRef.current >= MAX_CONSECUTIVE_NETWORK_FAILURES) {
        stop();
        setState({
          phase: "error",
          message: "lost contact with the server while the job was running",
        });
        return;
      }
      schedule(Math.min(delayRef.current * 2, MAX_DELAY_MS));
    }
  }, [schedule, stop]);
  pollRef.current = poll;

  const start = useCallback(
    (jobId: string) => {
      stop();
      stoppedRef.current = false;
      jobIdRef.current = jobId;
      delayRef.current = MIN_DELAY_MS;
      failuresRef.current = 0;
      setState({ phase: "polling", status: "queued" });
      schedule(MIN_DELAY_MS);
    },
    [schedule, stop],
  );

  const cancel = useCallback(async () => {
    const jobId = jobIdRef.current;
    if (jobId === null || stoppedRef.current) return;
    try {
      const job = await api.request<JobOut>(`/jobs/${jobId}/cancel`, { method: "POST" });
      stop();
      setState({ phase: "cancelled", status: job.status });
    } catch (error) {
      // 409 means the job finished between our poll and the cancel: the next poll (or
      // the terminal state it already set) wins; anything else is a real error.
      if (error instanceof ApiError && error.status === 409) return;
      if (!(error instanceof ApiError) || error.status === 0) return; // transient; keep polling
      stop();
      setState({ phase: "error", message: error.message });
    }
  }, [stop]);

  const reset = useCallback(() => {
    stop();
    stoppedRef.current = false;
    jobIdRef.current = null;
    setState({ phase: "idle" });
  }, [stop]);

  useEffect(() => stop, [stop]);

  return { state, start, cancel, reset };
}
