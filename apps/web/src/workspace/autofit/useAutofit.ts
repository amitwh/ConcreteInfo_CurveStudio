/**
 * The Auto-fit state machine, one per dataset. `ensure` is the auto-start entry point
 * the workspace calls after a dataset is uploaded or selected: it starts a run when the
 * dataset has none, and otherwise loads the latest stored one. A run is a job polled
 * with 300 ms → 1.5 s backoff; `progress` (done/total, counts, elapsed, live top-5)
 * streams into the state until the job succeeds (the stored run is fetched then),
 * fails, or is cancelled — a cancelled run keeps the rows ranked so far as a partial
 * view. Datasets are independent: polling for one never blocks another, and a stale
 * response never overwrites a newer run for the same dataset.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type AutofitOut,
  type AutofitSummaryOut,
  type JobOut,
  type ProgressOut,
  type RankedFit,
  type RankBy,
} from "../../api/client";
import type { WorkspaceMode } from "../mode";
import { invalidateFits } from "../workspaceCache";

const MIN_DELAY_MS = 300;
const MAX_DELAY_MS = 1500;
const MAX_CONSECUTIVE_NETWORK_FAILURES = 5;

export type AutofitPhase = "idle" | "checking" | "starting" | "running" | "loading" | "ready" | "error";

export interface AutofitState {
  phase: AutofitPhase;
  /** The Auto-fit job id while one is in flight. */
  jobId: string | null;
  /** The last progress snapshot (counts and the live top rows). */
  progress: ProgressOut | null;
  /** The loaded stored run, ranked by `rankBy`. */
  stored: AutofitOut | null;
  /** The ordering `stored` is ranked by. */
  rankBy: RankBy;
  /** Why the run could not start or the poll gave up (user-facing message). */
  error: string | null;
  /** Machine code for the failure (TOO_FEW_POINTS, QUOTA_EXCEEDED, …). */
  errorCode: string | null;
  /** True when the shown rows come from a cancelled run (nothing was stored). */
  cancelled: boolean;
  /** Rows ranked before a cancel; shown with a partial notice. */
  partialRows: RankedFit[] | null;
}

export const AUTOFIT_IDLE: AutofitState = {
  phase: "idle",
  jobId: null,
  progress: null,
  stored: null,
  rankBy: "best_overall",
  error: null,
  errorCode: null,
  cancelled: false,
  partialRows: null,
};

/** Options the workspace collects; `buildAutofitRequest` turns them into the body. */
export interface AutoFitOptions {
  /** Empty means every family (the request omits the field). */
  families: string[];
  exhaustive: boolean;
  /** Blank string means the server default. */
  maxParameters: string;
  nStarts: number;
  cvFolds: number;
  seed: string;
}

export const DEFAULT_AUTOFIT_OPTIONS: AutoFitOptions = {
  families: [],
  exhaustive: false,
  maxParameters: "",
  nStarts: 3,
  cvFolds: 5,
  seed: "",
};

/** Build the POST body, omitting everything left at its default. */
export function buildAutofitRequest(
  options: AutoFitOptions,
  rankBy: RankBy,
): Record<string, unknown> {
  const body: Record<string, unknown> = { rank_by: rankBy };
  if (options.families.length > 0) body.families = options.families;
  if (options.exhaustive) body.exhaustive = true;
  const maxParameters = Number.parseInt(options.maxParameters, 10);
  if (Number.isSafeInteger(maxParameters) && maxParameters >= 1 && maxParameters <= 10) {
    body.max_parameters = maxParameters;
  }
  if (options.nStarts !== 3) body.n_starts = options.nStarts;
  if (options.cvFolds !== 5) body.cv_folds = options.cvFolds;
  const seed = Number.parseInt(options.seed, 10);
  if (options.seed.trim() !== "" && Number.isSafeInteger(seed) && seed >= 0) body.seed = seed;
  return body;
}

interface Flight {
  datasetId: string;
  jobId: string;
  /** The dataset token that started this flight; a newer token makes it stale. */
  token: number;
  delay: number;
  failures: number;
}

function failureOf(error: unknown): { message: string; code: string | null } {
  if (error instanceof ApiError) {
    const detail = error.detail as { code?: string } | null;
    return { message: error.message, code: detail?.code ?? error.code };
  }
  return { message: "the run could not be started; check your connection", code: null };
}

export function useAutofit(): {
  stateFor: (datasetId: string | null) => AutofitState;
  ensure: (datasetId: string, mode?: WorkspaceMode) => void;
  run: (datasetId: string, options?: AutoFitOptions) => void;
  cancel: (datasetId: string) => Promise<void>;
  rerank: (datasetId: string, rankBy: RankBy) => Promise<void>;
  clear: (datasetId: string) => void;
} {
  const [states, setStates] = useState<Record<string, AutofitState>>({});
  const flights = useRef(new Map<string, Flight>()); // keyed by job id
  const timers = useRef(new Map<string, number>()); // keyed by job id
  /** Monotonic token per dataset: only the newest flow may apply its result. */
  const tokens = useRef<Record<string, number>>({});
  /** Latest state for callbacks that must not wait for a re-render. */
  const statesRef = useRef(states);
  statesRef.current = states;

  const patch = useCallback((datasetId: string, changes: Partial<AutofitState>): void => {
    setStates((current) => ({
      ...current,
      [datasetId]: { ...(current[datasetId] ?? AUTOFIT_IDLE), ...changes },
    }));
  }, []);

  const clearTimer = useCallback((jobId: string): void => {
    const timer = timers.current.get(jobId);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      timers.current.delete(jobId);
    }
  }, []);

  const stopFlight = useCallback(
    (jobId: string): void => {
      flights.current.delete(jobId);
      clearTimer(jobId);
    },
    [clearTimer],
  );

  const loadStored = useCallback(
    async (datasetId: string, autofitId: string, rankBy: RankBy, token: number): Promise<void> => {
      patch(datasetId, { phase: "loading" });
      try {
        const query = rankBy === "best_overall" ? "" : `?rank_by=${rankBy}`;
        const stored = await api.request<AutofitOut>(`/autofits/${autofitId}${query}`);
        if (tokens.current[datasetId] !== token) return; // a newer flow took over
        patch(datasetId, {
          phase: "ready",
          stored,
          rankBy: stored.rank_by,
          error: null,
          errorCode: null,
          cancelled: false,
          partialRows: null,
        });
      } catch (error) {
        if (tokens.current[datasetId] !== token) return;
        const failure = failureOf(error);
        patch(datasetId, { phase: "error", error: failure.message, errorCode: failure.code });
      }
    },
    [patch],
  );

  // Latest-ref indirection (the pattern of hooks/useJobPolling): `schedule` fires `poll`
  // from a timer and `poll` itself calls `schedule`.
  const pollRef = useRef<(jobId: string) => Promise<void>>(async () => {});

  const schedule = useCallback(
    (jobId: string, delay: number): void => {
      const flight = flights.current.get(jobId);
      if (flight === undefined) return;
      flight.delay = delay;
      clearTimer(jobId);
      timers.current.set(jobId, window.setTimeout(() => void pollRef.current(jobId), delay));
    },
    [clearTimer],
  );

  const poll = useCallback(
    async (jobId: string): Promise<void> => {
      const flight = flights.current.get(jobId);
      if (flight === undefined) return;
      const { datasetId } = flight;
      // A newer flow for this dataset owns the state now; this flight must not patch it.
      if (flight.token !== (tokens.current[datasetId] ?? 0)) {
        stopFlight(jobId);
        return;
      }
      try {
        const job = await api.request<JobOut>(`/jobs/${jobId}`);
        if (flights.current.get(jobId) !== flight) return;
        if (flight.token !== (tokens.current[datasetId] ?? 0)) {
          stopFlight(jobId);
          return;
        }
        flight.failures = 0;
        if (job.progress !== null) patch(datasetId, { progress: job.progress });
        switch (job.status) {
          case "queued":
          case "running":
            patch(datasetId, { phase: "running", jobId });
            schedule(jobId, Math.min(flight.delay * 2, MAX_DELAY_MS));
            return;
          case "succeeded": {
            stopFlight(jobId);
            // A finished run is treated as a change to the dataset's fit history: the
            // cached fits list is dropped, so the next mount revalidates it (one small
            // GET; the run itself stores no fits, but its rows become fits when opened).
            invalidateFits(datasetId);
            const autofitId = job.result_id ?? job.result_ref;
            if (autofitId === null) {
              patch(datasetId, {
                phase: "error",
                error: "the run finished but stored no result",
                errorCode: null,
              });
              return;
            }
            const rankBy = states[datasetId]?.rankBy ?? "best_overall";
            await loadStored(datasetId, autofitId, rankBy, flight.token);
            return;
          }
          case "failed":
            stopFlight(jobId);
            patch(datasetId, {
              phase: "error",
              error: job.error_message ?? "the run failed",
              errorCode: job.error_code,
            });
            return;
          case "cancelled":
            stopFlight(jobId);
            patch(datasetId, {
              phase: "ready",
              stored: null,
              cancelled: true,
              partialRows: job.progress?.top ?? null,
            });
            return;
        }
      } catch (error) {
        if (flights.current.get(jobId) !== flight) return;
        if (error instanceof ApiError && error.status !== 0) {
          stopFlight(jobId);
          patch(datasetId, { phase: "error", error: error.message, errorCode: error.code });
          return;
        }
        flight.failures += 1;
        if (flight.failures >= MAX_CONSECUTIVE_NETWORK_FAILURES) {
          stopFlight(jobId);
          patch(datasetId, {
            phase: "error",
            error: "lost contact with the server while the run was going",
            errorCode: null,
          });
          return;
        }
        schedule(jobId, Math.min(flight.delay * 2, MAX_DELAY_MS));
      }
    },
    [loadStored, patch, schedule, states, stopFlight],
  );
  pollRef.current = poll;

  const startRun = useCallback(
    (datasetId: string, options: AutoFitOptions | undefined, token: number): void => {
      const rankBy = statesRef.current[datasetId]?.rankBy ?? "best_overall";
      const body = buildAutofitRequest(options ?? DEFAULT_AUTOFIT_OPTIONS, rankBy);
      patch(datasetId, {
        phase: "starting",
        // The previous run's job id is gone: cancel must never target it.
        jobId: null,
        error: null,
        errorCode: null,
        cancelled: false,
        partialRows: null,
      });
      void api
        .request<{ job_id: string }>(`/datasets/${datasetId}/autofit`, {
          method: "POST",
          body: JSON.stringify(body),
        })
        .then((created) => {
          if (tokens.current[datasetId] !== token) return;
          flights.current.set(created.job_id, {
            datasetId,
            jobId: created.job_id,
            token,
            delay: MIN_DELAY_MS,
            failures: 0,
          });
          patch(datasetId, { phase: "running", jobId: created.job_id });
          schedule(created.job_id, MIN_DELAY_MS);
        })
        .catch((error: unknown) => {
          if (tokens.current[datasetId] !== token) return;
          const failure = failureOf(error);
          patch(datasetId, { phase: "error", error: failure.message, errorCode: failure.code });
        });
    },
    [patch, schedule],
  );

  // `ensure` is the auto-start entry point: a run when the dataset has none, otherwise
  // the latest stored one (the workspace calls it after upload and after selection). It
  // starts nothing outside Auto-fit mode: Statistics and Manual fit must not spend a run.

  const ensure = useCallback(
    (datasetId: string, mode: WorkspaceMode = "auto"): void => {
      if (mode !== "auto") return;
      // Something is already in flight for this dataset (re-selection, a double click):
      // leave it alone — starting again would fork a second run.
      const phase = statesRef.current[datasetId]?.phase;
      if (
        phase === "checking" ||
        phase === "starting" ||
        phase === "running" ||
        phase === "loading" ||
        [...flights.current.values()].some((flight) => flight.datasetId === datasetId)
      ) {
        return;
      }
      const token = (tokens.current[datasetId] ?? 0) + 1;
      tokens.current[datasetId] = token;
      patch(datasetId, { phase: "checking", error: null, errorCode: null });
      void api
        .request<AutofitSummaryOut[]>(`/datasets/${datasetId}/autofits`)
        .then((list) => {
          if (tokens.current[datasetId] !== token) return;
          if (list.length === 0) {
            startRun(datasetId, undefined, token);
            return;
          }
          const rankBy = statesRef.current[datasetId]?.rankBy ?? "best_overall";
          const latest = list[0];
          if (latest !== undefined) void loadStored(datasetId, latest.id, rankBy, token);
        })
        .catch((error: unknown) => {
          if (tokens.current[datasetId] !== token) return;
          const failure = failureOf(error);
          patch(datasetId, { phase: "error", error: failure.message, errorCode: failure.code });
        });
    },
    [loadStored, patch, startRun],
  );

  const run = useCallback(
    (datasetId: string, options?: AutoFitOptions): void => {
      const token = (tokens.current[datasetId] ?? 0) + 1;
      tokens.current[datasetId] = token;
      // A run in flight for this dataset is replaced by the new one.
      for (const flight of [...flights.current.values()]) {
        if (flight.datasetId === datasetId) stopFlight(flight.jobId);
      }
      startRun(datasetId, options, token);
    },
    [startRun, stopFlight],
  );

  const cancel = useCallback(
    async (datasetId: string): Promise<void> => {
      const state = statesRef.current[datasetId];
      // Only a running job can be cancelled: while a run is still starting there is no
      // job id yet, and a finished run's id must never be targeted.
      if (state?.phase !== "running" || state.jobId == null) return;
      const jobId = state.jobId;
      try {
        const job = await api.request<JobOut>(`/jobs/${jobId}/cancel`, { method: "POST" });
        stopFlight(jobId);
        patch(datasetId, {
          phase: "ready",
          stored: null,
          cancelled: true,
          partialRows: job.progress?.top ?? state.progress?.top ?? null,
        });
      } catch (error) {
        // 409: the job finished between our poll and the cancel — the poller's terminal
        // state wins. A network blip keeps polling; anything else is a real error.
        if (error instanceof ApiError && error.status === 409) return;
        if (!(error instanceof ApiError) || error.status === 0) return;
        stopFlight(jobId);
        patch(datasetId, { phase: "error", error: error.message, errorCode: error.code });
      }
    },
    [patch, stopFlight],
  );

  const rerank = useCallback(
    async (datasetId: string, rankBy: RankBy): Promise<void> => {
      const storedId = statesRef.current[datasetId]?.stored?.id;
      if (storedId == null) return;
      try {
        const stored = await api.request<AutofitOut>(`/autofits/${storedId}?rank_by=${rankBy}`);
        if (statesRef.current[datasetId]?.stored?.id !== stored.id) return; // the run changed meanwhile
        // A successful re-rank clears a previous re-rank failure.
        patch(datasetId, { stored, rankBy: stored.rank_by, error: null, errorCode: null });
      } catch (error) {
        if (error instanceof ApiError) {
          patch(datasetId, { error: error.message, errorCode: error.code });
        }
      }
    },
    [patch],
  );

  const clear = useCallback(
    (datasetId: string): void => {
      // Invalidate every flow for this dataset, so their late responses patch nothing.
      tokens.current[datasetId] = (tokens.current[datasetId] ?? 0) + 1;
      for (const flight of [...flights.current.values()]) {
        if (flight.datasetId === datasetId) stopFlight(flight.jobId);
      }
      setStates((current) => {
        const next = { ...current };
        delete next[datasetId];
        return next;
      });
    },
    [stopFlight],
  );

  const stateFor = useCallback(
    (datasetId: string | null): AutofitState =>
      datasetId === null ? AUTOFIT_IDLE : (states[datasetId] ?? AUTOFIT_IDLE),
    [states],
  );

  // Stop every timer when the workspace unmounts (the jobs keep running server-side).
  useEffect(
    () => () => {
      for (const jobId of [...timers.current.keys()]) stopFlight(jobId);
    },
    [stopFlight],
  );

  return { stateFor, ensure, run, cancel, rerank, clear };
}
