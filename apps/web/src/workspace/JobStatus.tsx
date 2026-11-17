/**
 * Live job status: what the poller sees, announced politely to screen readers, with a
 * Cancel button wired to POST /jobs/{id}/cancel. Double-click safe: hidden unless a job
 * is actually in flight.
 */

import type { ReactElement } from "react";
import type { JobState } from "../hooks/useJobPolling";

const PHASE_TEXT: Record<JobState["phase"], string> = {
  idle: "",
  polling: "Fit job in progress…",
  succeeded: "Fit finished.",
  failed: "Fit failed.",
  cancelled: "Fit cancelled.",
  error: "Job status unknown.",
};

export function JobStatus({
  state,
  onCancel,
  succeededText,
}: {
  state: JobState;
  onCancel: () => void;
  /** Overrides "Fit finished." — the workspace sets it once the loaded fit's status
   *  is known, so a screen reader hears "Fit did not converge." for such fits. */
  succeededText?: string;
}): ReactElement | null {
  if (state.phase === "idle") return null;
  const text =
    state.phase === "polling"
      ? `${PHASE_TEXT.polling} (${state.status ?? "queued"})`
      : state.phase === "succeeded"
        ? (succeededText ?? PHASE_TEXT.succeeded)
        : state.phase === "failed" || state.phase === "error"
          ? `${PHASE_TEXT[state.phase]} ${state.message ?? ""}`.trim()
          : PHASE_TEXT[state.phase];
  return (
    <div className="job-status" role="status" aria-live="polite">
      <span
        className={`job-status__dot job-status__dot--${state.phase === "polling" ? (state.status ?? "queued") : state.phase}`}
        aria-hidden="true"
      />
      <span className="job-status__text">{text}</span>
      {state.phase === "polling" && (
        <button type="button" className="btn btn--small" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  );
}
