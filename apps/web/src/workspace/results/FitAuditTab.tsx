/**
 * The fit audit: each check with its verdict, the evidence and the numbers behind it, and
 * what to do next.
 * The server computes it from the stored fit, and nothing is refitted here except by the
 * cross-validation job the control at the end queues. A verdict is a statement of evidence,
 * never of proof, and the wording is the server's: this tab shows it and rewrites none of it.
 */

import { useEffect, useId, useRef, useState, type ReactElement } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ApiError,
  api,
  type AuditCheckOut,
  type AuditVerdict,
  type FitAuditOut,
} from "../../api/client";
import { Empty, ErrorBox, Loading, errorText } from "../../components/States";
import { useJobPolling, type JobState } from "../../hooks/useJobPolling";
import { fmtInt, fmtNumber } from "../../format";

export interface FitAuditTabProps {
  fitId: string;
  /** Whether the fit converged. One that did not still has its audit, but no cross-validation to run. */
  converged: boolean;
}

const VERDICT_LABEL: Record<AuditVerdict, string> = {
  pass: "Pass",
  caution: "Caution",
  fail: "Fail",
  not_applicable: "Not applicable",
};

const FOLD_CHOICES = [
  { value: "5", label: "5 folds" },
  { value: "10", label: "10 folds" },
  { value: "loo", label: "Leave one out" },
] as const;

/** A verdict as a pill in the tones the alerts use: green, amber, red or grey. The word carries the meaning; the colour repeats it. */
function VerdictPill({
  verdict,
  overall = false,
}: {
  verdict: AuditVerdict;
  overall?: boolean;
}): ReactElement {
  return (
    <span className={`verdict verdict--${verdict}${overall ? " verdict--overall" : ""}`}>
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

/** A number or rule a check's verdict turned on, labelled. Only called for a check that has one: a check without it gets no line, and no label. */
function Figure({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <p className="audit-check__figure">
      <span className="audit-check__lead">{label}:</span>{" "}
      <span className="audit-check__value">{value}</span>
    </p>
  );
}

function CheckCard({ check }: { check: AuditCheckOut }): ReactElement {
  // A check that did not apply is no finding: quiet, with the reason it could not run.
  const quiet = check.verdict === "not_applicable";
  return (
    <li
      className={`audit-check audit-check--${check.verdict}${quiet ? " audit-check--quiet" : ""}`}
      data-check={check.id}
    >
      <h4 className="audit-check__title">
        <VerdictPill verdict={check.verdict} />
        <span>{check.title}</span>
      </h4>
      <p className="audit-check__evidence">{check.evidence}</p>
      {check.statistic !== null && <Figure label="Statistic" value={check.statistic} />}
      {check.threshold !== null && <Figure label="Threshold" value={check.threshold} />}
      {!quiet && <p className="muted">{check.explanation}</p>}
      {check.next_step !== null && (
        <p className="audit-check__next">
          <span className="audit-check__lead">Next step:</span> {check.next_step}
        </p>
      )}
    </li>
  );
}

function SimplerModelLine({
  simpler,
}: {
  simpler: NonNullable<FitAuditOut["simpler_model"]>;
}): ReactElement {
  return (
    <p className="audit-simpler">
      A model with fewer parameters fits as well: <strong>{simpler.name}</strong> (
      <code>{simpler.model_id}</code>), {fmtInt(simpler.n_parameters)} parameters,{" "}
      {fmtNumber(simpler.delta_aicc)} AICc from the best, cross-validation RMSE{" "}
      {fmtNumber(simpler.cv_rmse)}.
    </p>
  );
}

/** How many rows that took no part are named before the rest are only counted. */
const EXCLUDED_LISTED_MAX = 8;

/** "40 observations, 3 parameters": what the audit ran on. */
function countsText(observations: number, parameters: number): string {
  return `${fmtInt(observations)} ${observations === 1 ? "observation" : "observations"}, ${fmtInt(parameters)} ${parameters === 1 ? "parameter" : "parameters"}`;
}

/**
 * The rows the audit left out, each with its reason, as one sentence: the lowest few named,
 * then how many more. The server numbers rows from zero; a reader counts from one.
 */
function excludedRowsText(rows: FitAuditOut["excluded"]): string | null {
  if (rows.length === 0) return null;
  const inRowOrder = [...rows].sort((first, second) => first.index - second.index);
  const named = inRowOrder
    .slice(0, EXCLUDED_LISTED_MAX)
    .map((row) => `row ${fmtInt(row.index + 1)} (${row.reason})`);
  const rest = inRowOrder.length - named.length;
  const list = rest > 0 ? `${named.join(", ")}, and ${fmtInt(rest)} more` : named.join(", ");
  const count =
    rows.length === 1 ? "1 row took no part" : `${fmtInt(rows.length)} rows took no part`;
  return `${count}: ${list}.`;
}

/** The job a 409 names: a cross-validation of this fit that is already queued or running. */
function crossValidationJob(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const detail = error.detail as { code?: unknown; job_id?: unknown } | null;
  return detail?.code === "CROSS_VALIDATION_RUNNING" && typeof detail.job_id === "string"
    ? detail.job_id
    : null;
}

/** What to say about the cross-validation job: an announcement, or an error. */
function jobMessage(state: JobState): { error: boolean; text: string } | null {
  switch (state.phase) {
    case "idle":
      return null;
    case "polling":
      return {
        error: false,
        text: `Cross-validation is ${
          state.status === "running" ? "running" : "queued"
        } on the server. The audit is read again when the job finishes.`,
      };
    case "succeeded":
      return { error: false, text: "Cross-validation finished." };
    case "failed":
      return {
        error: true,
        text: `The cross-validation failed: ${state.message ?? "the job failed"}. The audit is unchanged.`,
      };
    case "cancelled":
      return { error: false, text: "Cross-validation was cancelled. The audit is unchanged." };
    case "error":
      return {
        error: true,
        text: `The cross-validation could not be followed: ${
          state.message ?? "the job's status is unknown"
        }. The audit is unchanged.`,
      };
  }
}

export function FitAuditTab({ fitId, converged }: FitAuditTabProps): ReactElement {
  const auditQuery = useQuery({
    queryKey: ["fit-audit", fitId],
    queryFn: () => api.fitAudit(fitId),
  });
  const [folds, setFolds] = useState("5");
  const [posting, setPosting] = useState(false);
  const [startError, setStartError] = useState<unknown>(null);
  const foldsId = useId();

  // The job stores its cross-validation on this fit, so the audit is read again when it
  // finishes. The hook's argument is the job's `result_ref`, the dataset the job ran on: for
  // this job the fit is `result_id`, which is the fit already shown here.
  const { state: job, start, cancel } = useJobPolling(() => void auditQuery.refetch());

  // A job is followed only while this tab is on screen: one whose request came back after
  // the tab was left is not this tab's to poll.
  const onScreen = useRef(true);
  useEffect(() => {
    onScreen.current = true;
    return () => {
      onScreen.current = false;
    };
  }, []);

  const audit = auditQuery.data;
  const excluded = audit === undefined ? null : excludedRowsText(audit.excluded);
  const crossValidation = audit?.checks.find((check) => check.id === "cross_validation");
  const decided = crossValidation !== undefined && crossValidation.verdict !== "not_applicable";
  const busy = posting || job.phase === "polling";
  const message = jobMessage(job);

  const run = async (): Promise<void> => {
    setStartError(null);
    setPosting(true);
    try {
      const { job_id } = await api.runFitAuditCrossValidation(
        fitId,
        folds === "loo" ? "loo" : Number(folds),
      );
      if (onScreen.current) start(job_id);
    } catch (error) {
      if (!onScreen.current) return;
      // A 409 is not a failure: the server names the job that is already running for this fit.
      const running = crossValidationJob(error);
      if (running !== null) start(running);
      else setStartError(error);
    } finally {
      setPosting(false);
    }
  };

  return (
    <div>
      <h3>Fit audit</h3>
      {auditQuery.isPending && <Loading label="Loading the fit audit…" />}
      {auditQuery.isError && (
        <>
          <ErrorBox error={auditQuery.error} />
          <button type="button" className="btn" onClick={() => void auditQuery.refetch()}>
            Retry
          </button>
        </>
      )}
      {audit !== undefined && (
        <>
          <div className="audit-summary">
            <span className="audit-summary__label">Overall verdict</span>
            <VerdictPill verdict={audit.overall} overall />
            <p className="audit-summary__text">{audit.summary}</p>
            <p className="audit-summary__counts">
              {countsText(audit.n_observations, audit.n_parameters)}
            </p>
            {excluded !== null && <p className="audit-summary__excluded">{excluded}</p>}
          </div>
          {audit.checks.length === 0 ? (
            <Empty>
              This fit&apos;s audit came back with no checks, which should not happen: every
              fit has its convergence check. Load the fit again; if the audit stays empty,
              report it.
            </Empty>
          ) : (
            <ol className="audit-checks">
              {audit.checks.map((check) => (
                <CheckCard key={check.id} check={check} />
              ))}
            </ol>
          )}
          {audit.simpler_model !== null && <SimplerModelLine simpler={audit.simpler_model} />}
          {auditQuery.isFetching && (
            <p className="hint" role="status">
              Reading the audit again…
            </p>
          )}
          <section className="audit-cv" aria-label="Cross-validation check">
            <h4>Cross-validation</h4>
            <p className="hint">
              {converged
                ? "Cross-validation refits the model on folds of this fit's data and scores the points left out. It runs as a job that uses no quota."
                : "Cross-validation refits the model, and a fit that did not converge has none to refit: refit first, then run it."}
            </p>
            <div className="audit-cv__controls">
              <div className="field audit-cv__folds">
                <label htmlFor={foldsId}>Folds</label>
                <select
                  id={foldsId}
                  value={folds}
                  disabled={busy}
                  onChange={(event) => setFolds(event.currentTarget.value)}
                >
                  {FOLD_CHOICES.map((choice) => (
                    <option key={choice.value} value={choice.value}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </div>
              <button
                type="button"
                className="btn btn--small"
                disabled={busy || !converged}
                aria-busy={busy || undefined}
                onClick={() => void run()}
              >
                {busy
                  ? "Running cross-validation…"
                  : decided
                    ? "Run again"
                    : "Run cross-validation check"}
              </button>
              {job.phase === "polling" && (
                <button type="button" className="btn btn--small" onClick={() => void cancel()}>
                  Cancel
                </button>
              )}
            </div>
            {startError !== null && (
              <p className="alert alert--error" role="alert">
                {errorText(startError, "the cross-validation could not be started")}
              </p>
            )}
            {message !== null &&
              (message.error ? (
                <p className="alert alert--error" role="alert">
                  {message.text}
                </p>
              ) : (
                <p className="audit-cv__status" role="status">
                  {message.text}
                </p>
              ))}
          </section>
        </>
      )}
    </div>
  );
}
