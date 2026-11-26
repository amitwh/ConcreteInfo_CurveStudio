/**
 * The fit's further tools, each queued as a job and rendered as a stored statistical result:
 *
 * - model discernibility: can these data tell the leading models apart?
 * - experiment feasibility: what would a planned design recover?
 * - where to measure next: which further x values sharpen the estimates most?
 *
 * One job runs in this tab at a time. Its status is the polled job's, its Cancel posts the
 * job's cancel route, and the last result stays on screen until a new run replaces it.
 * Nothing here computes a number: the server's result is shown by `StatsResult`.
 */

import { useCallback, useEffect, useId, useRef, useState, type ReactElement } from "react";
import {
  ApiError,
  api,
  type AdvisorRequest,
  type FitBundle,
  type JobCreatedOut,
  type SimulationRequest,
  type StatResultOut,
} from "../../api/client";
import { Empty, ErrorBox, Loading, apiErrorText } from "../../components/States";
import { useJobPolling, type JobState } from "../../hooks/useJobPolling";
import { fmtInt, fmtNumber } from "../../format";
import { explainCode } from "../../messages";
import { StatsResult } from "../stats/StatsResult";

export interface FitToolsTabProps {
  fitId: string;
  bundle: FitBundle;
}

type ToolKey = "discern" | "simulate" | "advise";

/** How each tool is named in a running line. */
const TOOL_NAME: Record<ToolKey, string> = {
  discern: "The discernibility report",
  simulate: "The simulation",
  advise: "The advice",
};

/** What each tool's button says when it is not the one running. */
const RUN_LABEL: Record<ToolKey, string> = {
  discern: "Run the discernibility report",
  simulate: "Run the simulation",
  advise: "Run the advisor",
};

/** A form's value, or the sentence that says why it is not one. */
type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

/** A whole number from a field, or null when the text is not one. */
function wholeNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  return Number.isInteger(value) ? value : null;
}

/** A finite number from a field, or null when the text is not one. */
function finiteNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

/** The `detail.code` an API error carries, when it carries one. */
function detailCode(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;
  const detail = error.detail as { code?: unknown } | null;
  return typeof detail?.code === "string" ? detail.code : null;
}

/** A refused request as the tab says it: the code's sentence, then the server's message. */
function refusalText(error: unknown): string {
  return explainCode(detailCode(error), apiErrorText(error, "the tool could not be started"));
}

/** A failed job as the tab says it: the code's sentence, then the server's message. */
function failureText(state: JobState): string {
  return explainCode(state.code, state.message, "the job failed");
}

/**
 * Parse "name=value, other=value" into a difference mapping. Blank is no difference at all;
 * anything else that is not a pair of a name and a finite number is refused with its text.
 */
function parseDifference(text: string): Parsed<Record<string, number> | null> {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: true, value: null };
  const pairs: Record<string, number> = {};
  for (const part of trimmed.split(",")) {
    const pieces = part.split("=");
    const name = pieces[0]?.trim() ?? "";
    const value = pieces.length === 2 ? finiteNumber(pieces[1] ?? "") : null;
    if (name === "" || value === null) {
      return {
        ok: false,
        error: `differences are name=value pairs separated by commas; "${part.trim()}" is not one`,
      };
    }
    pairs[name] = value;
  }
  return { ok: true, value: pairs };
}

/** A design from the form: an evenly spaced range, or an explicit list of locations. */
function parseDesign(
  mode: "range" | "list",
  low: string,
  high: string,
  count: string,
  locations: string,
  replicates: number,
): Parsed<SimulationRequest["design"]> {
  if (mode === "list") {
    const values = locations
      .split(/[\s,;]+/)
      .map((token) => token.trim())
      .filter((token) => token !== "")
      .map(Number);
    if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
      return {
        ok: false,
        error: "the design list needs at least one finite x value, separated by commas",
      };
    }
    return { ok: true, value: { locations: values, replicates } };
  }
  const lowValue = finiteNumber(low);
  const highValue = finiteNumber(high);
  const countValue = wholeNumber(count);
  if (lowValue === null || highValue === null) {
    return { ok: false, error: "the design range needs a finite low and high" };
  }
  if (lowValue >= highValue) {
    return { ok: false, error: "the design range needs low below high" };
  }
  if (countValue === null || countValue < 1) {
    return { ok: false, error: "the design range needs a whole count of one or more" };
  }
  return { ok: true, value: { low: lowValue, high: highValue, count: countValue, replicates } };
}

/** The progress the polled job reports, as a line: done of total, else elapsed seconds. */
function progressText(state: JobState): string {
  const progress = state.progress;
  if (progress == null) return "";
  if (progress.total > 0) {
    return ` ${fmtInt(progress.done)} of ${fmtInt(progress.total)} done.`;
  }
  if (progress.elapsed_s > 0) return ` ${fmtNumber(progress.elapsed_s, { digits: 1 })} s elapsed.`;
  return "";
}

export function FitToolsTab({ fitId, bundle }: FitToolsTabProps): ReactElement {
  const base = useId();
  const parameters = bundle.result.parameters;
  const truth = Object.fromEntries(parameters.map((parameter) => [parameter.name, parameter.value]));

  /* ---- The three forms ---- */
  const [discernLevel, setDiscernLevel] = useState("0.95");
  const [discernGrid, setDiscernGrid] = useState("200");
  const [feasRuns, setFeasRuns] = useState("200");
  const [feasSeed, setFeasSeed] = useState("0");
  const [feasNoiseKind, setFeasNoiseKind] = useState<"sd" | "cv">("sd");
  const [feasNoiseValue, setFeasNoiseValue] = useState("0.1");
  const [feasDesignMode, setFeasDesignMode] = useState<"range" | "list">("range");
  const [feasLow, setFeasLow] = useState("0");
  const [feasHigh, setFeasHigh] = useState("1");
  const [feasCount, setFeasCount] = useState("5");
  const [feasLocations, setFeasLocations] = useState("");
  const [feasReplicates, setFeasReplicates] = useState("1");
  const [feasDifference, setFeasDifference] = useState("");
  const [advicePoints, setAdvicePoints] = useState("3");
  const [adviceCandidates, setAdviceCandidates] = useState("400");
  const [adviceLow, setAdviceLow] = useState("");
  const [adviceHigh, setAdviceHigh] = useState("");
  const [adviceReplicates, setAdviceReplicates] = useState("1");
  const [adviceObjective, setAdviceObjective] = useState("d_optimal");

  /* ---- The one job, and what the tab is showing ---- */
  const [active, setActive] = useState<ToolKey | null>(null);
  const [posting, setPosting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [startError, setStartError] = useState<unknown>(null);
  const [result, setResult] = useState<StatResultOut | null>(null);
  const [resultError, setResultError] = useState<unknown>(null);
  const [reading, setReading] = useState(false);

  // A job is followed only while this tab is on screen: one whose result comes back after
  // the tab was left is not this tab's to show.
  const onScreen = useRef(true);
  useEffect(() => {
    onScreen.current = true;
    return () => {
      onScreen.current = false;
    };
  }, []);

  const onFinished = useCallback((resultId: string): void => {
    setReading(true);
    setResultError(null);
    void api.statResult(resultId).then(
      (stored) => {
        if (!onScreen.current) return;
        setResult(stored);
        setReading(false);
      },
      (error: unknown) => {
        if (!onScreen.current) return;
        setResultError(error);
        setReading(false);
      },
    );
  }, []);

  const { state: job, start, cancel } = useJobPolling(onFinished);
  const busy = posting || job.phase === "polling";

  // The tool's name is kept only while its job is in play; a settled job leaves none.
  useEffect(() => {
    if (!posting && job.phase !== "polling") setActive(null);
  }, [posting, job.phase]);

  async function submit(kind: ToolKey, send: () => Promise<JobCreatedOut>): Promise<void> {
    setFormError(null);
    setStartError(null);
    setActive(kind);
    setPosting(true);
    try {
      const { job_id } = await send();
      if (onScreen.current) start(job_id);
    } catch (error) {
      if (onScreen.current) setStartError(error);
    } finally {
      setPosting(false);
    }
  }

  /* ---- Model discernibility ---- */
  function runDiscernibility(): void {
    const level = finiteNumber(discernLevel);
    const grid = wholeNumber(discernGrid);
    if (level === null || !(level > 0 && level < 1)) {
      setFormError("the confidence level must be a number between 0 and 1, e.g. 0.95");
      return;
    }
    if (grid === null || grid < 2) {
      setFormError("the grid needs a whole number of at least 2 points");
      return;
    }
    void submit("discern", () => api.fitDiscernibility(fitId, { level, grid_points: grid }));
  }

  /* ---- Experiment feasibility ---- */
  function runSimulation(): void {
    if (parameters.length === 0) {
      setFormError("this fit reports no parameters, so there is no truth to simulate under");
      return;
    }
    const runs = wholeNumber(feasRuns);
    const seed = wholeNumber(feasSeed);
    const noise = finiteNumber(feasNoiseValue);
    const replicates = wholeNumber(feasReplicates);
    if (runs === null || runs < 2) {
      setFormError("the simulation needs a whole number of runs, at least 2");
      return;
    }
    if (seed === null || seed < 0) {
      setFormError("the seed must be a whole number of 0 or more");
      return;
    }
    if (noise === null || noise <= 0) {
      setFormError("the noise value must be a finite number above 0");
      return;
    }
    if (replicates === null || replicates < 1) {
      setFormError("replicates must be a whole number of 1 or more");
      return;
    }
    const design = parseDesign(
      feasDesignMode,
      feasLow,
      feasHigh,
      feasCount,
      feasLocations,
      replicates,
    );
    if (!design.ok) {
      setFormError(design.error);
      return;
    }
    const difference = parseDifference(feasDifference);
    if (!difference.ok) {
      setFormError(difference.error);
      return;
    }
    const request: SimulationRequest = {
      truth,
      design: design.value,
      noise: { kind: feasNoiseKind, value: noise },
      runs,
      seed,
      level: 0.95,
      difference: difference.value,
    };
    void submit("simulate", () => api.fitSimulation(fitId, request));
  }

  /* ---- Where to measure next ---- */
  function runAdvisor(): void {
    const points = wholeNumber(advicePoints);
    const candidates = wholeNumber(adviceCandidates);
    const replicates = wholeNumber(adviceReplicates);
    if (points === null || points < 1) {
      setFormError("points must be a whole number of 1 or more");
      return;
    }
    if (candidates === null || candidates < 1) {
      setFormError("candidates must be a whole number of 1 or more");
      return;
    }
    if (replicates === null || replicates < 1) {
      setFormError("replicates must be a whole number of 1 or more");
      return;
    }
    const low = finiteNumber(adviceLow);
    const high = finiteNumber(adviceHigh);
    if (adviceLow.trim() !== "" && low === null) {
      setFormError("low must be a finite number, or blank for the data's own range");
      return;
    }
    if (adviceHigh.trim() !== "" && high === null) {
      setFormError("high must be a finite number, or blank for the data's own range");
      return;
    }
    const narrowed = adviceObjective.startsWith("parameter:")
      ? adviceObjective.slice("parameter:".length)
      : null;
    const request: AdvisorRequest = {
      points,
      candidates,
      replicates,
      objective: narrowed === null ? "d_optimal" : "parameter",
      parameter: narrowed,
      level: 0.95,
      low,
      high,
    };
    void submit("advise", () => api.fitAdvisor(fitId, request));
  }

  const objectiveChoices = [
    { value: "d_optimal", label: "D-optimal (sharpen every estimate)" },
    ...parameters.map((parameter) => ({
      value: `parameter:${parameter.name}`,
      label: `One parameter's SE: ${parameter.name}`,
    })),
  ];

  /* ---- What the job is saying ---- */
  let status: { error: boolean; text: string } | null = null;
  if (posting) {
    status = { error: false, text: `Starting ${TOOL_NAME[active ?? "discern"].toLowerCase()}…` };
  } else {
    switch (job.phase) {
      case "idle":
        break;
      case "polling":
        status = {
          error: false,
          text: `${TOOL_NAME[active ?? "discern"]} is ${
            job.status === "running" ? "running" : "queued"
          } on the server.${progressText(job)}`,
        };
        break;
      case "succeeded":
        status = { error: false, text: "The tool finished; its stored result is below." };
        break;
      case "failed":
        status = { error: true, text: `The tool failed: ${failureText(job)}.` };
        break;
      case "cancelled":
        status = { error: false, text: "The tool was cancelled. The last result is unchanged." };
        break;
      case "error":
        status = {
          error: true,
          text: `The job could not be followed: ${job.message ?? "its status is unknown"}.`,
        };
        break;
    }
  }

  const busyLabel = (kind: ToolKey): string =>
    busy && active === kind ? "Running…" : RUN_LABEL[kind];

  return (
    <div>
      <h3>More tools</h3>
      <p className="hint">
        Each tool reads this stored fit and queues a job; the result is stored beside the
        fit's dataset. One runs at a time in this tab.
      </p>

      <fieldset className="fieldset">
        <legend>Model discernibility</legend>
        <p className="hint">
          Are the leading models of the fit&apos;s Auto-fit ranking distinguishable on these
          data? A fit with no ranking behind it fails with NO_RANKING.
        </p>
        <div className="field">
          <label htmlFor={`${base}-discern-level`}>Confidence level</label>
          <input
            id={`${base}-discern-level`}
            type="number"
            step="0.01"
            min="0.01"
            max="0.99"
            value={discernLevel}
            disabled={busy}
            onChange={(event) => setDiscernLevel(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-discern-grid`}>Grid points</label>
          <input
            id={`${base}-discern-grid`}
            type="number"
            step="1"
            min="2"
            value={discernGrid}
            disabled={busy}
            onChange={(event) => setDiscernGrid(event.currentTarget.value)}
          />
        </div>
        <button
          type="button"
          className="btn btn--small"
          disabled={busy}
          aria-busy={busy && active === "discern" ? true : undefined}
          onClick={runDiscernibility}
        >
          {busyLabel("discern")}
        </button>
      </fieldset>

      <fieldset className="fieldset">
        <legend>Experiment feasibility</legend>
        <p className="hint">
          Simulate a planned design under this fit&apos;s model and measure how the estimates
          recover the truth.
        </p>
        <div className="field">
          <label>Truth (this fit&apos;s parameter values)</label>
          <p className="fit-tools__truth">
            {parameters.length === 0
              ? "This fit reports no parameters, so there is no truth to simulate under."
              : parameters
                  .map((parameter) => `${parameter.name} = ${fmtNumber(parameter.value)}`)
                  .join(", ")}
          </p>
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-runs`}>Simulated runs</label>
          <input
            id={`${base}-feas-runs`}
            type="number"
            step="1"
            min="2"
            max="5000"
            value={feasRuns}
            disabled={busy}
            onChange={(event) => setFeasRuns(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-seed`}>Seed</label>
          <input
            id={`${base}-feas-seed`}
            type="number"
            step="1"
            min="0"
            value={feasSeed}
            disabled={busy}
            onChange={(event) => setFeasSeed(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-noise-kind`}>Noise kind</label>
          <select
            id={`${base}-feas-noise-kind`}
            value={feasNoiseKind}
            disabled={busy}
            onChange={(event) => setFeasNoiseKind(event.currentTarget.value as "sd" | "cv")}
          >
            <option value="sd">constant SD</option>
            <option value="cv">coefficient of variation</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-noise-value`}>Noise value</label>
          <input
            id={`${base}-feas-noise-value`}
            type="number"
            step="0.01"
            min="0"
            value={feasNoiseValue}
            disabled={busy}
            onChange={(event) => setFeasNoiseValue(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-design-mode`}>Design</label>
          <select
            id={`${base}-feas-design-mode`}
            value={feasDesignMode}
            disabled={busy}
            onChange={(event) => setFeasDesignMode(event.currentTarget.value as "range" | "list")}
          >
            <option value="range">range (low, high, count)</option>
            <option value="list">explicit list</option>
          </select>
        </div>
        {feasDesignMode === "range" ? (
          <>
            <div className="field">
              <label htmlFor={`${base}-feas-low`}>Design low</label>
              <input
                id={`${base}-feas-low`}
                type="number"
                step="any"
                value={feasLow}
                disabled={busy}
                onChange={(event) => setFeasLow(event.currentTarget.value)}
              />
            </div>
            <div className="field">
              <label htmlFor={`${base}-feas-high`}>Design high</label>
              <input
                id={`${base}-feas-high`}
                type="number"
                step="any"
                value={feasHigh}
                disabled={busy}
                onChange={(event) => setFeasHigh(event.currentTarget.value)}
              />
            </div>
            <div className="field">
              <label htmlFor={`${base}-feas-count`}>Design count</label>
              <input
                id={`${base}-feas-count`}
                type="number"
                step="1"
                min="1"
                value={feasCount}
                disabled={busy}
                onChange={(event) => setFeasCount(event.currentTarget.value)}
              />
            </div>
          </>
        ) : (
          <div className="field">
            <label htmlFor={`${base}-feas-locations`}>Design locations (x, comma separated)</label>
            <input
              id={`${base}-feas-locations`}
              type="text"
              placeholder="0, 2.5, 5"
              value={feasLocations}
              disabled={busy}
              onChange={(event) => setFeasLocations(event.currentTarget.value)}
            />
          </div>
        )}
        <div className="field">
          <label htmlFor={`${base}-feas-replicates`}>Replicates per location</label>
          <input
            id={`${base}-feas-replicates`}
            type="number"
            step="1"
            min="1"
            max="100"
            value={feasReplicates}
            disabled={busy}
            onChange={(event) => setFeasReplicates(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-feas-difference`}>
            Differences to measure power for (optional, name=value)
          </label>
          <input
            id={`${base}-feas-difference`}
            type="text"
            placeholder="a=0.5, b=0.1"
            value={feasDifference}
            disabled={busy}
            onChange={(event) => setFeasDifference(event.currentTarget.value)}
          />
        </div>
        <button
          type="button"
          className="btn btn--small"
          disabled={busy}
          aria-busy={busy && active === "simulate" ? true : undefined}
          onClick={runSimulation}
        >
          {busyLabel("simulate")}
        </button>
      </fieldset>

      <fieldset className="fieldset">
        <legend>Where to measure next</legend>
        <p className="hint">
          Recommend further x values that sharpen the estimates, assuming the model form holds.
        </p>
        <div className="field">
          <label htmlFor={`${base}-advice-points`}>Points</label>
          <input
            id={`${base}-advice-points`}
            type="number"
            step="1"
            min="1"
            max="20"
            value={advicePoints}
            disabled={busy}
            onChange={(event) => setAdvicePoints(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-advice-candidates`}>Candidate locations</label>
          <input
            id={`${base}-advice-candidates`}
            type="number"
            step="1"
            min="10"
            max="1000"
            value={adviceCandidates}
            disabled={busy}
            onChange={(event) => setAdviceCandidates(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-advice-low`}>Low (blank for the data&apos;s own range)</label>
          <input
            id={`${base}-advice-low`}
            type="number"
            step="any"
            value={adviceLow}
            disabled={busy}
            onChange={(event) => setAdviceLow(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-advice-high`}>High (blank for the data&apos;s own range)</label>
          <input
            id={`${base}-advice-high`}
            type="number"
            step="any"
            value={adviceHigh}
            disabled={busy}
            onChange={(event) => setAdviceHigh(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-advice-replicates`}>Replicates per location</label>
          <input
            id={`${base}-advice-replicates`}
            type="number"
            step="1"
            min="1"
            max="20"
            value={adviceReplicates}
            disabled={busy}
            onChange={(event) => setAdviceReplicates(event.currentTarget.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={`${base}-advice-objective`}>Objective</label>
          <select
            id={`${base}-advice-objective`}
            value={adviceObjective}
            disabled={busy}
            onChange={(event) => setAdviceObjective(event.currentTarget.value)}
          >
            {objectiveChoices.map((choice) => (
              <option key={choice.value} value={choice.value}>
                {choice.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="btn btn--small"
          disabled={busy}
          aria-busy={busy && active === "advise" ? true : undefined}
          onClick={runAdvisor}
        >
          {busyLabel("advise")}
        </button>
      </fieldset>

      <section className="fit-tools__status" aria-label="Tool status">
        {formError !== null && (
          <p className="alert alert--error" role="alert">
            {formError}
          </p>
        )}
        {startError !== null && (
          <p className="alert alert--error" role="alert">
            {refusalText(startError)}
          </p>
        )}
        {status !== null &&
          (status.error ? (
            <p className="alert alert--error" role="alert">
              {status.text}
            </p>
          ) : (
            <p className="hint" role="status">
              {status.text}
            </p>
          ))}
        {job.phase === "polling" && (
          <button type="button" className="btn btn--small" onClick={() => void cancel()}>
            Cancel
          </button>
        )}
      </section>

      <section className="fit-tools__result" aria-label="Stored result">
        {reading && <Loading label="Reading the tool's result…" />}
        {resultError !== null && <ErrorBox error={resultError} />}
        {result !== null ? (
          <StatsResult result={result.result} />
        ) : (
          reading === false &&
          resultError === null && (
            <Empty>No tool has finished in this tab yet. Its result will be shown here.</Empty>
          )
        )}
      </section>
    </div>
  );
}
