/**
 * The fit panel: model picker (registry + polynomial + custom equation), method/loss,
 * per-parameter bounds and initial values, weights, seed and time limit. Client-side
 * validation mirrors the server contract; a custom equation's grammar is validated by
 * the server and its message is shown inline on submit.
 */

import { useEffect, useMemo, useState, type ReactElement } from "react";
import { Search } from "lucide-react";
import type { DatasetDetailOut, ModelInfo } from "../api/client";
import { fmtEquation } from "../format";
import {
  LOSSES,
  MAX_POLYNOMIAL_DEGREE,
  METHODS,
  buildFitSpec,
  emptyAdvanced,
  type AdvancedOptions,
  type ModelSelection,
} from "./spec";

export interface FitPanelProps {
  dataset: DatasetDetailOut | null;
  models: ModelInfo[];
  busy: boolean;
  onFit: (spec: Record<string, unknown>) => void;
  fitError: string | null;
  /** "Use this model manually" from an Auto-fit row: prefills the picker once. */
  preset?: { kind: "registered"; id: string } | { kind: "polynomial"; degree: number } | null;
}

const POLY_VALUE = "__polynomial__";
const CUSTOM_VALUE = "__custom__";

/**
 * The API catalogue appends its own "polynomial" family entry to /models. Selecting it
 * would send a registered ref the fit engine rejects with UNKNOWN_MODEL (only the
 * dedicated degree-picker entry above uses the closed-form polynomial path), so the
 * catalogue duplicate is filtered out of the picker.
 */
const CATALOGUE_POLYNOMIAL_ID = "polynomial";

function searchMatches(model: ModelInfo, query: string): boolean {
  if (query === "") return true;
  const haystack = `${model.id} ${model.name} ${model.category} ${model.equation}`.toLowerCase();
  return haystack.includes(query);
}

/**
 * Per-parameter inputs are derived from the selected model, never carried over: they are
 * reset (and bounds prefilled from the model's defaults — blank where unbounded, and
 * always-blank initial values, which the catalogue does not provide) whenever the model
 * or the dataset changes.
 */
function prefillParameters(
  model: ModelInfo | null,
): Pick<AdvancedOptions, "initial" | "lower" | "upper"> {
  const initial: Record<string, string> = {};
  const lower: Record<string, string> = {};
  const upper: Record<string, string> = {};
  if (model !== null) {
    for (const parameter of model.parameters) {
      if (parameter.lower !== null) lower[parameter.name] = String(parameter.lower);
      if (parameter.upper !== null) upper[parameter.name] = String(parameter.upper);
    }
  }
  return { initial, lower, upper };
}

export function FitPanel({ dataset, models, busy, onFit, fitError, preset }: FitPanelProps): ReactElement {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string>("");
  const [degree, setDegree] = useState("1");
  const [equation, setEquation] = useState("");
  const [parameterNames, setParameterNames] = useState("");
  const [variable, setVariable] = useState("x");
  const [advanced, setAdvanced] = useState<AdvancedOptions>(emptyAdvanced);
  const [errors, setErrors] = useState<string[]>([]);

  const grouped = useMemo(() => {
    const query = search.trim().toLowerCase();
    const map = new Map<string, ModelInfo[]>();
    for (const model of models) {
      if (model.id === CATALOGUE_POLYNOMIAL_ID) continue;
      // A filter must never hide the model that is actually selected: a blank picker
      // over a live selection would let the panel fit a model it no longer names.
      if (!searchMatches(model, query) && model.id !== selectedId) continue;
      const list = map.get(model.category) ?? [];
      list.push(model);
      map.set(model.category, list);
    }
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [models, search, selectedId]);

  const selection: ModelSelection = useMemo(() => {
    if (selectedId === POLY_VALUE)
      return { kind: "polynomial", degree: Number.parseInt(degree, 10) || 0 };
    if (selectedId === CUSTOM_VALUE)
      return { kind: "custom", equation, parameters: parameterNames, variable };
    if (selectedId !== "") return { kind: "registered", id: selectedId };
    return { kind: "polynomial", degree: 1 };
  }, [selectedId, degree, equation, parameterNames, variable]);

  const isPolynomial = selectedId === POLY_VALUE;
  const isCustom = selectedId === CUSTOM_VALUE;
  const isRegistered = selectedId !== "" && !isPolynomial && !isCustom;
  const activeModel = useMemo(
    () => (isRegistered ? (models.find((m) => m.id === selectedId) ?? null) : null),
    [isRegistered, models, selectedId],
  );

  // Reset (and re-prefill) the per-parameter inputs whenever the model or the dataset
  // changes, so no stale bounds/initials silently ride along into the next fit.
  const datasetId = dataset?.id ?? null;
  useEffect(() => {
    setAdvanced((current) => ({ ...current, ...prefillParameters(activeModel) }));
  }, [datasetId, activeModel]);

  // An Auto-fit row's "Use this model manually": select that model (or polynomial
  // degree) once, when the preset arrives.
  useEffect(() => {
    if (preset === null || preset === undefined) return;
    if (preset.kind === "registered") {
      setSelectedId(preset.id);
    } else {
      setSelectedId(POLY_VALUE);
      setDegree(String(preset.degree));
    }
    setErrors([]);
  }, [preset]);

  const parameterRows: Array<{ name: string; description: string; unit: string }> = useMemo(() => {
    if (isRegistered && activeModel !== null) {
      return activeModel.parameters.map((p) => ({
        name: p.name,
        description: p.description,
        unit: p.unit,
      }));
    }
    if (isCustom) {
      return parameterNames
        .split(/[,\s]+/)
        .map((name) => name.trim())
        .filter((name) => name !== "")
        .map((name) => ({ name, description: "custom parameter", unit: "" }));
    }
    return [];
  }, [isRegistered, isCustom, activeModel, parameterNames]);

  const update = (patch: Partial<AdvancedOptions>): void => {
    setAdvanced((current) => ({ ...current, ...patch }));
  };

  const onFitClick = (): void => {
    setErrors([]);
    if (selectedId === "") {
      setErrors(["choose a model (or the polynomial / custom equation entries)"]);
      return;
    }
    const result = buildFitSpec(selection, advanced);
    if ("errors" in result) {
      setErrors(result.errors);
      return;
    }
    onFit(result.spec as unknown as Record<string, unknown>);
  };

  return (
    <section className="panel" aria-label="Fit controls">
      <h2 className="panel-title">Fit</h2>

      <div className="field">
        <label htmlFor="model-search">Search models</label>
        <span className="search-field">
          <Search className="icon" aria-hidden="true" focusable="false" />
          <input
            id="model-search"
            type="search"
            placeholder="id, name, category or equation"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </span>
      </div>

      <div className="field">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={selectedId}
          onChange={(e) => {
            setSelectedId(e.target.value);
            setErrors([]);
          }}
        >
          <option value="">Choose a model…</option>
          <option value={POLY_VALUE}>Polynomial (degree 1–{MAX_POLYNOMIAL_DEGREE})</option>
          <option value={CUSTOM_VALUE}>Custom equation…</option>
          {grouped.map(([category, list]) => (
            <optgroup key={category} label={category}>
              {list.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      {isPolynomial && (
        <div className="field">
          <label htmlFor="poly-degree">Degree</label>
          <input
            id="poly-degree"
            type="number"
            min={1}
            max={MAX_POLYNOMIAL_DEGREE}
            value={degree}
            onChange={(e) => setDegree(e.target.value)}
          />
          <p className="hint">1–{MAX_POLYNOMIAL_DEGREE}; fitted in closed form — method, loss, bounds and initial values do not apply.</p>
        </div>
      )}

      {isCustom && (
        <>
          <div className="field">
            <label htmlFor="custom-equation">Equation</label>
            <textarea
              id="custom-equation"
              rows={2}
              placeholder="y = a * exp(-b * x)"
              value={equation}
              onChange={(e) => setEquation(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="custom-parameters">Parameter names</label>
            <input
              id="custom-parameters"
              type="text"
              placeholder="a, b"
              autoComplete="off"
              value={parameterNames}
              onChange={(e) => setParameterNames(e.target.value)}
            />
            <p className="hint">comma separated, e.g. a, b</p>
          </div>
          <div className="field">
            <label htmlFor="custom-variable">Independent variable</label>
            <input
              id="custom-variable"
              type="text"
              value={variable}
              onChange={(e) => setVariable(e.target.value)}
            />
          </div>
          <p className="hint">The equation is checked by a safe grammar on the server; its message appears here on submit.</p>
        </>
      )}

      {activeModel !== null && (
        <div className="model-card">
          <p className="equation" title={fmtEquation(activeModel.equation)}>
            {fmtEquation(activeModel.equation)}
          </p>
          <ul className="param-list">
            {activeModel.parameters.map((parameter) => (
              <li key={parameter.name}>
                <code>{parameter.name}</code>
                {parameter.description !== "" && <> — {parameter.description}</>}
                {parameter.unit !== "" && <> [{parameter.unit}]</>}
              </li>
            ))}
          </ul>
          {activeModel.assumptions !== "" && (
            <p className="hint">Assumes: {activeModel.assumptions}</p>
          )}
        </div>
      )}

      <fieldset className="fieldset" disabled={isPolynomial}>
        <legend>Method</legend>
        <div className="field">
          <label htmlFor="fit-method">Method</label>
          <select
            id="fit-method"
            value={advanced.method}
            disabled={isPolynomial}
            onChange={(e) => update({ method: e.target.value })}
          >
            {METHODS.map((method) => (
              <option key={method.value} value={method.value}>
                {method.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="fit-loss">Loss</label>
          <select
            id="fit-loss"
            value={advanced.loss}
            disabled={isPolynomial}
            onChange={(e) => update({ loss: e.target.value })}
          >
            {LOSSES.map((loss) => (
              <option key={loss.value} value={loss.value}>
                {loss.label}
              </option>
            ))}
          </select>
        </div>
      </fieldset>

      {parameterRows.length > 0 && !isPolynomial && (
        <fieldset className="fieldset">
          <legend>Parameters — initial values and bounds</legend>
          <div className="table-wrap">
            {/* data-label carries each cell's column name for the stacked layout the
                stylesheet switches to below its breakpoint (item 14). */}
            <table className="table table--compact params-grid">
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">Initial</th>
                  <th scope="col">Lower</th>
                  <th scope="col">Upper</th>
                </tr>
              </thead>
              <tbody>
                {parameterRows.map((row) => (
                  <tr key={row.name} title={row.description}>
                    <td>
                      <code>{row.name}</code>
                    </td>
                    <td data-label="Initial">
                      <input
                        aria-label={`Initial ${row.name}`}
                        type="number"
                        step="any"
                        value={advanced.initial[row.name] ?? ""}
                        onChange={(e) =>
                          update({
                            initial: { ...advanced.initial, [row.name]: e.target.value },
                          })
                        }
                      />
                    </td>
                    <td data-label="Lower">
                      <input
                        aria-label={`Lower bound ${row.name}`}
                        type="number"
                        step="any"
                        placeholder={
                          isRegistered && activeModel !== null
                            ? String(
                                activeModel.parameters.find((p) => p.name === row.name)?.lower ??
                                  "−∞",
                              )
                            : "−∞"
                        }
                        value={advanced.lower[row.name] ?? ""}
                        onChange={(e) =>
                          update({
                            lower: { ...advanced.lower, [row.name]: e.target.value },
                          })
                        }
                      />
                    </td>
                    <td data-label="Upper">
                      <input
                        aria-label={`Upper bound ${row.name}`}
                        type="number"
                        step="any"
                        placeholder={
                          isRegistered && activeModel !== null
                            ? String(
                                activeModel.parameters.find((p) => p.name === row.name)?.upper ??
                                  "∞",
                              )
                            : "∞"
                        }
                        value={advanced.upper[row.name] ?? ""}
                        onChange={(e) =>
                          update({
                            upper: { ...advanced.upper, [row.name]: e.target.value },
                          })
                        }
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </fieldset>
      )}

      <fieldset className="fieldset" disabled={isPolynomial}>
        <legend>Options</legend>
        <div className="field field--check">
          <input
            id="fit-weights"
            type="checkbox"
            checked={advanced.useWeights}
            disabled={isPolynomial || dataset?.weight_column == null}
            onChange={(e) => update({ useWeights: e.target.checked })}
          />
          <label htmlFor="fit-weights">Use weights</label>
        </div>
        {dataset?.weight_column != null ? (
          <p className="hint">weights column: {dataset.weight_column}</p>
        ) : (
          <p className="hint">this dataset has no weights column</p>
        )}
        <div className="field">
          <label htmlFor="fit-seed">Seed</label>
          <input
            id="fit-seed"
            type="number"
            min={0}
            value={advanced.seed}
            disabled={isPolynomial}
            onChange={(e) => update({ seed: e.target.value })}
          />
          <p className="hint">0 to 2⁵³−1; empty uses the server default</p>
        </div>
        <div className="field">
          <label htmlFor="fit-timelimit">Time limit (seconds)</label>
          <input
            id="fit-timelimit"
            type="number"
            step="any"
            min={0}
            value={advanced.timeLimit}
            disabled={isPolynomial}
            onChange={(e) => update({ timeLimit: e.target.value })}
          />
        </div>
      </fieldset>

      {errors.length > 0 && (
        <div className="alert alert--error" role="alert">
          <ul>
            {errors.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </div>
      )}
      {fitError !== null && (
        <div className="alert alert--error" role="alert">
          {fitError}
        </div>
      )}

      <button
        type="button"
        className="btn btn--primary btn--block"
        onClick={onFitClick}
        disabled={busy || dataset === null}
      >
        Fit
      </button>
    </section>
  );
}
