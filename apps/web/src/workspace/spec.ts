/**
 * Building a FitSpec from the fit panel's form state, with client-side checks that
 * mirror the server's contract (curvestudio.workflow.spec). Anything the grammar of a
 * custom equation is left to the server — its message is shown inline on submit.
 */

import type { FitSpecJson, ModelInfo } from "../api/client";

export const MAX_POLYNOMIAL_DEGREE = 10;
export const MAX_CUSTOM_EQUATION_LENGTH = 2000;
export const MAX_CUSTOM_PARAMETERS = 20;
export const MAX_SEED = 2 ** 53 - 1;

export const METHODS = [
  { value: "auto", label: "auto" },
  { value: "lm", label: "Levenberg–Marquardt (lm)" },
  { value: "trf", label: "Trust Region Reflective (trf)" },
  { value: "dogbox", label: "Dogbox" },
  { value: "nelder", label: "Nelder–Mead" },
  { value: "powell", label: "Powell" },
  { value: "bfgs", label: "BFGS" },
  { value: "lbfgsb", label: "L-BFGS-B" },
  { value: "differential_evolution", label: "Differential evolution" },
  { value: "dual_annealing", label: "Dual annealing" },
] as const;

export const LOSSES = [
  { value: "linear", label: "linear (least squares)" },
  { value: "soft_l1", label: "soft L1" },
  { value: "huber", label: "Huber" },
  { value: "cauchy", label: "Cauchy" },
  { value: "arctan", label: "arctan" },
] as const;

export type ModelSelection =
  | { kind: "registered"; id: string }
  | { kind: "polynomial"; degree: number }
  | { kind: "custom"; equation: string; parameters: string; variable: string };

export interface AdvancedOptions {
  method: string;
  loss: string;
  useWeights: boolean;
  seed: string;
  timeLimit: string;
  initial: Record<string, string>;
  lower: Record<string, string>;
  upper: Record<string, string>;
}

export const emptyAdvanced: AdvancedOptions = {
  method: "auto",
  loss: "linear",
  useWeights: false,
  seed: "",
  timeLimit: "",
  initial: {},
  lower: {},
  upper: {},
};

function parseFloatStrict(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

function parseIntStrict(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "" || !/^\d+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : null;
}

/** Client-side validation + spec assembly. Errors are shown next to the Fit button. */
export function buildFitSpec(
  selection: ModelSelection,
  advanced: AdvancedOptions,
): { spec: FitSpecJson } | { errors: string[] } {
  const errors: string[] = [];
  const model: Record<string, unknown> = {};

  if (selection.kind === "polynomial") {
    if (
      !Number.isInteger(selection.degree) ||
      selection.degree < 1 ||
      selection.degree > MAX_POLYNOMIAL_DEGREE
    ) {
      errors.push(`the polynomial degree must be between 1 and ${MAX_POLYNOMIAL_DEGREE}`);
    } else {
      model.kind = "polynomial";
      model.degree = selection.degree;
    }
  } else if (selection.kind === "registered") {
    if (selection.id === "") {
      errors.push("choose a model");
    } else {
      model.kind = "registered";
      model.id = selection.id;
    }
  } else {
    const equation = selection.equation.trim();
    if (equation === "") errors.push("the custom equation is required");
    else if (equation.length > MAX_CUSTOM_EQUATION_LENGTH)
      errors.push(`the equation must be at most ${MAX_CUSTOM_EQUATION_LENGTH} characters`);
    const parameters = selection.parameters
      .split(/[,\s]+/)
      .map((name) => name.trim())
      .filter((name) => name !== "");
    if (parameters.length > MAX_CUSTOM_PARAMETERS)
      errors.push(`at most ${MAX_CUSTOM_PARAMETERS} parameter names are allowed`);
    const variable = selection.variable.trim() || "x";
    if (parameters.some((name) => name === variable))
      errors.push(`the parameter names must not include the variable '${variable}'`);
    if (errors.length === 0) {
      model.kind = "custom";
      model.equation = equation;
      model.parameters = parameters;
      model.variable = variable;
    }
  }

  const isPolynomial = selection.kind === "polynomial";
  const spec: Record<string, unknown> = { model };

  if (!isPolynomial) {
    // Server defaults ("auto"/"linear"/no weights) are omitted to keep the wire format
    // minimal and the recorded spec honest about what was actually chosen.
    if (advanced.method !== "auto") spec.method = advanced.method;
    if (advanced.loss !== "linear") spec.loss = advanced.loss;
    if (advanced.useWeights) spec.use_weights = true;

    const initial: Record<string, number> = {};
    const bounds: Record<string, [number | null, number | null]> = {};
    for (const [name, raw] of Object.entries(advanced.initial)) {
      if (raw.trim() === "") continue; // cleared means unset, exactly like the bound fields
      const value = parseFloatStrict(raw);
      if (value === null) errors.push(`the initial value for '${name}' is not a number`);
      else initial[name] = value;
    }
    for (const name of [...Object.keys(advanced.lower), ...Object.keys(advanced.upper)]) {
      if (name in bounds) continue;
      // A field that was never filled (missing key) is empty, i.e. unset — not an error.
      const lowerRaw = advanced.lower[name] ?? "";
      const upperRaw = advanced.upper[name] ?? "";
      const lower = parseFloatStrict(lowerRaw);
      const upper = parseFloatStrict(upperRaw);
      if (lowerRaw.trim() !== "" && lower === null)
        errors.push(`the lower bound for '${name}' is not a number`);
      if (upperRaw.trim() !== "" && upper === null)
        errors.push(`the upper bound for '${name}' is not a number`);
      if (lower !== null && upper !== null && lower >= upper)
        errors.push(`the lower bound for '${name}' must be below the upper bound`);
      bounds[name] = [lower, upper];
    }
    if (Object.keys(initial).length > 0) spec.initial = initial;
    if (Object.keys(bounds).length > 0)
      spec.bounds = Object.fromEntries(
        Object.entries(bounds).filter(([, pair]) => pair[0] !== null || pair[1] !== null),
      );

    if (advanced.seed.trim() !== "") {
      const seed = parseIntStrict(advanced.seed);
      if (seed === null || seed > MAX_SEED) errors.push("the seed must be an integer from 0 to 2^53-1");
      else spec.seed = seed;
    }
    if (advanced.timeLimit.trim() !== "") {
      const timeLimit = parseFloatStrict(advanced.timeLimit);
      if (timeLimit === null || timeLimit <= 0)
        errors.push("the time limit must be a positive number of seconds");
      else spec.time_limit = timeLimit;
    }
  }

  if (errors.length > 0) return { errors };
  return { spec: spec as unknown as FitSpecJson };
}

/** The label of the selected model, for the panel header and history rows. */
export function modelLabel(selection: ModelSelection, models: ModelInfo[]): string {
  if (selection.kind === "polynomial")
    return `Polynomial (degree ${selection.degree || "?"})`;
  if (selection.kind === "custom") return "Custom equation";
  return models.find((m) => m.id === selection.id)?.name ?? selection.id;
}
