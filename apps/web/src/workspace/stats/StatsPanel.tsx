/**
 * The Statistics mode's tool form: the catalogue grouped by family, and for the chosen tool
 * one control per request field — a numeric column, a group column, several columns, or an
 * option rendered from its schema. Run is refused with its reason while a required field is
 * empty, no dataset is selected for a tool that needs one, or a run is already in flight.
 */

import { useEffect, useId, useState, type ReactElement } from "react";
import {
  ApiError,
  type DatasetDetailOut,
  type RequestField,
  type ToolInfo,
} from "../../api/client";
import { errorText } from "../../components/States";
import type { StatsTools } from "./useStatsTools";

export interface StatsPanelProps {
  /** The dataset the run reads its columns from; null for a tool that needs none. */
  dataset: DatasetDetailOut | null;
  /** The catalogue; null until it has loaded (or when it failed — see `error`). */
  tools: StatsTools | null;
  onRun: (tool: string, choices: Record<string, unknown>) => void;
  /** True while a run is in flight: Run is disabled. */
  busy: boolean;
  /** A catalogue or run failure. */
  error: unknown;
}

/** The most columns a several-columns field lists (the tools take at most twenty). */
const MAX_LISTED_COLUMNS = 20;

/** A field's value in the form: column names, a list of them, an option's text or a flag. */
type FieldValue = string | boolean | string[];

type OptionKind = "boolean" | "integer" | "number" | "select" | "text";

function schemaBranches(schema: Record<string, unknown>): Record<string, unknown>[] {
  const anyOf = schema.anyOf;
  if (!Array.isArray(anyOf)) return [schema];
  return anyOf.filter(
    (branch): branch is Record<string, unknown> =>
      typeof branch === "object" && branch !== null && !Array.isArray(branch),
  );
}

/** How an option reads, from the JSON schema its field declares. */
function optionKind(schema: Record<string, unknown>): OptionKind {
  const branches = schemaBranches(schema);
  const types = new Set(
    branches
      .map((branch) => branch.type)
      .filter((type): type is string => typeof type === "string"),
  );
  const hasChoices = branches.some((branch) => Array.isArray(branch.enum));
  if (types.has("boolean")) return "boolean";
  if (types.has("integer")) return "integer";
  if (types.has("number")) return "number";
  if (types.has("string")) return hasChoices ? "select" : "text";
  return "text";
}

/** The values a select option offers, as strings. */
function optionChoices(schema: Record<string, unknown>): string[] {
  for (const branch of schemaBranches(schema)) {
    if (Array.isArray(branch.enum)) return branch.enum.map((value) => String(value));
  }
  return [];
}

/** The value a field starts at: an option's schema default, or empty. */
function defaultValue(field: RequestField): FieldValue {
  if (field.kind === "columns") return [];
  if (field.kind === "column" || field.kind === "group") return "";
  const schema = field.option_schema ?? {};
  const kind = optionKind(schema);
  const fallback = schema.default;
  if (kind === "boolean") return fallback === true;
  if (kind === "number" || kind === "integer") {
    return typeof fallback === "number" ? String(fallback) : "";
  }
  return typeof fallback === "string" ? fallback : "";
}

function defaultsFor(tool: ToolInfo): Record<string, FieldValue> {
  const values: Record<string, FieldValue> = {};
  for (const field of tool.fields) values[field.name] = defaultValue(field);
  return values;
}

/** The columns a field may name, from the dataset's stored table. */
function optionsFor(field: RequestField, dataset: DatasetDetailOut | null): string[] {
  if (dataset === null) return [];
  if (field.kind === "column") {
    return dataset.table_columns.filter((column) => column.kind === "number").map((c) => c.name);
  }
  if (field.kind === "group") return dataset.table_columns.map((column) => column.name);
  if (field.kind === "columns") {
    return dataset.table_columns
      .filter((column) => column.kind === "number")
      .map((column) => column.name)
      .slice(0, MAX_LISTED_COLUMNS);
  }
  return [];
}

/** A required column control starts on its first choice, so what it shows is what it sends. */
function shownValue(field: RequestField, raw: FieldValue, options: string[]): FieldValue {
  if (
    (field.kind === "column" || field.kind === "group") &&
    raw === "" &&
    field.required &&
    options.length > 0
  ) {
    return options[0] as string;
  }
  return raw;
}

function valueOf(
  field: RequestField,
  values: Record<string, FieldValue>,
  dataset: DatasetDetailOut | null,
): FieldValue {
  const raw = values[field.name] ?? defaultValue(field);
  return shownValue(field, raw, optionsFor(field, dataset));
}

/** Why Run is refused, or null when it may run. */
function blockReason(
  tool: ToolInfo,
  values: Record<string, FieldValue>,
  dataset: DatasetDetailOut | null,
): string | null {
  if (tool.needs_data && dataset === null) return "Choose a dataset first.";
  for (const field of tool.fields) {
    if (!field.required) continue;
    const value = valueOf(field, values, dataset);
    const empty = field.kind === "columns" ? (value as string[]).length === 0 : value === "";
    if (empty) return `Fill in ${field.label}.`;
  }
  return null;
}

/** The request body: every filled field, an option as the number or flag its schema declares. */
function buildChoices(
  tool: ToolInfo,
  values: Record<string, FieldValue>,
  dataset: DatasetDetailOut | null,
): Record<string, unknown> {
  const choices: Record<string, unknown> = {};
  for (const field of tool.fields) {
    const value = valueOf(field, values, dataset);
    if (field.kind === "column" || field.kind === "group") {
      if (value !== "") choices[field.name] = value;
      continue;
    }
    if (field.kind === "columns") {
      const chosen = value as string[];
      if (chosen.length > 0) choices[field.name] = chosen;
      continue;
    }
    const kind = optionKind(field.option_schema ?? {});
    if (kind === "boolean") {
      choices[field.name] = value === true;
      continue;
    }
    if (value === "") continue;
    if (kind === "integer") {
      const parsed = Number.parseInt(String(value), 10);
      if (Number.isInteger(parsed)) choices[field.name] = parsed;
      continue;
    }
    if (kind === "number") {
      const parsed = Number(String(value));
      if (Number.isFinite(parsed)) choices[field.name] = parsed;
      continue;
    }
    choices[field.name] = value;
  }
  return choices;
}

/** A service failure as its message and its code (the tool's code when the envelope names one). */
function statErrorText(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail as { code?: unknown } | null;
    const code = typeof detail?.code === "string" ? detail.code : error.code;
    return `${error.message} (${code})`;
  }
  return errorText(error);
}

export function StatsPanel({
  dataset,
  tools,
  onRun,
  busy,
  error,
}: StatsPanelProps): ReactElement {
  const baseId = useId();
  const [toolName, setToolName] = useState("");
  const [values, setValues] = useState<Record<string, FieldValue>>({});

  const tool = tools?.tools.find((entry) => entry.name === toolName) ?? null;
  const datasetKey = dataset?.id ?? null;

  // A new dataset's columns are not the old ones: clear every column choice, keep options.
  useEffect(() => {
    if (tool === null) return;
    setValues((current) => {
      const next = { ...current };
      for (const field of tool.fields) {
        if (field.kind === "column" || field.kind === "group") next[field.name] = "";
        else if (field.kind === "columns") next[field.name] = [];
      }
      return next;
    });
  }, [datasetKey, tool]);

  const chooseTool = (name: string): void => {
    setToolName(name);
    const chosen = tools?.tools.find((entry) => entry.name === name) ?? null;
    setValues(chosen === null ? {} : defaultsFor(chosen));
  };

  const setField = (name: string, value: FieldValue): void => {
    setValues((current) => ({ ...current, [name]: value }));
  };

  const reason = tool === null ? null : blockReason(tool, values, dataset);
  const blocked = tool === null || busy || reason !== null;
  const disabledReason =
    tool === null
      ? "Choose a tool first."
      : busy
        ? "A run is in progress."
        : reason;

  return (
    <section className="panel" aria-label="Statistics">
      <h2 className="panel-title">Statistics</h2>
      {error != null && (
        <p className="alert alert--error" role="alert">
          {statErrorText(error)}
        </p>
      )}
      {tools === null ? (
        error == null && <p className="hint">Loading the tool catalogue…</p>
      ) : (
        <>
          <div className="field">
            <label htmlFor={`${baseId}-tool`}>Tool</label>
            <select
              id={`${baseId}-tool`}
              value={toolName}
              onChange={(event) => chooseTool(event.target.value)}
            >
              <option value="">Choose a tool</option>
              {tools.families.map(([key, title]) => {
                const familyTools = tools.tools.filter((entry) => entry.family === key);
                if (familyTools.length === 0) return null;
                return (
                  <optgroup key={key} label={title}>
                    {familyTools.map((entry) => (
                      <option key={entry.name} value={entry.name}>
                        {entry.title}
                      </option>
                    ))}
                  </optgroup>
                );
              })}
            </select>
          </div>

          {tool !== null &&
            tool.fields.map((field) => {
              const options = optionsFor(field, dataset);
              const value = valueOf(field, values, dataset);
              const fieldId = `${baseId}-${field.name}`;
              const descId = `${fieldId}-desc`;
              const describedBy = field.description == null ? undefined : descId;
              return (
                <div key={field.name} className="field">
                  <label htmlFor={fieldId}>{field.label}</label>
                  {field.kind === "column" || field.kind === "group" ? (
                    <select
                      id={fieldId}
                      aria-describedby={describedBy}
                      value={String(value)}
                      onChange={(event) => setField(field.name, event.target.value)}
                    >
                      {field.kind === "group" && !field.required && (
                        <option value="">Not grouped</option>
                      )}
                      {options.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : field.kind === "columns" ? (
                    <select
                      id={fieldId}
                      multiple
                      aria-describedby={describedBy}
                      value={value as string[]}
                      onChange={(event) =>
                        setField(
                          field.name,
                          Array.from(event.target.selectedOptions, (option) => option.value),
                        )
                      }
                    >
                      {options.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    renderOption(field, value, fieldId, describedBy, setField)
                  )}
                  {field.description != null && (
                    <p className="hint" id={descId}>
                      {field.description}
                    </p>
                  )}
                </div>
              );
            })}

          {tool !== null && dataset !== null && !dataset.table_stored && (
            <p className="hint">
              This dataset was uploaded before its whole table was kept. Upload it again to
              analyse its other columns.
            </p>
          )}

          <div className="field">
            <button
              type="button"
              onClick={() => {
                if (tool !== null) onRun(tool.name, buildChoices(tool, values, dataset));
              }}
              disabled={blocked}
            >
              Run
            </button>
            {disabledReason != null && (
              <p className="hint" role="status">
                {disabledReason}
              </p>
            )}
          </div>
        </>
      )}
    </section>
  );
}

/** One option control, by the kind its schema declares. */
function renderOption(
  field: RequestField,
  value: FieldValue,
  id: string,
  describedBy: string | undefined,
  setField: (name: string, value: FieldValue) => void,
): ReactElement {
  const schema = field.option_schema ?? {};
  const kind = optionKind(schema);
  if (kind === "boolean") {
    return (
      <input
        id={id}
        type="checkbox"
        aria-describedby={describedBy}
        checked={value === true}
        onChange={(event) => setField(field.name, event.target.checked)}
      />
    );
  }
  if (kind === "integer" || kind === "number") {
    return (
      <input
        id={id}
        type="number"
        step={kind === "integer" ? "1" : "any"}
        aria-describedby={describedBy}
        value={String(value)}
        onChange={(event) => setField(field.name, event.target.value)}
      />
    );
  }
  if (kind === "select") {
    return (
      <select
        id={id}
        aria-describedby={describedBy}
        value={String(value)}
        onChange={(event) => setField(field.name, event.target.value)}
      >
        <option value="">Choose…</option>
        {optionChoices(schema).map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  }
  return (
    <input
      id={id}
      type="text"
      aria-describedby={describedBy}
      value={String(value)}
      onChange={(event) => setField(field.name, event.target.value)}
    />
  );
}
