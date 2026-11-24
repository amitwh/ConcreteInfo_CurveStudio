import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { DatasetDetailOut } from "../../api/client";
import { StatsPanel, type StatsPanelProps } from "./StatsPanel";
import type { StatsTools } from "./useStatsTools";

const DATASET: DatasetDetailOut = {
  id: "ds-1",
  name: "growth.csv",
  n_rows: 3,
  columns: ["age", "bmi", "sex"],
  x_column: "age",
  y_column: "bmi",
  weight_column: null,
  issues: [],
  created_at: "2026-11-17T05:00:00Z",
  points: null,
  table_stored: true,
  table_columns: [
    { name: "age", kind: "number", n_missing: 0 },
    { name: "bmi", kind: "number", n_missing: 0 },
    { name: "sex", kind: "text", n_missing: 1 },
  ],
};

const OLD_DATASET: DatasetDetailOut = {
  ...DATASET,
  table_stored: false,
  table_columns: [
    { name: "age", kind: "number", n_missing: 0 },
    { name: "bmi", kind: "number", n_missing: 0 },
  ],
};

const CATALOGUE: StatsTools = {
  families: [
    ["summarise", "Summarise"],
    ["showcase", "Showcase"],
    ["power", "Sample size and power"],
  ],
  tools: [
    {
      name: "descriptives",
      family: "summarise",
      title: "Descriptive statistics",
      needs_data: true,
      max_rows: null,
      fields: [
        {
          name: "values",
          label: "Values",
          kind: "column",
          required: true,
          description: "The numeric column to summarise.",
          option_schema: null,
        },
        {
          name: "groups",
          label: "Groups",
          kind: "group",
          required: false,
          description: "A grouping column; each level is summarised as its own group.",
          option_schema: null,
        },
        {
          name: "level",
          label: "Level",
          kind: "option",
          required: false,
          description: "The confidence level around each mean.",
          option_schema: { type: "number", default: 0.95 },
        },
      ],
    },
    {
      name: "showcase",
      family: "showcase",
      title: "Every option kind",
      needs_data: true,
      max_rows: null,
      fields: [
        {
          name: "values",
          label: "Values",
          kind: "column",
          required: true,
          description: null,
          option_schema: null,
        },
        {
          name: "columns",
          label: "Columns",
          kind: "columns",
          required: true,
          description: "The numeric columns to compare.",
          option_schema: null,
        },
        {
          name: "method",
          label: "Method",
          kind: "option",
          required: false,
          description: "The coefficient.",
          option_schema: { type: "string", enum: ["pearson", "spearman"], default: "pearson" },
        },
        {
          name: "level",
          label: "Level",
          kind: "option",
          required: false,
          description: "The confidence level.",
          option_schema: { type: "number", default: 0.95 },
        },
        {
          name: "seed",
          label: "Seed",
          kind: "option",
          required: false,
          description: "The random seed.",
          option_schema: { type: "integer", default: 7 },
        },
        {
          name: "exact",
          label: "Exact",
          kind: "option",
          required: false,
          description: "Use the exact test.",
          option_schema: { type: "boolean", default: false },
        },
        {
          name: "control",
          label: "Control",
          kind: "option",
          required: false,
          description: "A control group's name.",
          option_schema: {
            anyOf: [{ type: "string" }, { type: "null" }],
            default: null,
          },
        },
      ],
    },
    {
      name: "power",
      family: "power",
      title: "Sample size and power",
      needs_data: false,
      max_rows: null,
      fields: [
        {
          name: "design",
          label: "Design",
          kind: "option",
          required: true,
          description: "The design to compute for.",
          option_schema: { type: "string", enum: ["one_sample", "independent"] },
        },
        {
          name: "alpha",
          label: "Alpha",
          kind: "option",
          required: false,
          description: "The significance level.",
          option_schema: { type: "number", default: 0.05 },
        },
      ],
    },
  ],
};

function renderPanel(overrides: Partial<StatsPanelProps> = {}): StatsPanelProps {
  const props: StatsPanelProps = {
    dataset: DATASET,
    tools: CATALOGUE,
    onRun: vi.fn(),
    busy: false,
    error: null,
    ...overrides,
  };
  render(<StatsPanel {...props} />);
  return props;
}

function choose(name: string): void {
  fireEvent.change(screen.getByLabelText("Tool"), { target: { value: name } });
}

function runButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: "Run" }) as HTMLButtonElement;
}

describe("StatsPanel — the tool picker", () => {
  it("groups every tool under its family", () => {
    renderPanel();
    const summarise = screen.getByRole("group", { name: "Summarise" });
    expect(within(summarise).getByRole("option", { name: "Descriptive statistics" })).toBeTruthy();
    const power = screen.getByRole("group", { name: "Sample size and power" });
    expect(within(power).getByRole("option", { name: "Sample size and power" })).toBeTruthy();
  });

  it("shows a picker per field with its label and description once a tool is chosen", () => {
    renderPanel();
    choose("descriptives");
    expect(screen.getByLabelText("Values")).toBeTruthy();
    expect(screen.getByLabelText("Groups")).toBeTruthy();
    expect(screen.getByLabelText("Level")).toBeTruthy();
    expect(screen.getByText("The numeric column to summarise.")).toBeTruthy();
    expect(screen.getByText("The confidence level around each mean.")).toBeTruthy();
  });

  it("offers number columns to a column field and every column to a group field", () => {
    renderPanel();
    choose("descriptives");
    const values = screen.getByLabelText("Values") as HTMLSelectElement;
    const valueNames = Array.from(values.options).map((option) => option.value);
    expect(valueNames).toEqual(["age", "bmi"]);
    expect(valueNames).not.toContain("sex");
    const groups = screen.getByLabelText("Groups") as HTMLSelectElement;
    const groupNames = Array.from(groups.options).map((option) => option.value);
    expect(groupNames).toContain("sex");
  });

  it("renders an option of each kind with its default", () => {
    renderPanel();
    choose("showcase");
    expect((screen.getByLabelText("Method") as HTMLSelectElement).value).toBe("pearson");
    expect((screen.getByLabelText("Level") as HTMLInputElement).value).toBe("0.95");
    expect((screen.getByLabelText("Seed") as HTMLInputElement).value).toBe("7");
    expect((screen.getByLabelText("Exact") as HTMLInputElement).checked).toBe(false);
    expect((screen.getByLabelText("Control") as HTMLInputElement).value).toBe("");
    const columns = screen.getByLabelText("Columns") as HTMLSelectElement;
    expect(columns.multiple).toBe(true);
  });
});

describe("StatsPanel — Run's refusals", () => {
  it("refuses while a required field is empty", () => {
    renderPanel({ dataset: null });
    choose("power");
    expect(runButton().disabled).toBe(true);
    expect(screen.getByText("Fill in Design.")).toBeTruthy();
  });

  it("refuses a tool that needs data without a dataset", () => {
    renderPanel({ dataset: null });
    choose("descriptives");
    expect(runButton().disabled).toBe(true);
    expect(screen.getByText("Choose a dataset first.")).toBeTruthy();
  });

  it("refuses while a run is in flight", () => {
    renderPanel({ busy: true });
    choose("descriptives");
    expect(runButton().disabled).toBe(true);
    expect(screen.getByText("A run is in progress.")).toBeTruthy();
  });

  it("refuses before a tool is chosen", () => {
    renderPanel();
    expect(runButton().disabled).toBe(true);
    expect(screen.getByText("Choose a tool first.")).toBeTruthy();
  });
});

describe("StatsPanel — running", () => {
  it("sends exactly the filled choices", () => {
    const props = renderPanel();
    choose("descriptives");
    fireEvent.change(screen.getByLabelText("Values"), { target: { value: "age" } });
    fireEvent.change(screen.getByLabelText("Groups"), { target: { value: "sex" } });
    fireEvent.change(screen.getByLabelText("Level"), { target: { value: "0.99" } });
    expect(runButton().disabled).toBe(false);
    fireEvent.click(runButton());
    expect(props.onRun).toHaveBeenCalledWith("descriptives", {
      values: "age",
      groups: "sex",
      level: 0.99,
    });
  });

  it("runs a tool that needs no data with no dataset", () => {
    const props = renderPanel({ dataset: null });
    choose("power");
    fireEvent.change(screen.getByLabelText("Design"), { target: { value: "independent" } });
    expect(runButton().disabled).toBe(false);
    fireEvent.click(runButton());
    expect(props.onRun).toHaveBeenCalledWith(
      "power",
      expect.objectContaining({ design: "independent", alpha: 0.05 }),
    );
  });

  it("shows the sentence and only the fit's columns for a dataset from before", () => {
    renderPanel({ dataset: OLD_DATASET });
    choose("descriptives");
    expect(screen.getByText(/uploaded before its whole table was kept/)).toBeTruthy();
    const values = screen.getByLabelText("Values") as HTMLSelectElement;
    expect(Array.from(values.options).map((option) => option.value)).toEqual(["age", "bmi"]);
  });
});
