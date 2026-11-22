import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { FitPanel } from "./FitPanel";

type DatasetDetailOut = components["schemas"]["DatasetDetailOut"];
type ModelInfo = components["schemas"]["ModelInfo"];

const dataset = (weightColumn: string | null): DatasetDetailOut => ({
  id: "ds-1",
  name: "growth.csv",
  n_rows: 40,
  columns: ["t", "y", "w"],
  x_column: "t",
  y_column: "y",
  weight_column: weightColumn,
  issues: [],
  created_at: "2026-11-17T05:00:00Z",
  points: { n_total: 40, x: [1, 2], y: [2, 4], w: null },
});

const models: ModelInfo[] = [
  {
    id: "exp_decay",
    name: "Exponential decay",
    category: "Exponential",
    equation: "y = a * exp(-b * x)",
    latex: "",
    parameters: [
      { name: "a", description: "amplitude", unit: "", lower: null, upper: null },
      { name: "b", description: "rate", unit: "1/x", lower: 0, upper: null },
    ],
    variables: ["x"],
    references: [],
    assumptions: "x real",
    is_family: false,
  },
  {
    // The API catalogue appends its own polynomial family entry next to the
    // registered models — the picker must not offer it twice.
    id: "polynomial",
    name: "Polynomial",
    category: "polynomial",
    equation: "y = b0 + b1*x + ... + bn*x**n",
    latex: "",
    parameters: [
      { name: "b0", description: "Intercept (constant term)", unit: "", lower: null, upper: null },
      { name: "b1", description: "Coefficient of x", unit: "", lower: null, upper: null },
    ],
    variables: ["x"],
    references: [],
    assumptions: "",
    is_family: true,
  },
];

function renderPanel(overrides: Partial<Parameters<typeof FitPanel>[0]> = {}) {
  const onFit = vi.fn();
  render(
    <FitPanel
      dataset={dataset(null)}
      models={models}
      busy={false}
      onFit={onFit}
      fitError={null}
      {...overrides}
    />,
  );
  return onFit;
}

describe("FitPanel validation (mirrors the server contract)", () => {
  it("rejects a polynomial degree outside 1..10 without submitting", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "__polynomial__");
    const degree = screen.getByLabelText("Degree");
    await user.clear(degree);
    await user.type(degree, "11");
    await user.click(screen.getByRole("button", { name: "Fit" }));
    const error = await screen.findByRole("alert");
    expect(error.textContent).toContain("degree");
    expect(onFit).not.toHaveBeenCalled();
  });

  it("requires a non-empty custom equation", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.click(screen.getByRole("button", { name: "Fit" }));
    expect((await screen.findByRole("alert")).textContent).toContain("equation");
    expect(onFit).not.toHaveBeenCalled();
  });

  // 15 s timeout: 2200 synthetic keystrokes are inherently slow, and the default 5 s
  // flakes under heavy parallel load on the build machine (observed at load ~20).
  it("rejects a custom equation longer than 2000 characters", async () => {
    const onFit = renderPanel();
    // delay: null — 2200 keystrokes still fire every event, without waiting between keys.
    const user = userEvent.setup({ delay: null });
    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.type(screen.getByLabelText("Equation"), "a*".repeat(1100));
    await user.click(screen.getByRole("button", { name: "Fit" }));
    expect((await screen.findByRole("alert")).textContent).toContain("2000");
    expect(onFit).not.toHaveBeenCalled();
  }, 15_000);

  it("rejects a negative seed and a non-positive time limit", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    await user.type(screen.getByLabelText("Seed"), "-1");
    await user.click(screen.getByRole("button", { name: "Fit" }));
    expect((await screen.findByRole("alert")).textContent).toContain("seed");
    expect(onFit).not.toHaveBeenCalled();
  });

  it("submits a minimal registered-model spec; bounds prefilled from model defaults", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    await user.click(screen.getByRole("button", { name: "Fit" }));
    await waitFor(() => expect(onFit).toHaveBeenCalledTimes(1));
    expect(onFit).toHaveBeenCalledWith({
      model: { kind: "registered", id: "exp_decay" },
      // b's default lower bound (0) is prefilled into the table and therefore sent;
      // a is unbounded so nothing is sent for it.
      bounds: { b: [0, null] },
    });
  });

  it("sends degree, bounds and initial values for a polynomial-free custom model", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.type(screen.getByLabelText("Equation"), "a * x + b");
    await user.type(screen.getByLabelText("Parameter names"), "a, b");
    const initialA = screen.getByLabelText("Initial a");
    await user.type(initialA, "2");
    await user.click(screen.getByRole("button", { name: "Fit" }));
    await waitFor(() => expect(onFit).toHaveBeenCalledTimes(1));
    expect(onFit).toHaveBeenCalledWith({
      model: { kind: "custom", equation: "a * x + b", parameters: ["a", "b"], variable: "x" },
      initial: { a: 2 },
    });
  });

  it("treats a cleared Initial field as unset instead of blocking Fit", async () => {
    const onFit = renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.type(screen.getByLabelText("Equation"), "a * x + b");
    await user.type(screen.getByLabelText("Parameter names"), "a, b");
    const initialA = screen.getByLabelText("Initial a");
    await user.type(initialA, "2");
    await user.clear(initialA);
    await user.click(screen.getByRole("button", { name: "Fit" }));
    await waitFor(() => expect(onFit).toHaveBeenCalledTimes(1));
    // No `initial` key at all: the empty field means "unset", like the bound fields.
    expect(onFit).toHaveBeenCalledWith({
      model: { kind: "custom", equation: "a * x + b", parameters: ["a", "b"], variable: "x" },
    });
  });

  it("disables method, loss, bounds and initial inputs for polynomials", async () => {
    renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "__polynomial__");
    expect((screen.getByLabelText("Method") as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByLabelText("Loss") as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByLabelText("Seed") as HTMLInputElement).disabled).toBe(true);
  });

  it("enables the weights checkbox only when the dataset has weights", () => {
    renderPanel({ dataset: dataset("w") });
    expect((screen.getByLabelText("Use weights") as HTMLInputElement).disabled).toBe(false);
  });

  it("disables the weights checkbox when the dataset has no weight column", () => {
    renderPanel({ dataset: dataset(null) });
    expect((screen.getByLabelText("Use weights") as HTMLInputElement).disabled).toBe(true);
  });

  it("disables the Fit button while a job is running (double-click safe)", () => {
    renderPanel({ busy: true });
    expect((screen.getByRole("button", { name: "Fit" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it("shows a server-side validation message inline", () => {
    renderPanel({ fitError: "unknown function 'sinc' in equation" });
    expect(screen.getByRole("alert").textContent).toContain("unknown function");
  });
});

describe("bounds and initial values follow the selection (no stale carry-over)", () => {
  it("clears them when the model changes, in both directions", async () => {
    renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    await user.type(screen.getByLabelText("Initial a"), "5");
    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.type(screen.getByLabelText("Equation"), "a * x + b");
    await user.type(screen.getByLabelText("Parameter names"), "a, b");
    // the custom 'a' must not silently inherit exp_decay's stale initial 5
    expect((screen.getByLabelText("Initial a") as HTMLInputElement).value).toBe("");
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    // and switching back starts from the model defaults again
    expect((screen.getByLabelText("Initial a") as HTMLInputElement).value).toBe("");
  });

  it("prefills bounds from the model's parameter defaults, blank when unbounded", async () => {
    renderPanel();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    expect((screen.getByLabelText("Lower bound a") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Lower bound b") as HTMLInputElement).value).toBe("0");
    expect((screen.getByLabelText("Upper bound b") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Initial a") as HTMLInputElement).value).toBe("");
  });

  it("clears them when the dataset changes (model stays selected, defaults return)", async () => {
    const onFit = vi.fn();
    const props = (ds: DatasetDetailOut) => ({
      dataset: ds,
      models,
      busy: false,
      onFit,
      fitError: null,
    });
    const view = render(<FitPanel {...props(dataset(null))} />);
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Model"), "exp_decay");
    await user.type(screen.getByLabelText("Initial a"), "5");
    view.rerender(<FitPanel {...props({ ...dataset("w"), id: "ds-2", name: "other.csv" })} />);
    expect((screen.getByLabelText("Initial a") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Lower bound b") as HTMLInputElement).value).toBe("0");
  });

  it("lists Polynomial once even though the API catalogue also has a polynomial family", () => {
    renderPanel();
    const polynomialOptions = screen
      .getAllByRole("option")
      .filter((option) => option.textContent?.startsWith("Polynomial") === true);
    expect(polynomialOptions).toHaveLength(1);
  });
});

describe("FitPanel model picker", () => {
  it("never hides the selected model behind a search filter", async () => {
    const user = userEvent.setup();
    render(
      <FitPanel dataset={dataset(null)} models={models} busy={false} onFit={vi.fn()} fitError={null} />,
    );
    const select = screen.getByLabelText("Model") as HTMLSelectElement;
    await user.selectOptions(select, "exp_decay");
    expect(select.value).toBe("exp_decay");

    // a filter that matches nothing still leaves the selected model available
    const search = screen.getByLabelText(/search models/i);
    await user.type(search, "zzzz-no-such-model");
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("exp_decay");
    expect(screen.getByRole("option", { name: "Exponential decay" })).toBeTruthy();
  });

  it("filters the options by id, name, category or equation", async () => {
    const user = userEvent.setup();
    render(
      <FitPanel dataset={dataset(null)} models={models} busy={false} onFit={vi.fn()} fitError={null} />,
    );
    const search = screen.getByLabelText(/search models/i);
    await user.type(search, "decay");
    expect(screen.getByRole("option", { name: "Exponential decay" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: /polynomial/ })).toBeNull();
    // the fixed entries stay reachable at any time
    expect(screen.getByRole("option", { name: /Custom equation/ })).toBeTruthy();
  });
});
