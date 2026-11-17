import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useState, type ReactElement } from "react";
import type { components } from "../../api/schema";
import { RankingTable } from "./RankingTable";

type RankedFit = components["schemas"]["RankedFit"];

function row(modelId: string, overrides: Record<string, unknown> = {}): RankedFit {
  return {
    rank: 1,
    model_id: modelId,
    name: `Model ${modelId}`,
    category: "Exponential",
    equation_fitted: "y = 2.05 * exp(-0.32 * x)",
    metrics: {
      k: 2,
      n: 40,
      r2: 0.9982,
      adj_r2: 0.9981,
      rmse: 0.04521,
      aicc: -120.4,
      bic: -117.2,
      cv_rmse: 0.05112,
    },
    status: "converged",
    same_curve_as: [],
    warnings: [],
    overfitting_note: null,
    parameters: [],
    ...overrides,
  } as RankedFit;
}

function renderTable(
  rows: RankedFit[],
  props: Partial<Parameters<typeof RankingTable>[0]> = {},
) {
  const onSelect = vi.fn();
  const onOpen = vi.fn();
  render(
    <RankingTable
      rows={rows}
      selectedId={null}
      curvesAvailable={new Set(rows.map((r) => r.model_id))}
      onSelect={onSelect}
      onOpen={onOpen}
      {...props}
    />,
  );
  return { onSelect, onOpen };
}

describe("RankingTable", () => {
  it("renders the fitted equation, metrics and status for each row", () => {
    renderTable([row("exp_decay")]);
    expect(screen.getByText("y = 2.05 * exp(-0.32 * x)")).toBeTruthy();
    expect(screen.getByText("Exponential")).toBeTruthy();
    expect(screen.getAllByText("0.9982").length).toBeGreaterThan(0);
    expect(screen.getByText("converged")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy equation" })).toBeTruthy();
  });

  it("renders n/a for undefined metrics, never 0 or NaN", () => {
    renderTable([
      row("weibull", {
        metrics: { k: 3, n: 40, r2: null, adj_r2: null, rmse: null, aicc: null, bic: null, cv_rmse: null },
      }),
    ]);
    // 6 undefined metrics on the row
    expect(screen.getAllByText("n/a").length).toBeGreaterThanOrEqual(6);
  });

  it("shows the shared rank and the same-curve note for tied rows", () => {
    renderTable([
      row("exp_decay", { rank: 1 }),
      row("exp_decay_alias", { rank: 1, same_curve_as: ["exp_decay"] }),
      row("poly3", { rank: 3 }),
    ]);
    const note = screen.getByText(/same curve as:/);
    expect(note.textContent).toContain("exp_decay");
    const ranks = screen.getAllByText("1");
    expect(ranks.length).toBe(2);
  });

  it("selects a row on click and moves with arrow keys; Enter opens it", async () => {
    const rows = [row("a", { rank: 1 }), row("b", { rank: 2 }), row("c", { rank: 3 })];
    const onSelect = vi.fn();
    const onOpen = vi.fn();
    // A controlled harness: selection state lives outside the table, as in the page.
    function Harness(): ReactElement {
      const [selected, setSelected] = useState<string | null>("a");
      return (
        <RankingTable
          rows={rows}
          selectedId={selected}
          curvesAvailable={new Set(["a", "b", "c"])}
          onSelect={(id) => {
            setSelected(id);
            onSelect(id);
          }}
          onOpen={onOpen}
        />
      );
    }
    render(<Harness />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("row", { name: /Model b/ }));
    expect(onSelect).toHaveBeenCalledWith("b");

    const rowC = screen.getByRole("row", { name: /Model c/ });
    await user.click(rowC);
    // keyboard from the focused selected row: up selects the previous row, down the next
    expect(rowC.getAttribute("aria-selected")).toBe("true");
    rowC.focus();
    await user.keyboard("{ArrowUp}");
    expect(onSelect).toHaveBeenLastCalledWith("b");
    await user.keyboard("{ArrowDown}");
    expect(onSelect).toHaveBeenLastCalledWith("c");
    await user.keyboard("{Enter}");
    expect(onOpen).toHaveBeenCalledWith("c");
  });

  it("marks the selected row and shows which rows have a stored curve", () => {
    renderTable([row("a"), row("b")], {
      selectedId: "b",
      curvesAvailable: new Set(["a"]),
    });
    const selected = screen.getByRole("row", { name: /Model b/ });
    expect(selected.getAttribute("aria-selected")).toBe("true");
    const noCurve = within(screen.getByRole("row", { name: /Model b/ })).getByText(
      /no stored curve/i,
    );
    expect(noCurve).toBeTruthy();
  });
});
