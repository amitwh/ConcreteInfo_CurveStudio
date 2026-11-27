import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useState, type ReactElement } from "react";
import type { components } from "../../api/schema";
import { modelColor } from "../traces";
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
    degenerate: false,
    generated: false,
    shape_parameters: 0,
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

/** jsdom normalises an inline hex colour to `rgb(r, g, b)`; compare in one space. */
function hexOf(cssColor: string): string {
  const match = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(cssColor);
  if (match === null) return cssColor.toUpperCase();
  return (
    "#" +
    match
      .slice(1)
      .map((part) => Number(part).toString(16).padStart(2, "0"))
      .join("")
      .toUpperCase()
  );
}

/** A getBoundingClientRect stand-in for the geometry jsdom cannot compute. */
function domRect(x: number, y: number, width: number, height: number): DOMRect {
  return {
    x,
    y,
    width,
    height,
    top: y,
    bottom: y + height,
    left: x,
    right: x + width,
    toJSON: () => ({}),
  } as DOMRect;
}

describe("RankingTable", () => {
  it("colours the swatch by display position, so tied ranks stay distinct", () => {
    renderTable([row("a", { rank: 1 }), row("b", { rank: 2 }), row("c", { rank: 2 })]);
    const swatches = document.querySelectorAll<HTMLElement>(".rank-swatch");
    expect(swatches).toHaveLength(3);
    expect(hexOf(swatches[0]?.style.backgroundColor ?? "")).toBe(modelColor(1).toUpperCase());
    expect(hexOf(swatches[1]?.style.backgroundColor ?? "")).toBe(modelColor(2).toUpperCase());
    expect(hexOf(swatches[2]?.style.backgroundColor ?? "")).toBe(modelColor(3).toUpperCase());
  });

  it("offsets the swatch position on a later page of the ranking", () => {
    renderTable([row("a", { rank: 2 }), row("b", { rank: 2 })], { positionOffset: 10 });
    const swatches = document.querySelectorAll<HTMLElement>(".rank-swatch");
    expect(hexOf(swatches[0]?.style.backgroundColor ?? "")).toBe(modelColor(11).toUpperCase());
    expect(hexOf(swatches[1]?.style.backgroundColor ?? "")).toBe(modelColor(12).toUpperCase());
  });

  it("renders the fitted equation, metrics and status for each row", () => {
    renderTable([row("exp_decay")]);
    expect(screen.getByText("y = 2.05 * exp(-0.32 * x)")).toBeTruthy();
    expect(screen.getByText("Exponential")).toBeTruthy();
    expect(screen.getAllByText("0.9982").length).toBeGreaterThan(0);
    expect(screen.getByText("Converged")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy equation" })).toBeTruthy();
  });

  it("shows human status badges with a semantic tone", () => {
    renderTable([
      row("ok", { status: "converged" }),
      row("bad", { status: "not_converged" }),
      row("other", { status: "timed_out" }),
    ]);
    const converged = screen.getByText("Converged");
    expect(converged.className).toContain("badge--ok");
    const notConverged = screen.getByText("Not converged");
    expect(notConverged.className).toContain("badge--bad");
    const other = screen.getByText("timed_out");
    expect(other.className).toContain("badge--muted");
  });

  it("right-aligns the numeric columns via the num header class", () => {
    renderTable([row("a")]);
    for (const name of ["R²", "adj. R²", "RMSE", "AICc", "BIC", "CV-RMSE", "k"]) {
      expect(screen.getByRole("columnheader", { name }).className).toContain("num");
    }
    expect(screen.getByRole("columnheader", { name: "Model" }).className).not.toContain("num");
  });

  it("keeps the full action names for screen readers in the row menu", async () => {
    renderTable([row("a")], {
      onUseManually: vi.fn(),
      manuallySelectable: new Set(["a"]),
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Row actions for Model a" }));
    expect(screen.getByRole("menuitem", { name: "Open full results" })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: "Use this model manually" })).toBeTruthy();
  });

  it("opens the last row's menu fully inside the viewport, never off-screen", async () => {
    // jsdom has no layout, so the trigger box, the rendered menu's box and the viewport
    // are stubbed (one prototype-level spy — mixing it with per-element stubs makes the
    // stub leak onto every element) and the inline placement is asserted exactly.
    const user = userEvent.setup();
    const viewport = { width: 1280, height: 700 };
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(viewport.width);
    vi.spyOn(window, "innerHeight", "get").mockReturnValue(viewport.height);
    const realRect = Element.prototype.getBoundingClientRect;
    const rectSpy = vi
      .spyOn(Element.prototype, "getBoundingClientRect")
      .mockImplementation(function (this: Element) {
        // The last row's kebab: at the table's right edge, 12px above the viewport floor.
        if (this.classList.contains("row-menu__toggle")) {
          return domRect(1040, 600, 228, 88);
        }
        // The rendered menu: 208px wide (13rem), two items tall.
        if (this.classList.contains("row-menu")) {
          return domRect(1040, 508, 208, 88);
        }
        return realRect.call(this);
      });
    try {
      renderTable([row("a")], {
        onUseManually: vi.fn(),
        manuallySelectable: new Set(["a"]),
      });
      const toggle = screen.getByRole("button", {
        name: "Row actions for Model a",
      });

      await user.click(toggle);

      const menu = document.querySelector<HTMLElement>(".row-menu");
      expect(menu).not.toBeNull();
      const left = Number.parseFloat(menu?.style.left ?? "");
      const top = Number.parseFloat(menu?.style.top ?? "");
      expect(left).toBe(1060); // right-aligned to the kebab's right edge (1268 - 208)
      expect(top).toBe(508); // flipped above: 600 - 4 - 88
      expect(left + 208).toBeLessThanOrEqual(viewport.width);
      expect(top + 88).toBeLessThanOrEqual(viewport.height);
    } finally {
      rectSpy.mockRestore();
    }
  });

  it("renders the equation copy control compact with the full accessible name", () => {
    renderTable([row("a")]);
    const copy = screen.getByRole("button", { name: "Copy equation" });
    expect(copy.textContent).toBe("Copy");
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

  it("exposes the actions cell class that anchors the sticky column", () => {
    renderTable([row("a")]);
    const header = screen.getByRole("columnheader", { name: "Actions" });
    expect(header.className).toContain("ranking-row__actions");
    const tableRow = screen.getByRole("row", { name: /Model a/ });
    const cells = within(tableRow).getAllByRole("cell");
    const actionsCell = cells[cells.length - 1];
    expect(actionsCell?.className).toContain("ranking-row__actions");
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

  describe("column sorting (display only)", () => {
    function baseMetrics() {
      return {
        k: 2,
        n: 40,
        r2: 0.99,
        adj_r2: 0.98,
        rmse: 0.05,
        aicc: -100.0,
        bic: -97.0,
        cv_rmse: 0.06,
      };
    }

    const sortedRows = [
      row("a", { rank: 1, name: "Ceta", metrics: { ...baseMetrics(), rmse: 0.05 } }),
      row("b", { rank: 2, name: "Alpha", metrics: { ...baseMetrics(), rmse: 0.02 } }),
      row("c", { rank: 3, name: "Beta", metrics: { ...baseMetrics(), rmse: 0.09 } }),
    ];

    it("clicking a header reorders the visible rows, never the served ranking", async () => {
      const user = userEvent.setup();
      renderTable(sortedRows);
      const names = () =>
        screen
          .getAllByRole("row")
          .slice(1)
          .map((element) => within(element).getByText(/^(Alpha|Beta|Ceta)$/).textContent);

      expect(names()).toEqual(["Ceta", "Alpha", "Beta"]);
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(names()).toEqual(["Alpha", "Ceta", "Beta"]); // ascending RMSE
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(names()).toEqual(["Beta", "Ceta", "Alpha"]); // descending
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(names()).toEqual(["Ceta", "Alpha", "Beta"]); // back to ranked

      // the ranked order itself is untouched: the rank column still ascends
      const ranks = screen
        .getAllByRole("row")
        .slice(1)
        .map((element) => within(element).getAllByText(/^\d+$/)[0]?.textContent);
      expect(ranks).toEqual(["1", "2", "3"]);
    });

    it("reports the active sort to assistive technology", async () => {
      const user = userEvent.setup();
      renderTable(sortedRows);
      const header = screen.getByRole("columnheader", { name: "RMSE" });
      expect(header.getAttribute("aria-sort")).toBe("none");
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(header.getAttribute("aria-sort")).toBe("ascending");
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(header.getAttribute("aria-sort")).toBe("descending");
      await user.click(screen.getByRole("button", { name: "RMSE" }));
      expect(header.getAttribute("aria-sort")).toBe("none");
    });

    it("keeps the swatch on its ranked position while the display is sorted", async () => {
      const user = userEvent.setup();
      renderTable(sortedRows);
      await user.click(screen.getByRole("button", { name: "RMSE" })); // b, a, c displayed
      const swatches = document.querySelectorAll<HTMLElement>(".rank-swatch");
      // display order b, a, c — but the swatch still shows each row's ranked colour
      expect(hexOf(swatches[0]?.style.backgroundColor ?? "")).toBe(modelColor(2).toUpperCase());
      expect(hexOf(swatches[1]?.style.backgroundColor ?? "")).toBe(modelColor(1).toUpperCase());
      expect(hexOf(swatches[2]?.style.backgroundColor ?? "")).toBe(modelColor(3).toUpperCase());
    });

    it("sorts the Model column alphabetically", async () => {
      const user = userEvent.setup();
      renderTable(sortedRows);
      await user.click(screen.getByRole("button", { name: "Model" }));
      const names = screen
        .getAllByRole("row")
        .slice(1)
        .map((element) => within(element).getByText(/^(Alpha|Beta|Ceta)$/).textContent);
      expect(names).toEqual(["Alpha", "Beta", "Ceta"]);
    });
  });

  describe("row action menu", () => {
    it("fires the open action with the same handler the button used", async () => {
      const user = userEvent.setup();
      const { onOpen } = renderTable([row("a")]);
      await user.click(screen.getByRole("button", { name: "Row actions for Model a" }));
      await user.click(screen.getByRole("menuitem", { name: "Open full results" }));
      expect(onOpen).toHaveBeenCalledWith("a");
    });

    it("fires the use-manually action for rows that offer it", async () => {
      const user = userEvent.setup();
      const onUseManually = vi.fn();
      renderTable([row("a")], {
        onUseManually,
        manuallySelectable: new Set(["a"]),
      });
      await user.click(screen.getByRole("button", { name: "Row actions for Model a" }));
      await user.click(screen.getByRole("menuitem", { name: "Use this model manually" }));
      expect(onUseManually).toHaveBeenCalledWith(expect.objectContaining({ model_id: "a" }));
    });

    it("offers nothing that the row's buttons did not offer", async () => {
      const user = userEvent.setup();
      renderTable([row("a")]); // no onUseManually: only the open action exists
      await user.click(screen.getByRole("button", { name: "Row actions for Model a" }));
      const items = screen.getAllByRole("menuitem");
      expect(items.map((item) => item.textContent)).toEqual(["Open full results"]);
    });

    it("closes on Escape and returns focus to the kebab", async () => {
      const user = userEvent.setup();
      renderTable([row("a")]);
      const kebab = screen.getByRole("button", { name: "Row actions for Model a" });
      await user.click(kebab);
      await user.keyboard("{Escape}");
      expect(screen.queryByRole("menu")).toBeNull();
      expect(document.activeElement).toBe(kebab);
    });

    it("blocks the row's actions while a row is opening, as the buttons were", () => {
      renderTable([row("a"), row("b")], { openRowBusy: "a" });
      const busy = screen.getByRole("button", {
        name: "Row actions for Model a",
      }) as HTMLButtonElement;
      expect(busy.disabled).toBe(true);
      expect(busy.textContent).toBe("Opening…");
      const other = screen.getByRole("button", {
        name: "Row actions for Model b",
      }) as HTMLButtonElement;
      expect(other.disabled).toBe(true);
      expect(other.textContent).toBe("Actions");
    });
  });
});
