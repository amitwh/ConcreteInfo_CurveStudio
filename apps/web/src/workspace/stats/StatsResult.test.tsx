/**
 * The one view of a statistical result: every block is shown, a missing number reads as an
 * em dash, a p-value only through `pText`, and the methods fragment can be copied.
 */

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FIXTURE_RESULT } from "./fixture";
import { StatsResult } from "./StatsResult";

// jsdom cannot host a real Plotly graph; every chart is replaced by a stand-in here.
vi.mock("../../components/PlotChart", () => ({
  PlotChart: () => <div data-testid="plot" />,
}));

const writeText = vi.fn();

beforeEach(() => {
  writeText.mockReset();
  writeText.mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
});

function renderResult(): void {
  render(<StatsResult result={FIXTURE_RESULT} />);
}

/** The one row a label sits in. */
function rowOf(label: string): HTMLElement {
  const cell = screen.getByText(label);
  const row = cell.closest("tr");
  expect(row, `no row holds ${label}`).not.toBeNull();
  return row as HTMLElement;
}

describe("StatsResult — the heading and the level", () => {
  it("shows the result's title", () => {
    renderResult();
    expect(screen.getByRole("heading", { name: FIXTURE_RESULT.title })).toBeTruthy();
  });

  it("puts the confidence level in words", () => {
    renderResult();
    expect(screen.getByText("Confidence level: 95%")).toBeTruthy();
  });
});

describe("StatsResult — estimates", () => {
  it("gives every estimate a row with its SE and interval, a missing number as an em dash", () => {
    renderResult();
    const table = screen.getByRole("table", { name: "Estimates" });
    for (const column of ["Label", "Value", "SE", "Lower", "Upper"]) {
      expect(within(table).getByText(column)).toBeTruthy();
    }

    const complete = rowOf("Mean of A");
    expect(within(complete).getByText("24.5")).toBeTruthy();
    expect(within(complete).getByText("1.2")).toBeTruthy();
    expect(within(complete).getByText("21.9")).toBeTruthy();
    expect(within(complete).getByText("27.1")).toBeTruthy();

    // The difference has no interval: each missing end is an em dash, never a zero.
    const incomplete = rowOf("Difference in means (B - A)");
    expect(within(incomplete).getAllByText("—")).toHaveLength(2);
    expect(within(incomplete).queryByText("0")).toBeNull();
  });
});

describe("StatsResult — hypothesis tests", () => {
  it("shows each test's p-value text", () => {
    renderResult();
    const table = screen.getByRole("table", { name: "Tests" });
    for (const column of ["Test", "Method", "Statistic", "df", "p", "Adjusted p", "Adjustment"]) {
      expect(within(table).getByText(column)).toBeTruthy();
    }

    expect(within(rowOf("Difference in means")).getByText("p = 0.0324")).toBeTruthy();
    expect(
      within(rowOf("A against B, adjusted for the family")).getByText("p = 0.02"),
    ).toBeTruthy();
    expect(within(rowOf("Normality of A")).getByText("not computed")).toBeTruthy();
  });

  it("shows an adjusted p-value with the adjustment that produced it", () => {
    renderResult();
    const row = rowOf("A against B, adjusted for the family");
    expect(within(row).getByText("p = 0.05")).toBeTruthy();
    expect(within(row).getByText("Holm")).toBeTruthy();
  });

  it("puts the null hypothesis under a row, and the reason a test could not run", () => {
    renderResult();
    expect(screen.getByText(/The two population means are equal\./)).toBeTruthy();
    expect(screen.getByText(/The values of A come from a normal distribution\./)).toBeTruthy();
    expect(screen.getByText(/Group A has fewer than three observations\./)).toBeTruthy();
    // A test that produced no statistic, no df and no adjustment shows an em dash in each.
    expect(within(rowOf("Normality of A")).getAllByText("—")).toHaveLength(4);
  });
});

describe("StatsResult — assumptions", () => {
  it("shows one card per assumption, with the verdict's word and colour", () => {
    renderResult();
    const cards = Array.from(document.querySelectorAll<HTMLElement>(".audit-check"));
    expect(cards).toHaveLength(FIXTURE_RESULT.assumptions.length);
    const words = cards.map((card) => card.querySelector(".verdict")?.textContent);
    expect(words).toEqual(["Pass", "Caution", "Fail", "Not applicable"]);
    for (const card of cards) {
      const id = card.getAttribute("data-check");
      const check = FIXTURE_RESULT.assumptions.find((item) => item.id === id);
      expect(check).toBeDefined();
      expect(card.querySelector(".verdict")?.className).toContain(`verdict--${check?.verdict}`);
      expect(within(card).getByText(check?.evidence as string)).toBeTruthy();
      expect(within(card).getByText(check?.explanation as string)).toBeTruthy();
    }
  });

  it("shows a check's statistic, threshold and next step when the server sends them", () => {
    renderResult();
    const card = document.querySelector<HTMLElement>('[data-check="normality"]');
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText("Shapiro-Wilk p = 0.081")).toBeTruthy();
    expect(within(card as HTMLElement).getByText(/p >= 0.1 pass/)).toBeTruthy();
    expect(
      within(card as HTMLElement).getByText(
        "Read the result with the interval in mind, or compare the medians with a rank test.",
      ),
    ).toBeTruthy();
  });

  it("keeps a check that did not apply quiet, and gives it no next step", () => {
    renderResult();
    const card = document.querySelector<HTMLElement>('[data-check="group_sizes"]');
    expect(card?.className).toContain("audit-check--quiet");
    expect(card?.textContent).not.toContain("Next step");
  });
});

describe("StatsResult — warnings, tables and interpretation", () => {
  it("shows every warning", () => {
    renderResult();
    for (const warning of FIXTURE_RESULT.warnings) {
      expect(screen.getByText(warning)).toBeTruthy();
    }
  });

  it("shows every table with its title, a cell and its note", () => {
    renderResult();
    const cells: Record<string, string> = {
      "Summary by group": "A",
      "Pairwise comparisons": "B - A",
    };
    for (const table of FIXTURE_RESULT.tables) {
      expect(screen.getByRole("heading", { name: table.title })).toBeTruthy();
      const element = screen.getByRole("table", { name: table.title });
      expect(within(element).getByText(cells[table.title] as string)).toBeTruthy();
      expect(screen.getByText(table.note as string)).toBeTruthy();
    }
  });

  it("reads a p-value column through pText, and leaves a missing p as not computed", () => {
    renderResult();
    const summary = screen.getByRole("table", { name: "Summary by group" });
    expect(within(summary).getByText("p = 0.0324")).toBeTruthy();
    expect(within(summary).getByText("not computed")).toBeTruthy();
    const pairwise = screen.getByRole("table", { name: "Pairwise comparisons" });
    expect(within(pairwise).getByText("p = 0.05")).toBeTruthy();
  });

  it("shows the interpretation", () => {
    renderResult();
    expect(screen.getByRole("heading", { name: "Interpretation" })).toBeTruthy();
    expect(screen.getByText(FIXTURE_RESULT.interpretation)).toBeTruthy();
  });
});

describe("StatsResult — counts and excluded rows", () => {
  it("says how many rows were offered, used and excluded", () => {
    renderResult();
    expect(screen.getByText("40 offered, 36 used, 4 excluded.")).toBeTruthy();
  });

  it("lists the first 200 excluded rows, counts the rest, and starts collapsed", () => {
    renderResult();
    const summary = screen.getByText("Excluded rows (200)");
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    expect((details as HTMLDetailsElement).open).toBe(false);
    expect(details?.querySelectorAll("tbody tr")).toHaveLength(200);
    expect(screen.getByText("and 50 more")).toBeTruthy();
    const first = details?.querySelector("tbody tr");
    // The server numbers rows from zero; a reader counts from one.
    expect(first?.textContent).toContain("1");
    expect(first?.textContent).toContain("missing in y");
  });
});

describe("StatsResult — the methods fragment", () => {
  it("shows the fragment and copies it with the Copy methods text control", async () => {
    renderResult();
    expect(screen.getByText(FIXTURE_RESULT.methods_fragment)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Copy methods text" }));

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(FIXTURE_RESULT.methods_fragment),
    );
    expect(await screen.findByText("Copied")).toBeTruthy();
  });
});
