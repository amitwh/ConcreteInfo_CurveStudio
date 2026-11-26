import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SortableTh, cycleSort, sortRows, type SortState } from "./sorting";

type Row = { name: string; value: number | null };

const KEYS: Record<string, (row: Row) => number | string | null> = {
  name: (row) => row.name,
  value: (row) => row.value,
};

describe("cycleSort", () => {
  it("starts ascending, reverses on the second click, returns to the served order on the third", () => {
    const first = cycleSort(null, "value");
    expect(first).toEqual({ column: "value", direction: "asc" });
    const second = cycleSort(first, "value");
    expect(second).toEqual({ column: "value", direction: "desc" });
    const third = cycleSort(second, "value");
    expect(third).toBeNull();
  });

  it("switching columns starts a fresh ascending sort", () => {
    const state: SortState = { column: "value", direction: "desc" };
    expect(cycleSort(state, "name")).toEqual({ column: "name", direction: "asc" });
  });
});

describe("sortRows", () => {
  const rows: Row[] = [
    { name: "b", value: 2 },
    { name: "a", value: 10 },
    { name: "c", value: null },
  ];

  it("leaves the rows in the served order when no sort is active", () => {
    expect(sortRows(rows, null, KEYS)).toEqual(rows);
  });

  it("orders a numeric column ascending, undefined values last", () => {
    const sorted = sortRows(rows, { column: "value", direction: "asc" }, KEYS);
    expect(sorted.map((row) => row.name)).toEqual(["b", "a", "c"]);
  });

  it("orders a numeric column descending, undefined values still last", () => {
    const sorted = sortRows(rows, { column: "value", direction: "desc" }, KEYS);
    expect(sorted.map((row) => row.name)).toEqual(["a", "b", "c"]);
  });

  it("orders a text column ascending and descending", () => {
    const asc = sortRows(rows, { column: "name", direction: "asc" }, KEYS);
    expect(asc.map((row) => row.name)).toEqual(["a", "b", "c"]);
    const desc = sortRows(rows, { column: "name", direction: "desc" }, KEYS);
    expect(desc.map((row) => row.name)).toEqual(["c", "b", "a"]);
  });

  it("never mutates the rows it was given", () => {
    const copy = [...rows];
    sortRows(rows, { column: "value", direction: "asc" }, KEYS);
    expect(rows).toEqual(copy);
  });

  it("ignores a sort for a column it has no key for", () => {
    expect(sortRows(rows, { column: "unknown", direction: "asc" }, KEYS)).toEqual(rows);
  });
});

describe("SortableTh", () => {
  function Header({ state }: { state: SortState | null }): React.ReactElement {
    const onSort = vi.fn();
    return (
      <table>
        <thead>
          <tr>
            <SortableTh column="value" state={state} onSort={onSort} className="num">
              Value
            </SortableTh>
          </tr>
        </thead>
      </table>
    );
  }

  it("reports its direction to assistive tech only while it is the sorted column", () => {
    const { rerender } = render(<Header state={null} />);
    expect(screen.getByRole("columnheader", { name: "Value" }).getAttribute("aria-sort")).toBe(
      "none",
    );
    rerender(<Header state={{ column: "value", direction: "asc" }} />);
    expect(screen.getByRole("columnheader", { name: "Value" }).getAttribute("aria-sort")).toBe(
      "ascending",
    );
    rerender(<Header state={{ column: "value", direction: "desc" }} />);
    expect(screen.getByRole("columnheader", { name: "Value" }).getAttribute("aria-sort")).toBe(
      "descending",
    );
  });

  it("clicking the header asks for the next state of its own column", async () => {
    const onSort = vi.fn();
    render(
      <table>
        <thead>
          <tr>
            <SortableTh column="value" state={null} onSort={onSort}>
              Value
            </SortableTh>
          </tr>
        </thead>
      </table>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Value" }));
    expect(onSort).toHaveBeenCalledWith("value");
  });
});
