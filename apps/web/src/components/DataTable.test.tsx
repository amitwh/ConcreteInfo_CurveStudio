import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DataTable, type DataTableColumn } from "./DataTable";

interface Row {
  name: string;
  count: number;
  note: string;
}

const COLUMNS: DataTableColumn<Row>[] = [
  { header: "Name", cell: (cell) => cell.row.original.name },
  { header: "Count", meta: { num: true }, cell: (cell) => cell.row.original.count },
];

const ROWS: Row[] = [
  { name: "0.11.1", count: 12, note: "" },
  { name: "0.11.0", count: 100000, note: "" },
];

const EMPTY = <p>No rows.</p>;

describe("DataTable", () => {
  it("renders its headers, its rows and the caption that names the table", () => {
    render(<DataTable columns={COLUMNS} rows={ROWS} caption="Installs by version" empty={EMPTY} />);
    const table = screen.getByRole("table", { name: "Installs by version" });
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "Name",
      "Count",
    ]);
    expect(within(table).getAllByRole("cell").map((td) => td.textContent)).toEqual([
      "0.11.1",
      "12",
      "0.11.0",
      "100000",
    ]);
    expect(screen.queryByText("No rows.")).toBeNull();
  });

  it("shows the empty state in place of a table when there are no rows", () => {
    render(<DataTable columns={COLUMNS} rows={[]} caption="Installs by version" empty={EMPTY} />);
    expect(screen.getByText("No rows.")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("marks a numeric column so its header and cells align as numbers", () => {
    render(<DataTable columns={COLUMNS} rows={ROWS} caption="Counts" empty={EMPTY} />);
    const table = screen.getByRole("table", { name: "Counts" });
    const [name, count] = within(table).getAllByRole("columnheader");
    const cells = within(table).getAllByRole("cell");
    expect(name?.className).not.toContain("num");
    expect(count?.className).toContain("num");
    expect(cells[0]?.className).not.toContain("num");
    expect(cells[1]?.className).toContain("num");
  });

  it("cuts a long cell of a truncating column to one line and keeps the full text in a title", () => {
    const columns: DataTableColumn<Row>[] = [
      { header: "Name", cell: (cell) => cell.row.original.name },
      { header: "Note", meta: { truncate: true }, cell: (cell) => cell.row.original.note },
    ];
    const rows: Row[] = [{ name: "one", count: 1, note: '{"role":"admin","by":"someone"}' }];
    render(<DataTable columns={columns} rows={rows} caption="Notes" empty={EMPTY} />);
    const cell = within(screen.getByRole("table", { name: "Notes" })).getAllByRole("cell")[1];
    expect(cell?.className).toContain("truncate");
    expect(cell?.textContent).toBe('{"role":"admin","by":"someone"}');
    expect(cell?.getAttribute("title")).toBe('{"role":"admin","by":"someone"}');
  });

  it("accepts a plain number from a cell and a plain string for a header", () => {
    const columns: DataTableColumn<Row>[] = [{ header: "Count", cell: (cell) => cell.row.original.count }];
    render(<DataTable columns={columns} rows={ROWS} caption="Counts" empty={EMPTY} />);
    expect(within(screen.getByRole("table", { name: "Counts" })).getAllByRole("cell")).toHaveLength(2);
  });
});
