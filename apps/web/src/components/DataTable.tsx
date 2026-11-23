/**
 * A plain table over TanStack Table: a semantic <table> in the stylesheet's compact style,
 * with no sorting, filtering or pagination of its own. The console's lists arrive from the
 * API already filtered, ordered and paged, so the table only lays them out.
 *
 * A column is a header and a cell that are text (or a function of the row returning text
 * or a number). Two optional flags on `meta` style a column: `num` aligns it as a number,
 * `truncate` keeps it to one line with the full text in a title.
 */

import {
  flexRender,
  metaHelper,
  tableFeatures,
  useTable,
  type CellData,
  type ColumnDef,
  type RowData,
} from "@tanstack/react-table";
import type { ReactElement, ReactNode } from "react";

/** What a console table's column may ask of its header and cells. */
interface DataTableMeta {
  /** Right-align the header and the cells, as the stylesheet's numeric columns. */
  num?: boolean;
  /** Keep the cell to one line, cut with an ellipsis; the full text goes in a title. */
  truncate?: boolean;
}

/**
 * The console's tables enable no optional feature: the server does that work. The meta
 * slot is type-only; it is what gives `meta` its two flags on every column.
 */
const FEATURES = tableFeatures({ columnMeta: metaHelper<DataTableMeta>() });

/**
 * One column of a console table. Table v9 types a column's value as `CellData`; the cells
 * here read what they show from `cell.row.original` and return text or a number.
 */
export type DataTableColumn<T extends RowData> = ColumnDef<typeof FEATURES, T, CellData>;

export interface DataTableProps<T extends RowData> {
  columns: ReadonlyArray<DataTableColumn<T>>;
  rows: T[];
  /** The table's accessible name. */
  caption: string;
  /** What stands in for the table when there are no rows. */
  empty: ReactNode;
}

/** A stable empty array, so a table with nothing to show does not rebuild its model each render. */
const NO_ROWS: never[] = [];

function isText(value: unknown): value is string | number {
  return typeof value === "string" || typeof value === "number";
}

export function DataTable<T extends RowData>({
  columns,
  rows,
  caption,
  empty,
}: DataTableProps<T>): ReactElement {
  const table = useTable({ features: FEATURES, columns, data: rows.length > 0 ? rows : NO_ROWS });
  if (rows.length === 0) return <>{empty}</>;

  return (
    <table className="table table--compact">
      <caption className="sr-only">{caption}</caption>
      <thead>
        {table.getHeaderGroups().map((group) => (
          <tr key={group.id}>
            {group.headers.map((header) => (
              <th
                key={header.id}
                scope="col"
                className={header.column.columnDef.meta?.num ? "num" : undefined}
              >
                {header.isPlaceholder
                  ? null
                  : flexRender(header.column.columnDef.header, header.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getAllCells().map((cell) => {
              const meta = cell.column.columnDef.meta;
              const render = cell.column.columnDef.cell;
              const context = cell.getContext();
              // flexRender wraps a cell function in an element; its text is read here for the title.
              const value = typeof render === "function" ? render(context) : render;
              const content = isText(value) ? value : flexRender(render, context);
              const className = [meta?.num ? "num" : "", meta?.truncate ? "truncate" : ""]
                .filter(Boolean)
                .join(" ");
              const title = meta?.truncate && isText(value) ? String(value) : undefined;
              return (
                <td key={cell.id} className={className === "" ? undefined : className} title={title}>
                  {content}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
