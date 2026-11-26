/**
 * Display-only column sorting, shared by the workspace tables. The order rows arrive in
 * is somebody else's business — the ranking is the `RankBy` selector's, a fit's parameters
 * the engine's — so sorting here reorders what is *shown*, never what is stored or served:
 * clicking a header cycles ascending → descending → back to the served order, and a value
 * that is not defined ("n/a") stays at the bottom in both directions.
 */

import type { ReactElement, ReactNode } from "react";

export interface SortState {
  column: string;
  direction: "asc" | "desc";
}

/** The next sort state for `column`: asc → desc → null (the served order again). */
export function cycleSort(current: SortState | null, column: string): SortState | null {
  if (current === null || current.column !== column) return { column, direction: "asc" };
  if (current.direction === "asc") return { column, direction: "desc" };
  return null;
}

/** What one column of one row is sorted by; `null`/`undefined` is the "n/a" case. */
export type SortKey<T> = (row: T) => number | string | null | undefined;

function isMissing(value: number | string | null | undefined): value is null | undefined {
  return value === null || value === undefined;
}

/**
 * A copy of `rows` in the display order `state` asks for — the rows themselves, and the
 * array they came in, are untouched. An inactive or unknown sort returns them as served.
 */
export function sortRows<T>(
  rows: readonly T[],
  state: SortState | null,
  keys: Record<string, SortKey<T>>,
): T[] {
  if (state === null) return [...rows];
  const key = keys[state.column];
  if (key === undefined) return [...rows];
  const descending = state.direction === "desc";
  return [...rows].sort((left, right) => {
    const a = key(left);
    const b = key(right);
    // "n/a" is not a number to compare: it reads as missing information, so it sorts to
    // the bottom whichever way the column is ordered.
    if (isMissing(a) && isMissing(b)) return 0;
    if (isMissing(a)) return 1;
    if (isMissing(b)) return -1;
    if (a === b) return 0;
    if (descending) return b < a ? -1 : 1;
    return a < b ? -1 : 1;
  });
}

/**
 * A column header that cycles the display order when clicked. The caret shows the active
 * direction (and is hidden from the accessible name, so the header still reads as its
 * label alone); `aria-sort` tells assistive technology which column is ordered.
 */
export function SortableTh({
  column,
  state,
  onSort,
  className,
  children,
}: {
  column: string;
  state: SortState | null;
  onSort: (column: string) => void;
  className?: string;
  children: ReactNode;
}): ReactElement {
  const active = state?.column === column;
  const sort = active ? (state.direction === "asc" ? "ascending" : "descending") : "none";
  return (
    <th scope="col" className={className} aria-sort={sort}>
      <button type="button" className="th-sort" onClick={() => onSort(column)}>
        {children}
        <span className="th-sort__caret" aria-hidden="true">
          {active ? (state.direction === "asc" ? "▲" : "▼") : "⇅"}
        </span>
      </button>
    </th>
  );
}
