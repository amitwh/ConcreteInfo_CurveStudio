import type { ReactElement } from "react";
import { fmtInt } from "../format";

export interface PaginationProps {
  /** From 1. */
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
}

/**
 * The pager of a server-paged list: what is showing, and Previous and Next. It changes
 * nothing itself and announces nothing: the list it belongs to shows the new page.
 */
export function Pagination({ page, pageSize, total, onPageChange }: PaginationProps): ReactElement {
  const first = (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return (
    <div className="pagination">
      <button
        type="button"
        className="btn btn--small"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        Previous
      </button>
      <span>
        {total === 0
          ? `Showing 0 of 0`
          : `Showing ${fmtInt(first)}–${fmtInt(last)} of ${fmtInt(total)}`}
      </span>
      <button
        type="button"
        className="btn btn--small"
        disabled={last >= total}
        onClick={() => onPageChange(page + 1)}
      >
        Next
      </button>
    </div>
  );
}
