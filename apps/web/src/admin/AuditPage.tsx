/**
 * The audit trail, newest first, filtered by account, action and a range of IST days (both
 * ends inclusive). The filters apply on submit, never as they are typed; the server pages,
 * 25 rows at a time.
 */

import { useQuery } from "@tanstack/react-query";
import { useId, useState, type ChangeEvent, type FormEvent, type ReactElement } from "react";
import { ApiError } from "../api/client";
import { DataTable, type DataTableColumn } from "../components/DataTable";
import { Pagination } from "../components/Pagination";
import { Empty, ErrorBox, Loading, apiErrorText } from "../components/States";
import { fmtDateTime, toIstDateInput } from "../format";
import { audit, type AuditFilters, type AuditOut } from "./adminApi";

/** The trail's key names every filter it depends on, so each filtered page is its own entry. */
const AUDIT_KEY = ["admin", "audit"] as const;
const UNFILTERED: AuditFilters = { page: 1 };

/** What the inputs hold; they become filters only when applied. */
interface Draft {
  userId: string;
  action: string;
  from: string;
  to: string;
}

const EMPTY_DRAFT: Draft = { userId: "", action: "", from: "", to: "" };

const COLUMNS: DataTableColumn<AuditOut>[] = [
  { header: "When", cell: (cell) => fmtDateTime(cell.row.original.created_at) },
  { header: "Action", cell: (cell) => cell.row.original.action },
  {
    header: "What",
    cell: (cell) => {
      const row = cell.row.original;
      return row.entity_id === null ? row.entity : `${row.entity} ${row.entity_id}`;
    },
  },
  { header: "Account", cell: (cell) => cell.row.original.user_id ?? "System" },
  {
    header: "Detail",
    meta: { truncate: true },
    cell: (cell) => detailText(cell.row.original.detail),
  },
];

/** The detail as compact JSON; an empty one is shown as a dash rather than "{}". */
function detailText(detail: AuditOut["detail"]): string {
  return Object.keys(detail).length === 0 ? "–" : JSON.stringify(detail);
}

export function AuditPage(): ReactElement {
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [filters, setFilters] = useState<AuditFilters>(UNFILTERED);
  const query = useQuery({ queryKey: [...AUDIT_KEY, filters], queryFn: () => audit(filters) });
  const ids = { account: useId(), action: useId(), from: useId(), to: useId() };
  // A future day has no entries; a date input cannot offer one.
  const today = toIstDateInput(new Date());

  const apply = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setFilters({
      userId: draft.userId.trim() || undefined,
      action: draft.action.trim() || undefined,
      from: draft.from || undefined,
      to: draft.to || undefined,
      page: 1,
    });
  };

  const clear = (): void => {
    setDraft(EMPTY_DRAFT);
    setFilters(UNFILTERED);
  };

  const bind = (key: keyof Draft) => ({
    value: draft[key],
    onChange: (event: ChangeEvent<HTMLInputElement>) =>
      setDraft((current) => ({ ...current, [key]: event.target.value })),
  });

  // The server refuses an impossible range with a 422; its reason is what is shown.
  const refused = query.error instanceof ApiError && query.error.status === 422;

  return (
    <>
      <h2>Audit</h2>
      <form className="admin-filters" onSubmit={apply} aria-label="Audit filters">
        <div className="field">
          <label htmlFor={ids.account}>Account</label>
          <input
            id={ids.account}
            type="text"
            placeholder="User ID"
            autoComplete="off"
            {...bind("userId")}
          />
        </div>
        <div className="field">
          <label htmlFor={ids.action}>Action</label>
          <input
            id={ids.action}
            type="text"
            placeholder="user.role_change"
            autoComplete="off"
            {...bind("action")}
          />
        </div>
        <div className="field">
          <label htmlFor={ids.from}>From</label>
          <input id={ids.from} type="date" max={today} {...bind("from")} />
        </div>
        <div className="field">
          <label htmlFor={ids.to}>To</label>
          <input id={ids.to} type="date" max={today} {...bind("to")} />
        </div>
        <div className="btn-row">
          <button type="submit" className="btn btn--primary">
            Apply filters
          </button>
          <button type="button" className="btn" onClick={clear}>
            Clear filters
          </button>
        </div>
      </form>

      {refused && <p className="alert alert--error" role="alert">{apiErrorText(query.error)}</p>}

      {query.isPending ? (
        <Loading label="Loading the audit trail…" />
      ) : query.isError ? (
        refused ? null : (
          <>
            <ErrorBox error={query.error} />
            <button type="button" className="btn" onClick={() => void query.refetch()}>
              Retry
            </button>
          </>
        )
      ) : (
        <>
          <DataTable
            columns={COLUMNS}
            rows={query.data.items}
            caption="Audit trail"
            empty={<Empty>No audit entries match these filters.</Empty>}
          />
          <Pagination
            page={query.data.page}
            pageSize={query.data.page_size}
            total={query.data.total}
            onPageChange={(page) => setFilters((current) => ({ ...current, page }))}
          />
        </>
      )}
    </>
  );
}
