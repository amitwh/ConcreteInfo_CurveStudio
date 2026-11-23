/**
 * The accounts, newest first, filtered by name or address, plan, role and status. An
 * administrator acts on an account from its row: opens it, changes its plan or role,
 * deactivates it, or deletes it. The signed-in administrator's own row opens only.
 *
 * Changing a plan or a role is not destructive, so those two are small forms that save
 * at once and say so; deactivating and deleting ask first. Every refusal is shown where
 * the action was taken, and the row stays as the server holds it.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type ReactElement,
} from "react";
import { useAuth } from "../api/auth";
import { ApiError } from "../api/client";
import { DataTable, type DataTableColumn } from "../components/DataTable";
import { Meter } from "../components/Meter";
import { Modal } from "../components/Modal";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Pagination } from "../components/Pagination";
import { StatusBadge } from "../components/StatusBadge";
import { Empty, ErrorBox, apiErrorText } from "../components/States";
import { fmtBytes, fmtDate, fmtDateTime, fmtInt } from "../format";
import {
  deleteUser,
  plans,
  updateUser,
  userDetail,
  users,
  type AdminUserDetailOut,
  type AdminUserOut,
  type AuditOut,
  type PlanAdminOut,
  type PlanLimits,
  type UserChange,
  type UsersFilters,
  type UsersPageOut,
} from "./adminApi";
import { QueryState } from "./QueryState";

/** The list's key names every filter it depends on, so each filtered page is its own entry. */
const USERS_KEY = ["admin", "users"] as const;
/** One account's detail, keyed by the account's id. */
const USER_KEY = ["admin", "user"] as const;
/** The plans, shared with the Plans page. */
const PLANS_KEY = ["admin", "plans"] as const;
const UNFILTERED: UsersFilters = { page: 1 };

const PLAN_SAVED = "Plan saved. The change is in the audit trail.";
const ROLE_SAVED = "Role saved. The change is in the audit trail.";
const OWN_ROW_NOTE = "You cannot change your own role or active flag.";
/** The server's code for a change to the administrator's own role or active flag. */
const SELF_CHANGE_REFUSED = "SELF_CHANGE_REFUSED";

/** What the filter inputs hold; they become filters only when applied. */
interface Draft {
  q: string;
  plan: string;
  role: "" | "member" | "admin";
  active: "" | "active" | "inactive";
}

const EMPTY_DRAFT: Draft = { q: "", plan: "", role: "", active: "" };

const ROLE_OPTIONS: ReadonlyArray<{ value: "" | "member" | "admin"; label: string }> = [
  { value: "", label: "Any" },
  { value: "member", label: "Member" },
  { value: "admin", label: "Administrator" },
];

const STATUS_OPTIONS: ReadonlyArray<{ value: Draft["active"]; label: string }> = [
  { value: "", label: "Any" },
  { value: "active", label: "Active" },
  { value: "inactive", label: "Deactivated" },
];

/** The audit detail as compact JSON; an empty one is shown as a dash rather than "{}". */
function detailText(detail: AuditOut["detail"]): string {
  return Object.keys(detail).length === 0 ? "–" : JSON.stringify(detail);
}

const AUDIT_COLUMNS: DataTableColumn<AuditOut>[] = [
  { header: "When", cell: (cell) => fmtDateTime(cell.row.original.created_at) },
  { header: "Action", cell: (cell) => cell.row.original.action },
  {
    header: "What",
    cell: (cell) => {
      const row = cell.row.original;
      return row.entity_id === null ? row.entity : `${row.entity} ${row.entity_id}`;
    },
  },
  {
    header: "Detail",
    meta: { truncate: true },
    cell: (cell) => detailText(cell.row.original.detail),
  },
];

/** The plan in force, and the override behind it when it names another plan. */
function planText(user: AdminUserOut): string {
  const { plan, plan_override: override } = user;
  return override !== null && override !== "" && override !== plan
    ? `${plan} → overridden to ${override}`
    : plan;
}

/** Why a change was refused: in the page's own words where it has one, else the server's. */
function refusal(error: unknown): string {
  const code = (error as { detail?: { code?: unknown } } | null)?.detail?.code;
  if (error instanceof ApiError && code === SELF_CHANGE_REFUSED) return OWN_ROW_NOTE;
  return apiErrorText(error);
}

/** A change to one account. The server's answer is written into every loaded view of the account at once, then they are fetched again. */
function useAccountUpdate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, change }: { id: string; change: UserChange }) => updateUser(id, change),
    onSuccess: (updated) => {
      queryClient.setQueriesData<UsersPageOut>({ queryKey: USERS_KEY }, (page) =>
        page === undefined
          ? page
          : {
              ...page,
              items: page.items.map((user) => (user.id === updated.id ? updated : user)),
            },
      );
      queryClient.setQueriesData<AdminUserDetailOut>({ queryKey: USER_KEY }, (detail) =>
        detail !== undefined && detail.user.id === updated.id ? { ...detail, user: updated } : detail,
      );
      void queryClient.invalidateQueries({ queryKey: USERS_KEY });
      // The account's audit trail gained a row.
      void queryClient.invalidateQueries({ queryKey: [...USER_KEY, updated.id] });
    },
  });
}

/** What is open on top of the list: one of the account's actions, or nothing. */
type Panel =
  | { kind: "none" }
  | { kind: "plan" | "role" | "active" | "delete"; user: AdminUserOut };

const CLOSED: Panel = { kind: "none" };

export function UsersPage(): ReactElement {
  const { user: me } = useAuth();
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [filters, setFilters] = useState<UsersFilters>(UNFILTERED);
  const list = useQuery({ queryKey: [...USERS_KEY, filters], queryFn: () => users(filters) });
  const plansQuery = useQuery({ queryKey: PLANS_KEY, queryFn: plans });
  const ids = { q: useId(), plan: useId(), role: useId(), active: useId() };

  const [openedId, setOpenedId] = useState<string | null>(null);
  const [panel, setPanel] = useState<Panel>(CLOSED);
  const [notice, setNotice] = useState<string | null>(null);
  // The row control that opened the detail, to return the focus to when it closes.
  const opener = useRef<HTMLElement | null>(null);

  const apply = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setFilters({
      q: draft.q.trim() || undefined,
      plan: draft.plan || undefined,
      role: draft.role || undefined,
      active: draft.active === "" ? undefined : draft.active === "active",
      page: 1,
    });
  };

  const clear = (): void => {
    setDraft(EMPTY_DRAFT);
    setFilters(UNFILTERED);
  };

  const bind = (key: keyof Draft) => ({
    value: draft[key],
    onChange: (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
      setDraft((current) => ({ ...current, [key]: event.target.value })),
  });

  const open = useCallback((user: AdminUserOut): void => {
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setOpenedId(user.id);
  }, []);

  const closeDetail = useCallback((): void => {
    setOpenedId(null);
    opener.current?.focus();
  }, []);

  const closePanel = useCallback((): void => setPanel(CLOSED), []);

  // The forms and confirmations share these two: only one is open at a time.
  const update = useAccountUpdate();
  const remove = useMutation({
    mutationFn: (id: string) => deleteUser(id),
    onSuccess: (_removed, id) => {
      queryClient.removeQueries({ queryKey: [...USER_KEY, id] });
      // Not waited for: the server has done its work, and the dialog closes on that.
      void queryClient.invalidateQueries({ queryKey: USERS_KEY });
      setPanel(CLOSED);
      // The account is gone, so its detail goes with it; there is no row to return to.
      setOpenedId((current) => (current === id ? null : current));
      setNotice("Account deleted.");
    },
  });

  /** Opens one of an account's action forms or confirmations, clean of the last action's result. */
  const start = useCallback(
    (kind: Exclude<Panel["kind"], "none">, user: AdminUserOut): void => {
      // A refusal belongs to the action that was refused, not to the next one.
      // reset is the observer's own bound method, stable across renders.
      update.reset();
      remove.reset();
      setNotice(null);
      setPanel({ kind, user });
    },
    [update.reset, remove.reset],
  );

  const ownId = me?.id ?? null;
  const columns = useMemo<DataTableColumn<AdminUserOut>[]>(
    () => [
      { header: "Joined", cell: (cell) => fmtDate(cell.row.original.created_at) },
      { header: "Name", cell: (cell) => cell.row.original.name || "–" },
      { header: "E-mail", cell: (cell) => cell.row.original.email },
      { header: "Plan", cell: (cell) => planText(cell.row.original) },
      { header: "Role", cell: (cell) => <StatusBadge kind={cell.row.original.role} /> },
      { header: "Status", cell: (cell) => <AccountStatus user={cell.row.original} /> },
      {
        header: "Last active",
        cell: (cell) => cell.row.original.last_active_on === null
          ? "Never"
          : fmtDate(cell.row.original.last_active_on),
      },
      { header: "Datasets", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.usage.datasets) },
      { header: "Stored fits", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.usage.fits) },
      {
        header: "Actions",
        cell: (cell) => (
          <RowActions user={cell.row.original} own={cell.row.original.id === ownId} onOpen={open} onStart={start} />
        ),
      },
    ],
    [ownId, open, start],
  );

  const opened = openedId === null ? null : (list.data?.items.find((user) => user.id === openedId) ?? null);
  const planNames = plansQuery.data?.map((plan) => plan.name) ?? [];

  return (
    <>
      <h2>Accounts</h2>
      <form className="admin-filters" onSubmit={apply} aria-label="Account filters">
        <div className="field">
          <label htmlFor={ids.q}>Name or e-mail</label>
          <input id={ids.q} type="search" maxLength={100} autoComplete="off" {...bind("q")} />
        </div>
        <div className="field">
          <label htmlFor={ids.plan}>Plan</label>
          <select id={ids.plan} {...bind("plan")}>
            <option value="">Any</option>
            {planNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={ids.role}>Role</label>
          <select id={ids.role} {...bind("role")}>
            {ROLE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={ids.active}>Status</label>
          <select id={ids.active} {...bind("active")}>
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
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

      {plansQuery.isError && (
        <>
          <ErrorBox error={plansQuery.error} />
          <button type="button" className="btn" onClick={() => void plansQuery.refetch()}>
            Retry
          </button>
        </>
      )}

      {notice !== null && (
        <p className="alert alert--success" role="status">
          {notice}
        </p>
      )}

      <QueryState label="Loading the accounts…" query={list}>
        {(page) => (
          <>
            <p className="muted admin-count">
              {fmtInt(page.total)} {page.total === 1 ? "account" : "accounts"}
            </p>
            <DataTable
              columns={columns}
              rows={page.items}
              caption="Accounts"
              empty={<Empty>No accounts match</Empty>}
            />
            <Pagination
              page={page.page}
              pageSize={page.page_size}
              total={page.total}
              onPageChange={(next) => setFilters((current) => ({ ...current, page: next }))}
            />
          </>
        )}
      </QueryState>

      {openedId !== null && (
        <UserDetail
          key={openedId}
          id={openedId}
          summary={opened}
          plans={plansQuery.data}
          isSelf={openedId === ownId}
          onClose={closeDetail}
        />
      )}

      {panel.kind === "plan" && (
        <ChangePlanDialog
          user={panel.user}
          planNames={planNames}
          plansError={plansQuery.error}
          onPlansRetry={() => void plansQuery.refetch()}
          save={update}
          onSaved={(message) => {
            setNotice(message);
            setPanel(CLOSED);
          }}
          onClose={closePanel}
        />
      )}
      {panel.kind === "role" && (
        <ChangeRoleDialog
          user={panel.user}
          save={update}
          onSaved={(message) => {
            setNotice(message);
            setPanel(CLOSED);
          }}
          onClose={closePanel}
        />
      )}
      {panel.kind === "active" && (
        <ConfirmDialog
          title={panel.user.is_active ? "Deactivate account?" : "Reactivate account?"}
          message={
            <p>
              {panel.user.is_active ? (
                <>
                  <strong>{panel.user.email}</strong> will not be able to sign in until the
                  account is reactivated. Nothing in the account is deleted.
                </>
              ) : (
                <>
                  <strong>{panel.user.email}</strong> will be able to sign in again, with
                  everything the account held.
                </>
              )}
            </p>
          }
          confirmLabel={panel.user.is_active ? "Deactivate" : "Reactivate"}
          busy={update.isPending}
          error={update.error === null ? null : refusal(update.error)}
          onConfirm={() =>
            update.mutate(
              { id: panel.user.id, change: { is_active: !panel.user.is_active } },
              {
                onSuccess: (updated) => {
                  setNotice(`Account ${updated.is_active ? "reactivated" : "deactivated"}.`);
                  setPanel(CLOSED);
                },
              },
            )
          }
          onClose={closePanel}
        />
      )}
      {panel.kind === "delete" && (
        <ConfirmDialog
          title="Delete account?"
          message={
            <p>
              This erases {panel.user.name || "the account"} (<strong>{panel.user.email}</strong>)
              {" "}for good: its datasets, stored fits, jobs and sessions are deleted. The audit
              trail is kept, without the account's identity. This cannot be undone.
            </p>
          }
          confirmLabel="Delete account"
          phrase={panel.user.email}
          danger
          busy={remove.isPending}
          error={remove.error === null ? null : apiErrorText(remove.error)}
          onConfirm={() => remove.mutate(panel.user.id)}
          onClose={closePanel}
        />
      )}
    </>
  );
}

/** The limits of the plan an account is resolved to, or undefined when they are not known. */
function planLimitsFor(plans: PlanAdminOut[] | undefined, plan: string | undefined): PlanLimits | undefined {
  return plans?.find((candidate) => candidate.name === plan)?.limits;
}

/** Active or deactivated, with an unconfirmed address said beneath it. */
function AccountStatus({ user }: { user: AdminUserOut }): ReactElement {
  return (
    <>
      {user.is_active ? "Active" : "Deactivated"}
      {!user.email_verified && <small className="muted admin-cell-note">E-mail not verified</small>}
    </>
  );
}

interface RowActionsProps {
  user: AdminUserOut;
  /** The signed-in administrator's own row: it opens, and offers nothing else. */
  own: boolean;
  onOpen: (user: AdminUserOut) => void;
  onStart: (kind: Exclude<Panel["kind"], "none">, user: AdminUserOut) => void;
}

/** What can be done to one account. Each control names the account for anyone who cannot see the row. */
function RowActions({ user, own, onOpen, onStart }: RowActionsProps): ReactElement {
  const { email } = user;
  const activeLabel = user.is_active ? "Deactivate" : "Reactivate";
  return (
    <span className="row-actions">
      <button type="button" className="btn btn--small" aria-label={`Open ${email}`} onClick={() => onOpen(user)}>
        Open
      </button>
      {own ? null : (
        <>
          <button
            type="button"
            className="btn btn--small"
            aria-label={`Change plan ${email}`}
            onClick={() => onStart("plan", user)}
          >
            Change plan
          </button>
          <button
            type="button"
            className="btn btn--small"
            aria-label={`Change role ${email}`}
            onClick={() => onStart("role", user)}
          >
            Change role
          </button>
          <button
            type="button"
            className="btn btn--small"
            aria-label={`${activeLabel} ${email}`}
            onClick={() => onStart("active", user)}
          >
            {activeLabel}
          </button>
          <button
            type="button"
            className="btn btn--small btn--danger"
            aria-label={`Delete ${email}`}
            onClick={() => onStart("delete", user)}
          >
            Delete
          </button>
        </>
      )}
    </span>
  );
}

interface UserDetailProps {
  id: string;
  /** The list's row for the account, for the heading while the detail loads. */
  summary: AdminUserOut | null;
  /** The plans the server lists, for the limits the usage is set against. */
  plans: PlanAdminOut[] | undefined;
  /** The account is the signed-in administrator's own. */
  isSelf: boolean;
  onClose: () => void;
}

/**
 * One account below the list: its plan and usage against the plan's limits, its dates, and
 * the last fifty rows of its audit trail. It takes the focus when it opens.
 */
function UserDetail({ id, summary, plans, isSelf, onClose }: UserDetailProps): ReactElement {
  const headingId = useId();
  const section = useRef<HTMLElement>(null);
  const query = useQuery({ queryKey: [...USER_KEY, id], queryFn: () => userDetail(id) });

  useEffect(() => {
    section.current?.focus();
  }, []);

  const shown = query.data?.user ?? summary;
  const overridden =
    shown !== null &&
    shown.plan_override !== null &&
    shown.plan_override !== "" &&
    shown.plan_override !== shown.plan;

  return (
    <section ref={section} className="card admin-detail" aria-labelledby={headingId} tabIndex={-1}>
      <div className="admin-detail__head">
        <div>
          <h3 id={headingId}>{shown === null ? "Account" : shown.name || shown.email}</h3>
          {shown !== null && (
            <p className="hint admin-detail__meta">
              <span>{shown.email}</span>
              <StatusBadge kind={shown.role} />
              <span>{shown.is_active ? "Active" : "Deactivated"}</span>
              {shown.email_verified ? null : <span>E-mail not verified</span>}
            </p>
          )}
        </div>
        <button type="button" className="btn btn--small" onClick={onClose}>
          Close
        </button>
      </div>

      {isSelf && <p className="alert alert--warning">{OWN_ROW_NOTE}</p>}

      <QueryState label="Loading the account…" query={query}>
        {({ user, audit }) => (
          <>
            <p className="admin-detail__plan">
              Plan <strong>{user.plan}</strong>
              {overridden && (
                <span className="muted"> (overridden to {user.plan_override})</span>
              )}
            </p>
            <Usage user={user} limits={planLimitsFor(plans, user.plan)} />
            <dl className="admin-kv">
              <dt>Joined</dt>
              <dd>{fmtDateTime(user.created_at)}</dd>
              <dt>Last sign-in</dt>
              <dd>{user.last_login_at === null ? "Never" : fmtDateTime(user.last_login_at)}</dd>
              <dt>Last active</dt>
              <dd>{user.last_active_on === null ? "Never" : fmtDate(user.last_active_on)}</dd>
            </dl>
            <h4>Activity</h4>
            <p className="hint">The last fifty things this account did, newest first.</p>
            <DataTable
              columns={AUDIT_COLUMNS}
              rows={audit}
              caption={`Audit trail for ${user.email}`}
              empty={<Empty>This account has done nothing yet.</Empty>}
            />
          </>
        )}
      </QueryState>
    </section>
  );
}

/** Usage against the plan's limits as meters; in plain figures when the limits are not known. */
function Usage({ user, limits }: { user: AdminUserOut; limits: PlanLimits | undefined }): ReactElement {
  const { datasets, dataset_bytes, fits, autofits_today } = user.usage;
  if (limits === undefined) {
    return (
      <>
        <p className="alert alert--warning">
          The limits of the plan {user.plan} are not known. The usage is:
        </p>
        <dl className="admin-kv">
          <dt>Datasets</dt>
          <dd>{fmtInt(datasets)}</dd>
          <dt>Storage</dt>
          <dd>{fmtBytes(dataset_bytes)}</dd>
          <dt>Stored fits</dt>
          <dd>{fmtInt(fits)}</dd>
          <dt>Auto-fit runs today</dt>
          <dd>{fmtInt(autofits_today)}</dd>
        </dl>
      </>
    );
  }
  return (
    <div className="meters">
      <Meter label="Datasets" used={datasets} limit={limits.max_datasets} />
      <Meter label="Storage" used={dataset_bytes} limit={limits.max_dataset_bytes} format={fmtBytes} />
      <Meter label="Stored fits" used={fits} limit={limits.max_fits} />
      <Meter label="Auto-fit runs today" used={autofits_today} limit={limits.max_autofits_per_day} />
    </div>
  );
}

/** Dismiss a form on top of the list, but never while its request is in flight. */
function useDismiss(onClose: () => void, busy: boolean): () => void {
  const busyRef = useRef(busy);
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);
  // Stable while `onClose` is: Modal re-runs its focus handling whenever this changes.
  return useCallback(() => {
    if (!busyRef.current) onClose();
  }, [onClose]);
}

const DEFAULT_PLAN = "";

interface ChangePlanDialogProps {
  user: AdminUserOut;
  /** The plans the server lists, for the picker. */
  planNames: string[];
  plansError: Error | null;
  onPlansRetry: () => void;
  save: ReturnType<typeof useAccountUpdate>;
  onSaved: (message: string) => void;
  onClose: () => void;
}

/** The plan picker. It opens on the account's override, and can clear it. */
function ChangePlanDialog({
  user,
  planNames,
  plansError,
  onPlansRetry,
  save,
  onSaved,
  onClose,
}: ChangePlanDialogProps): ReactElement {
  const selectId = useId();
  const dismiss = useDismiss(onClose, save.isPending);
  const current = user.plan_override ?? DEFAULT_PLAN;
  const [choice, setChoice] = useState(current);

  // An override that names no plan row any more is still the account's, so it is still offered.
  const names =
    user.plan_override !== null && !planNames.includes(user.plan_override)
      ? [user.plan_override, ...planNames]
      : planNames;

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (choice === current) return;
    save.mutate(
      { id: user.id, change: { plan_override: choice === DEFAULT_PLAN ? null : choice } },
      { onSuccess: () => onSaved(PLAN_SAVED) },
    );
  };

  return (
    <Modal title="Change plan" onClose={dismiss}>
      <form className="admin-change" aria-label={`Change plan for ${user.email}`} onSubmit={submit}>
        <p className="hint">
          For {user.name || user.email} ({user.email}). The plan in force is {user.plan}.
        </p>
        {plansError !== null ? (
          <>
            <ErrorBox error={plansError} />
            <button type="button" className="btn" onClick={onPlansRetry}>
              Retry
            </button>
          </>
        ) : (
          <div className="field">
            <label htmlFor={selectId}>Plan</label>
            <select
              id={selectId}
              value={choice}
              disabled={save.isPending}
              onChange={(event: ChangeEvent<HTMLSelectElement>) => setChoice(event.target.value)}
            >
              <option value={DEFAULT_PLAN}>Default plan (no override)</option>
              {names.map((name) => (
                <option key={name} value={name}>
                  {name === user.plan_override && !planNames.includes(name) ? `${name} (no longer a plan)` : name}
                </option>
              ))}
            </select>
          </div>
        )}
        {save.error !== null && (
          <p className="alert alert--error" role="alert">
            {refusal(save.error)}
          </p>
        )}
        <div className="modal__actions">
          <button type="button" className="btn" onClick={dismiss} disabled={save.isPending}>
            Cancel
          </button>
          <button type="submit" className="btn btn--primary" disabled={save.isPending || choice === current}>
            Save
          </button>
        </div>
      </form>
    </Modal>
  );
}

const ROLE_CHOICES: ReadonlyArray<{ value: AdminUserOut["role"]; label: string }> = [
  { value: "member", label: "Member" },
  { value: "admin", label: "Administrator" },
];

interface ChangeRoleDialogProps {
  user: AdminUserOut;
  save: ReturnType<typeof useAccountUpdate>;
  onSaved: (message: string) => void;
  onClose: () => void;
}

/** The role picker: two choices, the account's current one selected. */
function ChangeRoleDialog({ user, save, onSaved, onClose }: ChangeRoleDialogProps): ReactElement {
  const groupId = useId();
  const dismiss = useDismiss(onClose, save.isPending);
  const [choice, setChoice] = useState<AdminUserOut["role"]>(user.role);

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (choice === user.role) return;
    save.mutate({ id: user.id, change: { role: choice } }, { onSuccess: () => onSaved(ROLE_SAVED) });
  };

  return (
    <Modal title="Change role" onClose={dismiss}>
      <form className="admin-change" aria-label={`Change role for ${user.email}`} onSubmit={submit}>
        <p className="hint">
          For {user.name || user.email} ({user.email}). An administrator can open this console.
        </p>
        <fieldset className="fieldset" disabled={save.isPending}>
          <legend>Role</legend>
          {ROLE_CHOICES.map((role) => (
            <label className="check" key={role.value}>
              <input
                type="radio"
                name={groupId}
                value={role.value}
                checked={choice === role.value}
                onChange={() => setChoice(role.value)}
              />
              {role.label}
            </label>
          ))}
        </fieldset>
        {save.error !== null && (
          <p className="alert alert--error" role="alert">
            {refusal(save.error)}
          </p>
        )}
        <div className="modal__actions">
          <button type="button" className="btn" onClick={dismiss} disabled={save.isPending}>
            Cancel
          </button>
          <button type="submit" className="btn btn--primary" disabled={save.isPending || choice === user.role}>
            Save
          </button>
        </div>
      </form>
    </Modal>
  );
}
