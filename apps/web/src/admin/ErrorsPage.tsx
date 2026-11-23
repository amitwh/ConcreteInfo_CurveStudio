/**
 * The error events by status (new, acknowledged, resolved, ignored), each list filtered by
 * category. An event never names an account, and nothing here suggests one. What a visitor
 * typed on the recovery screen is an event of its own: it is shown as a visitor's note on
 * the error it names, never as an error type.
 *
 * The status, the category and the page are part of the list's key, so each is its own
 * entry. The tab's count is the total the server gives for that tab as filtered.
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
  type KeyboardEvent,
  type ReactElement,
} from "react";
import { DataTable, type DataTableColumn } from "../components/DataTable";
import { StatusBadge } from "../components/StatusBadge";
import { Pagination } from "../components/Pagination";
import { Empty, apiErrorText } from "../components/States";
import { fmtDateTime, fmtInt } from "../format";
import {
  errorBundle,
  errorDetail,
  errors,
  updateErrorStatus,
  type ErrorCategory,
  type ErrorEventDetailOut,
  type ErrorEventOut,
  type ErrorsFilters,
  type ErrorsPageOut,
  type ErrorStatus,
} from "./adminApi";
import { QueryState } from "./QueryState";

const ERRORS_KEY = ["admin", "errors"] as const;
/** One event's detail, keyed by the event's id. */
const ERROR_KEY = ["admin", "error"] as const;

const TABS: ReadonlyArray<{ value: ErrorStatus; label: string }> = [
  { value: "new", label: "New" },
  { value: "acknowledged", label: "Acknowledged" },
  { value: "resolved", label: "Resolved" },
  { value: "ignored", label: "Ignored" },
];

const CATEGORIES: ReadonlyArray<{ value: ErrorCategory | ""; label: string }> = [
  { value: "", label: "All" },
  { value: "crash", label: "Crash" },
  { value: "sql", label: "SQL" },
  { value: "ui", label: "UI" },
  { value: "network", label: "Network" },
];

/** The error type a visitor's note carries, as the reporter sends it. */
const NOTE_TYPE = "UserDescription";
const NOTE_LABEL = "Visitor's note";

/** What is shown when the server had nothing for a field. */
const ABSENT = "–";

/** A value the server names, as the page names it: "sql" → "SQL". Anything else is shown as it came. */
const LABELS: Record<string, string> = {
  crash: "Crash",
  sql: "SQL",
  ui: "UI",
  network: "Network",
  client: "Client",
  server: "Server",
  error: "Error",
  fatal: "Fatal",
  new: "New",
  acknowledged: "Acknowledged",
  resolved: "Resolved",
  ignored: "Ignored",
};

function labelOf(value: string): string {
  return LABELS[value] ?? value;
}

function isNote(event: Pick<ErrorEventOut, "error_type">): boolean {
  return event.error_type === NOTE_TYPE;
}

/** What the Type column and the detail's heading call the event. */
function typeLabel(event: Pick<ErrorEventOut, "error_type">): string {
  return isNote(event) ? NOTE_LABEL : event.error_type;
}

/** The text of a field the server had: null, undefined and blank all mean none was recorded. */
function present(value: string | null | undefined): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

/**
 * The reference a visitor's note carries: the first twelve characters of the fingerprint of
 * the error it is about. Only a browser's report can carry one.
 */
function noteReference(event: ErrorEventDetailOut): string | null {
  if (event.source !== "client") return null;
  const reference = (event.context_json as { reference?: unknown } | null)?.reference;
  return typeof reference === "string" && reference !== "" ? reference : null;
}

/** The error a note's reference names, among the event pages loaded so far. */
function findReferenced(
  queryClient: ReturnType<typeof useQueryClient>,
  reference: string,
  notId: string,
): ErrorEventOut | undefined {
  for (const [, page] of queryClient.getQueriesData<ErrorsPageOut>({ queryKey: ERRORS_KEY })) {
    const match = page?.items.find(
      (item) => item.id !== notId && item.fingerprint.startsWith(reference),
    );
    if (match !== undefined) return match;
  }
  return undefined;
}

/** The changes offered to an event's status, in the order they are shown. */
function statusActions(status: ErrorStatus): Array<{ label: string; next: ErrorStatus }> {
  const actions: Array<{ label: string; next: ErrorStatus }> = [];
  if (status === "resolved" || status === "ignored") actions.push({ label: "Reopen", next: "new" });
  if (status === "new") actions.push({ label: "Acknowledge", next: "acknowledged" });
  if (status !== "resolved") actions.push({ label: "Resolve", next: "resolved" });
  if (status !== "ignored") actions.push({ label: "Ignore", next: "ignored" });
  return actions;
}

/** What became of Copy as Markdown. "Copied" is claimed only when the clipboard took the text. */
type Copy =
  | { state: "idle" }
  | { state: "working" }
  | { state: "copied" }
  | { state: "shown"; text: string }
  | { state: "failed"; reason: string };

export function ErrorsPage(): ReactElement {
  const [status, setStatus] = useState<ErrorStatus>("new");
  const [category, setCategory] = useState<ErrorCategory | "">("");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // The row's Open that opened the detail, to return the focus to when it closes.
  const opener = useRef<HTMLElement | null>(null);

  const filters: ErrorsFilters = { status, category: category === "" ? undefined : category, page };
  const list = useQuery({
    queryKey: [...ERRORS_KEY, status, category, page],
    queryFn: () => errors(filters),
  });

  // A different list is a different set of events: nothing is selected in it, and it starts
  // at its first page. Choosing what is already chosen changes nothing.
  const chooseStatus = (next: ErrorStatus): void => {
    if (next === status) return;
    setStatus(next);
    setPage(1);
    setSelectedId(null);
  };

  const chooseCategory = (next: ErrorCategory | ""): void => {
    if (next === category) return;
    setCategory(next);
    setPage(1);
    setSelectedId(null);
  };

  /**
   * Opens an event. From the list, the row's Open is remembered as the place to return the
   * focus to; from inside a detail (a note's "Open that error") that place is already right,
   * and the control just clicked is about to go away with its detail.
   */
  const open = useCallback((id: string, fromList: boolean): void => {
    if (fromList) {
      opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    setSelectedId(id);
  }, []);

  const close = useCallback((): void => {
    setSelectedId(null);
    opener.current?.focus();
  }, []);

  const openFromDetail = useCallback((id: string): void => open(id, false), [open]);

  const columns = useMemo<DataTableColumn<ErrorEventOut>[]>(
    () => [
      { header: "Last seen", cell: (cell) => fmtDateTime(cell.row.original.last_seen_at) },
      { header: "Category", cell: (cell) => labelOf(cell.row.original.category) },
      {
        header: "Severity",
        // Fatal is marked: the word alone already differs, the badge makes it visible at a glance.
        cell: (cell) =>
          cell.row.original.severity === "fatal" ? (
            <span className="badge badge--bad">Fatal</span>
          ) : (
            labelOf(cell.row.original.severity)
          ),
      },
      { header: "Type", cell: (cell) => typeLabel(cell.row.original) },
      { header: "Message", meta: { truncate: true }, cell: (cell) => cell.row.original.message },
      { header: "Route", cell: (cell) => cell.row.original.route ?? ABSENT },
      { header: "Times", meta: { num: true }, cell: (cell) => fmtInt(cell.row.original.occurrence_count) },
      {
        header: "Actions",
        // The columns do not depend on which event is open: rebuilt columns would replace
        // this button, and the focus could not return to it when the detail closes.
        cell: (cell) => {
          const event = cell.row.original;
          return (
            <button
              type="button"
              className="btn btn--small"
              aria-label={`Open ${typeLabel(event)}, last seen ${fmtDateTime(event.last_seen_at)}`}
              onClick={() => open(event.id, true)}
            >
              Open
            </button>
          );
        },
      },
    ],
    [open],
  );

  const selected = selectedId === null ? null : (list.data?.items.find((item) => item.id === selectedId) ?? null);
  const empty =
    `No ${status} events` + (category === "" ? "." : ` in the ${labelOf(category)} category.`);

  return (
    <>
      <h2>Errors</h2>
      <StatusTabs status={status} count={list.data?.total ?? null} onSelect={chooseStatus} />
      <div role="tabpanel" id="admin-errors-panel" aria-labelledby={`admin-errors-tab-${status}`}>
        <div className="admin-filters">
          <div className="field">
            <label htmlFor="admin-errors-category">Category</label>
            <select
              id="admin-errors-category"
              value={category}
              onChange={(event: ChangeEvent<HTMLSelectElement>) =>
                chooseCategory(event.target.value as ErrorCategory | "")
              }
            >
              {CATEGORIES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <QueryState label="Loading the error events…" query={list}>
          {(data) => (
            <>
              <DataTable
                columns={columns}
                rows={data.items}
                caption="Error events"
                empty={<Empty>{empty}</Empty>}
              />
              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={setPage}
              />
            </>
          )}
        </QueryState>
      </div>

      {selectedId !== null && (
        <ErrorDetail
          key={selectedId}
          id={selectedId}
          summary={selected}
          onOpen={openFromDetail}
          onClose={close}
        />
      )}
    </>
  );
}

interface StatusTabsProps {
  status: ErrorStatus;
  /** The total of the current tab as the server counts it; null until it is known. */
  count: number | null;
  onSelect: (status: ErrorStatus) => void;
}

/**
 * The four statuses as tabs. Only the current tab shows a count, because the total the
 * server gives is the one of the list as loaded; a number on any other tab would be a guess.
 * The arrow keys move between tabs and select as they go.
 */
function StatusTabs({ status, count, onSelect }: StatusTabsProps): ReactElement {
  const tabs = useRef<Array<HTMLButtonElement | null>>([]);
  const last = TABS.length - 1;

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    const moves: Record<string, number> = {
      ArrowRight: (index + 1) % TABS.length,
      ArrowLeft: (index - 1 + TABS.length) % TABS.length,
      Home: 0,
      End: last,
    };
    const target = moves[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = TABS[target];
    if (next === undefined) return;
    onSelect(next.value);
    tabs.current[target]?.focus();
  };

  return (
    <div className="tabs" role="tablist" aria-label="Status">
      {TABS.map((tab, index) => {
        const selected = tab.value === status;
        return (
          <button
            key={tab.value}
            ref={(node) => {
              tabs.current[index] = node;
            }}
            type="button"
            role="tab"
            id={`admin-errors-tab-${tab.value}`}
            aria-selected={selected}
            aria-controls="admin-errors-panel"
            tabIndex={selected ? 0 : -1}
            className={selected ? "tab tab--active" : "tab"}
            onKeyDown={(event) => onKeyDown(event, index)}
            onClick={() => onSelect(tab.value)}
          >
            {tab.label}
            {selected && count !== null && <span className="tab__count">{fmtInt(count)}</span>}
          </button>
        );
      })}
    </div>
  );
}

interface ErrorDetailProps {
  id: string;
  /** The list's row for the event, for the heading while the detail loads. */
  summary: ErrorEventOut | null;
  onOpen: (id: string) => void;
  onClose: () => void;
}

/**
 * One event below the list, taking the focus as it opens: every field, the stack, the
 * context and the console lines, what can be done to its status, and the Markdown document.
 * A refusal of either action is shown beside it.
 */
function ErrorDetail({ id, summary, onOpen, onClose }: ErrorDetailProps): ReactElement {
  const headingId = useId();
  const section = useRef<HTMLElement>(null);
  const bundleField = useRef<HTMLTextAreaElement>(null);
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: [...ERROR_KEY, id], queryFn: () => errorDetail(id) });
  const [done, setDone] = useState<string | null>(null);
  const [copy, setCopy] = useState<Copy>({ state: "idle" });

  useEffect(() => {
    section.current?.focus();
  }, []);

  const change = useMutation({
    mutationFn: (next: ErrorStatus) => updateErrorStatus(id, next),
    onSuccess: (updated) => {
      // The server's answer is the truth: the detail shows it at once, then the lists follow.
      queryClient.setQueryData<ErrorEventDetailOut>([...ERROR_KEY, id], (current) =>
        current === undefined ? current : { ...current, ...updated },
      );
      void queryClient.invalidateQueries({ queryKey: ERRORS_KEY });
      setDone(updated.status === "new" ? "Reopened." : `Marked as ${updated.status}.`);
    },
  });

  const copyMarkdown = async (): Promise<void> => {
    setCopy({ state: "working" });
    let text: string;
    try {
      text = await errorBundle(id);
    } catch (error) {
      setCopy({ state: "failed", reason: apiErrorText(error) });
      return;
    }
    // The clipboard object is missing in some browsers and on insecure origins; a write can
    // still be refused. In both cases the document is shown, and that is said.
    const clipboard = navigator.clipboard as Clipboard | undefined;
    if (clipboard !== undefined) {
      try {
        await clipboard.writeText(text);
        setCopy({ state: "copied" });
        return;
      } catch {
        // shown below
      }
    }
    setCopy({ state: "shown", text });
  };

  const shown = query.data ?? summary;

  return (
    <section ref={section} className="card admin-detail" aria-labelledby={headingId} tabIndex={-1}>
      <div className="admin-detail__head">
        <div>
          <h3 id={headingId}>{shown === null ? "Error event" : typeLabel(shown)}</h3>
          {shown !== null && (
            <p className="hint admin-detail__meta">
              <StatusBadge kind={shown.status} />
              <span>{labelOf(shown.category)}</span>
              <span>{labelOf(shown.severity)}</span>
              <span>Last seen {fmtDateTime(shown.last_seen_at)}</span>
            </p>
          )}
        </div>
        <button type="button" className="btn btn--small" onClick={onClose}>
          Close
        </button>
      </div>

      <QueryState label="Loading the event…" query={query}>
        {(event) => {
          const note = isNote(event);
          const reference = noteReference(event);
          const referenced =
            reference === null ? undefined : findReferenced(queryClient, reference, event.id);
          const context =
            event.context_json === null || event.context_json === undefined
              ? null
              : JSON.stringify(event.context_json, null, 2);
          return (
            <>
              <div className="btn-row admin-detail__actions">
                {statusActions(event.status).map((action) => (
                  <button
                    key={action.label}
                    type="button"
                    className="btn btn--small"
                    disabled={change.isPending}
                    onClick={() => change.mutate(action.next)}
                  >
                    {action.label}
                  </button>
                ))}
                <button
                  type="button"
                  className="btn btn--small"
                  disabled={copy.state === "working"}
                  aria-busy={copy.state === "working" || undefined}
                  onClick={() => void copyMarkdown()}
                >
                  Copy as Markdown
                </button>
              </div>

              {done !== null && (
                <p className="alert alert--success" role="status">
                  {done}
                </p>
              )}
              {change.error !== null && (
                <p className="alert alert--error" role="alert">
                  {apiErrorText(change.error)}
                </p>
              )}

              {copy.state === "copied" && (
                <p className="alert alert--success" role="status">
                  Copied.
                </p>
              )}
              {copy.state === "failed" && (
                <p className="alert alert--error" role="alert">
                  {copy.reason}
                </p>
              )}
              {copy.state === "shown" && (
                <div className="admin-bundle">
                  <p role="status">
                    The clipboard is not available here, so the document is shown below. Use
                    Select all, then copy.
                  </p>
                  <textarea
                    ref={bundleField}
                    aria-label="Error event as Markdown"
                    readOnly
                    wrap="off"
                    rows={12}
                    value={copy.text}
                  />
                  <button
                    type="button"
                    className="btn btn--small"
                    onClick={() => bundleField.current?.select()}
                  >
                    Select all
                  </button>
                </div>
              )}

              {note ? (
                <>
                  <p className="admin-note__text">{event.message}</p>
                  <p className="hint">
                    A visitor's answer to “What were you doing?” on the recovery screen. It is a
                    note about an error, not an error.
                  </p>
                  {reference !== null && (
                    <p className="admin-note__about">
                      A note on the error whose reference is <code>{reference}</code>.{" "}
                      {referenced === undefined ? (
                        "That error is not among the events loaded here."
                      ) : (
                        <>
                          It belongs to <strong>{typeLabel(referenced)}</strong>, last seen{" "}
                          {fmtDateTime(referenced.last_seen_at)}.{" "}
                          <button
                            type="button"
                            className="btn btn--small"
                            onClick={() => onOpen(referenced.id)}
                          >
                            Open that error
                          </button>
                        </>
                      )}
                    </p>
                  )}
                </>
              ) : null}

              <dl className="admin-kv">
                <dt>Source</dt>
                <dd>{labelOf(event.source)}</dd>
                <dt>Category</dt>
                <dd>{labelOf(event.category)}</dd>
                <dt>Severity</dt>
                <dd>{labelOf(event.severity)}</dd>
                <dt>Status</dt>
                <dd>{labelOf(event.status)}</dd>
                <dt>Type</dt>
                <dd>{typeLabel(event)}</dd>
                {note ? null : (
                  <>
                    <dt>Message</dt>
                    <dd>{event.message}</dd>
                  </>
                )}
                <dt>Route</dt>
                <dd>{event.route ?? ABSENT}</dd>
                <dt>Method</dt>
                <dd>{event.method ?? ABSENT}</dd>
                <dt>Status code</dt>
                <dd>{event.status_code === null ? ABSENT : fmtInt(event.status_code)}</dd>
                <dt>Request ID</dt>
                <dd>
                  {event.request_id === null ? ABSENT : <code>{event.request_id}</code>}
                </dd>
                <dt>App version</dt>
                <dd>{event.app_version ?? ABSENT}</dd>
                <dt>Occurrences</dt>
                <dd>{fmtInt(event.occurrence_count)}</dd>
                <dt>First seen</dt>
                <dd>{fmtDateTime(event.first_seen_at)}</dd>
                <dt>Last seen</dt>
                <dd>{fmtDateTime(event.last_seen_at)}</dd>
                <dt>Fingerprint</dt>
                <dd>
                  <code>{event.fingerprint}</code>
                </dd>
                <dt>User agent</dt>
                <dd>{event.user_agent ?? ABSENT}</dd>
                <CodeField label="Stack" text={present(event.stack)} />
                <CodeField label="Context" text={context} />
                <CodeField label="Console log" text={present(event.console_log)} />
              </dl>
            </>
          );
        }}
      </QueryState>
    </section>
  );
}

/** A field holding a text the reader may need to copy or read line by line, in a block that scrolls. */
function CodeField({ label, text }: { label: string; text: string | null }): ReactElement {
  if (text === null) {
    return (
      <>
        <dt>{label}</dt>
        <dd>{ABSENT}</dd>
      </>
    );
  }
  return (
    <>
      <dt>{label}</dt>
      <dd>
        {/* Focusable, so a keyboard user can scroll it. */}
        <figure className="admin-code" role="region" aria-label={`${label} code block`} tabIndex={0}>
          <pre>
            <code>{text}</code>
          </pre>
        </figure>
      </dd>
    </>
  );
}
