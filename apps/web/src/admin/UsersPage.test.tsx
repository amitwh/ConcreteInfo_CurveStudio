import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { AdminUserOut, AuditOut, PlanAdminOut, UserChange, UsersPageOut } from "./adminApi";
import { UsersPage } from "./UsersPage";

const { authState } = vi.hoisted(() => ({
  authState: {
    user: { id: "", email: "admin@example.test", name: "The Admin", role: "admin" } as {
      id: string;
      email: string;
      name: string;
      role: "member" | "admin";
    } | null,
  },
}));

vi.mock("../api/auth", () => ({
  useAuth: () => ({ user: authState.user }),
}));

const ADMIN_ID = "6e0f2f4a-1f7c-4b4e-9a4a-0c7d2f9b1c11";
const MEERA_ID = "9b1c2d3e-aaaa-4bbb-8ccc-0123456789ab";
const NEWBIE_ID = "7c8b9a0f-cccc-4ddd-8eee-222222222222";
const OLDER_ID = "5d4e3f2a-bbbb-4ccc-8ddd-111111111111";

function account(overrides: Partial<AdminUserOut>): AdminUserOut {
  return {
    id: "0a0a0a0a-0000-4000-8000-000000000000",
    email: "someone@example.test",
    name: "Someone",
    role: "member",
    plan: "free",
    plan_override: null,
    is_active: true,
    email_verified: true,
    created_at: "2026-06-15T10:00:00+05:30",
    last_login_at: "2026-11-10T10:00:00+05:30",
    last_active_on: "2026-11-10",
    usage: { datasets: 1, dataset_bytes: 2000, fits: 1, autofits_today: 0 },
    ...overrides,
  };
}

/** An account resolved to self-hosted that still carries an override to another plan. */
const MEERA = account({
  id: MEERA_ID,
  email: "meera@example.test",
  name: "Meera Nair",
  plan: "self-hosted",
  plan_override: "pro",
  created_at: "2026-11-20T10:00:00+05:30",
  last_login_at: "2026-11-22T09:30:00+05:30",
  last_active_on: "2026-11-22",
  usage: { datasets: 12, dataset_bytes: 48_000_000, fits: 40, autofits_today: 2 },
});
/** Active, but the address was never confirmed. */
const NEWBIE = account({
  id: NEWBIE_ID,
  email: "new@example.test",
  name: "New Person",
  role: "admin",
  email_verified: false,
  created_at: "2026-11-01T12:00:00+05:30",
  last_login_at: null,
  last_active_on: "2026-11-21",
});
/** Deactivated, never active since the day it joined, and with no name. */
const OLDER = account({
  id: OLDER_ID,
  email: "old@example.test",
  name: "",
  is_active: false,
  created_at: "2026-10-02T08:00:00Z",
  last_login_at: "2026-10-02T08:05:00Z",
  last_active_on: null,
  usage: { datasets: 3, dataset_bytes: 0, fits: 0, autofits_today: 0 },
});
/** An account whose override is the plan in force: the override is the plan it is on. */
const IN_FORCE = account({
  id: "0c0c0c0c-0000-4000-8000-00000000000c",
  email: "devi@example.test",
  name: "Devi Rao",
  plan: "pro",
  plan_override: "pro",
});
/** The administrator who is signed in. */
const SELF = account({
  id: ADMIN_ID,
  email: "admin@example.test",
  name: "The Admin",
  role: "admin",
  created_at: "2026-01-01T00:00:00+05:30",
});

const FREE: PlanAdminOut = {
  name: "free",
  limits: { max_datasets: 25, max_dataset_bytes: 20_000_000, max_fits: 100, max_autofits_per_day: 5 },
};
const PRO: PlanAdminOut = {
  name: "pro",
  limits: { max_datasets: 200, max_dataset_bytes: 500_000_000, max_fits: 2000, max_autofits_per_day: 500 },
};
const SELF_HOSTED: PlanAdminOut = {
  name: "self-hosted",
  limits: { max_datasets: 25, max_dataset_bytes: 200_000_000, max_fits: 1000, max_autofits_per_day: 50 },
};
const PLANS = [FREE, PRO, SELF_HOSTED];

const AUDIT_NEW: AuditOut = {
  id: "a-9",
  user_id: MEERA_ID,
  action: "fit.create",
  entity: "fit",
  entity_id: "f-77",
  detail: { dataset: "mix-20.csv", model: "plastic" },
  request_id: "req-9",
  created_at: "2026-11-22T09:31:00+05:30",
};
const AUDIT_OLD: AuditOut = {
  id: "a-8",
  user_id: MEERA_ID,
  action: "auth.login",
  entity: "session",
  entity_id: null,
  detail: {},
  request_id: null,
  // 18:30 UTC on the 21st is midnight on the 22nd in IST
  created_at: "2026-11-21T18:30:00Z",
};

const PLAN_SAVED = "Plan saved. The change is in the audit trail.";
const ROLE_SAVED = "Role saved. The change is in the audit trail.";
const OWN_ROW_NOTE = "You cannot change your own role or active flag.";

/** A spy on the client's one request method, with its call record typed. */
function spyRequest() {
  return vi.spyOn(api, "request");
}
type RequestSpy = ReturnType<typeof spyRequest>;

interface Refusals {
  /** A plan the server no longer has, refused on save. */
  plan?: string;
  /** What the server answers to a role change. */
  role?: ApiError;
  /** What the server answers to an activation change. */
  active?: ApiError;
  /** What the server answers to a deletion. */
  delete?: ApiError;
}

/**
 * Answers like the API: the list and the plans are read, a PATCH or DELETE on an account
 * is applied to what the list then shows; any other call fails the test.
 */
function serve(refusals: Refusals = {}, list: AdminUserOut[] = [MEERA, NEWBIE, OLDER, SELF]): RequestSpy {
  let stored = list;
  return spyRequest().mockImplementation(async (path, init) => {
    if (path === "/admin/plans") return PLANS;
    if (path.startsWith("/admin/users?")) {
      const page = Number(new URLSearchParams(path.split("?")[1]).get("page"));
      return { items: stored, total: 128, page, page_size: 25 } satisfies UsersPageOut;
    }
    const match = /^\/admin\/users\/([^/]+)$/.exec(path);
    if (match === null) throw new Error(`unexpected request ${path}`);
    const id = match[1] ?? "";
    const current = stored.find((user) => user.id === id);
    if (current === undefined) throw new ApiError(404, "NOT_FOUND", "user not found", { code: "NOT_FOUND" });
    if (init === undefined) {
      return { user: current, audit: id === MEERA_ID ? [AUDIT_NEW, AUDIT_OLD] : [] };
    }
    if (init.method === "PATCH") {
      const change = JSON.parse(String(init.body)) as UserChange;
      if (change.plan_override !== undefined && change.plan_override === refusals.plan) {
        throw new ApiError(422, "VALIDATION_ERROR", `there is no plan named '${change.plan_override}'`, {});
      }
      if (change.role !== undefined && refusals.role !== undefined) throw refusals.role;
      if (change.is_active !== undefined && refusals.active !== undefined) throw refusals.active;
      const updated: AdminUserOut = {
        ...current,
        role: change.role ?? current.role,
        is_active: change.is_active ?? current.is_active,
        plan_override: change.plan_override === undefined ? current.plan_override : change.plan_override,
        plan:
          change.plan_override === undefined
            ? current.plan
            : change.plan_override === null
              ? "free"
              : change.plan_override,
      };
      stored = stored.map((user) => (user.id === id ? updated : user));
      return updated;
    }
    if (init.method === "DELETE") {
      if (refusals.delete !== undefined) throw refusals.delete;
      stored = stored.filter((user) => user.id !== id);
      return undefined;
    }
    throw new Error(`unexpected request ${path} ${init.method}`);
  });
}

function table(): HTMLElement {
  return screen.getByRole("table", { name: "Accounts" });
}

/** The data rows of the accounts table, each cell's text. */
function rowTexts(): string[][] {
  return within(table())
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell").map((cell) => cell.textContent ?? ""));
}

function row(email: string): HTMLElement {
  return within(table()).getByText(email).closest("tr") as HTMLElement;
}

/** Each term with its definition, as "Joined20/11/2026", across every list in `container`. */
function facts(container: HTMLElement): string[] {
  const terms = within(container).getAllByRole("term");
  const definitions = within(container).getAllByRole("definition");
  return terms.map((term, index) => `${term.textContent}${definitions[index]?.textContent ?? ""}`);
}

/** Every call's path and init, as plain values. */
function calls(spy: RequestSpy): Array<[string, RequestInit | undefined]> {
  return spy.mock.calls.map(([path, init]) => [path, init]);
}

function sentBody(spy: RequestSpy, method: string): Array<{ path: string; body: unknown }> {
  return calls(spy)
    .filter(([, init]) => init?.method === method)
    .map(([path, init]) => ({ path, body: init?.body === undefined ? null : JSON.parse(String(init.body)) }));
}

function listCalls(spy: RequestSpy): string[] {
  return calls(spy)
    .map(([path]) => path)
    .filter((path) => path.startsWith("/admin/users?"));
}

beforeEach(() => {
  authState.user = { id: ADMIN_ID, email: "admin@example.test", name: "The Admin", role: "admin" };
});

describe("UsersPage — the list", () => {
  it("shows loading, then every column of each account, newest first", async () => {
    serve();
    renderWithQuery(<UsersPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the accounts");
    await screen.findByRole("table", { name: "Accounts" });

    expect(within(table()).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "Joined",
      "Name",
      "E-mail",
      "Plan",
      "Role",
      "Status",
      "Last active",
      "Datasets",
      "Stored fits",
      "Actions",
    ]);
    const rows = rowTexts();
    expect(rows.map((cells) => [cells[0], cells[1], cells[2]])).toEqual([
      ["20/11/2026", "Meera Nair", "meera@example.test"],
      ["01/11/2026", "New Person", "new@example.test"],
      ["02/10/2026", "–", "old@example.test"],
      ["01/01/2026", "The Admin", "admin@example.test"],
    ]);
    expect(rows.map((cells) => cells[3])).toEqual([
      "self-hosted → overridden to pro",
      "free",
      "free",
      "free",
    ]);
    expect(rows.map((cells) => cells[4])).toEqual(["Member", "Admin", "Member", "Admin"]);
    // Datasets and stored fits, in en-IN
    expect(rows[0]?.slice(7, 9)).toEqual(["12", "40"]);
    expect(screen.getByText("128 accounts")).toBeTruthy();
    expect(screen.getByText("Showing 1–25 of 128")).toBeTruthy();
  });

  it("marks a deactivated account, an unverified address and an account never active", async () => {
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });

    const meera = within(row("meera@example.test")).getAllByRole("cell");
    expect(meera[5]?.textContent).toBe("Active");
    expect(meera[6]?.textContent).toBe("22/11/2026");

    const newbie = within(row("new@example.test")).getAllByRole("cell");
    expect(within(newbie[5] as HTMLElement).getByText("Active")).toBeTruthy();
    const note = within(newbie[5] as HTMLElement).getByText("E-mail not verified");
    expect(note.className).toContain("muted");

    const older = within(row("old@example.test")).getAllByRole("cell");
    expect(older[5]?.textContent).toBe("Deactivated");
    expect(older[6]?.textContent).toBe("Never");
  });

  it("marks a plan override while it is in force, not only when it names another plan", async () => {
    const user = userEvent.setup();
    serve({}, [IN_FORCE, MEERA]);
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });

    // In the list: the override shows whether or not it is the plan in force.
    expect(rowTexts().map((cells) => cells[3])).toEqual([
      "pro (override)",
      "self-hosted → overridden to pro",
    ]);

    // And in the account's detail.
    await user.click(within(row("devi@example.test")).getByRole("button", { name: "Open devi@example.test" }));
    const detail = screen.getByRole("region", { name: "Devi Rao" });
    expect(await within(detail).findByText(/set by an override/)).toBeTruthy();
    expect(within(detail).queryByText(/overridden to/)).toBeNull();
  });

  it("offers Open on every row, and the changes on every row but the signed-in administrator's own", async () => {
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });

    const names = (email: string): string[] =>
      within(row(email))
        .getAllByRole("button")
        .map((button) => button.getAttribute("aria-label") ?? button.textContent ?? "");
    expect(names("meera@example.test")).toEqual([
      "Open meera@example.test",
      "Change plan meera@example.test",
      "Change role meera@example.test",
      "Deactivate meera@example.test",
      "Delete meera@example.test",
    ]);
    expect(names("old@example.test")).toEqual([
      "Open old@example.test",
      "Change plan old@example.test",
      "Change role old@example.test",
      "Reactivate old@example.test",
      "Delete old@example.test",
    ]);
    expect(names("admin@example.test")).toEqual(["Open admin@example.test"]);
  });

  it("shows an empty state when nothing matches", async () => {
    serve({}, []);
    renderWithQuery(<UsersPage />);
    expect(await screen.findByText("No accounts match")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows an error with a Retry that fetches the accounts again", async () => {
    const user = userEvent.setup();
    const spy = serve();
    const answer = spy.getMockImplementation();
    let fail = true;
    spy.mockImplementation(async (path, init) => {
      if (fail && path.startsWith("/admin/users?")) {
        throw new ApiError(500, "INTERNAL_ERROR", "the server failed", null);
      }
      return answer?.(path, init);
    });
    renderWithQuery(<UsersPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    expect(screen.queryByRole("table")).toBeNull();

    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "Accounts" })).toBeTruthy();
    expect(listCalls(spy)).toHaveLength(2);
  });

  it("keeps the list working when the plans cannot be loaded: the plan filter offers only Any, and a Retry loads them", async () => {
    const user = userEvent.setup();
    const ONE: UsersPageOut = { items: [MEERA], total: 1, page: 1, page_size: 25 };
    const spy = spyRequest().mockImplementation(async (path) => {
      if (path === "/admin/plans") throw new ApiError(500, "INTERNAL_ERROR", "the server failed", null);
      if (path.startsWith("/admin/users?")) return ONE;
      throw new Error(`unexpected request ${path}`);
    });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    const planFilter = screen.getByLabelText("Plan") as HTMLSelectElement;
    expect(Array.from(planFilter.options).map((option) => option.textContent)).toEqual(["Any"]);
    expect(screen.getByRole("alert").textContent).toBe("the server failed");

    spy.mockImplementation(async (path) => {
      if (path === "/admin/plans") return PLANS;
      if (path.startsWith("/admin/users?")) return ONE;
      throw new Error(`unexpected request ${path}`);
    });
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() =>
      expect(Array.from(planFilter.options).map((option) => option.textContent)).toEqual([
        "Any",
        "free",
        "pro",
        "self-hosted",
      ]),
    );
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("UsersPage — filters and pages", () => {
  it("applies the filters only on submit, as the API's query string", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    expect(listCalls(spy)).toEqual(["/admin/users?page=1"]);
    expect(screen.getByRole("option", { name: "pro" })).toBeTruthy();

    await user.type(screen.getByRole("searchbox", { name: "Name or e-mail" }), "Meera+1@example.test");
    await user.selectOptions(screen.getByLabelText("Plan"), "pro");
    await user.selectOptions(screen.getByLabelText("Role"), "admin");
    await user.selectOptions(screen.getByLabelText("Status"), "inactive");
    // typing and choosing are not applying
    expect(listCalls(spy)).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() =>
      expect(listCalls(spy).at(-1)).toBe(
        "/admin/users?q=Meera%2B1%40example.test&plan=pro&role=admin&active=false&page=1",
      ),
    );
    await screen.findByRole("table", { name: "Accounts" });
  });

  it("sends no empty parameters for the filters left at Any", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.selectOptions(screen.getByLabelText("Status"), "active");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(listCalls(spy).at(-1)).toBe("/admin/users?active=true&page=1"));
  });

  it("clears the filters back to the unfiltered first page", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.type(screen.getByRole("searchbox", { name: "Name or e-mail" }), "meera");
    await user.selectOptions(screen.getByLabelText("Role"), "admin");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(listCalls(spy).at(-1)).toBe("/admin/users?q=meera&role=admin&page=1"));
    await screen.findByRole("table", { name: "Accounts" });

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await waitFor(() => expect(listCalls(spy).at(-1)).toBe("/admin/users?page=1"));
    expect((screen.getByRole("searchbox", { name: "Name or e-mail" }) as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Role") as HTMLSelectElement).value).toBe("");
    await screen.findByRole("table", { name: "Accounts" });
  });

  it("asks for the next page, and starts a filtered list again from the first page", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(listCalls(spy).at(-1)).toBe("/admin/users?page=2"));

    await user.selectOptions(screen.getByLabelText("Plan"), "pro");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(listCalls(spy).at(-1)).toBe("/admin/users?plan=pro&page=1"));
  });
});

describe("UsersPage — one account's detail", () => {
  it("opens below the list, takes the focus, and shows the usage against the plan's limits", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    expect(screen.queryByRole("region", { name: "Meera Nair" })).toBeNull();

    const trigger = within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" });
    await user.click(trigger);
    const detail = screen.getByRole("region", { name: "Meera Nair" });
    // reachable by keyboard: the detail takes the focus as it opens
    expect(document.activeElement).toBe(detail);
    expect(await within(detail).findByRole("progressbar", { name: "Storage" })).toBeTruthy();

    const meters = within(detail).getAllByRole("progressbar");
    expect(meters.map((meter) => meter.getAttribute("aria-label"))).toEqual([
      "Datasets",
      "Storage",
      "Stored fits",
      "Auto-fit runs today",
    ]);
    expect(meters.map((meter) => meter.getAttribute("aria-valuetext"))).toEqual([
      "12 of 25",
      "48 MB of 200 MB",
      "40 of 1,000",
      "2 of 50",
    ]);
    // the plan the account is on, and the override that put it there
    expect(within(detail).getByText("self-hosted")).toBeTruthy();
    expect(within(detail).getByText(/overridden to pro/)).toBeTruthy();

    // The instants carry their time in IST; the active day is a date.
    const accountFacts = facts(detail);
    expect(accountFacts).toContain("Joined20/11/2026 10:00");
    expect(accountFacts).toContain("Last sign-in22/11/2026 09:30");
    expect(accountFacts).toContain("Last active22/11/2026");
  });

  it("shows the licence entitlement behind the plan, and that an override wins until a licence changes it", async () => {
    const user = userEvent.setup();
    const LICENCED_ID = "1d1d1d1d-0000-4000-8000-00000000000d";
    const licenced = account({
      id: LICENCED_ID,
      email: "licenced@example.test",
      name: "Licenced Person",
      plan: "paid",
      plan_override: "paid",
      licence: {
        status: "active",
        plan_code: "curvestudio-pro",
        expires_at: "2026-12-25T10:00:00+05:30",
        last_checked_at: "2026-11-24T03:00:00+05:30",
      },
    });
    serve({}, [licenced]);
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(
      within(row("licenced@example.test")).getByRole("button", { name: "Open licenced@example.test" }),
    );
    const detail = screen.getByRole("region", { name: "Licenced Person" });
    expect(await within(detail).findByText(/set by an override/)).toBeTruthy();
    expect(within(detail).getByText(/an override wins until a licence changes it/)).toBeTruthy();
    const licenceFacts = facts(detail);
    expect(licenceFacts).toContain("Licenceactive");
    expect(licenceFacts).toContain("Licence plan codecurvestudio-pro");
    expect(licenceFacts).toContain("Licence expiry25/12/2026 10:00");
    expect(licenceFacts).toContain("Licence last check24/11/2026 03:00");
  });

  it("shows no licence facts for an account without one", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" }));
    const detail = screen.getByRole("region", { name: "Meera Nair" });
    await within(detail).findByRole("progressbar", { name: "Datasets" });
    expect(within(detail).queryByText("Licence plan code")).toBeNull();
  });

  it("lists the account's audit rows, newest first", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" }));

    const audit = await screen.findByRole("table", { name: "Audit trail for meera@example.test" });
    const cells = within(audit)
      .getAllByRole("row")
      .slice(1)
      .map((tableRow) => within(tableRow).getAllByRole("cell").map((cell) => cell.textContent));
    expect(cells).toEqual([
      ["22/11/2026 09:31", "fit.create", "fit f-77", '{"dataset":"mix-20.csv","model":"plastic"}'],
      ["22/11/2026 00:00", "auth.login", "session", "–"],
    ]);
    expect(within(audit).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "When",
      "Action",
      "What",
      "Detail",
    ]);
  });

  it("shows the usage in plain figures when the plan's limits are not known", async () => {
    const user = userEvent.setup();
    // An override that names no plan row resolves to a plan the list of plans does not carry.
    const UNKNOWN_ID = "0b0b0b0b-0000-4000-8000-00000000000b";
    const unknown = account({ id: UNKNOWN_ID, email: "who@example.test", name: "Who", plan: "enterprise" });
    serve({}, [MEERA, unknown]);
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("who@example.test")).getByRole("button", { name: "Open who@example.test" }));
    const detail = screen.getByRole("region", { name: "Who" });
    expect(await within(detail).findByText(/The limits of the plan enterprise are not known/)).toBeTruthy();
    expect(within(detail).queryByRole("progressbar")).toBeNull();
    const figures = facts(detail);
    expect(figures).toContain("Datasets1");
    expect(figures).toContain("Storage2 KB");
    expect(figures).toContain("Auto-fit runs today0");
  });

  it("closes on request and returns the focus to the row's Open", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    const trigger = within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" });
    await user.click(trigger);
    await screen.findByRole("region", { name: "Meera Nair" });
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("region", { name: "Meera Nair" })).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("says the account is loading until the server answers", async () => {
    const user = userEvent.setup();
    const spy = serve();
    const answer = spy.getMockImplementation();
    let release = (): void => {};
    spy.mockImplementation((path, init) => {
      if (path === `/admin/users/${MEERA_ID}` && init === undefined) {
        return new Promise((resolve) => {
          release = () => resolve({ user: MEERA, audit: [] });
        });
      }
      return answer === undefined ? Promise.reject(new Error("nothing behind the spy")) : answer(path, init);
    });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" }));
    const detail = screen.getByRole("region", { name: "Meera Nair" });
    expect(within(detail).getByRole("status").textContent).toContain("Loading the account");
    expect(within(detail).queryByRole("progressbar")).toBeNull();

    release();
    expect(await within(detail).findByRole("progressbar", { name: "Datasets" })).toBeTruthy();
  });

  it("shows the server's reason when the account cannot be loaded, with a Retry", async () => {
    const user = userEvent.setup();
    const spy = serve();
    const answer = spy.getMockImplementation();
    let refuse = true;
    spy.mockImplementation(async (path, init) => {
      if (refuse && path === `/admin/users/${MEERA_ID}` && init === undefined) {
        throw new ApiError(500, "INTERNAL_ERROR", "the server failed", null);
      }
      return answer?.(path, init);
    });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Open meera@example.test" }));
    const detail = screen.getByRole("region", { name: "Meera Nair" });
    expect((await within(detail).findByRole("alert")).textContent).toBe("the server failed");

    refuse = false;
    await user.click(within(detail).getByRole("button", { name: "Retry" }));
    expect(await within(detail).findByRole("progressbar", { name: "Datasets" })).toBeTruthy();
  });

  it("says on the administrator's own account that its role and active flag cannot be changed here", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("admin@example.test")).getByRole("button", { name: "Open admin@example.test" }));
    const detail = screen.getByRole("region", { name: "The Admin" });
    expect(within(detail).getByText(OWN_ROW_NOTE)).toBeTruthy();
  });
});

describe("UsersPage — changing an account", () => {
  it("Change plan sends the chosen override, confirms it, and shows the plan the server resolved", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change plan meera@example.test" }));

    const dialog = screen.getByRole("dialog", { name: "Change plan" });
    const select = within(dialog).getByLabelText("Plan") as HTMLSelectElement;
    // fed by GET /admin/plans, with the way back to the default first; the current one chosen
    expect(Array.from(select.options).map((option) => option.textContent)).toEqual([
      "Default plan (no override)",
      "free",
      "pro",
      "self-hosted",
    ]);
    expect(select.value).toBe("pro");
    expect(within(dialog).getByRole("button", { name: "Save" })).toHaveProperty("disabled", true);

    await user.selectOptions(select, "free");
    await user.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await screen.findByText(PLAN_SAVED)).toBeTruthy();
    expect(sentBody(spy, "PATCH")).toEqual([{ path: `/admin/users/${MEERA_ID}`, body: { plan_override: "free" } }]);
    expect(screen.queryByRole("dialog")).toBeNull();
    // the list is fetched again and shows what the server now holds: free, by the override
    await waitFor(() => expect(listCalls(spy)).toHaveLength(2));
    expect(within(row("meera@example.test")).getAllByRole("cell")[3]?.textContent).toBe(
      "free (override)",
    );
  });

  it("Change plan can clear an override, sent as null", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change plan meera@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Change plan" });
    await user.selectOptions(within(dialog).getByLabelText("Plan"), "");
    await user.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await screen.findByText(PLAN_SAVED)).toBeTruthy();
    expect(sentBody(spy, "PATCH")[0]?.body).toEqual({ plan_override: null });
  });

  it("shows the server's reason for a plan it does not have, keeps the form open and the row as it was", async () => {
    const user = userEvent.setup();
    const spy = serve({ plan: "pro" });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    // Start from an account on the default plan and pick the plan the server refuses.
    await user.click(within(row("new@example.test")).getByRole("button", { name: "Change plan new@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Change plan" });
    await user.selectOptions(within(dialog).getByLabelText("Plan"), "pro");
    await user.click(within(dialog).getByRole("button", { name: "Save" }));

    expect((await within(dialog).findByRole("alert")).textContent).toBe("there is no plan named 'pro'");
    expect((within(dialog).getByLabelText("Plan") as HTMLSelectElement).value).toBe("pro");
    expect(sentBody(spy, "PATCH")).toEqual([{ path: `/admin/users/${NEWBIE_ID}`, body: { plan_override: "pro" } }]);
    expect(screen.queryByText(PLAN_SAVED)).toBeNull();
    expect(within(row("new@example.test")).getAllByRole("cell")[3]?.textContent).toBe("free");

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(1);
  });

  it("closes Change plan without a change on Cancel or Escape", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change plan meera@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Change plan" });
    await user.selectOptions(within(dialog).getByLabelText("Plan"), "free");
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(0);

    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change plan meera@example.test" }));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(0);
  });

  it("Change role sends the chosen role, confirms it, and the row shows it", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change role meera@example.test" }));

    const dialog = screen.getByRole("dialog", { name: "Change role" });
    expect((within(dialog).getByRole("radio", { name: "Member" }) as HTMLInputElement).checked).toBe(true);
    expect(within(dialog).getByRole("button", { name: "Save" })).toHaveProperty("disabled", true);

    await user.click(within(dialog).getByRole("radio", { name: "Administrator" }));
    await user.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await screen.findByText(ROLE_SAVED)).toBeTruthy();
    expect(sentBody(spy, "PATCH")).toEqual([{ path: `/admin/users/${MEERA_ID}`, body: { role: "admin" } }]);
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(within(row("meera@example.test")).getAllByRole("cell")[4]?.textContent).toBe("Admin"));
  });

  it("shows SELF_CHANGE_REFUSED as the page's own reason if the server ever answers with it", async () => {
    const user = userEvent.setup();
    serve({
      role: new ApiError(422, "VALIDATION_ERROR", "an administrator cannot change their own role or active flag", {
        code: "SELF_CHANGE_REFUSED",
      }),
    });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change role meera@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Change role" });
    await user.click(within(dialog).getByRole("radio", { name: "Administrator" }));
    await user.click(within(dialog).getByRole("button", { name: "Save" }));

    expect((await within(dialog).findByRole("alert")).textContent).toBe(OWN_ROW_NOTE);
    expect(screen.queryByText(ROLE_SAVED)).toBeNull();
    expect(within(row("meera@example.test")).getAllByRole("cell")[4]?.textContent).toBe("Member");
  });

  it("does not carry one action's refusal into the next action on the same account", async () => {
    const user = userEvent.setup();
    serve({ role: new ApiError(500, "INTERNAL_ERROR", "the server failed", null) });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change role meera@example.test" }));
    let dialog = screen.getByRole("dialog", { name: "Change role" });
    await user.click(within(dialog).getByRole("radio", { name: "Administrator" }));
    await user.click(within(dialog).getByRole("button", { name: "Save" }));
    expect((await within(dialog).findByRole("alert")).textContent).toBe("the server failed");
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Change plan meera@example.test" }));
    dialog = screen.getByRole("dialog", { name: "Change plan" });
    expect(within(dialog).queryByRole("alert")).toBeNull();
    expect((within(dialog).getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("Deactivate asks first, then sends the change and offers Reactivate", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Deactivate meera@example.test" }));

    const dialog = screen.getByRole("dialog", { name: "Deactivate account?" });
    expect(within(dialog).getByText(/meera@example\.test/)).toBeTruthy();
    expect(within(dialog).getByText(/will not be able to sign in/)).toBeTruthy();
    expect(sentBody(spy, "PATCH")).toHaveLength(0);

    await user.click(within(dialog).getByRole("button", { name: "Deactivate" }));
    expect(await screen.findByText("Account deactivated.")).toBeTruthy();
    expect(sentBody(spy, "PATCH")).toEqual([{ path: `/admin/users/${MEERA_ID}`, body: { is_active: false } }]);
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() =>
      expect(within(row("meera@example.test")).getAllByRole("cell")[5]?.textContent).toBe("Deactivated"),
    );
    expect(
      within(row("meera@example.test")).getByRole("button", { name: "Reactivate meera@example.test" }),
    ).toBeTruthy();
  });

  it("Reactivate asks first, then sends the change", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("old@example.test")).getByRole("button", { name: "Reactivate old@example.test" }));

    const dialog = screen.getByRole("dialog", { name: "Reactivate account?" });
    await user.click(within(dialog).getByRole("button", { name: "Reactivate" }));
    expect(await screen.findByText("Account reactivated.")).toBeTruthy();
    expect(sentBody(spy, "PATCH")).toEqual([{ path: `/admin/users/${OLDER_ID}`, body: { is_active: true } }]);
  });

  it("a refused activation change shows the reason inside the dialog and keeps the row as it was", async () => {
    const user = userEvent.setup();
    const spy = serve({ active: new ApiError(500, "INTERNAL_ERROR", "the server failed", null) });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Deactivate meera@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Deactivate account?" });
    await user.click(within(dialog).getByRole("button", { name: "Deactivate" }));

    expect((await within(dialog).findByRole("alert")).textContent).toBe("the server failed");
    expect(screen.queryByText("Account deactivated.")).toBeNull();
    expect(within(row("meera@example.test")).getAllByRole("cell")[5]?.textContent).toBe("Active");

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(1);
  });

  it("Cancel and Escape close a confirmation without sending anything", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Deactivate meera@example.test" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();

    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Delete meera@example.test" }));
    expect(screen.getByRole("dialog", { name: "Delete account?" })).toBeTruthy();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(0);
    expect(sentBody(spy, "DELETE")).toHaveLength(0);
    expect(within(row("meera@example.test")).getAllByRole("cell")[5]?.textContent).toBe("Active");
  });
});

describe("UsersPage — deleting an account", () => {
  const MESSAGE =
    /datasets, stored fits, jobs and sessions are deleted.*audit trail is kept, without the account's identity/;

  it("says what is erased and asks for the account's e-mail before the button enables", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("old@example.test")).getByRole("button", { name: "Delete old@example.test" }));

    const dialog = screen.getByRole("dialog", { name: "Delete account?" });
    const message = within(dialog).getByText(MESSAGE);
    expect(message.textContent).toContain("This cannot be undone.");
    // the account is named in the message, and again as the phrase to type
    expect(message.textContent).toContain("(old@example.test)");
    expect(within(dialog).getByLabelText(/type old@example\.test to confirm/i)).toBeTruthy();
    const confirm = within(dialog).getByRole("button", { name: "Delete account" }) as HTMLButtonElement;
    const input = within(dialog).getByLabelText(/type old@example\.test to confirm/i);
    expect(confirm.disabled).toBe(true);

    await user.type(input, "old@example.tes");
    expect(confirm.disabled).toBe(true);
    await user.type(input, "t");
    expect(confirm.disabled).toBe(false);
    expect(sentBody(spy, "DELETE")).toHaveLength(0);
  });

  it("sends the request on confirm, closes the detail, and says the account is deleted", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("old@example.test")).getByRole("button", { name: "Open old@example.test" }));
    await screen.findByRole("region", { name: "old@example.test" });

    await user.click(within(row("old@example.test")).getByRole("button", { name: "Delete old@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Delete account?" });
    await user.type(within(dialog).getByLabelText(/type old@example\.test to confirm/i), "old@example.test");
    await user.click(within(dialog).getByRole("button", { name: "Delete account" }));

    expect(await screen.findByText("Account deleted.")).toBeTruthy();
    expect(calls(spy).filter(([, init]) => init?.method === "DELETE")).toEqual([
      [`/admin/users/${OLDER_ID}`, { method: "DELETE" }],
    ]);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByRole("region", { name: "old@example.test" })).toBeNull();
    await waitFor(() => expect(screen.queryByText("old@example.test")).toBeNull());
  });

  it("shows the server's reason for a refused deletion and keeps the account's row", async () => {
    const user = userEvent.setup();
    const spy = serve({ delete: new ApiError(500, "INTERNAL_ERROR", "the server failed", null) });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("meera@example.test")).getByRole("button", { name: "Delete meera@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Delete account?" });
    await user.type(within(dialog).getByLabelText(/type meera@example\.test to confirm/i), "meera@example.test");
    await user.click(within(dialog).getByRole("button", { name: "Delete account" }));

    expect((await within(dialog).findByRole("alert")).textContent).toBe("the server failed");
    expect(screen.queryByText("Account deleted.")).toBeNull();
    expect(within(row("meera@example.test")).getAllByRole("cell")[2]?.textContent).toBe("meera@example.test");

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sentBody(spy, "PATCH")).toHaveLength(0);
  });

  it("closes as soon as the server has erased the account, without waiting for the list to refresh", async () => {
    const user = userEvent.setup();
    const spy = serve();
    const answer = spy.getMockImplementation();
    let erased = false;
    let release = (): void => {};
    // After the deletion, the list's refresh is held back until the test lets it go.
    spy.mockImplementation(async (path, init) => {
      if (path.startsWith("/admin/users?") && erased) {
        return new Promise((resolve) => {
          release = () => resolve(answer === undefined ? undefined : answer(path, init));
        });
      }
      if (init?.method === "DELETE") erased = true;
      return answer === undefined ? Promise.reject(new Error("nothing behind the spy")) : answer(path, init);
    });
    renderWithQuery(<UsersPage />);
    await screen.findByRole("table", { name: "Accounts" });
    await user.click(within(row("old@example.test")).getByRole("button", { name: "Delete old@example.test" }));
    const dialog = screen.getByRole("dialog", { name: "Delete account?" });
    await user.type(within(dialog).getByLabelText(/type old@example\.test to confirm/i), "old@example.test");
    await user.click(within(dialog).getByRole("button", { name: "Delete account" }));

    expect(await screen.findByText("Account deleted.")).toBeTruthy();
    expect(screen.queryByRole("dialog")).toBeNull();
    release();
    await waitFor(() => expect(screen.queryByText("old@example.test")).toBeNull());
  });
});
