import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { AuditOut } from "./adminApi";
import { AuditPage } from "./AuditPage";

const ADMIN = "6e0f2f4a-1f7c-4b4e-9a4a-0c7d2f9b1c11";
const OTHER = "9b1c2d3e-aaaa-4bbb-8ccc-0123456789ab";
const ROW_DETAIL = `{"role":"admin","by":"${ADMIN}"}`;

const ROW: AuditOut = {
  id: "a-2",
  user_id: ADMIN,
  action: "user.role_change",
  entity: "user",
  entity_id: OTHER,
  detail: { role: "admin", by: ADMIN },
  request_id: "req-2",
  created_at: "2026-11-20T10:15:00+05:30",
};
const SYSTEM_ROW: AuditOut = {
  id: "a-1",
  user_id: null,
  action: "request_stats.flush",
  entity: "request_stats",
  entity_id: null,
  detail: {},
  request_id: null,
  // 18:30 UTC is midnight on the 20th in IST
  created_at: "2026-11-19T18:30:00Z",
};

const PAGE = { items: [SYSTEM_ROW, ROW], total: 128, page: 1, page_size: 25 };
const PAGE_2 = {
  items: [{ ...ROW, id: "a-3", created_at: "2026-10-02T08:00:00Z" }],
  total: 128,
  page: 2,
  page_size: 25,
};
const NOTHING = { items: [], total: 0, page: 1, page_size: 25 };
const FILTERED = `/admin/audit?user_id=${ADMIN}&action=user.role_change&from=2026-11-01&to=2026-11-18&page=1`;

/** A spy on the client's one request method, with its call record typed. */
function spyRequest() {
  return vi.spyOn(api, "request");
}
type RequestSpy = ReturnType<typeof spyRequest>;

/** Answers like the API: the page each path asks for; any other path fails the test. */
function serve(pages: Record<string, unknown>): RequestSpy {
  return spyRequest().mockImplementation(async (path) => {
    const body = pages[path];
    if (body === undefined) throw new Error(`unexpected request ${path}`);
    return body;
  });
}

function table(): HTMLElement {
  return screen.getByRole("table", { name: "Audit trail" });
}

function rowTexts(): string[][] {
  return within(table())
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell").map((cell) => cell.textContent));
}

function setDate(label: string, value: string): void {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

describe("AuditPage", () => {
  it("shows loading, then the trail newest first with When in IST, What, Account and Detail", async () => {
    serve({ "/admin/audit?page=1": PAGE });
    renderWithQuery(<AuditPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the audit trail");
    await screen.findByRole("table", { name: "Audit trail" });

    expect(rowTexts()).toEqual([
      ["20/11/2026 00:00", "request_stats.flush", "request_stats", "System", "–"],
      ["20/11/2026 10:15", "user.role_change", `user ${OTHER}`, ADMIN, ROW_DETAIL],
    ]);
    expect(within(table()).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "When",
      "Action",
      "What",
      "Account",
      "Detail",
    ]);
    expect(screen.getByText("Showing 1–25 of 128")).toBeTruthy();
  });

  it("keeps the whole of a truncated detail in the cell's title", async () => {
    serve({ "/admin/audit?page=1": PAGE });
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });
    const detailCells = within(table())
      .getAllByRole("cell")
      .filter((cell) => cell.classList.contains("truncate"));
    expect(detailCells).toHaveLength(2);
    expect(detailCells[1]?.getAttribute("title")).toBe(ROW_DETAIL);
  });

  it("applies the filters only on submit, as the API's query string", async () => {
    const user = userEvent.setup();
    const spy = serve({ "/admin/audit?page=1": PAGE, [FILTERED]: PAGE });
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });
    expect(spy).toHaveBeenCalledTimes(1);

    await user.type(screen.getByRole("textbox", { name: "Account" }), ADMIN);
    await user.type(screen.getByRole("textbox", { name: "Action" }), "user.role_change");
    setDate("From", "2026-11-01");
    setDate("To", "2026-11-18");
    // typing is not applying
    expect(spy).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy).toHaveBeenLastCalledWith(FILTERED);
    await screen.findByRole("table", { name: "Audit trail" });
  });

  it("starts a filtered trail again from the first page", async () => {
    const user = userEvent.setup();
    const spy = serve({ "/admin/audit?page=2": PAGE_2, "/admin/audit?page=1": PAGE });
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("Showing 26–50 of 128");

    await user.type(screen.getByRole("textbox", { name: "Action" }), "user.role_change");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() =>
      expect(spy).toHaveBeenLastCalledWith("/admin/audit?action=user.role_change&page=1"),
    );
  });

  it("clears the filters back to the unfiltered first page", async () => {
    const user = userEvent.setup();
    const spy = serve({
      "/admin/audit?page=1": PAGE,
      "/admin/audit?action=user.role_change&page=1": PAGE,
    });
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });

    await user.type(screen.getByRole("textbox", { name: "Action" }), "user.role_change");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() =>
      expect(spy).toHaveBeenLastCalledWith("/admin/audit?action=user.role_change&page=1"),
    );
    await screen.findByRole("table", { name: "Audit trail" });

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await waitFor(() => expect(spy).toHaveBeenLastCalledWith("/admin/audit?page=1"));
    expect((screen.getByRole("textbox", { name: "Action" }) as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("From") as HTMLInputElement).value).toBe("");
    await screen.findByRole("table", { name: "Audit trail" });
  });

  it("asks for the next page and shows what it returns", async () => {
    const user = userEvent.setup();
    const spy = serve({ "/admin/audit?page=1": PAGE, "/admin/audit?page=2": PAGE_2 });
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });

    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(spy).toHaveBeenLastCalledWith("/admin/audit?page=2"));
    expect(await screen.findByText("Showing 26–50 of 128")).toBeTruthy();
    expect(rowTexts()).toEqual([
      ["02/10/2026 13:30", "user.role_change", `user ${OTHER}`, ADMIN, ROW_DETAIL],
    ]);
  });

  it("shows an empty state when nothing matches", async () => {
    serve({ "/admin/audit?page=1": NOTHING });
    renderWithQuery(<AuditPage />);
    expect(await screen.findByText("No audit entries match these filters.")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByText("Showing 0 of 0")).toBeTruthy();
  });

  it("shows the server's reason for a reversed date range where the table would be", async () => {
    const user = userEvent.setup();
    const spy = spyRequest()
      .mockResolvedValueOnce(PAGE)
      .mockRejectedValueOnce(
        new ApiError(422, "VALIDATION_ERROR", "the request is not valid", {
          errors: [{ loc: ["query", "from"], msg: "'from' must not be after 'to'", type: "value_error" }],
        }),
      );
    renderWithQuery(<AuditPage />);
    await screen.findByRole("table", { name: "Audit trail" });

    setDate("From", "2026-11-18");
    setDate("To", "2026-11-01");
    await user.click(screen.getByRole("button", { name: "Apply filters" }));
    expect((await screen.findByRole("alert")).textContent).toBe("'from' must not be after 'to'");
    expect(screen.queryByRole("table")).toBeNull();
    expect(spy).toHaveBeenLastCalledWith("/admin/audit?from=2026-11-18&to=2026-11-01&page=1");
    // the filters stay as typed, so the range can be corrected
    expect((screen.getByLabelText("From") as HTMLInputElement).value).toBe("2026-11-18");
  });

  it("shows an error with a Retry that fetches the trail again", async () => {
    const user = userEvent.setup();
    const spy = spyRequest()
      .mockRejectedValueOnce(new ApiError(500, "INTERNAL_ERROR", "the server failed", null))
      .mockResolvedValue(PAGE);
    renderWithQuery(<AuditPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "Audit trail" })).toBeTruthy();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
