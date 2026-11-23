import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { PlanAdminOut } from "./adminApi";
import { PlansPage } from "./PlansPage";

const FREE: PlanAdminOut = {
  name: "free",
  limits: { max_datasets: 25, max_dataset_bytes: 20_000_000, max_fits: 1000, max_autofits_per_day: 50 },
};
const PRO: PlanAdminOut = {
  name: "pro",
  limits: { max_datasets: 200, max_dataset_bytes: 500_000_000, max_fits: 2000, max_autofits_per_day: 500 },
};
const PLANS = [FREE, PRO];

/** The one refusal the page must make itself. */
const TOO_LOW = "Each limit must be a whole number of at least 1.";

/**
 * Answers like the API: GET /admin/plans lists what is stored, PATCH /admin/plans/{name}
 * stores the limits it is given (or refuses with `refusal`); any other call fails the test.
 */
/** A spy on the client's one request method, with its call record typed. */
function spyRequest() {
  return vi.spyOn(api, "request");
}
type RequestSpy = ReturnType<typeof spyRequest>;

function serve(refusal?: ApiError): RequestSpy {
  let stored: PlanAdminOut[] = PLANS;
  return spyRequest().mockImplementation(async (path, init) => {
    if (path === "/admin/plans" && init === undefined) return stored;
    if (init?.method === "PATCH") {
      if (refusal !== undefined) throw refusal;
      const limits = (JSON.parse(String(init.body)) as { limits: PlanAdminOut["limits"] }).limits;
      const name = path.slice("/admin/plans/".length);
      stored = stored.map((plan) => (plan.name === name ? { name, limits } : plan));
      return { name, limits };
    }
    throw new Error(`unexpected request ${path}`);
  });
}

function card(name: string): HTMLElement {
  return screen.getByRole("article", { name });
}

function limitTexts(name: string): string[] {
  return within(card(name))
    .getAllByRole("definition")
    .map((dd) => dd.textContent);
}

/** Every call made, its path and its init, as plain values the tests can read. */
function calls(spy: RequestSpy): Array<[string, RequestInit | undefined]> {
  return spy.mock.calls.map(([path, init]) => [path, init]);
}

function patchCalls(spy: RequestSpy): Array<[string, RequestInit | undefined]> {
  return calls(spy).filter(([, init]) => init?.method === "PATCH");
}

async function openEditor(user: ReturnType<typeof userEvent.setup>, plan: string): Promise<void> {
  await user.click(within(card(plan)).getByRole("button", { name: "Edit limits" }));
  await screen.findByRole("form", { name: `Edit limits for ${plan}` });
}

describe("PlansPage", () => {
  it("shows loading, then each plan's four limits in words", async () => {
    serve();
    renderWithQuery(<PlansPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the plans");
    expect(screen.queryByRole("article")).toBeNull();
    await screen.findByRole("article", { name: "free" });

    expect(screen.getAllByRole("article")).toHaveLength(2);
    expect(within(card("free")).getByRole("heading", { name: "free" })).toBeTruthy();
    expect(limitTexts("free")).toEqual(["25", "20 MB", "1,000", "50"]);
    expect(limitTexts("pro")).toEqual(["200", "500 MB", "2,000", "500"]);
    expect(
      within(card("free"))
        .getAllByRole("term")
        .map((dt) => dt.textContent),
    ).toEqual(["Datasets", "Storage", "Stored fits", "Auto-fit runs a day"]);
  });

  it("saves the typed limits, confirms it, and shows the plan as the server now holds it", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "free" });

    await openEditor(user, "free");
    const datasets = within(screen.getByRole("form", { name: "Edit limits for free" })).getByRole(
      "spinbutton",
      { name: "Datasets" },
    );
    await user.clear(datasets);
    await user.type(datasets, "30");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Limits saved. The change is in the audit trail.")).toBeTruthy();
    expect(patchCalls(spy)).toHaveLength(1);
    expect(patchCalls(spy)[0]?.[0]).toBe("/admin/plans/free");
    expect(JSON.parse(String((patchCalls(spy)[0]?.[1] as RequestInit).body))).toEqual({
      limits: { max_datasets: 30, max_dataset_bytes: 20_000_000, max_fits: 1000, max_autofits_per_day: 50 },
    });
    // the card is refreshed from the server after the change
    await waitFor(() =>
      expect(calls(spy).filter(([path]) => path === "/admin/plans")).toHaveLength(2),
    );
    expect(limitTexts("free")).toEqual(["30", "20 MB", "1,000", "50"]);
    expect(screen.queryByRole("form")).toBeNull();
    expect(screen.queryByText("Limits saved. The change is in the audit trail.")).toBeTruthy();
  });

  it("refuses a limit below 1 in the page and never sends it", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "free" });

    await openEditor(user, "free");
    const form = screen.getByRole("form", { name: "Edit limits for free" });
    const fits = within(form).getByRole("spinbutton", { name: "Stored fits" });
    await user.clear(fits);
    await user.type(fits, "0");
    await user.click(within(form).getByRole("button", { name: "Save" }));

    expect(within(form).getByRole("alert").textContent).toBe(TOO_LOW);
    expect(patchCalls(spy)).toHaveLength(0);
    expect(screen.queryByText("Limits saved. The change is in the audit trail.")).toBeNull();
  });

  it("refuses an empty and a fractional limit the same way", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "pro" });
    await openEditor(user, "pro");
    const form = screen.getByRole("form", { name: "Edit limits for pro" });

    const perDay = within(form).getByRole("spinbutton", { name: "Auto-fit runs a day" });
    await user.clear(perDay);
    await user.click(within(form).getByRole("button", { name: "Save" }));
    expect(within(form).getByRole("alert").textContent).toBe(TOO_LOW);

    await user.type(perDay, "2.5");
    await user.click(within(form).getByRole("button", { name: "Save" }));
    expect(within(form).getByRole("alert").textContent).toBe(TOO_LOW);
    expect(patchCalls(spy)).toHaveLength(0);
  });

  it("shows the server's reason for a refusal and keeps the form open with what was typed", async () => {
    const user = userEvent.setup();
    const spy = serve(
      new ApiError(422, "VALIDATION_ERROR", "the request is not valid", {
        errors: [{ loc: ["body", "limits", "max_datasets"], msg: "max_datasets must be at least 1", type: "value_error" }],
      }),
    );
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "free" });
    await openEditor(user, "free");
    const form = screen.getByRole("form", { name: "Edit limits for free" });
    const datasets = within(form).getByRole("spinbutton", { name: "Datasets" });
    await user.clear(datasets);
    await user.type(datasets, "3000");
    await user.click(within(form).getByRole("button", { name: "Save" }));

    expect((await within(form).findByRole("alert")).textContent).toBe("max_datasets must be at least 1");
    expect(patchCalls(spy)).toHaveLength(1);
    expect((datasets as HTMLInputElement).value).toBe("3000");
    expect(limitTexts("free")).toEqual(["25", "20 MB", "1,000", "50"]);
    expect(screen.queryByText("Limits saved. The change is in the audit trail.")).toBeNull();
  });

  it("keeps the form open, with its inputs and button disabled, while the save is in flight", async () => {
    const user = userEvent.setup();
    spyRequest().mockImplementation(async (path, init) => {
      if (path === "/admin/plans" && init === undefined) return PLANS;
      // the save never finishes: the form must hold still while it is in flight
      if (init?.method === "PATCH") return new Promise(() => {});
      throw new Error(`unexpected request ${path}`);
    });
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "free" });
    await openEditor(user, "free");
    const form = screen.getByRole("form", { name: "Edit limits for free" });

    await user.click(within(form).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(within(form).getByRole("button", { name: "Save" })).toHaveProperty("disabled", true));
    for (const input of within(form).getAllByRole("spinbutton")) {
      expect((input as HTMLInputElement).disabled).toBe(true);
    }
    expect(screen.queryByRole("form")).toBeTruthy();
  });

  it("closes the form without a change on Cancel", async () => {
    const user = userEvent.setup();
    const spy = serve();
    renderWithQuery(<PlansPage />);
    await screen.findByRole("article", { name: "free" });
    await openEditor(user, "free");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("form")).toBeNull();
    expect(patchCalls(spy)).toHaveLength(0);
    expect(limitTexts("free")).toEqual(["25", "20 MB", "1,000", "50"]);
  });

  it("shows an empty state when there is no plan", async () => {
    spyRequest().mockResolvedValue([]);
    renderWithQuery(<PlansPage />);
    expect(await screen.findByText("There are no plans.")).toBeTruthy();
    expect(screen.queryByRole("article")).toBeNull();
  });

  it("shows an error with a Retry that fetches the plans again", async () => {
    const user = userEvent.setup();
    const spy = spyRequest()
      .mockRejectedValueOnce(new ApiError(500, "INTERNAL_ERROR", "the server failed", null))
      .mockResolvedValue(PLANS);
    renderWithQuery(<PlansPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("article", { name: "pro" })).toBeTruthy();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
