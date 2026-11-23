import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, type MeOut } from "../api/client";
import { Modal } from "../components/Modal";
import { renderWithQuery } from "../test/render";
import { ProfilePage } from "./ProfilePage";

const { auth } = vi.hoisted(() => ({ auth: { updateUser: vi.fn(), logout: vi.fn() } }));

vi.mock("../api/auth", () => ({ useAuth: () => auth }));

const PLAN = {
  name: "Free",
  max_datasets: 10,
  max_dataset_bytes: 100_000_000,
  max_fits: 1_00_000,
  max_autofits_per_day: 20,
};

const ME: MeOut = {
  id: "u1",
  email: "amit.h@example.test",
  name: "Amit Haridas",
  created_at: "2026-11-01T10:00:00+05:30",
  email_verified: true,
  role: "member",
  preferences: { error_reports: true, report_byline: false },
  last_login_at: "2026-11-20T09:30:00+05:30",
  plan: PLAN,
  usage: { datasets: 3, dataset_bytes: 25_000_000, fits: 80_000, autofits_today: 0 },
};

/** The four meters, as "used of limit" and the percentage each one shows. */
const METER_TEXT = ["3 of 10", "25 MB of 100 MB", "80,000 of 1,00,000", "0 of 20"];
const METER_PERCENT = ["30%", "25%", "80%", "0%"];

const SENT = { message: "sent" };

/** A 422 as the API sends it: a generic message, the reason in detail.errors[0].msg. */
function rejected(msg: string): ApiError {
  return new ApiError(422, "VALIDATION_ERROR", "the request is not valid", {
    errors: [{ loc: ["body", "field"], msg, type: "value_error" }],
  });
}

/** The <section> whose <h3> is `name`. */
function section(name: string): HTMLElement {
  return screen.getByRole("heading", { level: 3, name }).closest("section") as HTMLElement;
}

function renderPage(): void {
  renderWithQuery(<ProfilePage />);
}

/**
 * Mocks every endpoint the page calls, the way the server answers them: each change
 * returns the whole profile with that change applied. A test overrides the one it is about.
 */
function mockEndpoints(me: MeOut = ME): void {
  let stored = me;
  vi.spyOn(api, "me").mockImplementation(async () => stored);
  vi.spyOn(api, "updateMe").mockImplementation(async (name) => {
    stored = { ...stored, name };
    return stored;
  });
  vi.spyOn(api, "updatePreferences").mockImplementation(async (change) => {
    stored = { ...stored, preferences: { ...stored.preferences, ...change } };
    return stored;
  });
  vi.spyOn(api, "changePassword").mockImplementation(async () => stored);
  vi.spyOn(api, "revokeAllSessions").mockImplementation(async () => stored);
  vi.spyOn(api, "deleteMe").mockResolvedValue(undefined);
  vi.spyOn(api, "resendVerification").mockResolvedValue(SENT);
  vi.spyOn(api, "forgotPassword").mockResolvedValue(SENT);
}

/** Waits for the page to finish loading and returns the user-event driver. */
async function opened(me: MeOut = ME): Promise<ReturnType<typeof userEvent.setup>> {
  mockEndpoints(me);
  renderPage();
  await screen.findByRole("heading", { name: "Account" });
  return userEvent.setup();
}

/** The confirmation the page shows once a change has gone through. */
async function statusText(): Promise<string> {
  return (await screen.findByRole("status")).textContent ?? "";
}

beforeEach(() => {
  auth.updateUser.mockReset();
  auth.logout.mockReset();
});

afterEach(() => {
  api.resetForTests();
});

describe("ProfilePage", () => {
  it("renders the six sections from MeOut", async () => {
    await opened();
    const headings = screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent);
    expect(headings).toEqual([
      "Account",
      "Plan and usage",
      "Reports",
      "Privacy",
      "Sessions",
      "Danger zone",
    ]);
    expect(within(section("Account")).getByText("amit.h@example.test")).toBeTruthy();
    expect(within(section("Account")).getByDisplayValue("Amit Haridas")).toBeTruthy();
    expect(within(section("Sessions")).getByText("You are signed in on this device.")).toBeTruthy();
  });

  it("shows the loading state until the profile arrives", async () => {
    let release!: (me: MeOut) => void;
    vi.spyOn(api, "me").mockReturnValue(
      new Promise<MeOut>((resolve) => {
        release = resolve;
      }),
    );
    renderPage();
    expect(screen.getByRole("status").textContent).toMatch(/loading/i);
    expect(screen.queryByRole("heading", { name: "Account" })).toBeNull();

    release(ME);
    expect(await screen.findByRole("heading", { name: "Account" })).toBeTruthy();
  });

  it("shows the error state, and Retry fetches the profile again", async () => {
    mockEndpoints();
    vi.mocked(api.me).mockRejectedValueOnce(
      new ApiError(0, "NETWORK", "could not reach the server", null),
    );
    renderPage();

    expect((await screen.findByRole("alert")).textContent).toBe("could not reach the server");
    expect(screen.queryByRole("heading", { name: "Account" })).toBeNull();

    await userEvent.setup().click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("heading", { name: "Account" })).toBeTruthy();
    expect(api.me).toHaveBeenCalledTimes(2);
  });
});

describe("ProfilePage — the session hears about every change", () => {
  it("updateUser replaces the profile the session holds", async () => {
    // The real provider, not the mock: the header's name comes from this value.
    const actual = await vi.importActual<typeof import("../api/auth")>("../api/auth");
    vi.spyOn(api, "refreshNow").mockResolvedValue(true);
    vi.spyOn(api, "me").mockResolvedValue(ME);
    api.setTokens("a1", "r1"); // a stored session for the provider to restore

    function Held(): ReactElement {
      const { status, user, updateUser } = actual.useAuth();
      return (
        <>
          <output data-testid="held">{`${status}:${user?.name ?? "-"}`}</output>
          <button type="button" onClick={() => updateUser({ ...ME, name: "Renamed" })}>
            rename
          </button>
        </>
      );
    }
    const user = userEvent.setup();
    renderWithQuery(<actual.AuthProvider><Held /></actual.AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("held").textContent).toBe("authed:Amit Haridas"));

    await user.click(screen.getByRole("button", { name: "rename" }));
    expect(screen.getByTestId("held").textContent).toBe("authed:Renamed");
  });
});

describe("ProfilePage — Account", () => {
  it("saves the name through PATCH /me and hands the returned profile to the session", async () => {
    const user = await opened();

    await user.clear(screen.getByLabelText("Name"));
    await user.type(screen.getByLabelText("Name"), "Amit H");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(api.updateMe).toHaveBeenCalledWith("Amit H");
    expect(auth.updateUser).toHaveBeenCalledWith({ ...ME, name: "Amit H" });
    expect(await statusText()).toBe("Name saved.");
    expect(screen.getByDisplayValue("Amit H")).toBeTruthy();
  });

  it("refuses a blank name in the page, without a request", async () => {
    const user = await opened();
    const input = screen.getByLabelText("Name");
    expect(input.getAttribute("maxlength")).toBe("100");
    // styles.css styles inputs by their type attribute: without it the field is unstyled
    expect(input.getAttribute("type")).toBe("text");

    await user.clear(input);
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/enter a name/i);

    await user.type(input, "   ");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(screen.getByRole("alert").textContent).toMatch(/enter a name/i);
    expect(api.updateMe).not.toHaveBeenCalled();
    expect(auth.updateUser).not.toHaveBeenCalled();
  });

  it("shows the server's reason when it refuses a name", async () => {
    const user = await opened();
    vi.mocked(api.updateMe).mockRejectedValue(rejected("name must be at most 100 characters"));

    await user.type(screen.getByLabelText("Name"), "!");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "name must be at most 100 characters",
    );
    expect(auth.updateUser).not.toHaveBeenCalled();
  });

  it("offers to resend the verification email while the address is unverified", async () => {
    const user = await opened({ ...ME, email_verified: false });
    await user.click(screen.getByRole("button", { name: "Resend verification email" }));

    expect(api.resendVerification).toHaveBeenCalledWith("amit.h@example.test");
    expect(await statusText()).toBe("If the address is unverified, a new link is on its way.");
  });

  it("offers no resend once the address is verified", async () => {
    await opened();
    expect(screen.queryByRole("button", { name: /resend verification/i })).toBeNull();
  });

  it("changes the password and confirms that other devices were signed out", async () => {
    const user = await opened();
    await user.type(screen.getByLabelText("Current password"), "old-pass-123");
    await user.type(screen.getByLabelText("New password"), "new-pass-456");
    await user.click(screen.getByRole("button", { name: "Change password" }));

    expect(api.changePassword).toHaveBeenCalledWith("old-pass-123", "new-pass-456");
    expect(api.changePassword).toHaveBeenCalledTimes(1);
    expect(auth.updateUser).toHaveBeenCalledWith(ME);
    expect(await statusText()).toBe("Password changed. Other devices were signed out.");
    expect(screen.getByLabelText("New password")).toHaveProperty("value", "");
  });

  it("shows the server's message for a wrong current password", async () => {
    const user = await opened();
    vi.mocked(api.changePassword).mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "the current password is not correct", {}),
    );

    await user.type(screen.getByLabelText("Current password"), "nope-pass-123");
    await user.type(screen.getByLabelText("New password"), "new-pass-456");
    await user.click(screen.getByRole("button", { name: "Change password" }));

    expect((await screen.findByRole("alert")).textContent).toBe(
      "the current password is not correct",
    );
    expect(screen.queryByText(/password changed/i)).toBeNull();
    expect(auth.updateUser).not.toHaveBeenCalled();
  });

  it("shows the reason when the new password is rejected", async () => {
    const user = await opened();
    vi.mocked(api.changePassword).mockRejectedValue(
      rejected("the new password must be different from the current one"),
    );

    await user.type(screen.getByLabelText("Current password"), "same-pass-123");
    await user.type(screen.getByLabelText("New password"), "same-pass-123");
    await user.click(screen.getByRole("button", { name: "Change password" }));

    expect((await screen.findByRole("alert")).textContent).toBe(
      "the new password must be different from the current one",
    );
  });

  it("sends a set-password link to the signed-in address", async () => {
    const user = await opened();
    expect(screen.getByText(/created with Google or Microsoft/i)).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Set a password by e-mail" }));
    expect(api.forgotPassword).toHaveBeenCalledWith("amit.h@example.test");
    expect(await statusText()).toBe(
      "If your address is registered, a link to set a password is on its way.",
    );
  });

  it("never sends a second request for a section while one is in flight", async () => {
    const user = await opened();
    let release!: (me: MeOut) => void;
    vi.mocked(api.updateMe).mockReturnValue(
      new Promise<MeOut>((resolve) => {
        release = resolve;
      }),
    );

    await user.type(screen.getByLabelText("Name"), " II");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(api.updateMe).toHaveBeenCalledTimes(1);
    // the whole Account section waits: its other actions are disabled until the save settles
    const setPassword = screen.getByRole("button", {
      name: "Set a password by e-mail",
    }) as HTMLButtonElement;
    expect(setPassword.disabled).toBe(true);
    await user.click(setPassword);
    expect(api.forgotPassword).not.toHaveBeenCalled();

    release({ ...ME, name: "Amit Haridas II" });
    await statusText();
    expect(setPassword.disabled).toBe(false);
  });
});

describe("ProfilePage — Plan and usage", () => {
  it("shows the plan and four meters with used and limit", async () => {
    await opened();
    expect(within(section("Plan and usage")).getByText("Free")).toBeTruthy();
    expect(screen.getAllByRole("progressbar")).toHaveLength(4);

    METER_TEXT.forEach((text) => expect(screen.getByText(text)).toBeTruthy());
    METER_PERCENT.forEach((text) => expect(screen.getByText(text)).toBeTruthy());

    const storage = screen.getByRole("progressbar", { name: "Storage" });
    expect(storage.getAttribute("aria-valuemin")).toBe("0");
    expect(storage.getAttribute("aria-valuenow")).toBe("25000000");
    expect(storage.getAttribute("aria-valuemax")).toBe("100000000");
    for (const name of ["Datasets", "Stored fits", "Auto-fit runs today"]) {
      expect(screen.getByRole("progressbar", { name })).toBeTruthy();
    }
    expect(screen.queryByText(/over 90%/i)).toBeNull();
  });

  it("marks a meter that is over 90% in words", async () => {
    await opened({ ...ME, usage: { ...ME.usage, fits: 95_000 } });
    const fits = section("Plan and usage");
    expect(within(fits).getByText(/over 90%/i)).toBeTruthy();
    expect(screen.getByText("95%")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "Stored fits" }).getAttribute("aria-valuenow")).toBe(
      "95000",
    );
  });

  it("keeps the bar honest when usage has passed its limit", async () => {
    await opened({ ...ME, usage: { ...ME.usage, datasets: 12 } });
    const datasets = screen.getByRole("progressbar", { name: "Datasets" });
    expect(screen.getByText("12 of 10")).toBeTruthy();
    expect(screen.getByText("120%")).toBeTruthy();
    expect(datasets.getAttribute("aria-valuenow")).toBe("12");
    expect(datasets.getAttribute("aria-valuemax")).toBe("12");
  });
});

describe("ProfilePage — switches", () => {
  it("each switch sends only its own key, and the session gets the result", async () => {
    const user = await opened();
    const byline = screen.getByRole("switch", { name: "Show my name on PDF reports" });
    const errorReports = screen.getByRole("switch", { name: "Send error reports" });
    expect(byline.getAttribute("aria-checked")).toBe("false");
    expect(errorReports.getAttribute("aria-checked")).toBe("true");

    await user.click(byline);
    expect(api.updatePreferences).toHaveBeenLastCalledWith({ report_byline: true });
    await user.click(errorReports);
    expect(api.updatePreferences).toHaveBeenLastCalledWith({ error_reports: false });
    expect(api.updatePreferences).toHaveBeenCalledTimes(2);

    expect(auth.updateUser).toHaveBeenLastCalledWith({
      ...ME,
      preferences: { error_reports: false, report_byline: true },
    });
    await waitFor(() => expect(byline.getAttribute("aria-checked")).toBe("true"));
    expect(errorReports.getAttribute("aria-checked")).toBe("false");
  });

  it("describes the byline with the user's name", async () => {
    await opened();
    expect(
      screen.getByText("Your name appears as “Prepared by Amit Haridas” on the PDF reports you export."),
    ).toBeTruthy();
  });

  it("asks for a name before the byline can be switched on", async () => {
    await opened({ ...ME, name: "" });
    const byline = screen.getByRole("switch", {
      name: "Show my name on PDF reports",
    }) as HTMLButtonElement;
    expect(byline.disabled).toBe(true);
    expect(screen.getByText(/add your name/i)).toBeTruthy();
  });

  it("reverts a switch whose change failed, and shows the error", async () => {
    const user = await opened();
    let fail!: (error: unknown) => void;
    vi.mocked(api.updatePreferences).mockReturnValue(
      new Promise<MeOut>((_resolve, reject) => {
        fail = reject;
      }),
    );

    const errorReports = screen.getByRole("switch", { name: "Send error reports" });
    await user.click(errorReports);
    // optimistic while the request is in flight
    expect(errorReports.getAttribute("aria-checked")).toBe("false");

    fail(new ApiError(0, "NETWORK", "could not reach the server — check your connection", null));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "could not reach the server — check your connection",
    );
    expect(errorReports.getAttribute("aria-checked")).toBe("true");
    expect(auth.updateUser).not.toHaveBeenCalled();
  });

  it("ignores a second click on a switch while its change is in flight", async () => {
    await opened();
    let release!: (me: MeOut) => void;
    vi.mocked(api.updatePreferences).mockReturnValue(
      new Promise<MeOut>((resolve) => {
        release = resolve;
      }),
    );
    const byline = screen.getByRole("switch", { name: "Show my name on PDF reports" });

    // Two clicks in the same tick, before React has disabled the control
    fireEvent.click(byline);
    fireEvent.click(byline);
    expect(api.updatePreferences).toHaveBeenCalledTimes(1);
    expect((byline as HTMLButtonElement).disabled).toBe(true);

    release({ ...ME, preferences: { ...ME.preferences, report_byline: true } });
    await statusText();
    expect(byline.getAttribute("aria-checked")).toBe("true");
    expect((byline as HTMLButtonElement).disabled).toBe(false);
  });
});

describe("ProfilePage — Sessions", () => {
  it("asks before signing out everywhere, then does it", async () => {
    const user = await opened();
    await user.click(screen.getByRole("button", { name: "Sign out everywhere" }));
    const dialog = await screen.findByRole("dialog", { name: "Sign out everywhere?" });
    expect(api.revokeAllSessions).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.revokeAllSessions).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Sign out everywhere" }));
    const again = await screen.findByRole("dialog", { name: "Sign out everywhere?" });
    await user.click(within(again).getByRole("button", { name: "Sign out everywhere" }));

    expect(api.revokeAllSessions).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(auth.updateUser).toHaveBeenCalledWith(ME);
    expect(await statusText()).toBe("Other devices were signed out.");
  });

  it("keeps the confirmation open with the error when it fails", async () => {
    const user = await opened();
    vi.mocked(api.revokeAllSessions).mockRejectedValue(
      new ApiError(0, "NETWORK", "could not reach the server", null),
    );
    await user.click(screen.getByRole("button", { name: "Sign out everywhere" }));
    await user.click(
      within(await screen.findByRole("dialog", { name: "Sign out everywhere?" })).getByRole(
        "button",
        { name: "Sign out everywhere" },
      ),
    );

    const dialog = screen.getByRole("dialog", { name: "Sign out everywhere?" });
    expect((await within(dialog).findByRole("alert")).textContent).toBe("could not reach the server");
    expect(auth.updateUser).not.toHaveBeenCalled();
  });

  it("Escape inside a confirmation closes only the confirmation, not the profile sheet", async () => {
    const closeSheet = vi.fn();
    mockEndpoints();
    const user = userEvent.setup();
    renderWithQuery(
      <Modal title="Your profile" variant="sheet" onClose={closeSheet}>
        <ProfilePage />
      </Modal>,
    );
    await screen.findByRole("button", { name: "Sign out everywhere" });

    await user.click(screen.getByRole("button", { name: "Sign out everywhere" }));
    expect(screen.getByRole("dialog", { name: "Sign out everywhere?" })).toBeTruthy();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Sign out everywhere?" })).toBeNull();
    expect(screen.getByRole("dialog", { name: "Your profile" })).toBeTruthy();
    expect(closeSheet).not.toHaveBeenCalled();
  });
});

describe("ProfilePage — Danger zone", () => {
  const PHRASE = /type amit\.h to confirm/i;

  async function openDelete(user: ReturnType<typeof userEvent.setup>): Promise<HTMLElement> {
    await user.click(screen.getByRole("button", { name: "Delete account" }));
    return screen.findByRole("dialog", { name: "Delete account?" });
  }

  it("deletes the account once the local part of the e-mail is typed, then ends the session", async () => {
    const user = await opened();
    expect(screen.getByText(/audit trail is kept/i)).toBeTruthy();

    const dialog = await openDelete(user);
    const confirm = within(dialog).getByRole("button", {
      name: "Delete account",
    }) as HTMLButtonElement;
    expect(within(dialog).getByLabelText(PHRASE)).toBeTruthy();
    expect(confirm.disabled).toBe(true);

    await user.type(within(dialog).getByLabelText(PHRASE), "amit");
    expect(confirm.disabled).toBe(true);
    await user.type(within(dialog).getByLabelText(PHRASE), ".h");
    expect(confirm.disabled).toBe(false);

    await user.click(confirm);
    expect(api.deleteMe).toHaveBeenCalledWith("amit.h");
    await waitFor(() => expect(auth.logout).toHaveBeenCalledTimes(1));
  });

  it("Cancel in the confirmation deletes nothing", async () => {
    const user = await opened();
    const dialog = await openDelete(user);
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.deleteMe).not.toHaveBeenCalled();
    expect(auth.logout).not.toHaveBeenCalled();
  });

  it("keeps the page and the confirmation open when the server refuses (422)", async () => {
    const user = await opened();
    // The gate in the page matches the server's rule, but the server has the last word.
    vi.mocked(api.deleteMe).mockRejectedValue(
      rejected("this does not match the part of your e-mail address before the @"),
    );
    const dialog = await openDelete(user);
    await user.type(within(dialog).getByLabelText(PHRASE), "amit.h");
    await user.click(within(dialog).getByRole("button", { name: "Delete account" }));

    expect(api.deleteMe).toHaveBeenCalledWith("amit.h");
    const alert = await within(screen.getByRole("dialog", { name: "Delete account?" })).findByRole(
      "alert",
    );
    expect(alert.textContent).toBe(
      "this does not match the part of your e-mail address before the @",
    );
    expect(screen.getByRole("heading", { name: "Danger zone" })).toBeTruthy();
    expect(auth.logout).not.toHaveBeenCalled();
  });

  it("keeps the session when the delete itself fails", async () => {
    const user = await opened();
    vi.mocked(api.deleteMe).mockRejectedValue(
      new ApiError(0, "NETWORK", "could not reach the server", null),
    );
    const dialog = await openDelete(user);
    await user.type(within(dialog).getByLabelText(PHRASE), "amit.h");
    await user.click(within(dialog).getByRole("button", { name: "Delete account" }));

    const alert = await within(screen.getByRole("dialog", { name: "Delete account?" })).findByRole(
      "alert",
    );
    expect(alert.textContent).toBe("could not reach the server");
    expect(auth.logout).not.toHaveBeenCalled();
  });
});
