import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { HomePage } from "./HomePage";

const { loginMock } = vi.hoisted(() => ({ loginMock: vi.fn() }));

vi.mock("../api/auth", () => ({
  useAuth: () => ({
    status: "anon",
    user: null,
    login: loginMock,
    register: vi.fn(),
    logout: vi.fn(),
    adoptSession: vi.fn(),
  }),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, oauthProviders: vi.fn(() => Promise.resolve([])) },
  };
});

function renderHome() {
  return render(
    <MemoryRouter initialEntries={["/start"]}>
      <Routes>
        <Route path="/start" element={<HomePage />} />
        <Route path="/" element={<main>workspace</main>} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  loginMock.mockReset();
  loginMock.mockResolvedValue(undefined);
  vi.mocked(api.oauthProviders).mockResolvedValue([]);
});

describe("HomePage", () => {
  it("renders the headline, the sign-in card and the section headings", async () => {
    renderHome();

    expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(
      /defensible model/i,
    );
    expect(screen.getByLabelText("Email")).toBeTruthy();
    expect(screen.getByLabelText("Password")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeTruthy();

    expect(screen.getByRole("heading", { name: /fit the data/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /four steps/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /library that covers/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /explain to a reviewer/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /ranking is the result/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /curves engineers/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /one engine/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /questions, answered/i })).toBeTruthy();

    // Settle the provider discovery so its state update happens inside act().
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
  });

  it("submits the login form and navigates to the workspace", async () => {
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(loginMock).toHaveBeenCalledWith("a@b.co", "longenoughpass");
    await waitFor(() => expect(screen.getByText("workspace")).not.toBeNull());
  });

  it("shows the server error inline when sign-in fails", async () => {
    loginMock.mockRejectedValue(
      new ApiError(401, "UNAUTHORIZED", "invalid email or password", {}),
    );
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid email or password/i);
  });

  it("shows the verify-your-email note on 403 EMAIL_NOT_VERIFIED", async () => {
    loginMock.mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "verify your email before signing in", {
        code: "EMAIL_NOT_VERIFIED",
      }),
    );
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/verify your email/i);
    expect(screen.getByRole("link", { name: "sign in" })).toBeTruthy();
  });

  it("hides the social buttons when no providers are configured", async () => {
    renderHome();
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /continue with google/i })).toBeNull();
  });

  it("shows the social button when a provider is configured", async () => {
    vi.mocked(api.oauthProviders).mockResolvedValue(["google"]);
    renderHome();
    expect(await screen.findByRole("button", { name: /continue with google/i })).toBeTruthy();
  });

  it("shows the ConcreteInfo product and open-source project links in the footer", () => {
    renderHome();
    const footer = screen.getByRole("contentinfo");

    const logo = within(footer).getByAltText("ConcreteInfo");
    expect(logo.getAttribute("src")).toBe("/concreteinfo-logo.png");
    expect(logo.getAttribute("width")).toBe("100");
    expect(logo.getAttribute("height")).toBe("28");
    const logoLink = logo.closest("a");
    expect(logoLink?.getAttribute("href")).toBe("https://www.concreteinfo.in");
    expect(logoLink?.getAttribute("target")).toBe("_blank");
    expect(logoLink?.getAttribute("rel")).toBe("noopener noreferrer");
    expect(within(footer).getByText("A ConcreteInfo product.")).toBeTruthy();

    expect(within(footer).getByRole("heading", { name: "Open source" })).toBeTruthy();
    const projects: Array<[RegExp, string]> = [
      [/CurveStudio source/i, "https://github.com/amitwh/ConcreteInfo_CurveStudio"],
      [/PC Power Calculator/i, "https://pcpower.concreteinfo.net/"],
      [/Markdown Converter/i, "https://github.com/amitwh/markdown-converter"],
      [/GitHub Desktop for Linux/i, "https://github.com/amitwh/GitHubDesktopLinux"],
    ];
    for (const [name, href] of projects) {
      const link = within(footer).getByRole("link", { name });
      expect(link.getAttribute("href")).toBe(href);
      expect(link.getAttribute("target")).toBe("_blank");
      expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    }
  });

  it("renders the six reproducible-fit chips and switches the displayed equation", async () => {
    const user = userEvent.setup();
    renderHome();
    const section = document.getElementById("fits");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(scoped.getAllByRole("button")).toHaveLength(6);
    const decay = scoped.getByRole("button", { name: "Exponential decay" });
    expect(decay.getAttribute("aria-pressed")).toBe("true");
    expect(scoped.getByText("y = a·exp(−b·x)+c", { selector: "code" })).toBeTruthy();
    expect(scoped.getByText("0.9964")).toBeTruthy();

    const firstCurve = section?.querySelector(".hp-fits__curve")?.getAttribute("d") ?? "";
    expect(firstCurve).toMatch(/^M[\d.]+ [\d.]+ L/);
    expect(firstCurve).not.toMatch(/NaN|undefined/);
    expect(section?.querySelector(".hp-fits__band")?.getAttribute("d") ?? "").not.toMatch(
      /NaN|undefined/,
    );
    expect(section?.querySelectorAll(".hp-fits__points circle")).toHaveLength(12);

    const logistic = scoped.getByRole("button", { name: "Logistic growth" });
    await user.click(logistic);

    expect(logistic.getAttribute("aria-pressed")).toBe("true");
    expect(decay.getAttribute("aria-pressed")).toBe("false");
    expect(scoped.getByText("y = L/(1+exp(−k(x−x₀)))", { selector: "code" })).toBeTruthy();
    expect(scoped.queryByText("y = a·exp(−b·x)+c", { selector: "code" })).toBeNull();
    expect(section?.querySelector(".hp-fits__curve")?.getAttribute("d")).not.toBe(firstCurve);
  });

  it("renders the five-model AICc comparison in the statistics section", () => {
    renderHome();
    expect(document.querySelectorAll(".hp-aicc__bar")).toHaveLength(5);
    expect(screen.getByText(/ΔAICc against the best model/i)).toBeTruthy();
  });
});
