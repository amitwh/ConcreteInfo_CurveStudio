import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import otherProjects from "../content/other-projects.json";
import { HomePage } from "./HomePage";

const { loginMock } = vi.hoisted(() => ({ loginMock: vi.fn() }));

/** Read once so source-level assertions (retired copy really gone) run against the file. */
const HOME_PAGE_SOURCE = readFileSync(
  resolve(dirname(fileURLToPath(import.meta.url)), "./HomePage.tsx"),
  "utf-8",
);

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
    api: {
      ...actual.api,
      oauthProviders: vi.fn(() => Promise.resolve([])),
      resendVerification: vi.fn(),
    },
  };
});

function renderHome(path = "/start") {
  return render(
    <MemoryRouter initialEntries={[path]}>
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
    expect(screen.getByRole("heading", { name: /measured data, in every discipline/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /one engine/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /other tools we build and maintain/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /questions we get asked/i })).toBeTruthy();

    // Settle the provider discovery so its state update happens inside act().
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
  });

  it("opens with the research hero copy", () => {
    renderHome();
    const hero = document.querySelector(".hp-hero");
    expect(hero).not.toBeNull();
    expect(hero?.querySelector(".hp-eyebrow")?.textContent).toBe(
      "CURVE FITTING AND STATISTICS · VALIDATED ON NIST STRD DATASETS",
    );
    expect(hero?.querySelector(".hp-lede")?.textContent?.startsWith(
      "CurveStudio fits every applicable model to your data, ranks them on the evidence, and g",
    )).toBe(true);
  });

  it("keeps the trust strip pills and stats", () => {
    renderHome();
    const strip = document.querySelector(".hp-trust");
    expect(strip).not.toBeNull();
    const scoped = within(strip as HTMLElement);
    for (const pill of [
      "NIST StRD reference sets",
      "~200 models",
      "AICc / BIC ranking",
      "Confidence & prediction bands",
      "AGPL-3.0",
      "Offline desktop app",
    ]) {
      expect(scoped.getByText(pill)).toBeTruthy();
    }
    expect(scoped.getByText("174")).toBeTruthy();
    expect(scoped.getByText("~200")).toBeTruthy();
    expect(scoped.getByText("export formats")).toBeTruthy();
  });

  it("renders the who-it-is-for section with exactly five persona cards", () => {
    renderHome();
    const section = document.getElementById("who");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(
      scoped.getByRole("heading", {
        level: 2,
        name: "For the analysis behind a paper, a thesis or a lab report.",
      }),
    ).toBeTruthy();
    expect(scoped.getByText("WHO IT IS FOR")).toBeTruthy();
    expect(section?.querySelectorAll("article")).toHaveLength(5);

    const personas: Array<[string, string]> = [
      [
        "Research scientists",
        "Test competing models on equal terms and report the one the data supports.",
      ],
      [
        "Laboratory analysts",
        "Calibration curves, peak fits and method comparisons, with the uncertainty attached.",
      ],
      [
        "Postgraduate students",
        "Learn what R², AICc and a confidence band actually say about your data.",
      ],
      [
        "Engineers in research and test",
        "Fit sensor, materials and reliability data on your own machine.",
      ],
      ["Lecturers", "Show a class why a high R² is not the end of the argument."],
    ];
    for (const [name, body] of personas) {
      expect(scoped.getByRole("heading", { name })).toBeTruthy();
      expect(scoped.getByText(body)).toBeTruthy();
    }
  });

  it("renders the eight research requirements as two-column rows", () => {
    renderHome();
    const section = document.getElementById("requirements");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(
      scoped.getByRole("heading", {
        level: 2,
        name: "What an analysis has to survive, and how CurveStudio answers.",
      }),
    ).toBeTruthy();
    expect(scoped.getByText("RESEARCH REQUIREMENTS")).toBeTruthy();

    for (const requirement of [
      "Choose a model on evidence",
      "Report uncertainty honestly",
      "Show that the fit hides nothing",
      "Trust the arithmetic",
      "Use your own equation",
      "Account for measurement error",
      "Reproduce it next year",
      "Publish and archive",
    ]) {
      expect(scoped.getByText(requirement)).toBeTruthy();
    }
    // Spot-check the answers beside three of the rows.
    expect(scoped.getByText(/AICc, BIC, adjusted R² and cross-validated error/)).toBeTruthy();
    expect(scoped.getByText("Weighted least squares.")).toBeTruthy();
    expect(scoped.getByText(/NIST StRD certified datasets/)).toBeTruthy();
  });

  it("renders the More from ConcreteInfo section from the shared project list", () => {
    renderHome();
    const section = document.getElementById("projects");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(
      scoped.getByRole("heading", { level: 2, name: "Other tools we build and maintain." }),
    ).toBeTruthy();
    expect(scoped.getByText("MORE FROM CONCRETEINFO")).toBeTruthy();

    // Open source first, commercial second.
    const groupNames = scoped
      .getAllByRole("heading", { level: 3 })
      .map((heading) => heading.textContent);
    expect(groupNames.indexOf("Open source")).toBeGreaterThanOrEqual(0);
    expect(groupNames.indexOf("Commercial software")).toBeGreaterThan(
      groupNames.indexOf("Open source"),
    );

    for (const entry of otherProjects.entries) {
      expect(scoped.getByText(entry.summary)).toBeTruthy();
      const link = scoped.getByRole("link", { name: entry.name });
      expect(link.getAttribute("href")).toBe(entry.url);
      expect(link.getAttribute("target")).toBe("_blank");
    }
    expect(
      scoped.getByRole("link", { name: "PC Power Calculator" }).getAttribute("href"),
    ).toBe("https://pcpower.concreteinfo.net/");
    expect(
      scoped.getByRole("link", { name: "ConcreteInfo Mix Design Platform" }).getAttribute("href"),
    ).toBe("https://cmdapp.concreteinfo.cloud");
  });

  it("replaces the concrete use cases with the seven research ones", () => {
    renderHome();
    const section = document.getElementById("use-cases");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(
      scoped.getByRole("heading", { level: 2, name: "Built for measured data, in every discipline." }),
    ).toBeTruthy();
    expect(section?.querySelectorAll("article")).toHaveLength(7);
    for (const title of [
      "Reaction kinetics",
      "Enzyme kinetics",
      "Growth curves",
      "Calibration and drift",
      "Spectra and chromatograms",
      "Dose–response",
      "Teaching-lab physics",
    ]) {
      expect(scoped.getByRole("heading", { name: title })).toBeTruthy();
    }

    // The retired concrete framing is gone from the page source entirely.
    expect(HOME_PAGE_SOURCE).not.toContain("Cube strength");
    expect(HOME_PAGE_SOURCE).not.toContain("water-binder");
    expect(HOME_PAGE_SOURCE).not.toContain("Built for the curves engineers actually measure");
  });

  it("answers how to report a fit in a paper with the real exports", () => {
    renderHome();
    const faq = document.getElementById("faq");
    expect(faq).not.toBeNull();
    const scoped = within(faq as HTMLElement);

    expect(scoped.getByText("How do I report a fit in a paper?")).toBeTruthy();
    expect(scoped.getByText(/CSV, JSON, XLSX and PDF/)).toBeTruthy();
    expect(scoped.getByText(/desktop project file records the full analysis/)).toBeTruthy();
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

  it("rejects a malformed email without calling the API", async () => {
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "not-an-email");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/email/i);
    expect(loginMock).not.toHaveBeenCalled();
  });

  it("rejects a password shorter than 10 characters", async () => {
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "short");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/10/);
    expect(loginMock).not.toHaveBeenCalled();
  });

  it("shows the verify-your-email note and a resend button on 403 EMAIL_NOT_VERIFIED", async () => {
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
    expect(screen.getByRole("button", { name: "Resend verification email" })).toBeTruthy();
  });

  it("resends the verification email to the address already typed", async () => {
    loginMock.mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "verify your email before signing in", {
        code: "EMAIL_NOT_VERIFIED",
      }),
    );
    const resend = vi.mocked(api.resendVerification).mockResolvedValue({ message: "sent" });
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await user.click(await screen.findByRole("button", { name: "Resend verification email" }));

    expect(resend).toHaveBeenCalledWith("a@b.co");
    expect(
      await screen.findByText(/if an unverified account exists, a new verification email/i),
    ).toBeTruthy();
  });

  it("shows the OAuth error when redirected back with ?oauth_error=provider", () => {
    renderHome("/start?oauth_error=provider");
    expect(screen.getByRole("alert").textContent).toMatch(/provider could not complete sign-in/i);
  });

  it("shows the session-expired message for ?oauth_error=state", () => {
    renderHome("/start?oauth_error=state");
    expect(screen.getByRole("alert").textContent).toMatch(/session expired or was opened/i);
  });

  it("falls back to a generic message for an unknown oauth_error", () => {
    renderHome("/start?oauth_error=something-new");
    expect(screen.getByRole("alert").textContent).toMatch(
      /social sign-in could not be completed/i,
    );
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

  it("links Contact and Privacy in the footer and mirrors the shared open-source list", () => {
    renderHome();
    const footer = screen.getByRole("contentinfo");

    const contact = within(footer).getByRole("link", { name: "Contact" });
    expect(contact.getAttribute("href")).toBe("mailto:amit@concreteinfo.in");
    expect(within(footer).getByRole("link", { name: "Privacy" }).getAttribute("href")).toBe(
      "/about",
    );

    // The Open source column mirrors the shared list, plus CurveStudio's own source.
    for (const entry of otherProjects.entries.filter((e) => e.kind === "open-source")) {
      expect(
        within(footer).getByRole("link", { name: entry.name }).getAttribute("href"),
      ).toBe(entry.url);
    }
    expect(
      within(footer).getByRole("link", { name: /CurveStudio source/i }).getAttribute("href"),
    ).toBe("https://github.com/amitwh/ConcreteInfo_CurveStudio");
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

  it("renders the Statistics section with its three sentences and the record of the check", () => {
    renderHome();
    const section = document.getElementById("statistics");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    expect(
      scoped.getByRole("heading", {
        level: 2,
        name: "Checked after the fit. Written up for the paper.",
      }),
    ).toBeTruthy();

    const sentences = [
      "After every fit, a fit audit checks the evidence for you: each check gets a verdict, the numbers behind it and what to do next.",
      "A methods text writes the fit up for you: the model, the estimator, the estimates with their standard errors and the citations, as plain text, Markdown or LaTeX.",
      "We looked for these two in the public documentation of CurveExpert and CurveExpert Pro, Origin, GraphPad Prism, LabPlot, SciDAVis, QtiPlot, jamovi and JASP in November 2026, and did not find either offered in its own right; those tools offer residual plots and tests, rankings, pairwise comparisons and equation export.",
    ];
    for (const sentence of sentences) {
      expect(scoped.getByText(sentence)).toBeTruthy();
    }

    // The record of that check is a file of this repository, on the address the footer's
    // source link uses.
    const record = scoped.getByRole("link", { name: "How we checked" });
    expect(record.getAttribute("href")).toBe(
      "https://github.com/amitwh/ConcreteInfo_CurveStudio/blob/main/docs/prior-art.md",
    );
    expect(record.getAttribute("target")).toBe("_blank");
    expect(record.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("lists the twenty-five statistical tools by family in the Statistics section", () => {
    renderHome();
    const section = document.getElementById("statistics");
    expect(section).not.toBeNull();
    const scoped = within(section as HTMLElement);

    // One sentence says what the tools are for and where they run.
    expect(
      scoped.getByText(
        "Twenty-five statistical tools, from summarising the columns of a dataset to planning " +
          "the sample a study needs, compare means and ranks, trace relationships, count " +
          "associations, compare two methods and check a score's discrimination, a scale's " +
          "consistency and a process's stability — in the web app, the desktop app and the " +
          "command line.",
      ),
    ).toBeTruthy();

    // Every family's title, in the catalogue's order, and every tool under its family.
    const families: Array<[string, readonly string[]]> = [
      ["Summarise", ["Descriptive statistics", "Distribution analysis"]],
      [
        "Compare means",
        [
          "Analysis of variance",
          "Two-way analysis of variance",
          "Repeated-measures analysis of variance",
          "t tests",
        ],
      ],
      [
        "Rank-based comparisons",
        [
          "Kruskal–Wallis test",
          "Friedman test",
          "Mann-Whitney U test",
          "Wilcoxon signed-rank test",
          "Sign test",
        ],
      ],
      [
        "Relationships between columns",
        [
          "Correlation",
          "Principal component analysis",
          "Factor analysis",
          "Logistic regression",
          "Linear regression",
        ],
      ],
      ["Counts", ["Contingency table", "McNemar's test"]],
      [
        "Method comparison",
        ["Bland–Altman agreement", "Deming regression", "Passing–Bablok regression"],
      ],
      [
        "Measurement and classification",
        [
          "Control chart and process capability",
          "Reliability (Cronbach's alpha)",
          "Receiver operating characteristic curve",
        ],
      ],
      ["Sample size and power", ["Sample size and power"]],
    ];

    const element = section as HTMLElement;
    const familyTitles = Array.from(element.querySelectorAll(".hp-stats-tools__family")).map(
      (family) => family.textContent,
    );
    expect(familyTitles).toEqual(families.map(([title]) => title));

    const toolTitles = Array.from(element.querySelectorAll(".hp-stats-tools__tool")).map(
      (tool) => tool.textContent,
    );
    expect(toolTitles).toEqual(families.flatMap(([, tools]) => tools));
    expect(toolTitles).toHaveLength(25);
  });

  it("claims no absence in the Statistics section: what was not found is not proof", () => {
    renderHome();
    const text = (document.getElementById("statistics")?.textContent ?? "").toLowerCase();
    for (const banned of ["nowhere else", "unique", "only tool"]) {
      expect(text).not.toContain(banned);
      // and it is not written anywhere else on the page either
      expect(HOME_PAGE_SOURCE.toLowerCase()).not.toContain(banned);
    }
  });
});
