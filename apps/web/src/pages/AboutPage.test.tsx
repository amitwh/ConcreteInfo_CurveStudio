import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { errorReportsEnabled, setConsentResolver } from "../lib/errorReporter";
import { AboutPage } from "./AboutPage";

const { auth } = vi.hoisted(() => ({
  auth: {
    status: "anon" as "anon" | "authed",
    user: null as { preferences: { error_reports: boolean } } | null,
  },
}));

vi.mock("../api/auth", () => ({ useAuth: () => auth }));

function renderAbout() {
  return render(
    <MemoryRouter initialEntries={["/about"]}>
      <AboutPage />
    </MemoryRouter>,
  );
}

function reportsSwitch(): HTMLElement | null {
  return screen.queryByRole("switch", { name: "Send error reports" });
}

beforeEach(() => {
  auth.status = "anon";
  auth.user = null;
  setConsentResolver(null);
});

afterEach(() => {
  setConsentResolver(null);
});

describe("AboutPage", () => {
  it("shows the ConcreteInfo logo on its own chip class", () => {
    renderAbout();
    const logo = screen.getByRole("img", { name: "ConcreteInfo" });
    expect(logo.getAttribute("src")).toBe("/concreteinfo-logo.png");
    expect(logo.classList.contains("about-logo")).toBe(true);
  });

  it("links to the website and the help page", () => {
    renderAbout();
    expect(
      screen.getAllByRole("link", { name: "www.concreteinfo.in" })[0]?.getAttribute("href"),
    ).toBe("https://www.concreteinfo.in");
    expect(
      screen.getByRole("link", { name: "Help and documentation" }).getAttribute("href"),
    ).toBe("/help");
  });

  it("keeps the version, licence and source links", () => {
    renderAbout();
    expect(screen.getByText("Version")).toBeTruthy();
    expect(screen.getAllByRole("link", { name: "AGPL-3.0-or-later" }).length).toBeGreaterThan(0);
    expect(
      screen.getByRole("link", { name: "github.com/amitwh/ConcreteInfo_CurveStudio" }),
    ).toBeTruthy();
  });

  it("lists every credit the desktop app shows", () => {
    renderAbout();
    for (const credit of [
      "Qt / PySide6 (desktop app)",
      "NumPy, SciPy, SymPy",
      "pyqtgraph",
      "Inter and JetBrains Mono typefaces",
      "Lucide icons",
    ]) {
      expect(screen.getByText(credit)).toBeTruthy();
    }
    for (const licence of [
      "LGPL-3.0",
      "BSD-3-Clause",
      "MIT",
      "SIL Open Font License 1.1",
      "ISC",
    ]) {
      expect(screen.getByText(licence)).toBeTruthy();
    }
  });
});

describe("AboutPage — projects, free software and policy", () => {
  it("lists the shared five projects in two groups under the shared heading", () => {
    renderAbout();
    expect(
      screen.getByRole("heading", { name: "Other tools we build and maintain." }),
    ).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Open source" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Commercial software" })).toBeTruthy();
    // rendered from the imported shared JSON: the URLs pass through unchanged
    expect(
      screen.getByRole("link", { name: "PC Power Calculator" }).getAttribute("href"),
    ).toBe("https://pcpower.concreteinfo.net/");
    expect(screen.getByRole("link", { name: "CAMS2" }).getAttribute("href")).toBe(
      "https://cams2.concreteinfo.co.in",
    );
    // open source first, commercial second — the shared list's order
    const listed = Array.from(document.querySelectorAll(".about-projects a")).map(
      (link) => link.textContent,
    );
    expect(listed).toEqual([
      "PC Power Calculator",
      "Markdown Converter",
      "GitHub Desktop for Linux",
      "ConcreteInfo Mix Design Platform",
      "CAMS2",
    ]);
    // each entry keeps its one-line summary, and the contact line stays
    expect(screen.getByText(/chemical admixture management for multi-plant producers/i)).toBeTruthy();
    expect(
      screen.getAllByRole("link", { name: "amit@concreteinfo.in" })[0]?.getAttribute("href"),
    ).toBe("mailto:amit@concreteinfo.in");
  });

  it("explains why it is free and AGPL, and states the disclaimer and policy", () => {
    renderAbout();
    expect(screen.getByRole("heading", { name: "Why it is free and open" })).toBeTruthy();
    expect(screen.getByText(/the desktop app stays free/i)).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Disclaimer" })).toBeTruthy();
    expect(screen.getByText(/fits models to data; it does not decide what the data mean/i)).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Privacy and fair-use policy" })).toBeTruthy();
    expect(screen.getByText(/not sold, and it is not shared for advertising/i)).toBeTruthy();
  });

  it("says the web service keeps a dataset as the whole table that was uploaded", () => {
    renderAbout();
    expect(screen.getByText(/the whole table you uploaded/)).toBeTruthy();
  });
});

describe("AboutPage — error reports", () => {
  it("a visitor sees the switch under the privacy policy, on by default", () => {
    renderAbout();
    const toggle = reportsSwitch();
    expect(toggle).not.toBeNull();
    expect(toggle?.getAttribute("aria-checked")).toBe("true");
    expect(errorReportsEnabled()).toBe(true);
    expect(
      screen.getByText(
        /it can send an anonymous report \(the error, where it happened, and which browser\) so we can fix it\. Reports never include your account, your projects or your data\./,
      ),
    ).toBeTruthy();
  });

  it("turning it off persists, and so does turning it back on", async () => {
    const user = userEvent.setup();
    renderAbout();
    await user.click(reportsSwitch() as HTMLElement);
    expect(reportsSwitch()?.getAttribute("aria-checked")).toBe("false");
    expect(localStorage.getItem("curvestudio.errorReports")).toBe("off");
    expect(errorReportsEnabled()).toBe(false);

    await user.click(reportsSwitch() as HTMLElement);
    expect(reportsSwitch()?.getAttribute("aria-checked")).toBe("true");
    expect(localStorage.getItem("curvestudio.errorReports")).toBeNull();
    expect(errorReportsEnabled()).toBe(true);
  });

  it("the switch shows the setting that counts now, not the one at mount", () => {
    // a stale session is still being sorted out as the page first renders
    setConsentResolver(() => false);
    renderAbout();
    expect(reportsSwitch()?.getAttribute("aria-checked")).toBe("false");
    act(() => {
      setConsentResolver(null);
    });
    expect(reportsSwitch()?.getAttribute("aria-checked")).toBe("true");
  });

  it("a signed-in user does not see the browser's switch, and reads that the setting is on the profile", () => {
    auth.status = "authed";
    auth.user = { preferences: { error_reports: false } };
    renderAbout();
    expect(reportsSwitch()).toBeNull();
    expect(screen.queryByRole("switch")).toBeNull();
    expect(screen.getByText(/Profile → Privacy/)).toBeTruthy();
  });
});
