import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AboutPage } from "./AboutPage";

function renderAbout() {
  return render(
    <MemoryRouter initialEntries={["/about"]}>
      <AboutPage />
    </MemoryRouter>,
  );
}

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
  it("links the other three free ConcreteInfo projects", () => {
    renderAbout();
    expect(
      screen.getByRole("link", { name: "GitHub Desktop for Linux" }).getAttribute("href"),
    ).toBe("https://github.com/amitwh/GitHubDesktopLinux");
    expect(
      screen.getByRole("link", { name: "Markdown Converter" }).getAttribute("href"),
    ).toBe("https://github.com/amitwh/markdown-converter");
    expect(
      screen.getByRole("link", { name: "PC Power Calculator" }).getAttribute("href"),
    ).toBe("https://pcpower.concreteinfo.net/");
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
});
