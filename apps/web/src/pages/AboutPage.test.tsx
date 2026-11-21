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
      screen.getByRole("link", { name: "www.concreteinfo.in" }).getAttribute("href"),
    ).toBe("https://www.concreteinfo.in");
    expect(
      screen.getByRole("link", { name: "Help and documentation" }).getAttribute("href"),
    ).toBe("/help");
  });

  it("keeps the version, licence and source links", () => {
    renderAbout();
    expect(screen.getByText("Version")).toBeTruthy();
    expect(screen.getByRole("link", { name: "AGPL-3.0-or-later" })).toBeTruthy();
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
