import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { HelpPage } from "./HelpPage";

const SECTION_TITLES = [
  "Getting started",
  "Importing data",
  "Auto-fit",
  "Manual fit and custom equations",
  "Understanding the numbers",
  "Prediction",
  "Projects (desktop)",
  "Exports",
  "Keyboard shortcuts (desktop)",
  "Troubleshooting",
  "About, licence, projects and policy",
];

function renderHelp() {
  return render(
    <MemoryRouter initialEntries={["/help"]}>
      <HelpPage />
    </MemoryRouter>,
  );
}

describe("HelpPage", () => {
  it("renders all eleven section headings in order", () => {
    renderHelp();
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.textContent)).toEqual(SECTION_TITLES);
  });

  it("gives every section a stable anchor", () => {
    renderHelp();
    for (const id of [
      "getting-started",
      "auto-fit",
      "keyboard-shortcuts",
      "about-licence-credits",
    ]) {
      const section = document.getElementById(id);
      expect(section).not.toBeNull();
      expect(section?.tagName).toBe("SECTION");
    }
  });

  it("filters the section index to a match and jumps to it, case-insensitively", async () => {
    const scrollIntoView = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      writable: true,
      value: scrollIntoView,
    });
    renderHelp();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Search help"), "SHORTCUT");

    const nav = screen.getByRole("navigation", { name: "Sections" });
    const links = screen.getAllByRole("link");
    expect(
      links.some((link) => link.textContent?.includes("Keyboard shortcuts (desktop)")),
    ).toBe(true);
    expect(nav.textContent).not.toContain("Getting started");
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("says explicitly when no section matches", async () => {
    renderHelp();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Search help"), "zzzz-no-such-section");

    expect(screen.getByRole("status").textContent).toMatch(/no section matches/i);
    expect(screen.queryByRole("navigation", { name: "Sections" })).toBeNull();
  });
});

describe("HelpPage screenshots", () => {
  it("renders the workspace and paste screenshots with captions", () => {
    renderHelp();
    const images = Array.from(document.querySelectorAll(".help-figure img"));
    expect(images.map((image) => image.getAttribute("src"))).toEqual([
      "/help/workspace.png",
      "/help/paste.png",
    ]);
    for (const image of images) {
      expect(image.getAttribute("alt")).toBeTruthy();
    }
    expect(screen.getByText(/the chart captions the selected model's fitted equation/i)).toBeTruthy();
    expect(screen.getByText(/drop it straight into the grid/i)).toBeTruthy();
  });
});
