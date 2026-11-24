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
  "Fit audit and methods text",
  "About, licence, projects and policy",
];

function renderHelp(entry = "/help") {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <HelpPage />
    </MemoryRouter>,
  );
}

/** Run `body` with the browser at `address`: a memory router leaves the real address alone. */
function atAddress(address: string, body: () => void): void {
  const before = window.location.href;
  window.history.replaceState(null, "", address);
  try {
    body();
  } finally {
    window.history.replaceState(null, "", before);
  }
}

describe("HelpPage", () => {
  it("renders all twelve section headings in order", () => {
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
      "fit-audit-and-methods-text",
      "about-licence-credits",
    ]) {
      const section = document.getElementById(id);
      expect(section).not.toBeNull();
      expect(section?.tagName).toBe("SECTION");
    }
  });

  it("lists the fit audit section with the anchor /help#fit-audit-and-methods-text", () => {
    atAddress("/help", () => {
      renderHelp();
      const link = screen.getByRole("link", { name: /Fit audit and methods text/ });
      const target = new URL(link.getAttribute("href") ?? "", window.location.href);
      expect(target.pathname + target.hash).toBe("/help#fit-audit-and-methods-text");
      // the index numbers it from its place: after Troubleshooting, before About
      expect(link.textContent).toMatch(/11\s*Fit audit and methods text/);

      const section = document.getElementById("fit-audit-and-methods-text");
      expect(section?.querySelector("h2")?.textContent).toBe("Fit audit and methods text");
      // and its content is the section's: the table of the eleven checks
      const table = section?.querySelector("table");
      expect(table?.querySelectorAll("th")).toHaveLength(2);
      expect(table?.querySelectorAll("tbody tr")).toHaveLength(11);
      expect(section?.textContent).toContain("A verdict is evidence, not proof");
      expect(section?.textContent).toContain("curvestudio methods --style markdown");
    });
  });

  it("scrolls to the fit audit section when the page opens at /help#fit-audit-and-methods-text", () => {
    // jsdom cannot scroll: the stand-in records the element the page asked to bring into view.
    const scrolled: Element[] = [];
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      writable: true,
      value(this: Element) {
        scrolled.push(this);
      },
    });
    atAddress("/help#fit-audit-and-methods-text", () => {
      renderHelp("/help#fit-audit-and-methods-text");
      expect(scrolled.map((element) => element.id)).toEqual(["fit-audit-and-methods-text"]);
    });
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
