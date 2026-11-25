/**
 * /help as rendered from the canonical Markdown (docs/help-content.md — design item
 * 10): the headings come from the document itself, every anchor the app has ever
 * linked to still resolves, the screenshots are the document's own, and the search
 * still filters the sections. The expected structure is read from docs/, not
 * restated by hand, so a canonical edit that drifts from the page fails here.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { HelpPage } from "./HelpPage";
import { HELP_SECTIONS } from "../content/helpDocument";

const here = dirname(fileURLToPath(import.meta.url));
const canonicalMarkdown = readFileSync(resolve(here, "../../../../docs/help-content.md"), "utf8");

/** The `##` headings of the canonical document, in file order. */
function canonicalHeadings(marker: "## " | "### "): string[] {
  return canonicalMarkdown
    .split(/\r?\n/)
    .filter((line) => line.startsWith(marker))
    .map((line) => line.slice(marker.length).trim());
}

function renderHelp(): void {
  render(<HelpPage />);
}

/** Run `body` with the browser at `address`: jsdom leaves the real address alone. */
function atAddress(address: string, body: () => void): void {
  const before = window.location.href;
  window.history.replaceState(null, "", address);
  try {
    body();
  } finally {
    window.history.replaceState(null, "", before);
  }
}

/** Every file under `dir` (recursively) whose name ends in one of `extensions`. */
function sourceFiles(dir: string, extensions: string[]): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = `${dir}/${entry}`;
    if (statSync(path).isDirectory()) {
      files.push(...sourceFiles(path, extensions));
    } else if (extensions.some((extension) => entry.endsWith(extension))) {
      files.push(path);
    }
  }
  return files;
}

describe("HelpPage", () => {
  it("renders the canonical document's section headings in order", () => {
    renderHelp();
    const expected = canonicalHeadings("## ");
    expect(expected, "the canonical document no longer holds twenty-five sections").toHaveLength(
      25,
    );
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.textContent)).toEqual(expected);
  });

  it("renders the canonical document's subheadings in order", () => {
    renderHelp();
    const headings = screen.getAllByRole("heading", { level: 3 });
    expect(headings.map((heading) => heading.textContent)).toEqual(canonicalHeadings("### "));
  });

  it("gives every section one of the anchors the app has always used", () => {
    renderHelp();
    for (const id of [
      "getting-started",
      "auto-fit",
      "keyboard-shortcuts",
      "fit-audit-and-methods-text",
      "tools-that-start-from-a-fit",
      "statistics-tools",
      "summarising-columns",
      "comparing-means",
      "rank-based-comparisons",
      "relationships-between-columns",
      "counts",
      "comparing-two-methods",
      "measurement-and-classification",
      "sample-size-and-power",
      "advanced-models",
      "evidence-synthesis",
      "bayesian-tests",
      "about-licence-credits",
    ]) {
      const section = document.getElementById(id);
      expect(section).not.toBeNull();
      expect(section?.tagName).toBe("SECTION");
    }
  });

  it("maps every canonical heading to a unique anchor, or refuses to render", () => {
    // A heading without a SECTION_IDS entry makes importing the document throw —
    // the loud failure that keeps a renamed section from shipping unanchorable.
    const ids = HELP_SECTIONS.map((section) => section.id);
    expect(HELP_SECTIONS).toHaveLength(canonicalHeadings("## ").length);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("resolves every /help#… link anywhere in the app to a rendered section", () => {
    renderHelp();
    const srcRoot = resolve(here, "..");
    const anchors = new Set<string>();
    for (const file of sourceFiles(srcRoot, [".ts", ".tsx"])) {
      if (file.includes(".test.")) continue;
      for (const match of readFileSync(file, "utf8").matchAll(/\/help#[\w-]+/g)) {
        anchors.add(match[0]);
      }
    }
    expect(anchors.size, "the sweep found no /help# links — it is looking in the wrong place").toBeGreaterThan(
      0,
    );
    for (const link of anchors) {
      const section = document.getElementById(link.split("#")[1] ?? "");
      expect(section, `${link} does not resolve to a rendered section`).not.toBeNull();
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
      expect(section?.querySelector("h2")?.textContent).toBe("11. Fit audit and methods text");
      // and its content is the section's: the table of the eleven checks
      const table = section?.querySelector("table");
      expect(table?.querySelectorAll("th")).toHaveLength(2);
      expect(table?.querySelectorAll("tbody tr")).toHaveLength(11);
      expect(section?.textContent).toContain("A verdict is evidence, not proof");
      expect(section?.textContent).toContain("curvestudio methods --style markdown");
    });
  });

  it("explains the three fit tools and where they start", () => {
    renderHelp();
    const section = document.getElementById("tools-that-start-from-a-fit");
    expect(section).not.toBeNull();
    expect(section?.querySelector("h2")?.textContent).toBe("12. Tools that start from a fit");
    expect(
      Array.from(section?.querySelectorAll("h3") ?? []).map((heading) => heading.textContent),
    ).toEqual(["Model discernibility", "Experiment feasibility", "Where to measure next"]);
    expect(section?.textContent).toContain("More tools");
    expect(section?.textContent).toContain("curvestudio discern");
    expect(section?.textContent).toContain("a convention, not a test");
  });

  it("says the web service keeps a dataset as the whole table that was uploaded", () => {
    renderHelp();
    const section = document.getElementById("about-licence-credits");
    expect(section?.textContent).toContain("the whole table you uploaded");
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
      renderHelp();
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
  it("renders the document's own screenshots, captioned, from the app's /help assets", () => {
    renderHelp();
    const figures = Array.from(document.querySelectorAll(".help-figure"));
    expect(figures.map((figure) => figure.querySelector("img")?.getAttribute("src"))).toEqual([
      "/help/workspace.png",
      "/help/paste-grid.png",
      "/help/manual-fit.png",
      "/help/details.png",
    ]);
    for (const figure of figures) {
      const image = figure.querySelector("img");
      expect(image?.getAttribute("alt")).toBeTruthy();
      expect(figure.querySelector("figcaption")?.textContent).toBe(image?.getAttribute("alt"));
    }
  });
});

describe("HelpPage links", () => {
  it("opens the document's external links in a new tab, safely", () => {
    renderHelp();
    const external = Array.from(document.querySelectorAll<HTMLAnchorElement>("a[target='_blank']"));
    expect(external.length).toBeGreaterThan(0);
    for (const link of external) {
      expect(link.getAttribute("rel")).toContain("noopener");
      expect(link.getAttribute("rel")).toContain("noreferrer");
    }
    expect(
      external.some((link) => link.getAttribute("href") === "https://www.concreteinfo.in"),
    ).toBe(true);
  });
});
