/**
 * The options disclosure's layout contract: opening it must never move the layout
 * around it (the open content is an out-of-flow, height-capped, internally scrolled
 * dropdown panel) and must not scroll the page or move focus off the toggle.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ModelInfo } from "../../api/client";
import { DEFAULT_AUTOFIT_OPTIONS } from "./useAutofit";
import { AutofitOptions } from "./AutofitOptions";

/** The app stylesheet, read from disk (a `?raw` import comes back empty in vitest,
 *  and Vite rewrites `new URL(..., import.meta.url)` as an asset reference). */
const STYLES_CSS = readFileSync(
  path.resolve(fileURLToPath(import.meta.url), "..", "..", "..", "styles.css"),
  "utf8",
);

const MODELS: ModelInfo[] = [
  {
    id: "exp_decay",
    name: "Exponential decay",
    category: "Exponential",
    equation: "y = a * exp(-b * x)",
    latex: "",
    parameters: [],
    variables: ["x"],
    references: [],
    assumptions: "",
    is_family: false,
  },
];

function renderOptions(): void {
  render(
    <AutofitOptions
      models={MODELS}
      options={DEFAULT_AUTOFIT_OPTIONS}
      onChange={() => {}}
    />,
  );
}

describe("AutofitOptions", () => {
  it("toggling open does not scroll the page or move focus off the summary", async () => {
    // jsdom has no scrollIntoView; a browser-shaped no-op lets the spy prove the
    // toggle path never calls it.
    if (typeof Element.prototype.scrollIntoView !== "function") {
      Object.defineProperty(Element.prototype, "scrollIntoView", {
        value: () => {},
        writable: true,
        configurable: true,
      });
    }
    const user = userEvent.setup();
    renderOptions();
    const summary = screen.getByText("Options").closest("summary") as HTMLElement;
    const scroller = document.documentElement;
    scroller.scrollTop = 120;

    const scrollIntoView = vi.spyOn(Element.prototype, "scrollIntoView");
    const windowScroll = vi.spyOn(window, "scroll");
    const windowScrollTo = vi.spyOn(window, "scrollTo");

    await user.click(summary);

    expect(scrollIntoView).not.toHaveBeenCalled();
    expect(windowScroll).not.toHaveBeenCalled();
    expect(windowScrollTo).not.toHaveBeenCalled();
    expect(scroller.scrollTop).toBe(120);
    expect(document.activeElement).toBe(summary);
  });

  it("keeps every option inside one wrapper, the details' only other child", () => {
    renderOptions();
    const details = document.querySelector("details.autofit-options") as HTMLDetailsElement;
    const children = Array.from(details.children);
    expect(children).toHaveLength(2);
    expect(children[0]?.tagName).toBe("SUMMARY");
    const body = children[1] as HTMLElement;
    expect(body.className).toBe("autofit-options__body");
    // Everything the disclosure reveals lives in the wrapper the stylesheet bounds.
    expect(body.querySelector("fieldset")).not.toBeNull();
    expect(screen.getByLabelText(/Max parameters/).closest("div")?.parentElement).toBe(body);
    expect(screen.getByLabelText("Seed").closest("div")?.parentElement).toBe(body);
  });

  it("the stylesheet renders the open content as a height-capped scrollable overlay", () => {
    // jsdom computes no layout and its CSS parser gives up on the full sheet, so the
    // bounding rules are matched textually: out of flow (the section container's height
    // cannot change on toggle), with a max-height and its own vertical scroll.
    const ruleBody = (selector: string): string => {
      const start = STYLES_CSS.indexOf(`${selector} {`);
      if (start === -1) return "";
      const open = STYLES_CSS.indexOf("{", start);
      return STYLES_CSS.slice(open + 1, STYLES_CSS.indexOf("}", open));
    };

    const body = ruleBody(".autofit-options__body");
    expect(body).toContain("position: absolute");
    expect(body).toContain("max-height:");
    expect(body).toContain("overflow-y: auto");

    expect(ruleBody(".autofit-options")).toContain("position: relative");
  });
});
