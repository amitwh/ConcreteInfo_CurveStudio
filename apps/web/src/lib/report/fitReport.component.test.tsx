/**
 * The report component itself: the byline is checked where a reader sees it, as text between
 * the title and the subtitle, and as the document's author. The pure value mapping is covered
 * in `fitReport.test.ts`.
 *
 * react-pdf's primitives belong to its own renderer, which react-dom cannot draw. Stand-ins
 * that render plain elements let the real component run unchanged and let the test read what
 * it hands the document.
 */

import { render, screen } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { FitReport, type FitReportModel } from "./fitReport";

vi.mock("@react-pdf/renderer", async () => {
  const box =
    (tag: string) =>
    ({ children }: { children?: ReactNode }): ReactNode =>
      createElement(tag, null, children);
  return {
    StyleSheet: { create: <T extends object>(styles: T): T => styles },
    Document: ({ author, children }: { author?: string; children?: ReactNode }): ReactNode =>
      createElement("pdf-document", { author }, children),
    Page: box("pdf-page"),
    View: box("pdf-view"),
    Text: ({ children }: { children?: ReactNode }): ReactNode =>
      createElement("pdf-text", null, children),
    Image: ({ src }: { src: string }): ReactNode => createElement("pdf-image", { src }),
  };
});

function model(overrides: Partial<FitReportModel> = {}): FitReportModel {
  return {
    title: "Fit report",
    subtitle: "decay.csv — exp_decay",
    statusLabel: "converged",
    converged: true,
    generatedAt: "22/11/2026 14:05",
    version: "0.11.0",
    dataset: [],
    model: [],
    statusRows: [],
    failureBanner: null,
    equation: "y = a * exp(-b * x)",
    fittedEquation: "y = 2.5 * exp(-0.3 * x)",
    warnings: [],
    statistics: [],
    parameters: [],
    chartImage: null,
    chartCaption: "Data and fitted curve (exported at full scale).",
    byline: null,
    ...overrides,
  };
}

function renderReport(report: FitReportModel): HTMLElement {
  return render(<FitReport report={report} logoDataUrl={null} watermarkDataUrl={null} />)
    .container;
}

/** The single element that carries the document's `author` attribute. */
function authorOf(container: HTMLElement): string | null {
  const documents = container.querySelectorAll("[author]");
  expect(documents).toHaveLength(1);
  return documents[0]?.getAttribute("author") ?? null;
}

describe("FitReport — prepared-by line", () => {
  it("prints 'Prepared by <name>' between the title and the subtitle, and names the author", () => {
    const container = renderReport(model({ byline: "Alice" }));

    const byline = screen.getByText("Prepared by Alice");
    expect(byline.previousElementSibling?.textContent).toBe("Fit report");
    expect(byline.nextElementSibling?.textContent).toBe("decay.csv — exp_decay");
    expect(authorOf(container)).toBe("Alice");
  });

  it("prints no line and keeps the tool as author when there is no byline", () => {
    const container = renderReport(model());

    expect(screen.queryByText(/Prepared by/)).toBeNull();
    const subtitle = screen.getByText("decay.csv — exp_decay");
    expect(subtitle.previousElementSibling?.textContent).toBe("Fit report");
    expect(authorOf(container)).toBe("CurveStudio (ConcreteInfo)");
  });
});
