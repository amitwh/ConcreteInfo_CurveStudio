import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, type FitAuditOut, type FitMethodsOut } from "../../api/client";
import type { components } from "../../api/schema";
import { renderWithQuery } from "../../test/render";
import { ResultsTabs } from "./ResultsTabs";

const exportMock = vi.hoisted(() => ({ exportFitPdf: vi.fn() }));
vi.mock("../../lib/report/export", () => ({ exportFitPdf: exportMock.exportFitPdf }));

/** The signed-in account ResultsTabs reads the byline preference from; null = signed out. */
const { authState } = vi.hoisted(() => ({
  authState: {
    user: null as null | { name: string; preferences: { report_byline: boolean } },
  },
}));
vi.mock("../../api/auth", () => ({ useAuth: () => ({ user: authState.user }) }));

type FitBundle = components["schemas"]["FitBundle"];

function bundle(status: string, termination: string): FitBundle {
  return {
    result: {
      model_id: "exp_decay",
      equation: "y = a * exp(-b * x)",
      equation_fitted: null,
      parameters: [{ name: "a", value: 2, std_err: null, ci_low: null, ci_high: null }],
      covariance: null,
      correlation: null,
      stats: null,
      fitted: [],
      residuals: [],
      status,
      termination_reason: termination,
      method: "lm",
      loss: "linear",
      initial_params: {},
      n_starts: 1,
      warnings: [],
      provenance: {
        software_versions: {},
        dataset_fingerprint: "fp",
        created_at: "2026-11-17T05:00:00Z",
        seed: 0,
      },
    },
    summary: null,
    summary_reason: "the fit did not converge",
    curve: null,
    curve_reason: "no curve is drawn for a fit that did not converge",
    confidence_band: null,
    prediction_band: null,
    bands_reason: "not converged",
    equation_text: "y = a * exp(-b * x)",
    warnings: [],
    data: {
      x: [0, 1],
      y: [2, 1.3],
      weights: null,
      x_name: "x",
      y_name: "y",
      n_total: 2,
      subsampled: false,
    },
  } as unknown as FitBundle;
}

describe("non-converged fit display (UI honesty)", () => {
  it("shows exactly one banner: 'Fit did not converge: <reason>'", () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("not_converged", "maximum number of iterations exceeded")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    const alerts = screen.getAllByRole("alert");
    expect(alerts).toHaveLength(1);
    expect(alerts[0]!.className).toContain("alert--error");
    expect(alerts[0]!.textContent).toBe(
      "Fit did not converge: maximum number of iterations exceeded.",
    );
  });

  it("uses the matching heading for a failed fit", () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("failed", "time limit of 60 s exceeded")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert").textContent).toBe("Fit failed: time limit of 60 s exceeded.");
  });

  it("hides the estimates of a non-converged fit behind 'Show anyway'", async () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("not_converged", "max iterations")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    expect(screen.getByText(/do not use/i)).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show anyway" }));
    expect(screen.getByRole("table")).toBeTruthy();
    expect(screen.getByText("a")).toBeTruthy();
  });

  it("shows the parameter table directly for a converged fit, without t/p columns", () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("converged", "converged: rtol")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    expect(screen.getByRole("table")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Show anyway" })).toBeNull();
    // t and p are not part of the bundle contract; the columns are gone, not "n/a".
    expect(screen.queryByRole("columnheader", { name: "t" })).toBeNull();
    expect(screen.queryByRole("columnheader", { name: "p" })).toBeNull();
    expect(screen.getByRole("columnheader", { name: "95% CI" })).toBeTruthy();
  });
});

describe("PDF report export", () => {
  beforeEach(() => {
    exportMock.exportFitPdf.mockReset();
    exportMock.exportFitPdf.mockResolvedValue(undefined);
    authState.user = null;
  });

  function renderTabs(
    captureChart?: () => Promise<string | null>,
  ): ReturnType<typeof renderWithQuery> {
    return renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("converged", "converged: rtol")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
        captureChart={captureChart}
        datasetName="decay.csv"
      />,
    );
  }

  it("renders a PDF report button in the tab strip", () => {
    renderTabs();
    expect(screen.getByRole("button", { name: "PDF report" })).toBeTruthy();
  });

  it("captures the chart and exports the bundle", async () => {
    const captureChart = vi.fn().mockResolvedValue("data:image/png;base64,CHART");
    renderTabs(captureChart);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "PDF report" }));

    expect(captureChart).toHaveBeenCalledTimes(1);
    expect(exportMock.exportFitPdf).toHaveBeenCalledTimes(1);
    const [exportedBundle, meta] = exportMock.exportFitPdf.mock.calls[0] as [
      FitBundle,
      Record<string, unknown>,
    ];
    expect(exportedBundle.result.model_id).toBe("exp_decay");
    expect(meta).toMatchObject({
      datasetName: "decay.csv",
      chartImage: "data:image/png;base64,CHART",
    });
  });

  it("still exports without a chart when no capture is wired", async () => {
    renderTabs();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "PDF report" }));

    const [, meta] = exportMock.exportFitPdf.mock.calls[0] as [
      FitBundle,
      Record<string, unknown>,
    ];
    expect(meta).toMatchObject({ chartImage: null });
  });

  it("survives a failed chart capture and exports without the chart", async () => {
    const captureChart = vi.fn().mockRejectedValue(new Error("Plotly is gone"));
    renderTabs(captureChart);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "PDF report" }));

    const [, meta] = exportMock.exportFitPdf.mock.calls[0] as [
      FitBundle,
      Record<string, unknown>,
    ];
    expect(meta).toMatchObject({ chartImage: null });
  });

  it("shows an inline error and re-enables the button when generation fails", async () => {
    exportMock.exportFitPdf.mockRejectedValueOnce(new Error("renderer unavailable"));
    const captureChart = vi.fn().mockResolvedValue(null);
    renderTabs(captureChart);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "PDF report" }));

    const alert = await screen.findByRole("alert");
    expect(alert.className).toContain("alert--error");
    expect(alert.textContent).toBe("renderer unavailable");
    const button = screen.getByRole("button", { name: "PDF report" }) as HTMLButtonElement;
    expect(button.disabled).toBe(false);
  });

  /** Exports once and returns the meta the report was generated with. */
  async function exportMeta(user: ReturnType<typeof userEvent.setup>): Promise<Record<string, unknown>> {
    renderTabs();
    await user.click(screen.getByRole("button", { name: "PDF report" }));
    const [, meta] = exportMock.exportFitPdf.mock.calls[0] as [FitBundle, Record<string, unknown>];
    return meta;
  }

  it("names the user as the byline when the preference is on", async () => {
    authState.user = { name: "Amit Haridas", preferences: { report_byline: true } };
    const meta = await exportMeta(userEvent.setup());
    expect(meta.byline).toBe("Amit Haridas");
  });

  it("sends no byline when the preference is off", async () => {
    authState.user = { name: "Amit Haridas", preferences: { report_byline: false } };
    expect((await exportMeta(userEvent.setup())).byline).toBeUndefined();
  });

  it("sends no byline when the name is empty, even with the preference on", async () => {
    authState.user = { name: "", preferences: { report_byline: true } };
    expect((await exportMeta(userEvent.setup())).byline).toBeUndefined();
  });

  it("sends no byline when nobody is signed in", async () => {
    expect((await exportMeta(userEvent.setup())).byline).toBeUndefined();
  });
});

describe("PredictTab inline validation", () => {
  it("shows 'every x value must be a finite number' for a non-numeric entry", async () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("converged", "converged: rtol")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("tab", { name: "Predict" }));
    await user.type(
      screen.getByLabelText("x values (separated by space, comma or new line)"),
      "5 10 abc",
    );
    await user.click(screen.getByRole("button", { name: "Predict" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "every x value must be a finite number",
    );
  });
});

describe("the tab strip", () => {
  it("lists the tabs in order, with Fit audit after Statistics", () => {
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("converged", "converged: rtol")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Parameters",
      "Statistics",
      "Fit audit",
      "More tools",
      "Residuals",
      "Equation",
      "Predict",
      "History",
    ]);
  });

  it("the Fit audit tab renders its heading and the audit's summary", async () => {
    const audit: FitAuditOut = {
      checks: [
        {
          id: "converged",
          title: "The fit converged",
          verdict: "pass",
          statistic: null,
          threshold: "status converged",
          evidence: "Status converged: converged: rtol",
          explanation: "A fit that stopped before it converged has no usable estimates.",
          next_step: null,
        },
      ],
      overall: "pass",
      summary: "Every check that could run passed.",
      methods_fragment: "",
      simpler_model: null,
      n_observations: 2,
      n_parameters: 1,
      excluded: [],
    };
    vi.spyOn(api, "fitAudit").mockResolvedValue(audit);
    renderWithQuery(
      <ResultsTabs
        fitId="fit-1"
        bundle={bundle("converged", "converged: rtol")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />,
    );

    await userEvent.setup().click(screen.getByRole("tab", { name: "Fit audit" }));

    const heading = await screen.findByRole("heading", { level: 3, name: "Fit audit" });
    expect(heading).toBeTruthy();
    expect(api.fitAudit).toHaveBeenCalledWith("fit-1");
    expect(screen.getByText("Every check that could run passed.")).toBeTruthy();
  });
});

describe("Copy methods", () => {
  const SHARED = {
    citation_curvestudio: "CurveStudio 0.11.1 (ConcreteInfo)",
    citation_model: ["Model exp_decay: CurveStudio model library"],
    to_complete: ["data_source: say where the data came from"],
    missing: [],
  };

  const METHODS: FitMethodsOut = {
    text: {
      paragraph: "Data (y against x) were fitted by non-linear least squares.",
      equation: "y = 2.0134·exp(−0.5210·x) + 0.1042",
      table: "Parameter | Estimate\na | 2.0134",
      ...SHARED,
    },
    markdown: {
      paragraph: "Data (y against x) were fitted by non-linear least squares.",
      equation: "$y = 2.0134 e^{-0.5210x}$",
      table: "| Parameter | Estimate |\n|---|---|",
      ...SHARED,
    },
    latex: {
      paragraph: "Data (y against x) were fitted by non-linear least squares.",
      equation: "\\begin{equation}\ny = 2.0134\\,e^{-0.5210x} + 0.1042\n\\end{equation}",
      table: "\\begin{tabular}{lr}\na & 2.0134 \\\\\n\\end{tabular}",
      ...SHARED,
    },
  };

  /** The document copied for a style: its paragraph, equation and table, a blank line between. */
  function methodsDocument(style: keyof FitMethodsOut): string {
    const text = METHODS[style];
    return [text.paragraph, text.equation, text.table].join("\n\n");
  }

  function tabs(fitId = "fit-1", status = "converged") {
    return (
      <ResultsTabs
        fitId={fitId}
        bundle={bundle(status, status === "converged" ? "converged: rtol" : "max iterations")}
        fits={[]}
        onLoadFit={vi.fn()}
        onDeleteFit={vi.fn()}
      />
    );
  }

  async function choose(user: ReturnType<typeof userEvent.setup>, entry: string): Promise<void> {
    await user.click(screen.getByRole("button", { name: "Copy methods" }));
    await user.click(screen.getByRole("button", { name: entry }));
  }

  it("copies each style as its paragraph, a blank line, its equation, a blank line and its table", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
    vi.spyOn(api, "fitMethods").mockResolvedValue(METHODS);
    renderWithQuery(tabs());

    const styles: Array<[string, keyof FitMethodsOut, string]> = [
      ["Text", "text", "plain text"],
      ["Markdown", "markdown", "Markdown"],
      ["LaTeX", "latex", "LaTeX"],
    ];
    for (const [entry, style, name] of styles) {
      await choose(user, entry);
      expect(
        await screen.findByText(`Copied. The ${name} methods text is on the clipboard.`),
      ).toBeTruthy();
      expect(writeText).toHaveBeenLastCalledWith(methodsDocument(style));
    }

    // All three came from one request: a second copy does not fetch again.
    expect(api.fitMethods).toHaveBeenCalledTimes(1);
    expect(api.fitMethods).toHaveBeenCalledWith("fit-1");
  });

  it("shows the document with Select all when the clipboard refuses, and does not say Copied", async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new Error("denied"));
    vi.spyOn(api, "fitMethods").mockResolvedValue(METHODS);
    renderWithQuery(tabs());

    await choose(user, "LaTeX");

    expect(
      await screen.findByText(
        "The clipboard is not available here, so the LaTeX methods text is shown below. Use Select all, then copy.",
      ),
    ).toBeTruthy();
    const shown = screen.getByLabelText("Methods text as LaTeX") as HTMLTextAreaElement;
    expect(shown.value).toBe(methodsDocument("latex"));
    expect(shown.readOnly).toBe(true);
    expect(screen.queryByText(/Copied\./)).toBeNull();

    await user.click(screen.getByRole("button", { name: "Select all" }));
    expect([shown.selectionStart, shown.selectionEnd]).toEqual([0, shown.value.length]);
  });

  it("shows the document the same way when the browser has no clipboard at all", async () => {
    vi.spyOn(api, "fitMethods").mockResolvedValue(METHODS);
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    renderWithQuery(tabs());

    fireEvent.click(screen.getByRole("button", { name: "Copy methods" }));
    fireEvent.click(screen.getByRole("button", { name: "Markdown" }));

    expect(await screen.findByLabelText("Methods text as Markdown")).toBeTruthy();
    expect(screen.queryByText(/Copied\./)).toBeNull();
  });

  it("shows why the methods text could not be fetched", async () => {
    const user = userEvent.setup();
    const reason =
      "the fit is not_converged (max iterations): a fit that did not converge has no estimates to report";
    vi.spyOn(api, "fitMethods").mockRejectedValue(
      new ApiError(422, "VALIDATION_ERROR", reason, { code: "FIT_NOT_CONVERGED_INPUT" }),
    );
    renderWithQuery(tabs());

    await choose(user, "Text");

    expect((await screen.findByRole("alert")).textContent).toBe(reason);
    expect(screen.queryByText(/Copied\./)).toBeNull();
  });

  it("is disabled, with its hint, for a fit that did not converge", async () => {
    const fitMethods = vi.spyOn(api, "fitMethods");
    renderWithQuery(tabs("fit-1", "not_converged"));

    const control = screen.getByRole("button", { name: "Copy methods" }) as HTMLButtonElement;
    expect(control.disabled).toBe(true);
    expect(screen.getByText("No methods text for a fit that did not converge.")).toBeTruthy();

    await userEvent.setup().click(control);
    expect(fitMethods).not.toHaveBeenCalled();
  });

  it("closes its menu on Escape, returning focus to the button, and once Tab leaves the menu", async () => {
    const user = userEvent.setup();
    renderWithQuery(tabs());
    const trigger = screen.getByRole("button", { name: "Copy methods" });

    await user.click(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("button", { name: "LaTeX" })).toBeNull();
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(trigger);

    // Tab walks the entries, and the menu goes with the focus once it moves past them.
    await user.click(trigger);
    for (const entry of ["Text", "Markdown", "LaTeX"]) {
      await user.tab();
      expect(document.activeElement).toBe(screen.getByRole("button", { name: entry }));
    }
    await user.tab();
    expect(screen.queryByRole("button", { name: "LaTeX" })).toBeNull();
  });

  it("carries nothing said about one fit's methods text over to another fit", async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new Error("denied"));
    vi.spyOn(api, "fitMethods").mockResolvedValue(METHODS);
    const view = renderWithQuery(tabs());

    await choose(user, "Text");
    expect(await screen.findByLabelText("Methods text as plain text")).toBeTruthy();

    view.rerender(
      <QueryClientProvider client={view.queryClient}>{tabs("fit-2")}</QueryClientProvider>,
    );

    expect(screen.queryByLabelText("Methods text as plain text")).toBeNull();
  });
});
