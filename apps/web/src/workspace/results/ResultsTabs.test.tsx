import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, type FitAuditOut } from "../../api/client";
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
    render(
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
    render(
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
    render(
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
    render(
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
  ): ReturnType<typeof render> {
    return render(
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
    render(
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
    render(
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
