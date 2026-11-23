import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { reportModel, reportFileName, istTimestamp } from "./export";

type FitBundle = components["schemas"]["FitBundle"];

function convergedBundle(overrides: Partial<FitBundle["result"]> = {}): FitBundle {
  return {
    result: {
      model_id: "exp_decay",
      equation: "y = a * exp(-b * x)",
      equation_fitted: "y = 2.5 * exp(-0.3 * x)",
      parameters: [
        { name: "a", value: 2.5, std_err: 0.25, ci_low: 2.0, ci_high: 3.0 },
        { name: "b", value: 0.3, std_err: null, ci_low: null, ci_high: null },
      ],
      covariance: null,
      correlation: null,
      stats: {
        n: 40,
        k: 2,
        df: 38,
        r2: 0.9876543210987654,
        adj_r2: 0.9876543210987001,
        rmse: 0.0512,
        mae: 0.04,
        mse: 0.0026,
        sse: 0.104,
        residual_se: 0.0523,
        r2_definition: "centered",
      },
      fitted: [1, 2],
      residuals: [0.1, -0.1],
      status: "converged",
      termination_reason: "converged: rtol",
      method: "lm",
      loss: "linear",
      initial_params: {},
      n_starts: 5,
      warnings: ["one fit warning"],
      provenance: {
        software_versions: {},
        dataset_fingerprint: "fp",
        created_at: "2026-11-17T05:00:00Z",
        seed: 42,
      },
      ...overrides,
    },
    summary: null,
    summary_reason: null,
    curve: null,
    curve_reason: null,
    confidence_band: null,
    prediction_band: null,
    bands_reason: null,
    equation_text: "y = a * exp(-b * x)",
    warnings: ["one bundle warning", "one fit warning"],
    data: {
      x: [0, 1],
      y: [2, 1],
      weights: null,
      x_name: "time",
      y_name: "response",
      n_total: 40,
      subsampled: false,
    },
  };
}

function nonConvergedBundle(): FitBundle {
  const bundle = convergedBundle({
    status: "not_converged",
    stats: null,
    termination_reason: "maximum number of iterations exceeded",
    equation_fitted: null,
    parameters: [{ name: "a", value: 9.9, std_err: null, ci_low: null, ci_high: null }],
  });
  return bundle;
}

const META = {
  datasetName: "decay.csv",
  chartImage: "data:image/png;base64,AAAA",
  version: "0.11.0",
  generatedAt: new Date("2026-11-22T08:35:00Z"),
};

describe("reportModel — converged fit", () => {
  it("prints R² and adj. R² at full float precision", () => {
    const report = reportModel(convergedBundle(), META);
    const value = (label: string): string | undefined =>
      report.statistics.find((stat) => stat.label === label)?.value;
    expect(value("R²")).toBe("0.9876543210987654");
    expect(value("adj. R²")).toBe("0.9876543210987001");
  });

  it("formats the parameter table and its confidence intervals", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.parameters[0]).toEqual({
      name: "a",
      estimate: "2.5",
      stdErr: "0.25",
      ci: "[2, 3]",
    });
  });

  it("reports a converged status pill and no failure banner", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.statusLabel).toBe("converged");
    expect(report.converged).toBe(true);
    expect(report.failureBanner).toBeNull();
    expect(report.statusRows).toEqual([{ key: "Status", value: "converged" }]);
  });

  it("names the dataset and model, and timestamps in IST", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.subtitle).toBe("decay.csv — exp_decay");
    expect(report.generatedAt).toBe("22/11/2026 14:05");
    expect(report.version).toBe("0.11.0");
  });

  it("prints the columns and the seed without digit grouping", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.dataset).toContainEqual({ key: "Columns", value: "time / response" });
    expect(report.model).toContainEqual({ key: "Seed", value: "42" });
    expect(report.model).toContainEqual({ key: "Starts", value: "5" });
  });

  it("shows the fitted equation and de-duplicates warnings", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.equation).toBe("y = a * exp(-b * x)");
    expect(report.fittedEquation).toBe("y = 2.5 * exp(-0.3 * x)");
    expect(report.warnings).toEqual(["one fit warning", "one bundle warning"]);
  });

  it("prints 'n/a' for a missing confidence interval", () => {
    const report = reportModel(convergedBundle(), META);
    expect(report.parameters[1]).toEqual({
      name: "b",
      estimate: "0.3",
      stdErr: "n/a",
      ci: "n/a",
    });
  });
});

describe("reportModel — non-converged fit (UI honesty)", () => {
  it("shows the banner with the termination reason and makes no curve claim", () => {
    const report = reportModel(nonConvergedBundle(), { ...META, chartImage: null });
    expect(report.converged).toBe(false);
    expect(report.failureBanner).toBe(
      "Fit did not converge — maximum number of iterations exceeded",
    );
    expect(report.fittedEquation).toBeNull();
    expect(report.chartCaption).toContain("no curve is claimed");
    expect(report.statusRows).toContainEqual({
      key: "Termination",
      value: "maximum number of iterations exceeded",
    });
  });

  it("renders every undefined statistic as 'n/a', never 0", () => {
    const report = reportModel(nonConvergedBundle(), META);
    for (const stat of report.statistics) {
      expect(stat.value).toBe("n/a");
    }
  });
});

describe("reportModel — metadata fallbacks", () => {
  it("uses 'Untitled dataset' when no dataset name is supplied", () => {
    const report = reportModel(convergedBundle(), { ...META, datasetName: null });
    expect(report.dataset[0]).toEqual({ key: "Dataset", value: "Untitled dataset" });
    expect(report.subtitle.startsWith("Untitled dataset — ")).toBe(true);
  });

  it("passes the chart through only when one was captured", () => {
    expect(reportModel(convergedBundle(), META).chartImage).toBe(
      "data:image/png;base64,AAAA",
    );
    expect(
      reportModel(convergedBundle(), { ...META, chartImage: null }).chartImage,
    ).toBeNull();
  });
});

describe("reportModel — prepared-by line", () => {
  it("passes a byline through, trimmed", () => {
    expect(reportModel(convergedBundle(), { ...META, byline: "  Alice  " }).byline).toBe("Alice");
  });

  it("treats a missing or blank byline as none", () => {
    expect(reportModel(convergedBundle(), META).byline).toBeNull();
    for (const byline of [null, "", "   "]) {
      expect(reportModel(convergedBundle(), { ...META, byline }).byline).toBeNull();
    }
  });
});

describe("report file name", () => {
  it("is a sanitised model id plus an IST timestamp", () => {
    expect(reportFileName("Exp Decay/v2", new Date("2026-11-22T08:35:00Z"))).toBe(
      "curvestudio-fit-exp-decay-v2-20261122-1405.pdf",
    );
  });

  it("formats the IST timestamp with a 24-hour clock", () => {
    expect(istTimestamp(new Date("2026-11-22T18:35:00Z"))).toBe("23/11/2026 00:05");
  });
});
