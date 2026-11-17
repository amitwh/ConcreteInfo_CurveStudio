import { afterEach, describe, expect, it } from "vitest";
import type { components } from "../api/schema";
import {
  buildTraces,
  logAxesAllowed,
  mergeLayout,
  setWebglSupportedForTests,
  themeLayout,
} from "./traces";

type FitBundle = components["schemas"]["FitBundle"];
type PointsOut = components["schemas"]["PointsOut"];

const points = (x: number[], y: number[]): PointsOut => ({
  n_total: x.length,
  x,
  y,
  w: null,
});

function bundle(status: string, overrides: Record<string, unknown> = {}): FitBundle {
  return {
    result: {
      model_id: "exp_decay",
      equation: "y = a * exp(-b * x)",
      equation_fitted: "y = 2 * exp(-0.5 * x)",
      parameters: [{ name: "a", value: 2, std_err: 0.1, ci_low: 1.8, ci_high: 2.2 }],
      covariance: [[0.01, 0], [0, 0.0004]],
      correlation: [[1, 0], [0, 1]],
      stats: null,
      fitted: [2, 1.2],
      residuals: [0, 0.1],
      status,
      termination_reason: "converged: rtol",
      method: "lm",
      loss: "linear",
      initial_params: { a: 1 },
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
    summary_reason: null,
    curve: { x: [0, 1, 2], y: [2, 1.2, 0.7] },
    curve_reason: null,
    confidence_band: {
      kind: "confidence",
      level: 0.95,
      x: [0, 1, 2],
      fit: [2, 1.2, 0.7],
      lower: [1.9, 1.1, 0.6],
      upper: [2.1, 1.3, 0.8],
      extrapolated: [false, false, false],
      df: 38,
      note: "pointwise",
      pointwise: true,
    },
    prediction_band: {
      kind: "prediction",
      level: 0.95,
      x: [0, 1, 2],
      fit: [2, 1.2, 0.7],
      lower: [1.5, 0.8, 0.4],
      upper: [2.5, 1.6, 1.0],
      extrapolated: [false, false, false],
      df: 38,
      note: "pointwise",
      pointwise: true,
    },
    bands_reason: null,
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
    ...overrides,
  } as unknown as FitBundle;
}

const allOn = { confidence: true, prediction: true, logX: false, logY: false };

afterEach(() => {
  setWebglSupportedForTests(null);
});

describe("the no-curve invariant", () => {
  it("NEVER adds a fitted curve or bands for a non-converged fit", () => {
    for (const status of ["not_converged", "failed", "invalid_input"]) {
      const { traces } = buildTraces(
        points([0, 1, 2], [2, 1.3, 0.9]),
        bundle(status, { termination_reason: "max iterations exceeded" }),
        allOn,
        { x: "x", y: "y" },
      );
      expect(traces).toHaveLength(1);
      expect(traces[0]?.name).toBe("Data");
      expect(traces.map((t) => t.name)).not.toContain("Fitted curve");
    }
  });

  it("adds the curve and both bands for a converged fit when toggled on", () => {
    const { traces } = buildTraces(points([0, 1, 2], [2, 1.3, 0.9]), bundle("converged"), allOn, {
      x: "x",
      y: "y",
    });
    const names = traces.map((t) => t.name);
    expect(names).toContain("Data");
    expect(names).toContain("Fitted curve");
    expect(names).toContain("95% confidence band");
    expect(names).toContain("95% prediction band");
  });

  it("omits a band that the bundle does not define", () => {
    const { traces } = buildTraces(
      points([0, 1], [2, 1.3]),
      bundle("converged", { prediction_band: null, bands_reason: "n < k + 2" }),
      allOn,
      { x: "x", y: "y" },
    );
    const names = traces.map((t) => t.name);
    expect(names).toContain("95% confidence band");
    expect(names).not.toContain("95% prediction band");
  });

  it("renders the data trace as plain scatter for up to 5000 points (no WebGL needed)", () => {
    const xy = Array.from({ length: 5000 }, (_, i) => i + 1);
    const { traces } = buildTraces(points(xy, xy), null, allOn, { x: "x", y: "y" });
    expect(traces[0]?.type).toBe("scatter");
    expect(traces[0]?.mode).toBe("markers");
    expect(traces[0]?.x).toHaveLength(5000);
  });

  it("uses scattergl above 5000 points when the browser has WebGL", () => {
    setWebglSupportedForTests(true);
    const xy = Array.from({ length: 5001 }, (_, i) => i + 1);
    const { traces } = buildTraces(points(xy, xy), null, allOn, { x: "x", y: "y" });
    expect(traces[0]?.type).toBe("scattergl");
    expect(traces[0]?.x).toHaveLength(5001);
  });

  it("falls back to stride-downsampled plain scatter when WebGL is unavailable", () => {
    setWebglSupportedForTests(false);
    const xy = Array.from({ length: 20001 }, (_, i) => i + 1);
    const { traces } = buildTraces(points(xy, xy), null, allOn, { x: "x", y: "y" });
    expect(traces[0]?.type).toBe("scatter");
    const dataX = traces[0]?.x ?? [];
    const dataY = traces[0]?.y ?? [];
    expect(dataX.length).toBeLessThanOrEqual(5000);
    // Stride sampling keeps the extremes, so the plot spans the full data range.
    expect(dataX[0]).toBe(1);
    expect(dataX[dataX.length - 1]).toBe(20001);
    // x and y are sampled at the same indices: pairing is preserved.
    expect(dataY[dataY.length - 1]).toBe(20001);
  });

  it("escapes < > & in axis titles so Plotly renders them literally", () => {
    const { layout } = buildTraces(points([1, 2], [3, 4]), null, allOn, {
      x: "a<b & c",
      y: "y>d",
    });
    expect(layout.xaxis?.title?.text).toBe("a&lt;b &amp; c");
    expect(layout.yaxis?.title?.text).toBe("y&gt;d");
  });

  it("deep-merges xaxis/yaxis in mergeLayout so the theme's gridcolor survives", () => {
    const { layout } = buildTraces(points([1, 2], [3, 4]), null, allOn, { x: "t", y: "y" });
    const merged = mergeLayout(themeLayout(true), layout);
    expect(merged.xaxis?.gridcolor).toBe("#2A2E35");
    expect(merged.xaxis?.title?.text).toBe("t");
    expect(merged.yaxis?.gridcolor).toBe("#2A2E35");
    expect(merged.yaxis?.title?.text).toBe("y");
  });

  it("never sends a non-positive-valued axis to log scale", () => {
    const ok = logAxesAllowed(points([1, 2, 3], [4, 5, 6]), null);
    expect(ok).toEqual({ x: true, y: true });
    const badX = logAxesAllowed(points([-1, 2, 3], [4, 5, 6]), null);
    expect(badX).toEqual({ x: false, y: true });
    const badY = logAxesAllowed(points([1, 2, 3], [0, 5, 6]), null);
    expect(badY).toEqual({ x: true, y: false });
    // The curve extends past the data range (here to x = 0), and it is plotted on the
    // same axis — so log x must be refused even though the data points are positive.
    const withCurve = logAxesAllowed(points([1, 2, 3], [4, 5, 6]), bundle("converged"));
    expect(withCurve).toEqual({ x: false, y: true });
    const curveBadY = logAxesAllowed(
      points([1, 2, 3], [4, 5, 6]),
      bundle("converged", {
        curve: { x: [1, 2], y: [-3, 4] },
      }),
    );
    expect(curveBadY).toEqual({ x: true, y: false });
  });
});
