import { afterEach, describe, expect, it } from "vitest";
import type { components } from "../api/schema";
import {
  BAND_CONFIDENCE_FILL,
  BAND_PREDICTION_FILL,
  DATA_COLOR,
  MODEL_COLORS,
  GL_DISPLAY_MAX_POINTS,
  GL_MAX_SAMPLE_POINTS,
  SVG_SAMPLE_POINTS,
  buildTraces,
  chartSamplePoints,
  decimateMinMax,
  logAxesAllowed,
  mergeLayout,
  modelColor,
  equationLayout,
  scatterPoints,
  selectedEquation,
  setWebglSupportedForTests,
  themeLayout,
  wrapEquation,
} from "./traces";

type FitBundle = components["schemas"]["FitBundle"];
type RankedFit = components["schemas"]["RankedFit"];
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

const allOn = {
  confidence: true,
  prediction: true,
  logX: false,
  logY: false,
  top3: false,
};

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

  it("decimates the WebGL data trace to the display budget, keeping the envelope", () => {
    setWebglSupportedForTests(true);
    // Well past the display budget, with a unique spike hiding mid-block every 50k rows.
    const n = GL_DISPLAY_MAX_POINTS + 50_000;
    const xs = Array.from({ length: n }, (_, i) => i);
    const ys = xs.map((v) => (v % 50_000 === 25_000 ? 100_000 : 1));
    const { type, x: dx, y: dy } = scatterPoints(xs, ys);
    expect(type).toBe("scattergl");
    expect(dx.length).toBeLessThanOrEqual(GL_DISPLAY_MAX_POINTS + 2);
    // Every spike survived: each block's bucket maximum is drawn, not strided away.
    expect(dy.filter((v) => v === 100_000)).toHaveLength(5);
    // The drawn x-span is the full data span (the least-x and greatest-x samples stay).
    expect(Math.min(...dx)).toBe(0);
    expect(Math.max(...dx)).toBe(n - 1);
  });

  it("keeps the WebGL data trace undecimated up to the display budget", () => {
    setWebglSupportedForTests(true);
    const xy = Array.from({ length: GL_DISPLAY_MAX_POINTS }, (_, i) => i + 1);
    const { type, x, y } = scatterPoints(xy, xy);
    expect(type).toBe("scattergl");
    expect(x).toHaveLength(GL_DISPLAY_MAX_POINTS);
    expect(y).toHaveLength(GL_DISPLAY_MAX_POINTS);
  });

  it("NEVER decimates the fitted curve or the bands, however large the data trace", () => {
    setWebglSupportedForTests(true);
    // Data far above the display budget forces the Data trace through decimation.
    const n = GL_DISPLAY_MAX_POINTS + 10_000;
    const xs = Array.from({ length: n }, (_, i) => i);
    const ys = xs.map((v) => v);
    const curveX = Array.from({ length: 200 }, (_, i) => (i * (n - 1)) / 199);
    const curveY = curveX.map((v) => 2 * Math.exp(-0.0005 * v));
    const fit = bundle("converged", {
      curve: { x: curveX, y: curveY },
      confidence_band: {
        kind: "confidence",
        level: 0.95,
        x: curveX,
        fit: curveY,
        lower: curveY.map((v) => v - 0.1),
        upper: curveY.map((v) => v + 0.1),
        extrapolated: curveX.map(() => false),
        df: 38,
        note: "pointwise",
        pointwise: true,
      },
      prediction_band: null,
    });
    const { traces } = buildTraces(points(xs, ys), fit, allOn, { x: "x", y: "y" });
    const data = traces.find((t) => t.name === "Data");
    expect(data?.x?.length).toBeLessThanOrEqual(GL_DISPLAY_MAX_POINTS + 2);
    // The smoothness invariant: the curve and band are the engine's own 200-point
    // polygons, every point, verbatim — decimation touches the data trace alone.
    const curve = traces.find((t) => t.name === "Fitted curve");
    expect(curve?.x).toEqual(curveX);
    expect(curve?.y).toEqual(curveY);
    const band = traces.find((t) => t.name === "95% confidence band");
    expect(band?.x).toEqual([...curveX, ...[...curveX].reverse()]);
    expect(band?.y).toEqual([
      ...curveY.map((v) => v + 0.1),
      ...curveY.map((v) => v - 0.1).reverse(),
    ]);
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

  it("colours the single curve by the selected row's rank", () => {
    const { traces } = buildTraces(points([0, 1], [2, 1.3]), bundle("converged"), allOn, {
      x: "x",
      y: "y",
    }, { selectedPosition: 3 });
    const curve = traces.find((t) => t.name === "Fitted curve");
    expect(curve?.line?.color).toBe(modelColor(3));
  });

  it("keeps the data points neutral in both themes", () => {
    for (const dark of [false, true]) {
      const { traces } = buildTraces(points([0, 1], [2, 1.3]), null, allOn, { x: "x", y: "y" }, { dark });
      expect(traces[0]?.marker?.color).toBe(DATA_COLOR[dark ? "dark" : "light"]);
    }
  });

  it("fills the bands neutrally — never a model colour or the curve colour", () => {
    const { traces } = buildTraces(points([0, 1], [2, 1.3]), bundle("converged"), allOn, {
      x: "x",
      y: "y",
    });
    const confidence = traces.find((t) => t.name === "95% confidence band");
    const prediction = traces.find((t) => t.name === "95% prediction band");
    expect(confidence?.fillcolor).toBe(BAND_CONFIDENCE_FILL.light);
    expect(prediction?.fillcolor).toBe(BAND_PREDICTION_FILL.light);
    const palette = new Set<string>(MODEL_COLORS);
    const curveColor = String(traces.find((t) => t.name === "Fitted curve")?.line?.color);
    for (const fill of [String(confidence?.fillcolor), String(prediction?.fillcolor)]) {
      expect(palette.has(fill)).toBe(false);
      expect(fill).not.toBe(curveColor);
    }
  });

  it("show top 3 adds exactly two more curves in ranks 2 and 3's colours", () => {
    const overlay = [
      { name: "Best", position: 1, curve: { x: [0, 1], y: [2, 1.4] } },
      { name: "Second", position: 2, curve: { x: [0, 1], y: [2, 1.3] } },
      { name: "Third", position: 3, curve: { x: [0, 1], y: [2, 1.2] } },
    ];
    const off = buildTraces(points([0, 1], [2, 1.3]), bundle("converged"), allOn, {
      x: "x",
      y: "y",
    }, { overlay }).traces;
    const on = buildTraces(
      points([0, 1], [2, 1.3]),
      bundle("converged"),
      { ...allOn, top3: true },
      { x: "x", y: "y" },
      { overlay },
    ).traces;
    const curveTraces = (list: typeof on) =>
      list.filter((t) => t.mode === "lines" && t.fill === undefined);
    expect(curveTraces(off)).toHaveLength(1);
    const shown = curveTraces(on);
    expect(shown).toHaveLength(3);
    expect(shown.map((t) => t.line?.color)).toEqual([modelColor(1), modelColor(2), modelColor(3)]);
    expect(shown.map((t) => t.name)).toEqual(["Best", "Second", "Third"]);
  });

  it("turning show top 3 off returns to one curve", () => {
    const overlay = [
      { name: "Best", position: 1, curve: { x: [0, 1], y: [2, 1.4] } },
      { name: "Second", position: 2, curve: { x: [0, 1], y: [2, 1.3] } },
    ];
    const { traces } = buildTraces(points([0, 1], [2, 1.3]), bundle("converged"), allOn, {
      x: "x",
      y: "y",
    }, { overlay });
    expect(traces.filter((t) => t.mode === "lines" && t.fill === undefined)).toHaveLength(1);
    expect(traces.map((t) => t.name)).not.toContain("Second");
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

describe("chart equation caption", () => {
  const row = {
    rank: 1,
    model_id: "power",
    name: "Power",
    category: "power",
    equation_fitted: "y = 2.35*x**0.89",
    metrics: { k: 2, n: 8, r2: 0.99, adj_r2: 0.99, rmse: 0.1 },
    status: "converged",
    same_curve_as: [],
    warnings: [],
    parameters: [],
    generated: false,
    shape_parameters: 0,
    degenerate: false,
  } as unknown as RankedFit;

  it("captions the selected row's fitted equation with ^ and its name", () => {
    const equation = selectedEquation([row], "power", null);
    expect(equation).toEqual({ name: "Power", equation: "y = 2.35*x^0.89" });
    const layout = equationLayout(equation, false);
    const annotations = (layout.annotations ?? []) as Array<{ text?: string }>;
    expect(annotations[0]?.text).toContain("Power");
    expect(annotations[0]?.text).toContain("y = 2.35*x^0.89");
    expect(annotations[0]?.text).not.toContain("**");
  });

  it("falls back to the shown bundle when no row is selected", () => {
    const bundle = {
      result: { model_id: "power", equation: "y = a*x**b", equation_fitted: "y = 2.0*x**0.5" },
    } as unknown as FitBundle;
    expect(selectedEquation([], null, bundle)).toEqual({
      name: "power",
      equation: "y = 2.0*x^0.5",
    });
  });

  it("has no caption without a model, and wraps long equations", () => {
    expect(selectedEquation([], null, null)).toBeNull();
    expect(equationLayout(null, false)).toEqual({});
    const long = "y = 0.680918 + 2.37953*x + 0.0900383*x^2 + 0.0127839*x^3 + -0.000548964*x^4";
    const lines = wrapEquation(long, 40);
    expect(lines.length).toBeGreaterThan(1);
    // the wrap splits between terms and drops the separator space it broke at
    expect(lines.join("").replaceAll(" ", "")).toBe(long.replaceAll(" ", ""));
    expect(lines.every((line) => line.length <= 44)).toBe(true);
  });
});

describe("the chart's requested sample size (the WebGL path's larger fetch)", () => {
  it("is exactly the old request at or below the threshold, WebGL or not", () => {
    for (const webgl of [true, false]) {
      expect(chartSamplePoints(0, webgl)).toBe(SVG_SAMPLE_POINTS);
      expect(chartSamplePoints(20_000, webgl)).toBe(SVG_SAMPLE_POINTS);
    }
  });

  it("asks for the whole dataset above the threshold with WebGL, capped at the API maximum", () => {
    expect(chartSamplePoints(20_001, true)).toBe(20_001);
    expect(chartSamplePoints(100_000, true)).toBe(100_000);
    expect(chartSamplePoints(2_500_000, true)).toBe(GL_MAX_SAMPLE_POINTS);
  });

  it("keeps the old request without WebGL, however large or unknown the dataset", () => {
    expect(chartSamplePoints(1_000_000, false)).toBe(SVG_SAMPLE_POINTS);
    expect(chartSamplePoints(null, false)).toBe(SVG_SAMPLE_POINTS);
    expect(chartSamplePoints(undefined, false)).toBe(SVG_SAMPLE_POINTS);
  });
});

describe("min/max envelope decimation for display", () => {
  it("returns the input unchanged (as a copy) at or under the budget", () => {
    const x = [0, 1, 2, 3];
    const y = [1, 9, 2, 8];
    const out = decimateMinMax(x, y, 4);
    expect(out).toEqual({ x, y });
    expect(out.x).not.toBe(x); // a copy: the caller's arrays are never aliased
  });

  it("preserves every bucket's y-extreme exactly — a spike cannot vanish", () => {
    // 1000 points, 4 buckets of 250; a unique spike sits mid-bucket in each.
    const x: number[] = [];
    const y: number[] = [];
    for (let i = 0; i < 1000; i += 1) {
      x.push(i);
      y.push(i % 250 === 125 ? 10_000 + i : (i % 7) + (i % 3));
    }
    const { x: dx, y: dy } = decimateMinMax(x, y, 8);
    expect(dy.filter((v) => v >= 10_000)).toHaveLength(4);
    // The invariant, bucket by bucket: the output's min and max y over an x-bucket equal
    // the input's — the drawn envelope is exact, not approximate.
    const width = 1000 / 4;
    for (let bucket = 0; bucket < 4; bucket += 1) {
      const lo = bucket * width;
      const hi = lo + width;
      const inY = y.filter((_, i) => lo <= (x[i] ?? Infinity) && (x[i] ?? Infinity) < hi);
      const outY = dy.filter((_, i) => lo <= (dx[i] ?? Infinity) && (dx[i] ?? Infinity) < hi);
      expect(inY.length).toBeGreaterThan(0);
      expect(Math.min(...outY)).toBe(Math.min(...inY));
      expect(Math.max(...outY)).toBe(Math.max(...inY));
    }
  });

  it("keeps the exact x-extremes and stays within the budget", () => {
    const x = Array.from({ length: 10_001 }, (_, i) => (i * 7) % 10_003); // unsorted
    const y = x.map((v) => (v % 11) - 5);
    const { x: dx, y: dy } = decimateMinMax(x, y, 100);
    expect(dx.length).toBeLessThanOrEqual(102);
    expect(dy.length).toBe(dx.length);
    expect(Math.min(...dx)).toBe(Math.min(...x));
    expect(Math.max(...dx)).toBe(Math.max(...x));
  });

  it("skips non-finite samples (they draw no marker) and is deterministic", () => {
    const x = [0, 1, 2, 3, 4];
    const y = [1, Number.NaN, 3, Number.POSITIVE_INFINITY, 2];
    const first = decimateMinMax(x, y, 2);
    const second = decimateMinMax(x, y, 2);
    expect(first).toEqual(second);
    // One bucket over [0, 4]: its least-y (1 at x=0) and greatest-y (3 at x=2) survive;
    // the NaN and +Infinity samples never appear, and the x-extremes are kept.
    expect(first.x).toEqual([0, 2, 4]);
    expect(first.y).toEqual([1, 3, 2]);
  });

  it("emits one point when a single sample wins a bucket's min and max", () => {
    // Flat data: every bucket's min-y and max-y is its first sample (ties keep first).
    const x = [0, 1, 2, 3, 4, 5, 6, 7];
    const y = [5, 5, 5, 5, 5, 5, 5, 5];
    const { x: dx } = decimateMinMax(x, y, 2); // 1 bucket + the x-extremes
    expect(dx).toEqual([0, 7]);
  });

  it("returns nothing drawable when every sample is non-finite", () => {
    const out = decimateMinMax([1, 2, 3], [Number.NaN, Infinity, -Infinity], 2);
    expect(out).toEqual({ x: [], y: [] });
  });
});
