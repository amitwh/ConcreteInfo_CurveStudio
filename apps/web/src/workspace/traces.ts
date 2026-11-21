/**
 * Pure builders from API payloads to Plotly traces/layout. The invariant enforced here:
 * a fit whose status is not "converged" NEVER produces a curve or band trace, no matter
 * what the toggles say — the chart may show the data points and nothing else.
 */

import type { FitBundle, PointsOut } from "../api/client";
import type { PlotlyLayout, PlotlyTrace } from "plotly.js-dist-min";

export interface ChartToggles {
  confidence: boolean;
  prediction: boolean;
  logX: boolean;
  logY: boolean;
}

export interface AxisNames {
  x: string;
  y: string;
}

/** Fitted-curve/marker colour: the cams2 brand accent (matches --accent in styles.css). */
export const ACCENT = "#E5461F";

/** Above this many points the SVG `scatter` trace stops being fluid; below it there is
 *  no reason to spin up a WebGL context at all (headless and software-rendered browsers
 *  included). */
export const SCATTER_MAX_SVG_POINTS = 5000;

let webglCache: boolean | null = null;

/** Whether this browser can create a WebGL context (probed once, cached). */
export function webglSupported(): boolean {
  if (webglCache === null) {
    try {
      const canvas = document.createElement("canvas");
      webglCache =
        canvas.getContext("webgl2") !== null ||
        canvas.getContext("webgl") !== null ||
        canvas.getContext("experimental-webgl") !== null;
    } catch {
      webglCache = false;
    }
  }
  return webglCache;
}

/** Test-only: force the cached WebGL verdict (null restores the real probe). */
export function setWebglSupportedForTests(value: boolean | null): void {
  webglCache = value;
}

/** `scattergl` only pays off above SCATTER_MAX_SVG_POINTS points AND with WebGL. */
export function scatterTraceType(count: number): "scatter" | "scattergl" {
  return count > SCATTER_MAX_SVG_POINTS && webglSupported() ? "scattergl" : "scatter";
}

/** Evenly strided sample that always keeps the first and last element. */
export function downsampleStride<T>(values: readonly T[], max: number): T[] {
  if (values.length <= max) return [...values];
  const stride = (values.length - 1) / (max - 1);
  const sampled: T[] = [];
  for (let i = 0; i < max; i++) {
    sampled.push(values[Math.round(i * stride)] as T);
  }
  return sampled;
}

/** The (x, y) pair for a markers trace, downsampled in lockstep when WebGL is off. */
export function scatterPoints(
  x: readonly number[],
  y: readonly number[],
): { type: "scatter" | "scattergl"; x: number[]; y: number[] } {
  const type = scatterTraceType(x.length);
  if (type === "scatter" && x.length > SCATTER_MAX_SVG_POINTS) {
    return {
      type,
      x: downsampleStride(x, SCATTER_MAX_SVG_POINTS),
      y: downsampleStride(y, SCATTER_MAX_SVG_POINTS),
    };
  }
  return { type, x: [...x], y: [...y] };
}

/** Plotly parses pseudo-HTML in titles/names; escaping keeps user text literal. */
export function escapePlotlyText(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function isConverged(bundle: FitBundle | null): boolean {
  return bundle?.result.status === "converged";
}

function allPositive(values: readonly number[]): boolean {
  return values.every((v) => Number.isFinite(v) && v > 0);
}

/** Whether log axes are legal for this data (every plotted value must be positive). */
export function logAxesAllowed(
  points: PointsOut | null,
  bundle: FitBundle | null,
): { x: boolean; y: boolean } {
  const xs = [...(points?.x ?? [])];
  const ys = [...(points?.y ?? [])];
  if (isConverged(bundle)) {
    if (bundle?.curve !== null) {
      xs.push(...(bundle?.curve?.x ?? []));
      ys.push(...(bundle?.curve?.y ?? []));
    }
    for (const band of [bundle?.confidence_band, bundle?.prediction_band]) {
      if (band !== null && band !== undefined) {
        ys.push(...band.lower, ...band.upper);
      }
    }
  }
  return { x: allPositive(xs), y: allPositive(ys) };
}

function bandTraces(
  band: NonNullable<FitBundle["confidence_band"]>,
  label: string,
  color: string,
): PlotlyTrace[] {
  const upper = band.x.map((x, i): [number, number | null] => [x, band.upper[i] ?? null]);
  const lowerReversed = [...band.x]
    .map((x, i): [number, number | null] => [x, band.lower[i] ?? null])
    .reverse();
  const polygon = [...upper, ...lowerReversed];
  return [
    {
      type: "scatter",
      mode: "lines",
      name: label,
      x: polygon.map((p) => p[0]),
      y: polygon.map((p) => p[1]),
      line: { width: 0 },
      fill: "toself",
      fillcolor: color,
      hoverinfo: "skip",
      showlegend: true,
    },
  ];
}

/**
 * Build the chart's traces. Data points use WebGL `scattergl` only above
 * SCATTER_MAX_SVG_POINTS points and only when the browser has WebGL — otherwise a plain
 * (and, if needed, stride-downsampled) SVG `scatter` trace, so a software-rendered or
 * headless browser never sees "WebGL is not supported". The curve and bands are plain
 * SVG lines (200/2×200 points).
 */
export function buildTraces(
  points: PointsOut | null,
  bundle: FitBundle | null,
  toggles: ChartToggles,
  axisNames: AxisNames,
): { traces: PlotlyTrace[]; layout: Partial<PlotlyLayout> } {
  const traces: PlotlyTrace[] = [];
  if (points !== null) {
    const data = scatterPoints(points.x, points.y);
    traces.push({
      type: data.type,
      mode: "markers",
      name: "Data",
      x: data.x,
      y: data.y,
      marker: { color: "#3A3F47", size: 5, opacity: 0.85 },
    });
  }
  // The no-curve invariant: only a converged fit may draw a curve or bands.
  if (isConverged(bundle) && bundle !== null && bundle.curve !== null) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Fitted curve",
      x: bundle.curve.x,
      y: bundle.curve.y,
      line: { color: ACCENT, width: 2 },
    });
    if (toggles.confidence && bundle.confidence_band !== null) {
      traces.push(
        ...bandTraces(
          bundle.confidence_band,
          "95% confidence band",
          "rgba(15,108,189,0.15)",
        ),
      );
    }
    if (toggles.prediction && bundle.prediction_band !== null) {
      traces.push(
        ...bandTraces(
          bundle.prediction_band,
          "95% prediction band",
          "rgba(15,108,189,0.07)",
        ),
      );
    }
  }
  const logAllowed = logAxesAllowed(points, bundle);
  const layout: Partial<PlotlyLayout> = {
    margin: { t: 8, r: 8, b: 40, l: 48 },
    xaxis: {
      title: { text: escapePlotlyText(axisNames.x) },
      type: toggles.logX && logAllowed.x ? "log" : "linear",
    },
    yaxis: {
      title: { text: escapePlotlyText(axisNames.y) },
      type: toggles.logY && logAllowed.y ? "log" : "linear",
    },
  };
  return { traces, layout };
}

/** Layout colours/typography that follow the active theme (Plotly cannot read CSS
 *  variables): the UI font stack, transparent backgrounds, and gridline/axis text
 *  colours matching the dark prop. */
export function themeLayout(dark: boolean): Partial<PlotlyLayout> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: "Inter Variable, Inter, system-ui, sans-serif",
      color: dark ? "#C9CDD3" : "#1F2328",
    },
    xaxis: { gridcolor: dark ? "#2A2E35" : "#E4E7EB" },
    yaxis: { gridcolor: dark ? "#2A2E35" : "#E4E7EB" },
  };
}

/**
 * Merge layout fragments. Top-level keys follow "later wins", but xaxis/yaxis are merged
 * key-by-key so a theme fragment's gridcolor survives a fragment that only sets the axis
 * title or scale (and vice versa).
 */
export function mergeLayout(...parts: Array<Partial<PlotlyLayout>>): Partial<PlotlyLayout> {
  const merged: Record<string, unknown> = {};
  for (const part of parts) {
    for (const [key, value] of Object.entries(part)) {
      if (
        (key === "xaxis" || key === "yaxis") &&
        typeof value === "object" &&
        value !== null &&
        !Array.isArray(value) &&
        typeof merged[key] === "object" &&
        merged[key] !== null
      ) {
        merged[key] = { ...(merged[key] as Record<string, unknown>), ...(value as Record<string, unknown>) };
      } else {
        merged[key] = value;
      }
    }
  }
  return merged as Partial<PlotlyLayout>;
}
