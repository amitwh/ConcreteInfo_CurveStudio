/**
 * Pure builders from API payloads to Plotly traces/layout. The invariant enforced here:
 * a fit whose status is not "converged" NEVER produces a curve or band trace, no matter
 * what the toggles say — the chart may show the data points and nothing else.
 */

import type { FitBundle, PointsOut, RankedFit } from "../api/client";
import type { PlotlyLayout, PlotlyTrace } from "../lib/plotly-bundle";
import { fmtEquation } from "../format";

export interface ChartToggles {
  confidence: boolean;
  prediction: boolean;
  logX: boolean;
  logY: boolean;
  /** Overlay the three best-ranked models' stored curves. */
  top3: boolean;
}

export interface AxisNames {
  x: string;
  y: string;
}

/**
 * The categorical model palette. SOURCE OF TRUTH:
 * `packages/plotting/src/curvestudio_plotting/palette.py` — the desktop app and the
 * exported figures use the same eight colours, position 1 first (the brand accent), so a
 * model is recognisable everywhere. Keep the two lists in sync.
 */
export const MODEL_COLORS = [
  "#E5461F",
  "#3B82F6",
  "#10B981",
  "#F59E0B",
  "#8B5CF6",
  "#EC4899",
  "#06B6D4",
  "#84CC16",
] as const;

/**
 * The colour of a 1-based **display position** in the ranking, never a `RankedFit.rank`
 * (ranks tie, and tied rows must stay distinct). Positions beyond eight wrap in order.
 */
export function modelColor(position: number): string {
  if (position < 1) throw new RangeError(`position is 1-based; got ${position}`);
  return MODEL_COLORS[(position - 1) % MODEL_COLORS.length] as string;
}

/** The top row is slightly heavier so it still leads in a multi-curve overlay. */
export function curveWidth(position: number): number {
  return position === 1 ? 2.2 : 1.6;
}

/** Neutral band fills per theme (no hue, so a band is never read as a model curve). */
export const BAND_CONFIDENCE_FILL = {
  light: "rgba(86,86,86,0.20)",
  dark: "rgba(200,200,200,0.24)",
} as const;

/** The prediction band is lighter than the confidence band in both themes. */
export const BAND_PREDICTION_FILL = {
  light: "rgba(140,140,140,0.12)",
  dark: "rgba(200,200,200,0.12)",
} as const;

/** Neutral data-point colour per theme (never competes with a model colour). */
export const DATA_COLOR = { light: "#4B4B4B", dark: "#B0B0B0" } as const;

/** Fitted-curve/marker colour: the cams2 brand accent (matches --accent in styles.css). */
export const ACCENT = "#E5461F";

/** One model curve in a "show top 3" overlay. */
export interface OverlayCurve {
  name: string;
  /** 1-based position in the displayed ranking (not a tied rank value). */
  position: number;
  curve: { x: number[]; y: number[] };
}

export interface TraceOptions {
  dark?: boolean;
  /** The selected row's 1-based display position, used to colour the single curve. */
  selectedPosition?: number;
  /** The three best-ranked rows' curves, drawn when `toggles.top3` is on. */
  overlay?: readonly OverlayCurve[];
}

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
  options: TraceOptions = {},
): { traces: PlotlyTrace[]; layout: Partial<PlotlyLayout> } {
  const traces: PlotlyTrace[] = [];
  const theme = options.dark === true ? "dark" : "light";
  if (points !== null) {
    const data = scatterPoints(points.x, points.y);
    traces.push({
      type: data.type,
      mode: "markers",
      name: "Data",
      x: data.x,
      y: data.y,
      marker: { color: DATA_COLOR[theme], size: 5, opacity: 0.85 },
    });
  }
  // The no-curve invariant: only a converged fit may draw a curve or bands.
  if (isConverged(bundle) && bundle !== null && bundle.curve !== null) {
    const overlay = options.overlay ?? [];
    if (toggles.top3 && overlay.length > 0) {
      // Each ranked model keeps its own colour and its own legend name; the selected
      // row's bands below still belong to the selected model only.
      for (const entry of overlay) {
        traces.push({
          type: "scatter",
          mode: "lines",
          name: entry.name,
          x: entry.curve.x,
          y: entry.curve.y,
          line: { color: modelColor(entry.position), width: curveWidth(entry.position) },
        });
      }
    } else {
      const position = options.selectedPosition ?? 1;
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Fitted curve",
        x: bundle.curve.x,
        y: bundle.curve.y,
        line: { color: modelColor(position), width: curveWidth(position) },
      });
    }
    if (toggles.confidence && bundle.confidence_band !== null) {
      traces.push(
        ...bandTraces(bundle.confidence_band, "95% confidence band", BAND_CONFIDENCE_FILL[theme]),
      );
    }
    if (toggles.prediction && bundle.prediction_band !== null) {
      traces.push(
        ...bandTraces(bundle.prediction_band, "95% prediction band", BAND_PREDICTION_FILL[theme]),
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

// -- the selected model's equation, pinned to the chart -----------------------------------

export interface ChartEquation {
  name: string;
  equation: string;
}

/**
 * The equation the chart should caption: the selected ranking row's fitted equation when
 * there is one, else the shown bundle's. The engine stores `**`; it is displayed as `^`
 * (the same presentation rule as the ranking, details and reports).
 */
export function selectedEquation(
  rows: readonly RankedFit[],
  selectedRowId: string | null,
  bundle: FitBundle | null,
): ChartEquation | null {
  if (selectedRowId !== null) {
    const row = rows.find((candidate) => candidate.model_id === selectedRowId);
    const fitted = row?.equation_fitted ?? null;
    if (row !== undefined && fitted !== null && fitted !== "") {
      return { name: fmtEquation(row.name), equation: fmtEquation(fitted) };
    }
  }
  if (bundle === null) return null;
  // a payload may omit both fields entirely (a non-converged fit), so guard undefined too
  const fitted = bundle.result.equation_fitted ?? bundle.result.equation;
  if (fitted === null || fitted === undefined || fitted === "") return null;
  const row = rows.find((candidate) => candidate.model_id === bundle.result.model_id);
  return { name: fmtEquation(row?.name ?? bundle.result.model_id), equation: fmtEquation(fitted) };
}

/** Soft-wrap a long equation at its ` + ` / ` - ` terms, so the caption stays on the chart. */
export function wrapEquation(text: string, width = 64): string[] {
  const parts = text.split(/(?= [+-] )/);
  const lines: string[] = [];
  let current = "";
  for (const part of parts) {
    if (current !== "" && current.length + part.length > width) {
      lines.push(current);
      current = part.trimStart();
    } else {
      current += part;
    }
  }
  if (current !== "") lines.push(current);
  return lines;
}

/** A Plotly layout fragment carrying the equation caption (empty when there is none). */
export function equationLayout(
  equation: ChartEquation | null,
  dark: boolean,
): Partial<PlotlyLayout> {
  if (equation === null) return {};
  const lines = wrapEquation(equation.equation);
  return {
    annotations: [
      {
        x: 0.012,
        y: 0.985,
        xref: "paper",
        yref: "paper",
        xanchor: "left",
        yanchor: "top",
        align: "left",
        showarrow: false,
        text: `<b>${escapeHtml(equation.name)}</b><br>${lines.map(escapeHtml).join("<br>")}`,
        bgcolor: dark ? "rgba(28,25,23,0.88)" : "rgba(255,255,255,0.92)",
        bordercolor: dark ? "#4a4643" : "#e3e3e3",
        borderwidth: 1,
        borderpad: 6,
        font: {
          size: 11,
          color: dark ? "#f1f1f1" : "#0d0b09",
          family: "JetBrains Mono Variable, JetBrains Mono, ui-monospace, monospace",
        },
      },
    ],
  };
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
