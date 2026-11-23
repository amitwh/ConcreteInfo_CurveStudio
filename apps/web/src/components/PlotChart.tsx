/**
 * The one place Plotly is touched: a thin `Plotly.react` wrapper (no react-plotly.js).
 * The library arrives via dynamic import() on first mount, so it lives in its own chunk
 * and /login, /register and /about never download it; the component re-renders
 * declaratively from props and purges the graph on unmount so route changes never leak
 * Plotly state into a dead DOM node.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from "react";
import type {
  PlotlyAnnotation,
  PlotlyConfig,
  PlotlyLayout,
  PlotlyTrace,
} from "plotly.js-dist-min";
import { saveBlob } from "../api/client";

type PlotlyLib = (typeof import("plotly.js-dist-min"))["default"];

let plotlyPromise: Promise<PlotlyLib> | null = null;

/** First chart mount fetches Plotly once; later mounts (and PNG export) reuse it. */
function loadPlotly(): Promise<PlotlyLib> {
  plotlyPromise ??= import("plotly.js-dist-min").then((mod) => mod.default);
  return plotlyPromise;
}

export interface PlotChartProps {
  traces: PlotlyTrace[];
  layout: Partial<PlotlyLayout>;
  ariaLabel: string;
  dark?: boolean;
  height?: number;
  /** Receives the Plotly root element (for downloadImage) without breaking encapsulation. */
  outerRef?: RefObject<HTMLDivElement | null>;
}

const BASE_CONFIG: Partial<PlotlyConfig> = {
  responsive: true,
  displaylogo: false,
};

/** Below this container width the legend moves below the plot (horizontal), where it
 *  cannot overlap the drawing area. CSS cannot reposition Plotly's legend, so the
 *  wrapper watches the container width and re-renders. */
const NARROW_LEGEND_PX = 640;

/** Below this viewport width the chart shortens so it does not crowd the page. */
const SHORT_VIEWPORT_PX = 800;
const SHORT_HEIGHT = 300;

/** Faint attribution in the bottom-right corner of every chart. It lives in the live
 *  layout, so `Plotly.downloadImage` (which serialises that same graph) includes it in
 *  the exported PNG for free. */
export const CHART_WATERMARK = "ConcreteInfo CurveStudio";

export function PlotChart({
  traces,
  layout,
  ariaLabel,
  dark = false,
  height = 360,
  outerRef,
}: PlotChartProps): ReactElement {
  const rootRef = useRef<HTMLDivElement>(null);
  const libRef = useRef<PlotlyLib | null>(null);
  const [narrow, setNarrow] = useState(false);
  const [shortViewport, setShortViewport] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadPlotly().then((Plotly) => {
      if (!cancelled) libRef.current = Plotly;
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const root = rootRef.current;
    if (root === null || typeof ResizeObserver === "undefined") return;
    let frame = 0;
    const observer = new ResizeObserver((entries) => {
      setNarrow((entries[0]?.contentRect.width ?? 0) < NARROW_LEGEND_PX);
      if (typeof requestAnimationFrame !== "function") return;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const lib = libRef.current;
        if (lib !== null) lib.Plots.resize(root);
      });
    });
    observer.observe(root);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(`(max-width: ${SHORT_VIEWPORT_PX}px)`);
    const update = (): void => setShortViewport(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  const chartHeight = shortViewport ? Math.min(height, SHORT_HEIGHT) : height;

  /** The modebar is redundant next to the app's own "Download PNG" button, and on a
   *  phone it is drawn over the plotted area, so it is hidden there. The wide case
   *  names "hover" explicitly: Plotly reads an `undefined` value as false, which would
   *  also remove the modebar on desktop and would not restore it after a resize. */
  const plotConfig = useMemo<Partial<PlotlyConfig>>(
    () => ({ ...BASE_CONFIG, displayModeBar: narrow ? false : "hover" }),
    [narrow],
  );

  const fullLayout = useMemo<Partial<PlotlyLayout>>(() => {
    const base: Partial<PlotlyLayout> = narrow
      ? {
          ...layout,
          height: chartHeight,
          legend: { orientation: "h", y: -0.18, ...layout.legend },
          margin: { b: 64, ...layout.margin },
        }
      : { ...layout, height: chartHeight };
    const existing = layout.annotations ?? [];
    const watermark: PlotlyAnnotation = {
      text: CHART_WATERMARK,
      xref: "paper",
      yref: "paper",
      x: 0.995,
      y: 0.012,
      xanchor: "right",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 10, color: dark ? "rgba(241,241,241,0.22)" : "rgba(70,70,70,0.20)" },
    };
    return {
      ...base,
      annotations: existing.some((annotation) => annotation.text === CHART_WATERMARK)
        ? existing
        : [...existing, watermark],
    };
  }, [layout, chartHeight, narrow, dark]);

  useEffect(() => {
    const root = rootRef.current;
    if (root === null) return;
    let cancelled = false;
    void loadPlotly().then((Plotly) => {
      if (!cancelled) void Plotly.react(root, traces, fullLayout, plotConfig);
    });
    return () => {
      cancelled = true;
    };
  }, [traces, fullLayout, plotConfig]);

  useEffect(() => {
    const root = rootRef.current;
    return () => {
      if (root === null) return;
      void loadPlotly().then((Plotly) => {
        Plotly.purge(root);
      });
      if (outerRef !== undefined) outerRef.current = null;
    };
  }, [outerRef]);

  return (
    // No role="img" on the Plotly root: it would hide the modebar buttons from the
    // accessibility tree. The labelled <figure> names the chart instead.
    <figure
      className="plot-figure"
      aria-label={ariaLabel}
      style={{ width: "100%", height: `${chartHeight}px` }}
    >
      <div
        ref={(element) => {
          rootRef.current = element;
          if (outerRef !== undefined) outerRef.current = element;
        }}
        style={{ width: "100%", height: "100%" }}
      />
    </figure>
  );
}

/** Download the current graph as a PNG (bound to a button next to the chart). */
export function downloadPlotPng(root: HTMLElement | null, filename: string): void {
  if (root === null) return;
  void loadPlotly().then((Plotly) => {
    void Plotly.downloadImage(root, { format: "png", filename, scale: 2 });
  });
}

/** Download the current graph as an SVG document, named like the PNG but .svg: Plotly
 *  serialises the live graph to a data URL and the bytes are saved as a blob download. */
export function downloadPlotSvg(root: HTMLElement | null, filename: string): void {
  if (root === null) return;
  void loadPlotly().then(async (Plotly) => {
    const dataUrl = await Plotly.toImage(root, { format: "svg", scale: 2 });
    const bytes = Uint8Array.from(atob(dataUrl.slice(dataUrl.indexOf(",") + 1)), (char) =>
      char.charCodeAt(0),
    );
    saveBlob(new Blob([bytes], { type: "image/svg+xml" }), `${filename}.svg`);
  });
}

/**
 * Serialise the current graph to a PNG data URL for the PDF report. A root that is not
 * mounted (or a serialisation failure) yields null: the report is generated without the
 * chart rather than failing outright.
 */
export async function exportPlotPngDataUrl(
  root: HTMLElement | null,
  options: { width?: number; height?: number; scale?: number } = {},
): Promise<string | null> {
  if (root === null) return null;
  const { width = 1000, height = 620, scale = 2 } = options;
  try {
    const Plotly = await loadPlotly();
    const url = await Plotly.toImage(root, { format: "png", width, height, scale });
    return typeof url === "string" && url !== "" ? url : null;
  } catch {
    return null;
  }
}
