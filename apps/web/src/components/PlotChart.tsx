/**
 * The one place Plotly is touched: a thin `Plotly.react` wrapper (no react-plotly.js).
 * The library arrives via dynamic import() on first mount, so it lives in its own chunk
 * and /login, /register and /about never download it; the component re-renders
 * declaratively from props and purges the graph on unmount so route changes never leak
 * Plotly state into a dead DOM node.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from "react";
import type { PlotlyConfig, PlotlyLayout, PlotlyTrace } from "plotly.js-dist-min";

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

export function PlotChart({
  traces,
  layout,
  ariaLabel,
  height = 360,
  outerRef,
}: PlotChartProps): ReactElement {
  const rootRef = useRef<HTMLDivElement>(null);
  const [narrow, setNarrow] = useState(false);
  const [shortViewport, setShortViewport] = useState(false);

  useEffect(() => {
    const root = rootRef.current;
    if (root === null || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      setNarrow((entries[0]?.contentRect.width ?? 0) < NARROW_LEGEND_PX);
    });
    observer.observe(root);
    return () => observer.disconnect();
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

  const fullLayout = useMemo<Partial<PlotlyLayout>>(() => {
    if (!narrow) return { ...layout, height: chartHeight };
    return {
      ...layout,
      height: chartHeight,
      legend: { orientation: "h", y: -0.18, ...layout.legend },
      margin: { b: 64, ...layout.margin },
    };
  }, [layout, chartHeight, narrow]);

  useEffect(() => {
    const root = rootRef.current;
    if (root === null) return;
    let cancelled = false;
    void loadPlotly().then((Plotly) => {
      if (!cancelled) void Plotly.react(root, traces, fullLayout, BASE_CONFIG);
    });
    return () => {
      cancelled = true;
    };
  }, [traces, fullLayout]);

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
