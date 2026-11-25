/**
 * Minimal typed surface of the module-built Plotly — only what the thin wrapper in
 * `components/PlotChart.tsx` (via `lib/plotly-bundle.ts`) uses. The `plotly.js` package
 * ships TypeScript declarations only for its full-bundle entry (`lib/index.d.ts`), not
 * for the `lib/core` partial-bundle route; keeping this hand-written keeps the surface
 * lean and drift-free. Broaden it only when the wrapper genuinely needs more of the API.
 */

declare module "plotly.js/lib/core" {
  export interface PlotlyTrace {
    type?: string;
    mode?: string;
    name?: string;
    /** Numbers, or the strings a date or category axis takes (YYYY-MM-DD for a date axis). */
    x?: ReadonlyArray<number | null | string>;
    y?: ReadonlyArray<number | null>;
    width?: ReadonlyArray<number>;
    text?: string;
    marker?: { color?: string; size?: number; opacity?: number; line?: { width?: number } };
    line?: { color?: string; width?: number; dash?: string; shape?: string };
    fill?: string;
    fillcolor?: string;
    opacity?: number;
    hoverinfo?: string;
    showlegend?: boolean;
    /** Precomputed box statistics: a box drawn without the raw values. */
    boxpoints?: string | false;
    q1?: number;
    median?: number;
    q3?: number;
    lowerfence?: number;
    upperfence?: number;
    [key: string]: unknown;
  }

  export interface PlotlyAnnotation {
    text: string;
    xref?: string;
    yref?: string;
    x?: number;
    y?: number;
    xanchor?: string;
    yanchor?: string;
    showarrow?: boolean;
    font?: { size?: number; color?: string; family?: string; [key: string]: unknown };
    [key: string]: unknown;
  }

  export interface PlotlyLayout {
    title?: { text: string };
    annotations?: PlotlyAnnotation[];
    font?: { family?: string; size?: number; color?: string };
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    showlegend?: boolean;
    legend?: { orientation?: string; [key: string]: unknown };
    margin?: { t?: number; r?: number; b?: number; l?: number };
    xaxis?: {
      title?: { text: string };
      type?: "linear" | "log" | "date" | "category";
      zeroline?: boolean;
      gridcolor?: string;
      /** Tick positions, and the text shown at them (a category axis's labels). */
      tickvals?: ReadonlyArray<number | string>;
      ticktext?: ReadonlyArray<string>;
      [key: string]: unknown;
    };
    yaxis?: {
      title?: { text: string };
      type?: "linear" | "log";
      zeroline?: boolean;
      gridcolor?: string;
      [key: string]: unknown;
    };
    barmode?: string;
    [key: string]: unknown;
  }

  export interface PlotlyConfig {
    responsive?: boolean;
    displaylogo?: boolean;
    staticPlot?: boolean;
    displayModeBar?: boolean | "hover" | "static";
    [key: string]: unknown;
  }

  export interface DownloadImageOptions {
    format?: "png" | "jpeg" | "webp" | "svg";
    width?: number;
    height?: number;
    scale?: number;
    filename?: string;
  }

  /** A trace or component module accepted by `register` (the structural minimum). */
  export interface PlotlyModule {
    moduleType: string;
    name: string;
  }

  const Plotly: {
    react(
      root: HTMLElement,
      data: PlotlyTrace[],
      layout?: Partial<PlotlyLayout>,
      config?: Partial<PlotlyConfig>,
    ): Promise<unknown>;
    newPlot(
      root: HTMLElement,
      data: PlotlyTrace[],
      layout?: Partial<PlotlyLayout>,
      config?: Partial<PlotlyConfig>,
    ): Promise<unknown>;
    purge(root: HTMLElement): void;
    /** Grow the partial bundle by the given trace/component modules. */
    register(modules: ReadonlyArray<PlotlyModule>): void;
    downloadImage(root: HTMLElement, options: DownloadImageOptions): Promise<unknown>;
    /** Serialise the current graph to a data URL (PNG by default). */
    toImage(root: HTMLElement, options?: DownloadImageOptions): Promise<string>;
    Plots: { resize(root: HTMLElement): void };
  };

  export default Plotly;
}

/** The trace modules `lib/plotly-bundle.ts` registers; each re-exports its src module. */
declare module "plotly.js/lib/bar" {
  import type { PlotlyModule } from "plotly.js/lib/core";
  const bar: PlotlyModule;
  export default bar;
}

declare module "plotly.js/lib/box" {
  import type { PlotlyModule } from "plotly.js/lib/core";
  const box: PlotlyModule;
  export default box;
}

declare module "plotly.js/lib/scattergl" {
  import type { PlotlyModule } from "plotly.js/lib/core";
  const scattergl: PlotlyModule;
  export default scattergl;
}
