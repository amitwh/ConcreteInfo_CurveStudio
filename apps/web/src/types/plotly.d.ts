/**
 * Minimal typed surface of `plotly.js-dist-min` — only what the thin wrapper in
 * `components/PlotChart.tsx` uses. The bundle ships no TypeScript declarations of its
 * own and the full `@types/plotly.js` tree is deliberately not a dependency; keeping
 * this hand-written keeps the dependency set lean and drift-free. Broaden it only when
 * the wrapper genuinely needs more of the API.
 *
 * The full bundle (not `plotly.js-basic-dist-min`) is required: in plotly.js 4.x the
 * basic bundle has only scatter/bar/pie and no partial bundle ships the WebGL
 * `scattergl` trace, which the 20 000-point chart requirement needs.
 */

declare module "plotly.js-dist-min" {
  export interface PlotlyTrace {
    type?: string;
    mode?: string;
    name?: string;
    x?: ReadonlyArray<number | null>;
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
    [key: string]: unknown;
  }

  export interface PlotlyLayout {
    title?: { text: string };
    font?: { family?: string; size?: number; color?: string };
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    showlegend?: boolean;
    legend?: { orientation?: string; [key: string]: unknown };
    margin?: { t?: number; r?: number; b?: number; l?: number };
    xaxis?: {
      title?: { text: string };
      type?: "linear" | "log";
      zeroline?: boolean;
      gridcolor?: string;
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
    [key: string]: unknown;
  }

  export interface DownloadImageOptions {
    format?: "png" | "jpeg" | "webp" | "svg";
    width?: number;
    height?: number;
    scale?: number;
    filename?: string;
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
    downloadImage(root: HTMLElement, options: DownloadImageOptions): Promise<unknown>;
  };

  export default Plotly;
}
