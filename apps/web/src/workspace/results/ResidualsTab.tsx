/**
 * Residual diagnostics as plots: residual vs fitted, residual vs x, a histogram of the
 * weighted residuals (drawn from the pre-binned edges/counts the API sends) and a Q-Q
 * plot when the bundle carries the quantiles.
 */

import type { ReactElement } from "react";
import { PlotChart } from "../../components/PlotChart";
import type { FitBundle } from "../../api/client";
import { ACCENT, escapePlotlyText, scatterPoints } from "../traces";

function Scatter({
  x,
  y,
  xLabel,
  yLabel,
  title,
}: {
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  title: string;
}): ReactElement {
  const data = scatterPoints(x, y);
  return (
    <figure className="residual-plot">
      <figcaption>{title}</figcaption>
      <PlotChart
        ariaLabel={title}
        height={220}
        traces={[
          {
            type: data.type,
            mode: "markers",
            name: "residuals",
            x: data.x,
            y: data.y,
            marker: { color: ACCENT, size: 4, opacity: 0.8 },
          },
        ]}
        layout={{
          showlegend: false,
          margin: { t: 8, r: 8, b: 36, l: 44 },
          xaxis: { title: { text: escapePlotlyText(xLabel) } },
          yaxis: { title: { text: escapePlotlyText(yLabel) } },
        }}
      />
    </figure>
  );
}

export function ResidualsTab({ bundle }: { bundle: FitBundle }): ReactElement {
  const residuals = bundle.summary?.residuals ?? null;
  if (residuals === null) {
    return (
      <p className="alert alert--warning">
        Residual diagnostics are not available
        {bundle.summary_reason !== null ? `: ${bundle.summary_reason}` : "."}
      </p>
    );
  }
  const hasQQ = residuals.qq_theoretical.length > 0 && residuals.qq_ordered.length > 0;
  const qq = hasQQ ? scatterPoints(residuals.qq_theoretical, residuals.qq_ordered) : null;
  // Bars from pre-binned counts: one bar per interval, centred on it.
  const widths = residuals.hist_edges
    .slice(0, -1)
    .map((edge, i) => residuals.hist_edges[i + 1]! - edge);
  const centres = residuals.hist_edges
    .slice(0, -1)
    .map((edge, i) => edge + widths[i]! / 2);

  return (
    <div className="residual-grid">
      <Scatter
        x={residuals.fitted}
        y={residuals.residuals}
        xLabel="fitted"
        yLabel="residual"
        title="Residuals vs fitted"
      />
      <Scatter
        x={residuals.x}
        y={residuals.residuals}
        xLabel={bundle.data.x_name}
        yLabel="residual"
        title="Residuals vs x"
      />
      <figure className="residual-plot">
        <figcaption>Histogram of weighted residuals</figcaption>
        <PlotChart
          ariaLabel="Histogram of weighted residuals"
          height={220}
          traces={[
            {
              type: "bar",
              name: "count",
              x: centres,
              y: residuals.hist_counts,
              width: widths,
              marker: { color: ACCENT, opacity: 0.7 },
            },
          ]}
          layout={{
            showlegend: false,
            margin: { t: 8, r: 8, b: 36, l: 44 },
            xaxis: { title: { text: "weighted residual" } },
            yaxis: { title: { text: "count" } },
          }}
        />
      </figure>
      {qq !== null && (
        <figure className="residual-plot">
          <figcaption>Normal Q-Q</figcaption>
          <PlotChart
            ariaLabel="Normal Q-Q plot of the residuals"
            height={220}
            traces={[
              {
                type: qq.type,
                mode: "markers",
                name: "quantiles",
                x: qq.x,
                y: qq.y,
                marker: { color: ACCENT, size: 4 },
              },
            ]}
            layout={{
              showlegend: false,
              margin: { t: 8, r: 8, b: 36, l: 44 },
              xaxis: { title: { text: "theoretical quantiles" } },
              yaxis: { title: { text: "ordered residuals" } },
            }}
          />
        </figure>
      )}
    </div>
  );
}
