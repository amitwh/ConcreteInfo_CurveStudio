/**
 * The plots of one statistical result: the selected one drawn from the result's own block,
 * named by a chooser when there are several. Nothing is recomputed here — a plot's numbers
 * are the server's — and a kind this view cannot draw says so instead of an empty frame.
 */

import { useId, type ReactElement } from "react";
import type { StatPlot } from "../../api/client";
import { PlotChart } from "../../components/PlotChart";
import { plotTraces } from "./plotTraces";

export interface StatsPlotsProps {
  plots: StatPlot[];
  /** Which plot is shown; the parent holds the choice. */
  index: number;
  onSelect: (index: number) => void;
}

export function StatsPlots({ plots, index, onSelect }: StatsPlotsProps): ReactElement | null {
  const baseId = useId();
  if (plots.length === 0) return null;

  const current = Math.min(Math.max(index, 0), plots.length - 1);
  const active = plots[current] as StatPlot;
  const drawing = plotTraces(active);
  const kind = (active as { kind?: string }).kind ?? "unknown";

  const figure =
    drawing === null ? (
      <p className="hint">
        {`This plot cannot be drawn: its kind (${kind}) is not one this view draws.`}
      </p>
    ) : (
      <PlotChart traces={drawing.traces} layout={drawing.layout} ariaLabel={active.title} />
    );

  const body = (
    <>
      {figure}
      {active.note != null && <p className="hint">{active.note}</p>}
    </>
  );

  return (
    <div>
      {plots.length > 1 && (
        <div className="tabs" role="tablist" aria-label="Plots">
          {plots.map((plot, position) => (
            <button
              key={`${plot.title}-${position}`}
              type="button"
              role="tab"
              id={`${baseId}-plot-${position}`}
              aria-selected={position === current}
              aria-controls={`${baseId}-panel`}
              className={position === current ? "tab tab--active" : "tab"}
              onClick={() => onSelect(position)}
            >
              {plot.title}
            </button>
          ))}
        </div>
      )}
      {plots.length > 1 ? (
        <div
          id={`${baseId}-panel`}
          role="tabpanel"
          aria-labelledby={`${baseId}-plot-${current}`}
        >
          {body}
        </div>
      ) : (
        body
      )}
    </div>
  );
}
