import { useId, type ReactElement, type ReactNode } from "react";

export interface StatTileProps {
  label: string;
  /** The figure: one number, or several. */
  value: ReactNode;
  /** One line saying what the figure counts. */
  explanation: string;
}

/**
 * A figure with its name and what it counts. The three read as one titled group, so a
 * screen reader meets "Accounts, 128, registered accounts" as a single item.
 */
export function StatTile({ label, value, explanation }: StatTileProps): ReactElement {
  const labelId = useId();
  return (
    <div className="stat-tile" role="group" aria-labelledby={labelId}>
      <p className="stat-tile__label" id={labelId}>
        {label}
      </p>
      <p className="stat-tile__value">{value}</p>
      <p className="stat-tile__explanation">{explanation}</p>
    </div>
  );
}
