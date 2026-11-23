import type { ReactElement } from "react";
import { fmtInt } from "../format";

export interface MeterProps {
  label: string;
  used: number;
  limit: number;
  /** How to print both numbers; counts by default, bytes for storage. */
  format?: (value: number) => string;
}

/** "used of limit", a progress bar and the percentage; over 90 % is said in words. */
export function Meter({ label, used, limit, format = fmtInt }: MeterProps): ReactElement {
  const percent = limit > 0 ? (used / limit) * 100 : used > 0 ? 100 : 0;
  const shown = Math.round(percent);
  const over = shown > 90;
  const count = `${format(used)} of ${format(limit)}`;
  return (
    <div className="meter">
      <div className="meter__head">
        <span className="meter__label">{label}</span>
        <span className="meter__count">{count}</span>
      </div>
      <div
        className="meter__track"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        // Usage can pass a limit that was lowered; the bar fills, the numbers stay true.
        aria-valuemax={Math.max(limit, used, 0)}
        aria-valuenow={used}
        aria-valuetext={count}
      >
        <div
          className={over ? "meter__fill meter__fill--over" : "meter__fill"}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
      <div className="meter__foot">
        <span>{shown}%</span>
        {over && <span className="badge badge--bad">Over 90% used</span>}
      </div>
    </div>
  );
}
