/**
 * The one prominent switch at the top of the analysis area: Auto-fit (default) | Manual
 * fit. Two toggle buttons in a group, so screen readers announce the pressed state; a
 * muted status line says what Auto-fit is doing even while Manual fit is showing.
 */

import type { ReactElement } from "react";
import type { WorkspaceMode } from "./mode";

export interface ModeSwitchProps {
  mode: WorkspaceMode;
  onModeChange: (mode: WorkspaceMode) => void;
  /** Short, always-current Auto-fit summary (e.g. "running…", "done"). */
  autofitStatus: string;
}

const MODES: Array<{ value: WorkspaceMode; label: string; hint: string }> = [
  { value: "auto", label: "Auto-fit", hint: "every model is tried and ranked" },
  { value: "manual", label: "Manual fit", hint: "you pick the model" },
];

export function ModeSwitch({ mode, onModeChange, autofitStatus }: ModeSwitchProps): ReactElement {
  return (
    <div className="mode-switch">
      <div className="mode-switch__group" role="group" aria-label="Analysis mode">
        {MODES.map((entry) => (
          <button
            key={entry.value}
            type="button"
            aria-pressed={mode === entry.value}
            className={`mode-switch__mode${mode === entry.value ? " mode-switch__mode--active" : ""}`}
            onClick={() => onModeChange(entry.value)}
            title={entry.hint}
          >
            {entry.label}
          </button>
        ))}
      </div>
      <span className="muted mode-switch__status">Auto-fit: {autofitStatus}</span>
    </div>
  );
}
