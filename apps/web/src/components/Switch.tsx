/** An on/off switch named by the text beside it. */

import { useId, type ReactElement } from "react";

export function Switch({
  label,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}): ReactElement {
  const labelId = useId();
  return (
    <div className="switch-row">
      <button
        type="button"
        role="switch"
        className="switch"
        aria-checked={checked}
        aria-labelledby={labelId}
        disabled={disabled}
        onClick={onChange}
      >
        <span className="switch__knob" aria-hidden="true" />
      </button>
      <span id={labelId} className="switch-row__label">
        {label}
      </span>
    </div>
  );
}
