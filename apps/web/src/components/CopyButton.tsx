/** Copy text to the clipboard with visible feedback; keyboard and screen-reader friendly. */

import { useRef, useState, type ReactElement } from "react";
import { Check, Copy } from "lucide-react";

export function CopyButton({
  text,
  label,
  compact = false,
}: {
  text: string;
  label: string;
  /** Compact renders just "Copy" (e.g. inline with an equation); the full label stays
   *  the accessible name. */
  compact?: boolean;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  const onClick = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard access can be denied; say so rather than pretending it worked.
      setCopied(false);
      return;
    }
    setCopied(true);
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setCopied(false), 1500);
  };

  const Icon = copied ? Check : Copy;

  return (
    <button
      type="button"
      className="btn btn--small"
      onClick={() => void onClick()}
      {...(compact
        ? { "aria-label": copied ? `Copied ${label}` : `Copy ${label}` }
        : {})}
    >
      <Icon className="icon" aria-hidden="true" focusable="false" />
      {copied ? "Copied" : compact ? "Copy" : `Copy ${label}`}
    </button>
  );
}
