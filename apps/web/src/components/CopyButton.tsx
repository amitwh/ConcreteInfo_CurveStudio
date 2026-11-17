/** Copy text to the clipboard with visible feedback; keyboard and screen-reader friendly. */

import { useRef, useState, type ReactElement } from "react";

export function CopyButton({ text, label }: { text: string; label: string }): ReactElement {
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

  return (
    <button type="button" className="btn btn--small" onClick={() => void onClick()}>
      {copied ? "Copied" : `Copy ${label}`}
    </button>
  );
}
