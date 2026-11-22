/**
 * An accessible overlay panel: Help and About open in one, so the workspace underneath
 * stays mounted (looking something up never costs the user their selection or a running
 * analysis). Escape and the backdrop close it; focus enters the panel and returns to the
 * trigger afterwards; the page behind does not scroll.
 */

import { useEffect, useRef, type ReactElement, type ReactNode } from "react";
import { X } from "lucide-react";

export interface ModalProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** "sheet" is the tall, wide panel Help and About need; "panel" the compact default. */
  variant?: "panel" | "sheet";
  labelledBy?: string;
}

export function Modal({
  title,
  onClose,
  children,
  variant = "panel",
  labelledBy,
}: ModalProps): ReactElement {
  const panelRef = useRef<HTMLDivElement>(null);
  const headingId = labelledBy ?? "modal-title";

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panelRef.current?.focus();
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      previous?.focus?.();
    };
  }, [onClose]);

  const className =
    variant === "sheet" ? "modal__panel modal__panel--sheet" : "modal__panel";
  return (
    <div className="modal" role="presentation" onMouseDown={onClose}>
      <div
        ref={panelRef}
        className={className}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal__header">
          <h2 id={headingId}>{title}</h2>
          <button type="button" className="btn btn--small btn--icon" aria-label="Close" onClick={onClose}>
            <X className="icon" aria-hidden="true" focusable="false" />
          </button>
        </header>
        <div className="modal__body">{children}</div>
      </div>
    </div>
  );
}
