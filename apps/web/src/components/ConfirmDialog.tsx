/**
 * The confirmation every destructive or far-reaching action asks first, built on Modal
 * (the compact panel). It knows nothing about the action: the caller supplies the words,
 * says whether it is a danger action, and may demand that a phrase be typed before Confirm
 * enables. While `busy` the dialog holds still (a request in flight cannot be abandoned
 * half-way); an `error` is shown inside it and keeps it open.
 *
 * Escape, the backdrop and Cancel all close it. Escape is handled here, before Modal can:
 * Modal listens for it on window, so with a sheet open underneath one keypress would close
 * both. This listener runs first (capture phase) and stops the key there.
 */

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { Modal } from "./Modal";

export interface ConfirmDialogProps {
  title: string;
  message: ReactNode;
  confirmLabel: string;
  /** Defaults to "Cancel". */
  cancelLabel?: string;
  /** The confirm button is solid red: the action destroys something. */
  danger?: boolean;
  /**
   * Text that must be typed before Confirm enables. The match ignores case and surrounding
   * spaces: the dialog guards against a slip, the server checks whatever it requires.
   */
  phrase?: string;
  /** A request is in flight: both buttons are disabled and the dialog cannot be dismissed. */
  busy?: boolean;
  /** Why the last attempt failed; shown inside the dialog. */
  error?: string | null;
  /** Called with the typed text (empty when there is no phrase). */
  onConfirm: (typed: string) => void;
  onClose: () => void;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  cancelLabel = "Cancel",
  danger = false,
  phrase,
  busy = false,
  error = null,
  onConfirm,
  onClose,
}: ConfirmDialogProps): ReactElement {
  const [typed, setTyped] = useState("");
  const titleId = useId();
  const inputId = useId();
  const busyRef = useRef(busy);
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  const matches =
    phrase === undefined || typed.trim().toLowerCase() === phrase.trim().toLowerCase();

  // Stable while `onClose` is: Modal re-runs its focus handling whenever this changes.
  const dismiss = useCallback(() => {
    if (!busyRef.current) onClose();
  }, [onClose]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      dismiss();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [dismiss]);

  return (
    <Modal title={title} labelledBy={titleId} onClose={dismiss}>
      <div className="confirm">
        <div className="confirm__message">{message}</div>
        {phrase !== undefined && (
          <div className="field">
            <label htmlFor={inputId}>
              Type <strong>{phrase}</strong> to confirm
            </label>
            <input
              id={inputId}
              type="text"
              value={typed}
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => setTyped(event.target.value)}
            />
          </div>
        )}
        {error !== null && error !== "" && (
          <p className="alert alert--error" role="alert">
            {error}
          </p>
        )}
        <div className="modal__actions">
          <button type="button" className="btn" onClick={dismiss} disabled={busy}>
            {cancelLabel}
          </button>
          <button
            type="button"
            className={danger ? "btn btn--danger-solid" : "btn btn--primary"}
            onClick={() => onConfirm(typed.trim())}
            disabled={busy || !matches}
            aria-busy={busy || undefined}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  );
}
