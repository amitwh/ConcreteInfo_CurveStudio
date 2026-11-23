/**
 * The last line of defence. A component that throws while rendering lands here instead of
 * taking the whole app down with it: the error goes to the reporter as a fatal `ui` event
 * and the user gets a screen in the app's own styles.
 *
 * Reload is what recovers the app, and so is the link to the help page: it is a plain
 * anchor on purpose, so it is a full page load. The box asks what the user was doing; the
 * note travels as an event of its own, quoting the reference on the screen, so it never
 * changes the error's count. With reports off there is nothing to send and no reference to
 * quote, so the screen says where reports are switched on instead.
 *
 * The screen follows consent as it stands, not as it stood when the error arrived: a crash
 * at startup can land while the session (and with it the account's choice) is still being
 * restored. Nothing is reported while reports are off; the moment they are on, the error is.
 * It also applies the theme itself, since a crash with the first render means the app's own
 * provider never got to.
 */

import {
  Component,
  useEffect,
  useId,
  useState,
  useSyncExternalStore,
  type ErrorInfo,
  type FormEvent,
  type ReactElement,
  type ReactNode,
} from "react";
import {
  errorReportsEnabled,
  reportUiError,
  sendDescription,
  subscribeConsent,
} from "../lib/errorReporter";
import { preferredTheme } from "./ThemeToggle";

export interface ErrorBoundaryProps {
  children: ReactNode;
}

/** The error React caught, with the component stack it names. */
interface Caught {
  error: unknown;
  componentStack: string | null;
}

interface ErrorBoundaryState {
  failed: boolean;
  /** Set by `componentDidCatch`, which React calls once the fallback is already on screen. */
  caught: Caught | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false, caught: null };

  static getDerivedStateFromError(): Partial<ErrorBoundaryState> {
    return { failed: true };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    this.setState({ caught: { error, componentStack: info.componentStack ?? null } });
  }

  render(): ReactNode {
    const { failed, caught } = this.state;
    return failed ? <RecoveryScreen caught={caught} /> : this.props.children;
  }
}

/** The server's limit on a note, enforced as the user types. */
const NOTE_MAX_LENGTH = 1_000;

function RecoveryScreen({ caught }: { caught: Caught | null }): ReactElement {
  const reportsOn = useSyncExternalStore(subscribeConsent, errorReportsEnabled);
  const [reference, setReference] = useState("");
  const [note, setNote] = useState("");
  const [outcome, setOutcome] = useState<"idle" | "sending" | "sent" | "refused">("idle");
  const headingId = useId();
  const noteId = useId();

  // A crash with the first render means the app's ThemeProvider never ran, and so never put
  // the theme on <html>: the screen would come out light for a dark-mode visitor. Put it
  // there the same way, unless a theme is already in place. Should even this fail (storage
  // the browser refuses), the stylesheet's own light theme shows, and the screen still does.
  useEffect(() => {
    if (document.documentElement.dataset.theme !== undefined) return;
    try {
      document.documentElement.dataset.theme = preferredTheme();
    } catch {
      // nothing more can be done for the theme here; the screen must not fail over it
    }
  }, []);

  // The report goes out the first moment reports are on for this browser. Until a reference
  // comes back nothing was reported, so a change of consent asks again; the reporter itself
  // drops a repeat, so the error is still counted once.
  useEffect(() => {
    if (!reportsOn || caught === null || reference !== "") return;
    let superseded = false;
    void reportUiError(caught.error, { componentStack: caught.componentStack }).then((id) => {
      if (!superseded && id !== "") setReference(id);
    });
    return () => {
      superseded = true;
    };
  }, [reportsOn, caught, reference]);

  // The note quotes the reference; until the fingerprint is in, there is nothing to quote.
  const canSend = reference !== "" && note.trim() !== "";

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (!canSend || outcome === "sending") return;
    setOutcome("sending");
    void sendDescription(reference, note).then((sent) => setOutcome(sent ? "sent" : "refused"));
  };

  return (
    <main className="recovery">
      <section className="card recovery__card" aria-labelledby={headingId}>
        <h1 id={headingId}>Something went wrong</h1>
        <p>
          CurveStudio hit an error it could not recover from, so this screen replaced the
          app; your saved work (your datasets and fits) is safe.
        </p>
        {reportsOn && reference !== "" && (
          <p className="recovery__reference">
            <span>Reference</span>
            <code>{reference}</code>
          </p>
        )}
        <div className="btn-row recovery__actions">
          <button type="button" className="btn btn--primary" onClick={() => location.reload()}>
            Reload
          </button>
          <a className="btn" href="/help">
            Help and documentation
          </a>
        </div>
        {reportsOn ? (
          <form className="recovery__note" onSubmit={submit}>
            <div className="field">
              <label htmlFor={noteId}>What were you doing?</label>
              <textarea
                id={noteId}
                rows={4}
                maxLength={NOTE_MAX_LENGTH}
                value={note}
                disabled={outcome === "sent"}
                onChange={(event) => setNote(event.target.value)}
              />
            </div>
            {outcome === "sent" ? (
              <p className="alert alert--success" role="status">
                Thanks. Your note was sent with the reference above.
              </p>
            ) : (
              <>
                <button
                  type="submit"
                  className="btn"
                  disabled={!canSend || outcome === "sending"}
                >
                  {outcome === "sending" ? "Sending…" : "Send"}
                </button>
                {outcome === "refused" && (
                  <p className="alert alert--error" role="alert">
                    The note could not be sent. You can try again, or reload without it.
                  </p>
                )}
              </>
            )}
          </form>
        ) : (
          <p className="recovery__off">
            Error reports are off for this browser or your account: to turn them on, use the
            “Send error reports” switch on the About page, or Profile → Privacy if you are
            signed in.
          </p>
        )}
      </section>
    </main>
  );
}
