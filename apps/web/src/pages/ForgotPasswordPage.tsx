/**
 * /forgot-password — ask for a reset link.
 *
 * The API always answers 202 so it never reveals whether the address exists; the page
 * mirrors that by showing the same generic success panel either way. Only a network
 * failure (status 0) is surfaced as an error, because the request never landed.
 */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { errorText } from "../components/States";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function ForgotPasswordPage(): ReactElement {
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    if (!EMAIL_RE.test(email)) {
      setError("enter a valid email address");
      return;
    }
    setBusy(true);
    try {
      await api.forgotPassword(email);
      setSent(true);
    } catch (err) {
      if (err instanceof ApiError && err.status === 0) {
        setError(errorText(err, "could not reach the server — check your connection and try again"));
      } else {
        // Any other answer still resolves the same way: never leak account existence.
        setSent(true);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="card auth-card">
        <h1>Reset password</h1>
        {sent ? (
          <div role="status">
            <p className="alert alert--success">
              If an account exists, a password reset email has been sent.
            </p>
            <p className="muted">
              <Link to="/#login-card">Back to sign in</Link>
            </p>
          </div>
        ) : (
          <form onSubmit={(e) => void onSubmit(e)} noValidate>
            <p className="muted">
              Enter your email and we will send a reset link if an account exists.
            </p>
            {error !== null && (
              <p className="alert alert--error" role="alert">
                {error}
              </p>
            )}
            <div className="field">
              <label htmlFor="forgot-email">Email</label>
              <input
                id="forgot-email"
                name="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <button type="submit" className="btn btn--primary" disabled={busy}>
              {busy ? "Sending…" : "Send reset link"}
            </button>
            <p className="muted">
              Remembered it? <Link to="/#login-card">Sign in</Link>
            </p>
          </form>
        )}
      </section>
    </main>
  );
}
