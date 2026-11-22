/**
 * /reset-password?token=… — set a new password from the emailed link.
 *
 * The token names the account; the server enforces the weakness check, while this page
 * mirrors the length floor (10–128) and the confirmation field.
 */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { errorText } from "../components/States";

const INVALID_LINK = "this reset link is invalid or has expired";

export function ResetPasswordPage(): ReactElement {
  const [params] = useSearchParams();
  const token = params.get("token");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    if (password.length < 10 || password.length > 128) {
      setError("the password must be between 10 and 128 characters");
      return;
    }
    if (password !== confirm) {
      setError("the passwords do not match");
      return;
    }
    if (token === null || token === "") return;
    setBusy(true);
    try {
      await api.resetPassword(token, password);
      setDone(true);
    } catch (err) {
      if (err instanceof ApiError && err.status === 400) {
        setError(INVALID_LINK);
      } else if (err instanceof ApiError && err.status === 422) {
        setError(err.message);
      } else {
        setError(errorText(err, "could not reset the password; try again"));
      }
    } finally {
      setBusy(false);
    }
  };

  const missingToken = token === null || token === "";

  return (
    <main className="auth-page">
      <section className="card auth-card">
        <h1>Reset password</h1>
        {missingToken ? (
          <>
            <p className="alert alert--error" role="alert">
              {INVALID_LINK}
            </p>
            <p>
              <Link to="/login">Go to sign in</Link>
            </p>
            <p className="muted">
              Need a new link? <Link to="/forgot-password">Request a password reset</Link>.
            </p>
          </>
        ) : done ? (
          <div role="status">
            <p className="alert alert--success">
              Your password has been reset. Sign in with the new one.
            </p>
            <p>
              <Link to="/login">Sign in</Link>
            </p>
          </div>
        ) : (
          <form onSubmit={(e) => void onSubmit(e)} noValidate>
            {error !== null && (
              <p className="alert alert--error" role="alert">
                {error}
              </p>
            )}
            <div className="field">
              <label htmlFor="reset-password">New password (10–128 characters)</label>
              <input
                id="reset-password"
                name="password"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <div className="field">
              <label htmlFor="reset-confirm">Confirm new password</label>
              <input
                id="reset-confirm"
                name="confirm"
                type="password"
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
              />
            </div>
            <button type="submit" className="btn btn--primary" disabled={busy}>
              {busy ? "Resetting…" : "Reset password"}
            </button>
            <p className="muted">
              <Link to="/login">Back to sign in</Link>
            </p>
          </form>
        )}
      </section>
    </main>
  );
}
