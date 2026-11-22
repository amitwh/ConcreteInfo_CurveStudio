/** /login — email + password, client-side checks mirroring the server, inline errors. */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../api/auth";
import { api, ApiError } from "../api/client";
import { OAuthButtons } from "../components/OAuthButtons";
import { errorText } from "../components/States";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const RESEND_NOTE = "if an unverified account exists, a new verification email has been sent.";

/** The API redirects failed social sign-ins back to /login?oauth_error=<reason>. */
const OAUTH_ERRORS: Record<string, string> = {
  state: "your sign-in session expired or was opened in another tab — please try again",
  provider: "the provider could not complete sign-in — please try again",
};

export function clientSideAuthErrors(email: string, password: string): string | null {
  if (!EMAIL_RE.test(email)) return "enter a valid email address";
  if (password.length < 10 || password.length > 128)
    return "the password must be between 10 and 128 characters";
  return null;
}

/** The API rejects an unverified sign-in with 403 + detail.code (see openapi.json). */
function isEmailNotVerified(error: unknown): error is ApiError {
  return (
    error instanceof ApiError &&
    error.status === 403 &&
    (error.detail as { code?: string } | null)?.code === "EMAIL_NOT_VERIFIED"
  );
}

export function LoginPage(): ReactElement {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const oauthError = searchParams.get("oauth_error");
  const oauthMessage =
    oauthError === null
      ? null
      : (OAUTH_ERRORS[oauthError] ?? "social sign-in could not be completed; try again");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [unverified, setUnverified] = useState<string | null>(null);
  const [resendNote, setResendNote] = useState<string | null>(null);
  const [resendError, setResendError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    setUnverified(null);
    setResendNote(null);
    setResendError(null);
    const invalid = clientSideAuthErrors(email, password);
    if (invalid !== null) {
      setError(invalid);
      return;
    }
    setBusy(true);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      if (isEmailNotVerified(err)) setUnverified(err.message);
      else
        setError(
          err instanceof ApiError ? err.message : errorText(err, "sign-in failed; try again"),
        );
    } finally {
      setBusy(false);
    }
  };

  const onResend = async (): Promise<void> => {
    if (!EMAIL_RE.test(email)) return;
    setResendNote(null);
    setResendError(null);
    setBusy(true);
    try {
      await api.resendVerification(email);
      setResendNote(RESEND_NOTE);
    } catch (err) {
      setResendError(errorText(err, "could not resend the verification email; try again"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="card auth-card" onSubmit={(e) => void onSubmit(e)} noValidate>
        <h1>Sign in</h1>
        <p className="muted">CurveStudio — curve fitting in your browser.</p>
        {oauthMessage !== null && (
          <p className="alert alert--error" role="alert">
            {oauthMessage}
          </p>
        )}
        {error !== null && (
          <p className="alert alert--error" role="alert">
            {error}
          </p>
        )}
        {unverified !== null && (
          <div className="alert alert--warning">
            <p role="alert">{unverified}</p>
            <button
              type="button"
              className="btn btn--small"
              disabled={busy || !EMAIL_RE.test(email)}
              onClick={() => void onResend()}
            >
              Resend verification email
            </button>
            {resendNote !== null && <p role="status">{resendNote}</p>}
            {resendError !== null && <p role="alert">{resendError}</p>}
          </div>
        )}
        <div className="field">
          <label htmlFor="login-email">Email</label>
          <input
            id="login-email"
            name="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="login-password">Password</label>
          <input
            id="login-password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <p className="muted">
          <Link to="/forgot-password">Forgot password?</Link>
        </p>
        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <OAuthButtons />
        <p className="muted">
          No account? <Link to="/register">Create one</Link> ·{" "}
          <Link to="/about">About CurveStudio</Link>
        </p>
      </form>
    </main>
  );
}
