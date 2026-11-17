/** /login — email + password, client-side checks mirroring the server, inline errors. */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { ApiError } from "../api/client";
import { errorText } from "../components/States";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function clientSideAuthErrors(email: string, password: string): string | null {
  if (!EMAIL_RE.test(email)) return "enter a valid email address";
  if (password.length < 10 || password.length > 128)
    return "the password must be between 10 and 128 characters";
  return null;
}

export function LoginPage(): ReactElement {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
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
      setError(
        err instanceof ApiError ? err.message : errorText(err, "sign-in failed; try again"),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="card auth-card" onSubmit={(e) => void onSubmit(e)} noValidate>
        <h1>Sign in</h1>
        <p className="muted">CurveStudio — curve fitting in your browser.</p>
        {error !== null && (
          <p className="alert alert--error" role="alert">
            {error}
          </p>
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
        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <p className="muted">
          No account? <Link to="/register">Create one</Link> ·{" "}
          <Link to="/about">About CurveStudio</Link>
        </p>
      </form>
    </main>
  );
}
