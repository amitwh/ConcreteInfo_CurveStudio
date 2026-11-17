/**
 * /register — mirrors the server's RegisterIn contract: an email address and a password
 * of 10–128 characters; the optional display name is capped at 100 characters. Server
 * rejections (duplicate email, registration disabled) are shown inline.
 */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { ApiError } from "../api/client";
import { errorText } from "../components/States";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function RegisterPage(): ReactElement {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    if (!EMAIL_RE.test(email)) {
      setError("enter a valid email address");
      return;
    }
    if (password.length < 10 || password.length > 128) {
      setError("the password must be between 10 and 128 characters");
      return;
    }
    if (name.length > 100) {
      setError("the name must be at most 100 characters");
      return;
    }
    setBusy(true);
    try {
      await register(email, password, name);
      navigate("/", { replace: true });
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : errorText(err, "registration failed; try again"),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="card auth-card" onSubmit={(e) => void onSubmit(e)} noValidate>
        <h1>Create an account</h1>
        <p className="muted">Registration mirrors the server rules: email + 10–128 character password.</p>
        {error !== null && (
          <p className="alert alert--error" role="alert">
            {error}
          </p>
        )}
        <div className="field">
          <label htmlFor="register-name">Name (optional)</label>
          <input
            id="register-name"
            name="name"
            type="text"
            autoComplete="name"
            maxLength={100}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="register-email">Email</label>
          <input
            id="register-email"
            name="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="register-password">Password (10–128 characters)</label>
          <input
            id="register-password"
            name="password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? "Creating…" : "Create account"}
        </button>
        <p className="muted">
          Already registered? <Link to="/login">Sign in</Link>
        </p>
      </form>
    </main>
  );
}
