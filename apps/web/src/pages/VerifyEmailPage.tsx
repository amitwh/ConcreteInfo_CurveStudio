/**
 * /verify-email?token=… — consumes the one-time link from the verification email.
 *
 * The token is single-use, so the effect is guarded with a ref: React's StrictMode runs
 * effects twice in development and a second request would fail on an already-used token.
 */

import { useEffect, useRef, useState, type ReactElement } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { Loading } from "../components/States";

type VerifyState = "verifying" | "success" | "missing" | "invalid";

const INVALID_FALLBACK = "this verification link is invalid or has expired";

export function VerifyEmailPage(): ReactElement {
  const [params] = useSearchParams();
  const token = params.get("token");
  const started = useRef(false);
  const [state, setState] = useState<VerifyState>(
    token === null || token === "" ? "missing" : "verifying",
  );
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (token === null || token === "") {
      setState("missing");
      return;
    }
    if (started.current) return;
    started.current = true;
    void (async () => {
      try {
        await api.verifyEmail(token);
        setState("success");
      } catch (err) {
        setMessage(err instanceof ApiError ? err.message : null);
        setState("invalid");
      }
    })();
  }, [token]);

  return (
    <main className="auth-page">
      <section className="card auth-card">
        <h1>Verify email</h1>
        {state === "verifying" && <Loading label="Verifying your email…" />}
        {state === "success" && (
          <>
            <p className="alert alert--success" role="status">
              Your email is verified. You can sign in now.
            </p>
            <p>
              <Link to="/login">Sign in</Link>
            </p>
          </>
        )}
        {state === "missing" && (
          <>
            <p className="alert alert--error" role="alert">
              this verification link is missing its token
            </p>
            <p>
              <Link to="/login">Go to sign in</Link>
            </p>
            <p className="muted">
              Need a new link? Use “Resend verification” on the sign-in page.
            </p>
          </>
        )}
        {state === "invalid" && (
          <>
            <p className="alert alert--error" role="alert">
              {message ?? INVALID_FALLBACK}
            </p>
            <p>
              <Link to="/login">Go to sign in</Link>
            </p>
            <p className="muted">
              Need a new link? Use “Resend verification” on the sign-in page.
            </p>
          </>
        )}
      </section>
    </main>
  );
}
