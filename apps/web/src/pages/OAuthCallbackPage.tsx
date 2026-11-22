/**
 * /oauth/callback — the API redirects here once the provider has vouched for the user.
 *
 * The query carries a single-use exchange code; claim it, adopt the token pair through
 * the auth context, then land in the workspace. The ref guard is what makes the effect
 * safe under React's StrictMode double-invocation (a code may only be claimed once).
 */

import { useEffect, useRef, useState, type ReactElement } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { api } from "../api/client";
import { Loading } from "../components/States";

const INVALID = "This sign-in link is invalid or has expired";

export function OAuthCallbackPage(): ReactElement {
  const { adoptSession } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const started = useRef(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const code = new URLSearchParams(location.search).get("code");
    if (code === null || code === "") {
      setError(INVALID);
      return;
    }
    void (async () => {
      try {
        const tokens = await api.oauthExchange(code);
        await adoptSession(tokens);
        navigate("/", { replace: true });
      } catch {
        setError(INVALID);
      }
    })();
  }, [adoptSession, location.search, navigate]);

  return (
    <main className="auth-page">
      <section className="card auth-card">
        <h1>Signing you in</h1>
        {error === null ? (
          <Loading label="Completing sign-in…" />
        ) : (
          <>
            <p className="alert alert--error" role="alert">
              {error}
            </p>
            <p className="muted">
              <Link to="/#login-card">Back to sign in</Link>
            </p>
          </>
        )}
      </section>
    </main>
  );
}
