/**
 * Social sign-in buttons for the public pages.
 *
 * Providers are discovered from the API so a deployment only offers what it has
 * configured; when none are, this renders nothing at all (no empty divider). The
 * buttons are ordinary links in effect: they do a full-page navigation into the API's
 * start endpoint, which sets the PKCE flow cookie and redirects to the provider.
 */

import { useEffect, useState, type ReactElement, type ReactNode } from "react";
import { API_BASE, api } from "../api/client";

const BRAND_NAMES: Record<string, string> = {
  google: "Google",
  microsoft: "Microsoft",
  linkedin: "LinkedIn",
};

function GoogleMark(): ReactElement {
  return (
    <svg viewBox="0 0 48 48" width="16" height="16" aria-hidden="true" focusable="false">
      <path
        fill="#ea4335"
        d="M24 9.5c3.4 0 6.4 1.2 8.8 3.5l6.6-6.6A23.9 23.9 0 0 0 24 0C14.7 0 6.7 5.3 2.9 13l7.7 6a14.3 14.3 0 0 1 13.4-9.5Z"
      />
      <path
        fill="#4285f4"
        d="M46.1 24.6c0-1.6-.1-2.8-.4-4.1H24v8.4h12.5a10.7 10.7 0 0 1-4.6 7l7.4 5.7c4.3-4 6.8-9.9 6.8-17Z"
      />
      <path
        fill="#fbbc05"
        d="M10.6 28.9a14.4 14.4 0 0 1 0-9.8l-7.7-6A24 24 0 0 0 0 24c0 3.9.9 7.5 2.9 10.9l7.7-6Z"
      />
      <path
        fill="#34a853"
        d="M24 48c6.5 0 11.9-2.1 15.8-5.8l-7.4-5.7c-2 1.4-4.8 2.3-8.4 2.3a14.3 14.3 0 0 1-13.4-9.5l-7.7 6A24 24 0 0 0 24 48Z"
      />
    </svg>
  );
}

function MicrosoftMark(): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
      <rect x="1" y="1" width="10" height="10" fill="#f25022" />
      <rect x="13" y="1" width="10" height="10" fill="#7fba00" />
      <rect x="1" y="13" width="10" height="10" fill="#00a4ef" />
      <rect x="13" y="13" width="10" height="10" fill="#ffb900" />
    </svg>
  );
}

function LinkedInMark(): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
      <rect x="1" y="1" width="22" height="22" rx="4" fill="#0a66c2" />
      <text
        x="12"
        y="17"
        textAnchor="middle"
        fill="#fff"
        fontFamily="Inter, system-ui, sans-serif"
        fontSize="12"
        fontWeight="700"
      >
        in
      </text>
    </svg>
  );
}

const MARKS: Record<string, ReactNode> = {
  google: <GoogleMark />,
  microsoft: <MicrosoftMark />,
  linkedin: <LinkedInMark />,
};

export function OAuthButtons(): ReactElement | null {
  const [providers, setProviders] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    void api
      .oauthProviders()
      .then((list) => {
        if (alive) setProviders(list);
      })
      .catch(() => {
        // Discovery failing is not an error the visitor can act on; hide the options.
        if (alive) setProviders([]);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (providers.length === 0) return null;

  const start = (provider: string): void => {
    // Full-page navigation: the API must set its PKCE flow cookie before redirecting.
    window.location.assign(`${API_BASE}/auth/oauth/${provider}/start`);
  };

  return (
    <div className="oauth-block">
      <p className="auth-divider">
        <span>or</span>
      </p>
      <div className="oauth-buttons">
        {providers.map((provider) => (
          <button
            key={provider}
            type="button"
            className="btn oauth-btn"
            onClick={() => start(provider)}
          >
            {MARKS[provider] ?? null}
            <span>Continue with {BRAND_NAMES[provider] ?? provider}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
