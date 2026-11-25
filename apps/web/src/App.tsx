/**
 * Routes and the application shell.
 *
 * Signed in, everything lives under one splat route that mounts the signed-in shell
 * (AuthenticatedApp, its own lazy chunk) so the workspace stays mounted while Help, About
 * or the profile is open — those three open as overlays over the workspace, and their
 * URLs still work for deep links and the back button. The admin console (/admin) is the
 * one exception: a page of its own ahead of the splat, loaded on demand. Signed out,
 * Help and About stay ordinary pages behind a slim brand header (the profile needs a
 * session, so a visitor who opens its URL lands on the home page).
 *
 * Every page but the public home is lazy: the home page is all an anonymous visitor
 * downloads, so it carries none of the workspace's code — neither the workspace page
 * itself nor the signed-in shell that frames it. Each boundary shows the app's own
 * loading state (LazyBoundary), so a page whose chunk is still downloading reads as
 * loading — never as a blank flash.
 */

import { lazy, type ReactElement } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "./api/auth";
import { LazyBoundary, Loading } from "./components/States";
import { ThemeProvider } from "./components/ThemeToggle";
import { useChartPrefetch } from "./lib/chartPrefetch";
import { HomePage } from "./pages/HomePage";

/**
 * The admin console is loaded only when an administrator opens it: none of its code, the
 * table library included, belongs in the bundle every visitor downloads.
 */
const AdminConsole = lazy(() => import("./admin/AdminConsole"));

/** The signed-in shell: the workspace tab bar, the workspace page and its overlays. */
const AuthenticatedApp = lazy(() =>
  import("./authenticated/AuthenticatedApp").then((m) => ({ default: m.AuthenticatedApp })),
);

/**
 * The rest of the on-demand pages. Each is one chunk fetched at first use; the pages
 * export named components, so each loader picks its name out of the module.
 */
const HelpPage = lazy(() => import("./pages/HelpPage").then((m) => ({ default: m.HelpPage })));
const AboutPage = lazy(() => import("./pages/AboutPage").then((m) => ({ default: m.AboutPage })));
const RegisterPage = lazy(() => import("./pages/RegisterPage").then((m) => ({ default: m.RegisterPage })));
const VerifyEmailPage = lazy(() =>
  import("./pages/VerifyEmailPage").then((m) => ({ default: m.VerifyEmailPage })),
);
const ForgotPasswordPage = lazy(() =>
  import("./pages/ForgotPasswordPage").then((m) => ({ default: m.ForgotPasswordPage })),
);
const ResetPasswordPage = lazy(() =>
  import("./pages/ResetPasswordPage").then((m) => ({ default: m.ResetPasswordPage })),
);
const OAuthCallbackPage = lazy(() =>
  import("./pages/OAuthCallbackPage").then((m) => ({ default: m.OAuthCallbackPage })),
);

/**
 * `/login` is no longer a page of its own. The home card is the only sign-in surface, so
 * an old link (or the API's OAuth error redirect) keeps its query string and lands on
 * that card.
 */
function LoginRedirect(): ReactElement {
  const { search } = useLocation();
  return <Navigate replace to={`/${search}#login-card`} />;
}

/** The slim white header every public page wears. */
function PublicHeader(): ReactElement {
  return (
    <header className="app-header app-header--public">
      <div className="app-header__bar">
        <Link to="/" className="brand" aria-label="CurveStudio">
          <img
            className="brand__logo"
            src="/curvestudio-logo.png"
            alt="ConcreteInfo CurveStudio"
            width={165}
            height={44}
          />
        </Link>
        <nav aria-label="Main" className="public-nav">
          <Link to="/#features">Features</Link>
          <Link to="/#how-it-works">How it works</Link>
          <Link to="/#models">Models</Link>
          <Link to="/#use-cases">Use cases</Link>
          <Link to="/#faq">FAQ</Link>
        </nav>
        <div className="app-header__right">
          <Link to="/#login-card" className="btn btn--accent-outline btn--pill">
            Sign in
          </Link>
          <Link to="/register" className="btn btn--primary btn--pill">
            Create account
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function App(): ReactElement {
  const { status } = useAuth();
  // Once a session is authed, the chart bundle starts downloading at the browser's next
  // idle moment, so the first chart after sign-in waits on its render, not its download.
  useChartPrefetch(status);
  return (
    <ThemeProvider>
      <div className="app-shell">
        {status === "loading" ? (
          <Loading label="Restoring session…" />
        ) : status === "authed" ? (
          <Routes>
            {/* A page of its own before the splat: nothing of the workspace mounts under /admin. */}
            <Route
              path="/admin/*"
              element={
                <LazyBoundary label="Opening the admin console…">
                  <AdminConsole />
                </LazyBoundary>
              }
            />
            {/* One splat route: the signed-in shell keeps the workspace mounted across overlays. */}
            <Route
              path="*"
              element={
                <LazyBoundary label="Opening your workspace…">
                  <AuthenticatedApp />
                </LazyBoundary>
              }
            />
          </Routes>
        ) : (
          <>
            <PublicHeader />
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginRedirect />} />
              <Route
                path="/register"
                element={
                  <LazyBoundary label="Opening account sign-up…">
                    <RegisterPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/verify-email"
                element={
                  <LazyBoundary label="Opening e-mail verification…">
                    <VerifyEmailPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/forgot-password"
                element={
                  <LazyBoundary label="Opening password recovery…">
                    <ForgotPasswordPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/reset-password"
                element={
                  <LazyBoundary label="Opening password reset…">
                    <ResetPasswordPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/oauth/callback"
                element={
                  <LazyBoundary label="Completing sign-in…">
                    <OAuthCallbackPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/help"
                element={
                  <LazyBoundary label="Opening help…">
                    <HelpPage />
                  </LazyBoundary>
                }
              />
              <Route
                path="/about"
                element={
                  <LazyBoundary label="Opening about…">
                    <AboutPage />
                  </LazyBoundary>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </>
        )}
      </div>
    </ThemeProvider>
  );
}
