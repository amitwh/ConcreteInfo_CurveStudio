/**
 * Routes and the application shell.
 *
 * Signed in, everything lives under one splat route so the workspace stays mounted while
 * Help, About or the profile is open — those three open as overlays over the workspace,
 * and their URLs still work for deep links and the back button. The admin console (/admin)
 * is the one exception: a page of its own ahead of the splat, loaded on demand. Signed
 * out, Help and About stay ordinary pages behind a slim brand header (the profile needs a
 * session, so a visitor who opens its URL lands on the home page).
 *
 * Workspaces (the named tabs under the header) are held here: each remembers the dataset
 * it shows and its analysis mode, so switching tabs returns to exactly where you were.
 */

import { lazy, Suspense, useCallback, useEffect, useState, type ReactElement } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "./api/auth";
import { AppNav, type Overlay } from "./components/AppNav";
import { Modal } from "./components/Modal";
import { Loading } from "./components/States";
import { ThemeProvider } from "./components/ThemeToggle";
import { AboutPage } from "./pages/AboutPage";
import { HelpPage } from "./pages/HelpPage";
import { ForgotPasswordPage } from "./pages/ForgotPasswordPage";
import { HomePage } from "./pages/HomePage";
import { OAuthCallbackPage } from "./pages/OAuthCallbackPage";
import { ProfilePage } from "./pages/ProfilePage";
import { RegisterPage } from "./pages/RegisterPage";
import { ResetPasswordPage } from "./pages/ResetPasswordPage";
import { VerifyEmailPage } from "./pages/VerifyEmailPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import {
  activateWorkspace,
  activeWorkspace,
  addWorkspace,
  loadWorkspaces,
  removeWorkspace,
  renameWorkspace,
  saveWorkspaces,
  updateWorkspace,
} from "./workspace/workspaces";

/**
 * The admin console is loaded only when an administrator opens it: none of its code, the
 * table library included, belongs in the bundle every visitor downloads.
 */
const AdminConsole = lazy(() => import("./admin/AdminConsole"));

/** Which overlay the URL asks for (Help, About and the profile open over the workspace). */
export function overlayFor(pathname: string): Overlay {
  if (pathname === "/help") return "help";
  if (pathname === "/about") return "about";
  if (pathname === "/profile") return "profile";
  return null;
}

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

export function AuthenticatedApp(): ReactElement {
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const userId = user?.id ?? null;
  const [workspaceSet, setWorkspaceSet] = useState(() => loadWorkspaces(userId));
  const overlay = overlayFor(location.pathname);
  const active = activeWorkspace(workspaceSet);

  useEffect(() => {
    saveWorkspaces(userId, workspaceSet);
  }, [userId, workspaceSet]);

  const closeOverlay = useCallback(() => navigate("/"), [navigate]);

  return (
    <>
      <AppNav
        workspaces={workspaceSet.workspaces}
        activeId={active.id}
        overlay={overlay}
        onSelectWorkspace={(id) => setWorkspaceSet((current) => activateWorkspace(current, id))}
        onAddWorkspace={() => setWorkspaceSet((current) => addWorkspace(current))}
        onCloseWorkspace={(id) => setWorkspaceSet((current) => removeWorkspace(current, id))}
        onRenameWorkspace={(id, name) =>
          setWorkspaceSet((current) => renameWorkspace(current, id, name))
        }
      />
      <WorkspacePage
        key={active.id}
        initialDatasetId={active.datasetId}
        initialMode={active.mode}
        onSelectionChange={(datasetId) =>
          setWorkspaceSet((current) => updateWorkspace(current, active.id, { datasetId }))
        }
        onModeChange={(mode) => setWorkspaceSet((current) => updateWorkspace(current, active.id, { mode }))}
      />
      {overlay === "help" && (
        <Modal title="Help contents" variant="sheet" onClose={closeOverlay}>
          <HelpPage />
        </Modal>
      )}
      {overlay === "about" && (
        <Modal title="About CurveStudio" variant="sheet" onClose={closeOverlay}>
          <AboutPage />
        </Modal>
      )}
      {overlay === "profile" && (
        <Modal title="Your profile" variant="sheet" onClose={closeOverlay}>
          <ProfilePage />
        </Modal>
      )}
    </>
  );
}

export default function App(): ReactElement {
  const { status } = useAuth();
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
                <Suspense fallback={<Loading label="Opening the admin console…" />}>
                  <AdminConsole />
                </Suspense>
              }
            />
            {/* One splat route: opening /help, /about or /profile must not remount the workspace. */}
            <Route path="*" element={<AuthenticatedApp />} />
          </Routes>
        ) : (
          <>
            <PublicHeader />
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginRedirect />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/verify-email" element={<VerifyEmailPage />} />
              <Route path="/forgot-password" element={<ForgotPasswordPage />} />
              <Route path="/reset-password" element={<ResetPasswordPage />} />
              <Route path="/oauth/callback" element={<OAuthCallbackPage />} />
              <Route path="/help" element={<HelpPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </>
        )}
      </div>
    </ThemeProvider>
  );
}
