/**
 * Routes and the application shell.
 *
 * Signed in, everything lives under one splat route so the workspace stays mounted while
 * Help or About is open — those two open as overlays over the workspace, and their URLs
 * still work for deep links and the back button. Signed out, Help and About stay ordinary
 * pages behind a slim brand header.
 *
 * Workspaces (the named tabs under the header) are held here: each remembers the dataset
 * it shows and its analysis mode, so switching tabs returns to exactly where you were.
 */

import { useCallback, useEffect, useState, type ReactElement } from "react";
import { BookOpen, Info, Moon, Sun } from "lucide-react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "./api/auth";
import { AppNav } from "./components/AppNav";
import { Modal } from "./components/Modal";
import { Loading } from "./components/States";
import { ThemeProvider, useTheme } from "./components/ThemeToggle";
import { AboutPage } from "./pages/AboutPage";
import { HelpPage } from "./pages/HelpPage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
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

/** Which overlay the URL asks for (Help and About are overlays over the workspace). */
export function overlayFor(pathname: string): "help" | "about" | null {
  if (pathname === "/help") return "help";
  if (pathname === "/about") return "about";
  return null;
}

/** The brand-only header the sign-in and public pages wear. */
function PublicHeader(): ReactElement {
  const { theme, toggle } = useTheme();
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
        <nav aria-label="Main" className="app-nav">
          <Link to="/help">
            <BookOpen className="icon" aria-hidden="true" focusable="false" />
            <span className="nav__label">Help</span>
          </Link>
          <Link to="/about">
            <Info className="icon" aria-hidden="true" focusable="false" />
            <span className="nav__label">About</span>
          </Link>
        </nav>
        <div className="app-header__right">
          <button
            type="button"
            className="btn btn--small btn--icon"
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? (
              <Sun className="icon" aria-hidden="true" focusable="false" />
            ) : (
              <Moon className="icon" aria-hidden="true" focusable="false" />
            )}
          </button>
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
            {/* One splat route: navigating to /help or /about must not remount the workspace. */}
            <Route path="*" element={<AuthenticatedApp />} />
          </Routes>
        ) : (
          <>
            <PublicHeader />
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/help" element={<HelpPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
          </>
        )}
      </div>
    </ThemeProvider>
  );
}
