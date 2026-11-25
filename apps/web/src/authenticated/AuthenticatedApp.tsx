/**
 * The signed-in application: the workspace shell.
 *
 * Everything an authenticated session sees lives here — the navigation bar with its
 * workspace tabs, the workspace page itself, and the Help / About / profile overlays
 * that open over it (their URLs still work for deep links and the back button). It is
 * one lazy chunk of its own, so an anonymous visitor's home page downloads none of it;
 * App.tsx mounts it under the splat route once a session is restored.
 *
 * Workspaces (the named tabs under the header) are held here: each remembers the dataset
 * it shows and its analysis mode, so switching tabs returns to exactly where you were.
 */

import { lazy, useCallback, useEffect, useState, type ReactElement } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { AppNav, type Overlay } from "../components/AppNav";
import { Modal } from "../components/Modal";
import { LazyBoundary } from "../components/States";
import {
  activateWorkspace,
  activeWorkspace,
  addWorkspace,
  loadWorkspaces,
  removeWorkspace,
  renameWorkspace,
  saveWorkspaces,
  updateWorkspace,
} from "../workspace/workspaces";

const WorkspacePage = lazy(() => import("../pages/WorkspacePage").then((m) => ({ default: m.WorkspacePage })));
const HelpPage = lazy(() => import("../pages/HelpPage").then((m) => ({ default: m.HelpPage })));
const AboutPage = lazy(() => import("../pages/AboutPage").then((m) => ({ default: m.AboutPage })));
const ProfilePage = lazy(() => import("../pages/ProfilePage").then((m) => ({ default: m.ProfilePage })));

/** Which overlay the URL asks for (Help, About and the profile open over the workspace). */
export function overlayFor(pathname: string): Overlay {
  if (pathname === "/help") return "help";
  if (pathname === "/about") return "about";
  if (pathname === "/profile") return "profile";
  return null;
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
      <LazyBoundary label="Opening your workspace…">
        <WorkspacePage
          key={active.id}
          initialDatasetId={active.datasetId}
          initialMode={active.mode}
          onSelectionChange={(datasetId) =>
            setWorkspaceSet((current) => updateWorkspace(current, active.id, { datasetId }))
          }
          onModeChange={(mode) => setWorkspaceSet((current) => updateWorkspace(current, active.id, { mode }))}
        />
      </LazyBoundary>
      {overlay === "help" && (
        <Modal title="Help contents" variant="sheet" onClose={closeOverlay}>
          <LazyBoundary label="Opening help…">
            <HelpPage />
          </LazyBoundary>
        </Modal>
      )}
      {overlay === "about" && (
        <Modal title="About CurveStudio" variant="sheet" onClose={closeOverlay}>
          <LazyBoundary label="Opening about…">
            <AboutPage />
          </LazyBoundary>
        </Modal>
      )}
      {overlay === "profile" && (
        <Modal title="Your profile" variant="sheet" onClose={closeOverlay}>
          <LazyBoundary label="Opening your profile…">
            <ProfilePage />
          </LazyBoundary>
        </Modal>
      )}
    </>
  );
}
