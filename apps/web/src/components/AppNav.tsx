/**
 * The application header: brand, primary navigation, the workspace tabs and the session
 * controls. Help and About are links to their own URLs (deep links and the back button
 * keep working), but the authenticated shell renders them as overlays, so the workspace
 * underneath is never unmounted.
 *
 * The workspace tabs are the "multiple workspaces" surface: each tab is a named workspace
 * with its own dataset and mode. Double-click a tab to rename it; the × removes a
 * workspace (never the last one); + starts another.
 */

import { useEffect, useRef, useState, type ReactElement } from "react";
import {
  BookOpen,
  Info,
  Layers,
  LayoutDashboard,
  LogOut,
  MonitorDown,
  Moon,
  Plus,
  Sun,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../api/auth";
import { useTheme } from "./ThemeToggle";
import { useInstallPrompt } from "./useInstallPrompt";
import type { Workspace } from "../workspace/workspaces";

export interface AppNavProps {
  workspaces: Workspace[];
  activeId: string;
  overlay: "help" | "about" | null;
  onSelectWorkspace: (id: string) => void;
  onAddWorkspace: () => void;
  onCloseWorkspace: (id: string) => void;
  onRenameWorkspace: (id: string, name: string) => void;
}

function WorkspaceTab({
  workspace,
  active,
  closable,
  onSelect,
  onClose,
  onRename,
}: {
  workspace: Workspace;
  active: boolean;
  closable: boolean;
  onSelect: () => void;
  onClose: () => void;
  onRename: (name: string) => void;
}): ReactElement {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(workspace.name);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const commit = (): void => {
    setEditing(false);
    onRename(draft);
  };

  if (editing) {
    return (
      <li className="ws-tab ws-tab--editing">
        <input
          ref={inputRef}
          className="ws-tab__input"
          aria-label={`Rename ${workspace.name}`}
          value={draft}
          maxLength={60}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setDraft(workspace.name);
              setEditing(false);
            }
          }}
        />
      </li>
    );
  }

  return (
    <li className={`ws-tab${active ? " ws-tab--active" : ""}`}>
      <button
        type="button"
        className="ws-tab__label"
        aria-current={active ? "true" : undefined}
        title={`${workspace.name} — double-click to rename`}
        onClick={onSelect}
        onDoubleClick={() => {
          setDraft(workspace.name);
          setEditing(true);
        }}
      >
        <Layers className="ws-tab__icon" aria-hidden="true" focusable="false" />
        <span className="ws-tab__name truncate">{workspace.name}</span>
      </button>
      {closable && (
        <button
          type="button"
          className="ws-tab__close"
          aria-label={`Close ${workspace.name}`}
          title={`Close ${workspace.name}`}
          onClick={onClose}
        >
          <X className="icon" aria-hidden="true" focusable="false" />
        </button>
      )}
    </li>
  );
}

export function AppNav({
  workspaces,
  activeId,
  overlay,
  onSelectWorkspace,
  onAddWorkspace,
  onCloseWorkspace,
  onRenameWorkspace,
}: AppNavProps): ReactElement {
  const { status, user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const { canInstall, install } = useInstallPrompt();
  const email = user?.email ?? "";
  const initial = email.slice(0, 1).toUpperCase() || "?";

  return (
    <header className="app-header">
      <div className="app-header__bar">
        <Link to="/" className="brand" aria-label="CurveStudio — back to the workspace">
          <img
            className="brand__logo"
            src="/curvestudio-logo.png"
            alt="ConcreteInfo CurveStudio"
            width={165}
            height={44}
          />
        </Link>
        <nav aria-label="Main" className="app-nav">
          <Link to="/" aria-current={overlay === null ? "page" : undefined}>
            <LayoutDashboard className="icon" aria-hidden="true" focusable="false" />
            <span className="nav__label">Workspace</span>
          </Link>
          <Link to="/help" aria-current={overlay === "help" ? "page" : undefined}>
            <BookOpen className="icon" aria-hidden="true" focusable="false" />
            <span className="nav__label">Help</span>
          </Link>
          <Link to="/about" aria-current={overlay === "about" ? "page" : undefined}>
            <Info className="icon" aria-hidden="true" focusable="false" />
            <span className="nav__label">About</span>
          </Link>
        </nav>
        <div className="app-header__right">
          {canInstall && (
            <button
              type="button"
              className="btn btn--small btn--install"
              onClick={() => void install()}
              title="Install CurveStudio on this device"
            >
              <MonitorDown className="icon" aria-hidden="true" focusable="false" />
              <span className="btn__label">Install</span>
            </button>
          )}
          <button
            type="button"
            className="btn btn--small btn--icon"
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? (
              <Sun className="icon" aria-hidden="true" focusable="false" />
            ) : (
              <Moon className="icon" aria-hidden="true" focusable="false" />
            )}
          </button>
          {status === "authed" && user !== null && (
            <>
              <span className="account" title={email}>
                <span className="account__avatar" aria-hidden="true">
                  {initial}
                </span>
                <span className="account__email truncate">{email}</span>
              </span>
              <button
                type="button"
                className="btn btn--small btn--icon"
                onClick={() => void logout()}
                aria-label="Sign out"
                title="Sign out"
              >
                <LogOut className="icon" aria-hidden="true" focusable="false" />
              </button>
            </>
          )}
        </div>
      </div>
      <div className="ws-bar" aria-label="Workspaces">
        <ul className="ws-tabs" role="tablist" aria-label="Workspaces">
          {workspaces.map((workspace) => (
            <WorkspaceTab
              key={workspace.id}
              workspace={workspace}
              active={workspace.id === activeId}
              closable={workspaces.length > 1}
              onSelect={() => onSelectWorkspace(workspace.id)}
              onClose={() => onCloseWorkspace(workspace.id)}
              onRename={(name) => onRenameWorkspace(workspace.id, name)}
            />
          ))}
        </ul>
        <button
          type="button"
          className="ws-add"
          aria-label="New workspace"
          title="New workspace"
          onClick={onAddWorkspace}
        >
          <Plus className="icon" aria-hidden="true" focusable="false" />
        </button>
      </div>
    </header>
  );
}
