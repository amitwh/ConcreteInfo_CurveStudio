/**
 * The application header: brand, primary navigation, the workspace tabs and the session
 * controls. Help and About are links to their own URLs (deep links and the back button
 * keep working), but the authenticated shell renders them as overlays, so the workspace
 * underneath is never unmounted. The profile is the third overlay, reached from the
 * account menu.
 *
 * The workspace tabs are the "multiple workspaces" surface: each tab is a named workspace
 * with its own dataset and mode. Double-click a tab to rename it; the × removes a
 * workspace (never the last one); + starts another.
 */

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
} from "react";
import {
  BookOpen,
  ChevronDown,
  Info,
  Layers,
  LayoutDashboard,
  LogOut,
  MonitorDown,
  Moon,
  Plus,
  Sun,
  User,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../api/auth";
import type { MeOut } from "../api/client";
import { useTheme } from "./ThemeToggle";
import { useInstallPrompt } from "./useInstallPrompt";
import type { Workspace } from "../workspace/workspaces";

/** The overlays the shell can open over the workspace; the URL names one of them. */
export type Overlay = "help" | "about" | "profile" | null;

export interface AppNavProps {
  workspaces: Workspace[];
  activeId: string;
  overlay: Overlay;
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

/**
 * The account chip and the menu it opens: the account as a header, Profile, Sign out. The
 * chip is the only way to either, so on narrow screens it shrinks to the avatar and is
 * never hidden. Opening moves focus to the first item; the arrow keys move between items;
 * Escape closes and returns focus to the chip; a press anywhere outside closes it too.
 */
function AccountMenu({ user, onSignOut }: { user: MeOut; onSignOut: () => void }): ReactElement {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const firstItemRef = useRef<HTMLAnchorElement>(null);
  const buttonId = useId();
  const menuId = useId();
  const email = user.email;
  const name = user.name.trim();
  const initial = (name !== "" ? name : email).slice(0, 1).toUpperCase() || "?";

  const close = useCallback((restoreFocus: boolean): void => {
    setOpen(false);
    if (restoreFocus) buttonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (open) firstItemRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent): void => {
      if (!wrapRef.current?.contains(event.target as Node)) close(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, close]);

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      close(true);
      return;
    }
    // Tab leaves the menu: focus moves on, the menu goes.
    if (event.key === "Tab") {
      close(false);
      return;
    }
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]'),
    );
    const index = items.indexOf(document.activeElement as HTMLElement);
    const last = items.length - 1;
    let target: number;
    switch (event.key) {
      case "ArrowDown":
        target = index < 0 || index === last ? 0 : index + 1;
        break;
      case "ArrowUp":
        target = index <= 0 ? last : index - 1;
        break;
      case "Home":
        target = 0;
        break;
      case "End":
        target = last;
        break;
      default:
        return;
    }
    event.preventDefault();
    items[target]?.focus();
  };

  return (
    <div className="account-wrap" ref={wrapRef}>
      <button
        ref={buttonRef}
        id={buttonId}
        type="button"
        className="account"
        title={email}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={name !== "" ? `${name} (${email})` : email}
        onClick={() => (open ? close(false) : setOpen(true))}
      >
        <span className="account__avatar" aria-hidden="true">
          {initial}
        </span>
        <span className="account__text">
          {name !== "" && <span className="account__name truncate">{name}</span>}
          <span className="account__email truncate">{email}</span>
        </span>
        <ChevronDown className="account__chevron" aria-hidden="true" focusable="false" />
      </button>
      {open && (
        <div className="account-menu">
          <p className="account-menu__header">
            {name !== "" && <span className="account-menu__name">{name}</span>}
            <span className="account-menu__email">{email}</span>
          </p>
          <div role="menu" id={menuId} aria-labelledby={buttonId} onKeyDown={onMenuKeyDown}>
            <Link
              ref={firstItemRef}
              role="menuitem"
              className="account-menu__item"
              to="/profile"
              onClick={() => close(true)}
            >
              <User className="icon" aria-hidden="true" focusable="false" />
              Profile
            </Link>
            <button
              type="button"
              role="menuitem"
              className="account-menu__item"
              onClick={() => {
                close(false);
                onSignOut();
              }}
            >
              <LogOut className="icon" aria-hidden="true" focusable="false" />
              Sign out
            </button>
          </div>
        </div>
      )}
    </div>
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
            <AccountMenu user={user} onSignOut={() => void logout()} />
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
