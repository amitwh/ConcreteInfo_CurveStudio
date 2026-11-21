/** Routes and the app shell: header with nav, theme toggle and session controls. */

import type { ReactElement, ReactNode } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./api/auth";
import { ThemeProvider, useTheme } from "./components/ThemeToggle";
import { Loading } from "./components/States";
import { AboutPage } from "./pages/AboutPage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { WorkspacePage } from "./pages/WorkspacePage";

function RequireAuth({ children }: { children: ReactNode }): ReactElement {
  const { status } = useAuth();
  if (status === "loading") return <Loading label="Restoring session…" />;
  if (status === "anon") return <Navigate to="/login" replace />;
  return children as ReactElement;
}

function Header(): ReactElement {
  const { status, user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  return (
    <header className="app-header">
      <Link to="/" className="brand">
        <img
          className="brand__logo"
          src="/concreteinfo-logo.png"
          alt="ConcreteInfo"
          width={100}
          height={28}
        />
        <span className="brand__product">CurveStudio</span>
      </Link>
      <nav aria-label="Main" className="app-nav">
        <Link to="/">Workspace</Link>
        <Link to="/about">About</Link>
      </nav>
      <div className="app-header__right">
        <button
          type="button"
          className="btn btn--small"
          onClick={toggle}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        {status === "authed" && user !== null && (
          <>
            <span className="muted truncate" title={user.email}>
              {user.email}
            </span>
            <button type="button" className="btn btn--small" onClick={() => void logout()}>
              Sign out
            </button>
          </>
        )}
      </div>
    </header>
  );
}

export default function App(): ReactElement {
  return (
    <ThemeProvider>
      <div className="app-shell">
        <Header />
        <Routes>
          <Route
            path="/"
            element={
              <RequireAuth>
                <WorkspacePage />
              </RequireAuth>
            }
          />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </ThemeProvider>
  );
}
