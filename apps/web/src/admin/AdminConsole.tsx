/**
 * The admin console: a page of its own, outside the workspace, for administrators. The
 * server refuses a member on every call, so the check here only spares one the error page.
 *
 * The navigation lists the pages that exist and no others: a link that leads nowhere is
 * worse than none. Accounts lives at /admin/users, under the API's name for them.
 */

import type { ReactElement } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "../api/auth";
import { AuditPage } from "./AuditPage";
import { ErrorsPage } from "./ErrorsPage";
import { InstallsPage } from "./InstallsPage";
import { OverviewPage } from "./OverviewPage";
import { PlansPage } from "./PlansPage";
import { SystemPage } from "./SystemPage";
import { UsersPage } from "./UsersPage";

const SECTIONS = [
  { path: "/admin", label: "Overview" },
  { path: "/admin/users", label: "Accounts" },
  { path: "/admin/installs", label: "Installs" },
  { path: "/admin/errors", label: "Errors" },
  { path: "/admin/audit", label: "Audit" },
  { path: "/admin/system", label: "System" },
  { path: "/admin/plans", label: "Plans" },
] as const;

export default function AdminConsole(): ReactElement {
  const { user } = useAuth();
  const { pathname } = useLocation();
  if (user === null || user.role !== "admin") return <Navigate to="/" replace />;

  // "/admin/plans/" is the Plans page too; a trailing slash is not a different place.
  const current = pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;

  return (
    <div className="admin">
      <header className="admin__header">
        <Link to="/" className="admin__back">
          Back to CurveStudio
        </Link>
        <h1>Admin console</h1>
      </header>
      <div className="admin__body">
        <nav className="admin__nav" aria-label="Admin">
          <ul>
            {SECTIONS.map((section) => (
              <li key={section.path}>
                <Link
                  to={section.path}
                  aria-current={current === section.path ? "page" : undefined}
                >
                  {section.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        {/* Nested under the app's /admin/* route: these paths are relative to /admin. */}
        <main className="admin__content">
          <Routes>
            <Route index element={<OverviewPage />} />
            <Route path="users" element={<UsersPage />} />
            <Route path="installs" element={<InstallsPage />} />
            <Route path="errors" element={<ErrorsPage />} />
            <Route path="audit" element={<AuditPage />} />
            <Route path="system" element={<SystemPage />} />
            <Route path="plans" element={<PlansPage />} />
            <Route path="*" element={<Navigate to="/admin" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
