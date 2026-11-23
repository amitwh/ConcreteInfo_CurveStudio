/**
 * The admin console's calls: one typed function each, every one through the shared client.
 * All live under /admin and the server refuses a member with 403 on each, so the console's
 * own check only spares a member the error page.
 */

import { api, type Schemas } from "../api/client";

export type OverviewOut = Schemas["OverviewOut"];
export type InstallSummaryOut = Schemas["InstallSummaryOut"];
export type AuditOut = Schemas["AuditOut"];
export type AuditPageOut = Schemas["PageOut_AuditOut_"];
export type SystemOut = Schemas["SystemOut"];
export type PlanAdminOut = Schemas["PlanAdminOut"];
export type PlanLimits = Schemas["PlanLimitsIn"];

/** One row of GET /admin/system/latency. */
export type RouteLatencyOut = Schemas["RouteLatencyOut"];

/** The periods the latency figures come in, as the server names them. */
export type LatencyPeriod = Schemas["LatencyWindow"];

/** The audit trail's filters as the page holds them; `from` and `to` are IST days (YYYY-MM-DD). */
export interface AuditFilters {
  userId?: string;
  action?: string;
  from?: string;
  to?: string;
  /** From 1. */
  page: number;
  pageSize?: number;
}

/** An account as the console lists it; `plan` is the resolved plan's name. */
export type AdminUserOut = Schemas["AdminUserOut"];

/** An account with the last fifty rows of its audit trail, newest first. */
export type AdminUserDetailOut = Schemas["AdminUserDetailOut"];

export type UsersPageOut = Schemas["PageOut_AdminUserOut_"];

/**
 * What to change on an account. A key left out keeps its value; `plan_override: null`
 * clears it, back to the default plan.
 */
export interface UserChange {
  role?: AdminUserOut["role"];
  plan_override?: string | null;
  is_active?: boolean;
}

/** The account list's filters as the page holds them; empty values are not sent. */
export interface UsersFilters {
  /** A case-insensitive substring of the name or the address. */
  q?: string;
  /** A resolved plan's name. */
  plan?: string;
  role?: AdminUserOut["role"];
  active?: boolean;
  /** From 1. */
  page: number;
  pageSize?: number;
}

/** An error event as the console lists it, without the large texts. Nothing in it names an account. */
export type ErrorEventOut = Schemas["ErrorEventOut"];

/** An error event with everything stored about it. */
export type ErrorEventDetailOut = Schemas["ErrorEventDetailOut"];

export type ErrorsPageOut = Schemas["PageOut_ErrorEventOut_"];

/** The statuses an error event can be given. */
export type ErrorStatus = ErrorEventOut["status"];

/** The categories a report is filed under. */
export type ErrorCategory = "crash" | "sql" | "ui" | "network";

/** The error list's filters as the page holds them. */
export interface ErrorsFilters {
  status: ErrorStatus;
  /** Left out for every category. */
  category?: ErrorCategory;
  /** From 1. */
  page: number;
  pageSize?: number;
}

/** A query string from the defined, non-empty parameters only; empty when there are none. */
function queryString(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text === "" ? "" : `?${text}`;
}

export function overview(): Promise<OverviewOut> {
  return api.request<OverviewOut>("/admin/overview");
}

export function installsSummary(): Promise<InstallSummaryOut> {
  return api.request<InstallSummaryOut>("/admin/installs/summary");
}

export function audit(filters: AuditFilters): Promise<AuditPageOut> {
  const { userId, action, from, to, page, pageSize } = filters;
  return api.request<AuditPageOut>(
    `/admin/audit${queryString({
      user_id: userId,
      action,
      from,
      to,
      page,
      page_size: pageSize,
    })}`,
  );
}

export function system(): Promise<SystemOut> {
  return api.request<SystemOut>("/admin/system");
}

export function systemLatency(days: LatencyPeriod = 1): Promise<RouteLatencyOut[]> {
  return api.request<RouteLatencyOut[]>(`/admin/system/latency${queryString({ days })}`);
}

export function plans(): Promise<PlanAdminOut[]> {
  return api.request<PlanAdminOut[]>("/admin/plans");
}

export function updatePlan(name: string, limits: PlanLimits): Promise<PlanAdminOut> {
  return api.request<PlanAdminOut>(`/admin/plans/${encodeURIComponent(name)}`, {
    method: "PATCH",
    body: JSON.stringify({ limits }),
  });
}

/** The accounts matching `filters`, newest first, one page of them. */
export function users(filters: UsersFilters): Promise<UsersPageOut> {
  const { q, plan, role, active, page, pageSize } = filters;
  return api.request<UsersPageOut>(
    `/admin/users${queryString({
      q: q?.trim() || undefined,
      plan,
      role,
      // false is a filter of its own, so it is sent; only an unset one is left out.
      active,
      page,
      page_size: pageSize,
    })}`,
  );
}

export function userDetail(id: string): Promise<AdminUserDetailOut> {
  return api.request<AdminUserDetailOut>(`/admin/users/${encodeURIComponent(id)}`);
}

/** Change an account's role, plan or active flag; the server refuses an empty change. */
export function updateUser(id: string, change: UserChange): Promise<AdminUserOut> {
  return api.request<AdminUserOut>(`/admin/users/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(change),
  });
}

/** Erase an account and everything in it; resolves on the empty 204. */
export function deleteUser(id: string): Promise<void> {
  return api.request<void>(`/admin/users/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/** The error events matching `filters`, newest first by their last occurrence, one page of them. */
export function errors(filters: ErrorsFilters): Promise<ErrorsPageOut> {
  const { status, category, page, pageSize } = filters;
  return api.request<ErrorsPageOut>(
    `/admin/errors${queryString({ status, category, page, page_size: pageSize })}`,
  );
}

export function errorDetail(id: string): Promise<ErrorEventDetailOut> {
  return api.request<ErrorEventDetailOut>(`/admin/errors/${encodeURIComponent(id)}`);
}

/** Give an error event a new status; the server answers with the event. */
export function updateErrorStatus(id: string, status: ErrorStatus): Promise<ErrorEventOut> {
  return api.request<ErrorEventOut>(`/admin/errors/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

/**
 * The event as a Markdown document. Its body is text, not JSON, so it goes through
 * `api.text`: the one call in the console that does.
 */
export function errorBundle(id: string): Promise<string> {
  return api.text(`/admin/errors/${encodeURIComponent(id)}/bundle`);
}
