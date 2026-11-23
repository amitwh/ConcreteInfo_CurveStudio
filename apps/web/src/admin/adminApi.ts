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

/** A query string from the defined, non-empty parameters only; empty when there are none. */
function queryString(params: Record<string, string | number | undefined>): string {
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
