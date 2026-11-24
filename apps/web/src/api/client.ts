/**
 * Typed fetch wrapper around the CurveStudio API (contract: ../api/openapi.json,
 * types generated into ./schema.d.ts).
 *
 * Token strategy (a documented trade-off): the short-lived access token lives only in
 * memory — it dies with the tab and never touches storage — while the rotating refresh
 * token sits in localStorage so a reload keeps the session. XSS can therefore read the
 * refresh token; the deployment's strict CSP (`script-src 'self'`, no CDN, no inline
 * scripts) is what makes that acceptable, and refresh-token rotation + reuse detection
 * on the server limits the blast radius of a stolen token.
 */

import type { ServerErrorLike } from "../lib/errorReporter";
import { API_BASE } from "./apiBase";
import type { components } from "./schema";

export { API_BASE };

/** Must match the API's CURVESTUDIO_MAX_UPLOAD_BYTES default (apps/api/.env.example). */
export const MAX_UPLOAD_BYTES = 20_000_000;

const REFRESH_KEY = "curvestudio.refresh";

export type Schemas = components["schemas"];
export type TokenOut = Schemas["TokenOut"];
export type UserOut = Schemas["UserOut"];
/** The signed-in account in full: profile, preferences, the plan's limits and usage so far. */
export type MeOut = Schemas["MeOut"];
export type EmailVerificationOut = Schemas["EmailVerificationOut"];
export type MessageOut = Schemas["MessageOut"];
export type ModelInfo = Schemas["ModelInfo"];
export type DatasetOut = Schemas["DatasetOut"];
export type DatasetDetailOut = Schemas["DatasetDetailOut"];
export type PointsOut = Schemas["PointsOut"];
export type IssueOut = Schemas["IssueOut"];
export type JobOut = Schemas["JobOut"];
export type JobCreatedOut = Schemas["JobCreatedOut"];
export type FitAuditOut = Schemas["FitAuditOut"];
export type AuditCheckOut = Schemas["AuditCheckOut"];
export type FitMethodsOut = Schemas["FitMethodsOut"];
export type MethodsTextOut = Schemas["MethodsTextOut"];
/** A check's verdict, and the audit's overall verdict: a statement of evidence, never of proof. */
export type AuditVerdict = FitAuditOut["overall"];
export type FitBundle = Schemas["FitBundle"];
export type FitSummaryOut = Schemas["FitSummaryOut"];
export type Prediction = Schemas["Prediction"];
export type FitSpecJson = Schemas["FitSpec"];
export type RankBy = Schemas["RankBy"];
export type AutoFitRequest = Schemas["AutoFitRequest"];
export type AutofitOut = Schemas["AutofitOut"];
export type AutofitSummaryOut = Schemas["AutofitSummaryOut"];
export type AutoFitResult = Schemas["AutoFitResult"];
export type RankedFit = Schemas["RankedFit"];
export type CandidateOutcome = Schemas["CandidateOutcome"];
export type ProgressOut = Schemas["ProgressOut"];
export type RowCurve = Schemas["RowCurve"];
export type OpenRowOut = Schemas["OpenRowOut"];

/* ---- Statistics: the blocks of one statistical result ---- */
/** What one statistical tool returns: the only thing a result renderer is shown. */
export type ToolResult = Schemas["ToolResult"];
/** One estimated number, with its standard error and interval where the tool has them. */
export type Estimate = Schemas["Estimate"];
/** One hypothesis test as every surface shows it. */
export type HypothesisTest = Schemas["HypothesisTest"];
/** One assumption check of a statistical result (the generated `AuditCheck`, not a fit's
 *  `AuditCheckOut`). */
export type StatCheck = Schemas["AuditCheck"];
/** A table a result shows, and one of its columns. */
export type StatTable = Schemas["Table"];
export type StatTableColumn = Schemas["TableColumn"];
/** One cell of a table, already restricted to the plain values a table holds. */
export type TableCell = StatTable["rows"][number][number];
/** The five plots a result can carry, and the union a renderer switches on. */
export type ScatterPlot = Schemas["ScatterPlot"];
export type HistogramPlot = Schemas["HistogramPlot"];
export type QQPlot = Schemas["QQPlot"];
export type BoxPlot = Schemas["BoxPlot"];
export type MultiLinePlot = Schemas["MultiLinePlot"];
export type StatPlot = ToolResult["plots"][number];
/** The statistics catalogue: the families as (key, title) pairs and every tool. */
export type StatsToolsOut = Schemas["StatsToolsOut"];
/** One tool as a surface lists it: its family, its request fields and whether it needs data. */
export type ToolInfo = Schemas["ToolInfo"];
/** One request field: a column, a group, several columns, or an option with its schema. */
export type RequestField = Schemas["RequestField"];
/** One stored statistical result: the choices as sent and the result the core returned. */
export type StatResultOut = Schemas["StatResultOut"];
/** One stored result as a listing shows it: no result body. */
export type StatResultSummaryOut = Schemas["StatResultSummaryOut"];

export class ApiError extends Error {
  /** HTTP status; 0 means the server could not be reached at all. */
  readonly status: number;
  /** Machine-readable code from the error envelope (or NETWORK / HTTP_n). */
  readonly code: string;
  /** The envelope's `detail` object, when present. */
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

function isEnvelope(body: unknown): body is { error: string; message: string; detail: unknown } {
  if (typeof body !== "object" || body === null) return false;
  const candidate = body as Record<string, unknown>;
  return typeof candidate.error === "string" && typeof candidate.message === "string";
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // non-JSON body: fall through with the status text
  }
  if (isEnvelope(body)) {
    return new ApiError(response.status, body.error, body.message, body.detail);
  }
  // FastAPI's own 422 shape: {detail: [{loc, msg, type}, ...]}
  if (
    typeof body === "object" &&
    body !== null &&
    Array.isArray((body as { detail?: unknown }).detail)
  ) {
    const first = ((body as { detail: unknown[] }).detail[0] ?? {}) as { msg?: string };
    return new ApiError(
      response.status,
      "VALIDATION_ERROR",
      first.msg ?? "the request was rejected",
      body,
    );
  }
  return new ApiError(
    response.status,
    `HTTP_${response.status}`,
    response.statusText || "the request failed",
    body,
  );
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function networkError(): ApiError {
  return new ApiError(
    0,
    "NETWORK",
    "could not reach the server — check your connection and try again",
    null,
  );
}

interface RequestOptions {
  /** Skip attaching/refreshing the bearer token (auth endpoints). */
  auth?: boolean;
}

interface UploadFields {
  name?: string;
  xColumn?: string;
  yColumn?: string;
  weightColumn?: string;
}

/** The preferences to change; a key that is not named keeps its value. */
export interface PreferencesChange {
  error_reports?: boolean;
  report_byline?: boolean;
}

export interface UploadProgress {
  loaded: number;
  total: number;
}

class ApiClient {
  /** Set by the AuthProvider; invoked when the session is definitively dead. */
  onAuthFailure: (() => void) | null = null;
  /**
   * Set at startup with the error reporter's `reportApiError`. Invoked for every answer of
   * 500 and above, before the error is thrown to the caller, which is told nothing about it.
   */
  onServerError: ((response: ServerErrorLike, path: string) => void) | null = null;

  private accessToken: string | null = null;
  private refreshInFlight: Promise<boolean> | null = null;

  get hasRefreshToken(): boolean {
    return this.refreshToken !== null;
  }

  /** The stored refresh token: the one request that names the session to keep or to end. */
  get refreshToken(): string | null {
    return localStorage.getItem(REFRESH_KEY);
  }

  setTokens(access: string, refresh: string): void {
    this.accessToken = access;
    localStorage.setItem(REFRESH_KEY, refresh);
  }

  /** Adopt a token pair issued by any auth flow (password sign-in, OAuth exchange). */
  adoptTokens(tokens: TokenOut): void {
    this.setTokens(tokens.access_token, tokens.refresh_token);
  }

  clearTokens(): void {
    this.accessToken = null;
    localStorage.removeItem(REFRESH_KEY);
  }

  /** Test-only: detach listeners and forget tokens and any refresh still in flight, between tests. */
  resetForTests(): void {
    this.clearTokens();
    this.onAuthFailure = null;
    this.onServerError = null;
    this.refreshInFlight = null;
  }

  /** Hand a 5xx answer to the error reporter; the body is not its business. */
  private noteServerError(
    status: number,
    url: string,
    requestId: string | null,
    path: string,
  ): void {
    if (status >= 500) this.onServerError?.({ status, url, requestId }, path);
  }

  /** Exchange the stored refresh token for a fresh pair; single-flight. */
  async refreshNow(): Promise<boolean> {
    if (this.refreshInFlight !== null) return this.refreshInFlight;
    // Asked before any flight is opened, so it leaves nothing behind: nothing to refresh
    // with is not a logout, and the 401 is the caller's to surface.
    const refreshToken = localStorage.getItem(REFRESH_KEY);
    if (refreshToken === null) return false;
    const flight = this.exchangeRefreshToken(refreshToken);
    this.refreshInFlight = flight;
    try {
      return await flight;
    } finally {
      // Only its own: resetForTests may have forgotten it, or a later flight replaced it.
      if (this.refreshInFlight === flight) this.refreshInFlight = null;
    }
  }

  /**
   * One refresh call. Its first step awaits, so `refreshNow` records it as in flight before
   * it can finish.
   */
  private async exchangeRefreshToken(refreshToken: string): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE}/auth/refresh`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      // The body is read first, so that the check below is the last step before a token changes.
      const tokens = response.ok ? ((await response.json()) as TokenOut) : null;
      // The answer belongs to the session that asked, named by the token it sent. Signed out
      // or replaced by another sign-in while the request was away, it is dropped: the pair
      // is not stored, and a refusal must not end the session that took its place.
      if (this.refreshToken !== refreshToken) return false;
      if (tokens === null) {
        // A rejected refresh token means the session is over (revoked or reused).
        this.clearTokens();
        this.onAuthFailure?.();
        return false;
      }
      this.setTokens(tokens.access_token, tokens.refresh_token);
      return true;
    } catch {
      // Network trouble is not a logout: keep the token, let the caller surface the 401.
      return false;
    }
  }

  async request<T>(path: string, init?: RequestInit, options?: RequestOptions): Promise<T> {
    return this.send(path, init, options?.auth !== false, async (response) => {
      if (response.status === 204) return undefined as T;
      return (await response.json()) as T;
    });
  }

  /**
   * The same request as `request` — bearer token, one refresh on a 401, the same errors —
   * for the one endpoint whose body is a document rather than JSON: the body comes back
   * as text, unparsed. It holds no state of its own, so `resetForTests` covers it.
   */
  async text<T extends string = string>(path: string, init?: RequestInit): Promise<T> {
    return this.send(path, init, true, async (response) => (await response.text()) as T);
  }

  /** What `request` and `text` share: everything but how a successful body is read. */
  private async send<T>(
    path: string,
    init: RequestInit | undefined,
    useAuth: boolean,
    read: (response: Response) => Promise<T>,
  ): Promise<T> {
    const call = async (retried: boolean): Promise<T> => {
      const headers = new Headers(init?.headers);
      if (!headers.has("content-type") && init?.body != null) {
        headers.set("content-type", "application/json");
      }
      if (useAuth && this.accessToken !== null) {
        headers.set("Authorization", `Bearer ${this.accessToken}`);
      }
      let response: Response;
      try {
        response = await fetch(`${API_BASE}${path}`, { ...init, headers });
      } catch (error) {
        if (isAbortError(error)) throw error;
        throw networkError();
      }
      if (response.status === 401 && useAuth && !retried && !path.startsWith("/auth/")) {
        if (await this.refreshNow()) return call(true);
        // refreshNow already cleared tokens and fired onAuthFailure on rejection.
      }
      this.noteServerError(
        response.status,
        response.url,
        response.headers.get("x-request-id"),
        path,
      );
      if (!response.ok) throw await toApiError(response);
      return read(response);
    };
    return call(false);
  }

  async login(email: string, password: string): Promise<TokenOut> {
    const tokens = await this.request<TokenOut>(
      "/auth/login",
      { method: "POST", body: JSON.stringify({ email, password }) },
      { auth: false },
    );
    // The session starts here, not at the next refresh: store the pair immediately so
    // the very next request (GET /me) is authenticated.
    this.setTokens(tokens.access_token, tokens.refresh_token);
    return tokens;
  }

  async register(email: string, password: string, name?: string): Promise<UserOut> {
    return this.request<UserOut>(
      "/auth/register",
      { method: "POST", body: JSON.stringify({ email, password, name }) },
      { auth: false },
    );
  }

  /** The social providers this deployment has configured; empty when none are. */
  async oauthProviders(): Promise<string[]> {
    const out = await this.request<{ providers: string[] }>(
      "/auth/oauth/providers",
      undefined,
      { auth: false },
    );
    return out.providers;
  }

  /**
   * Claim a single-use OAuth exchange code for a token pair. The caller adopts the pair
   * through the auth context, which also loads the profile (/me).
   */
  async oauthExchange(code: string): Promise<TokenOut> {
    return this.request<TokenOut>(
      "/auth/oauth/exchange",
      { method: "POST", body: JSON.stringify({ code }) },
      { auth: false },
    );
  }

  /** Consume a verification token (anonymous: auth links arrive from an email). */
  async verifyEmail(token: string): Promise<EmailVerificationOut> {
    return this.request<EmailVerificationOut>(
      "/auth/verify-email",
      { method: "POST", body: JSON.stringify({ token }) },
      { auth: false },
    );
  }

  /** Ask for a fresh verification link. Always 202, so the address is never revealed. */
  async resendVerification(email: string): Promise<MessageOut> {
    return this.request<MessageOut>(
      "/auth/verify-email/resend",
      { method: "POST", body: JSON.stringify({ email }) },
      { auth: false },
    );
  }

  /** Ask for a password-reset link. Always 202, so the address is never revealed. */
  async forgotPassword(email: string): Promise<MessageOut> {
    return this.request<MessageOut>(
      "/auth/password/forgot",
      { method: "POST", body: JSON.stringify({ email }) },
      { auth: false },
    );
  }

  /** Set a new password from a reset token. */
  async resetPassword(token: string, newPassword: string): Promise<MessageOut> {
    return this.request<MessageOut>(
      "/auth/password/reset",
      { method: "POST", body: JSON.stringify({ token, new_password: newPassword }) },
      { auth: false },
    );
  }

  async logout(): Promise<void> {
    const refreshToken = localStorage.getItem(REFRESH_KEY);
    if (refreshToken === null) return;
    try {
      await this.request(
        "/auth/logout",
        { method: "POST", body: JSON.stringify({ refresh_token: refreshToken }) },
        { auth: false },
      );
    } catch {
      // The server may be gone; the local session still ends.
    } finally {
      this.clearTokens();
    }
  }

  async me(): Promise<MeOut> {
    return this.request<MeOut>("/me");
  }

  /** Change the displayed name (the server strips it; 1–100 characters). */
  async updateMe(name: string): Promise<MeOut> {
    return this.request<MeOut>("/me", { method: "PATCH", body: JSON.stringify({ name }) });
  }

  async updatePreferences(change: PreferencesChange): Promise<MeOut> {
    return this.request<MeOut>("/me/preferences", {
      method: "PUT",
      body: JSON.stringify(change),
    });
  }

  /**
   * Set a new password. Every other session is revoked; the session holding the stored
   * refresh token (sent along, so the server knows which one to keep) stays signed in.
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<MeOut> {
    return this.request<MeOut>("/me/password", {
      method: "POST",
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
        refresh_token: this.refreshToken,
      }),
    });
  }

  /** Sign out every other session; this one, named by its refresh token, stays. */
  async revokeAllSessions(): Promise<MeOut> {
    return this.request<MeOut>("/me/sessions/revoke-all", {
      method: "POST",
      body: JSON.stringify({ refresh_token: this.refreshToken }),
    });
  }

  /** Delete the account. `confirm` must be the part of the e-mail before the @. */
  async deleteMe(confirm: string): Promise<void> {
    return this.request<void>("/me", {
      method: "DELETE",
      body: JSON.stringify({ confirm }),
    });
  }

  /**
   * Multipart upload with progress and abort. Uses XHR because fetch cannot report
   * upload progress; a 401 mid-upload goes through the same single-flight refresh and
   * the upload is repeated once.
   */
  uploadDataset(
    file: File,
    fields: UploadFields,
    onProgress?: (progress: UploadProgress) => void,
    signal?: AbortSignal,
  ): Promise<DatasetOut> {
    const send = (retried: boolean): Promise<DatasetOut> =>
      new Promise<DatasetOut>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/datasets`);
        if (this.accessToken !== null) {
          xhr.setRequestHeader("Authorization", `Bearer ${this.accessToken}`);
        }
        xhr.upload.onprogress = (event): void => {
          if (event.lengthComputable) onProgress?.({ loaded: event.loaded, total: event.total });
        };
        const onAbort = (): void => xhr.abort();
        signal?.addEventListener("abort", onAbort);
        xhr.onload = (): void => {
          signal?.removeEventListener("abort", onAbort);
          const finish = async (): Promise<void> => {
            if (xhr.status === 401 && !retried && (await this.refreshNow())) {
              resolve(await send(true));
              return;
            }
            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                resolve(JSON.parse(String(xhr.response)) as DatasetOut);
              } catch {
                reject(new ApiError(xhr.status, "HTTP_" + xhr.status, "malformed response", null));
              }
              return;
            }
            this.noteServerError(
              xhr.status,
              xhr.responseURL,
              xhr.getResponseHeader("x-request-id"),
              "/datasets",
            );
            let body: unknown = null;
            try {
              body = JSON.parse(String(xhr.response));
            } catch {
              // leave null
            }
            if (isEnvelope(body)) {
              reject(new ApiError(xhr.status, body.error, body.message, body.detail));
            } else {
              reject(
                new ApiError(
                  xhr.status,
                  `HTTP_${xhr.status}`,
                  `upload failed (${xhr.status})`,
                  body,
                ),
              );
            }
          };
          void finish().catch((error: unknown) => reject(error));
        };
        xhr.onerror = (): void => {
          signal?.removeEventListener("abort", onAbort);
          reject(networkError());
        };
        xhr.onabort = (): void => {
          signal?.removeEventListener("abort", onAbort);
          reject(new DOMException("aborted", "AbortError"));
        };
        const form = new FormData();
        form.append("file", file);
        if (fields.name) form.append("name", fields.name);
        if (fields.xColumn) form.append("x_column", fields.xColumn);
        if (fields.yColumn) form.append("y_column", fields.yColumn);
        if (fields.weightColumn) form.append("weight_column", fields.weightColumn);
        xhr.send(form);
      });
    return send(false);
  }

  /** Download a file (fit export) as a Blob with the bearer token attached. */
  async download(path: string): Promise<{ blob: Blob; filename: string | null }> {
    const headers = new Headers();
    if (this.accessToken !== null) headers.set("Authorization", `Bearer ${this.accessToken}`);
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${path}`, { headers });
    } catch (error) {
      if (isAbortError(error)) throw error;
      throw networkError();
    }
    this.noteServerError(
      response.status,
      response.url,
      response.headers.get("x-request-id"),
      path,
    );
    if (!response.ok) throw await toApiError(response);
    const disposition = response.headers.get("content-disposition");
    const match = disposition?.match(/filename="?([^";]+)"?/);
    return { blob: await response.blob(), filename: match?.[1] ?? null };
  }

  /** The audit of a stored fit: each check with its verdict, evidence and next step. */
  async fitAudit(fitId: string): Promise<FitAuditOut> {
    return this.request<FitAuditOut>(`/fits/${fitId}/audit`);
  }

  /**
   * Queue the cross-validation check of a stored fit; the job uses no quota and stores its
   * result on the fit, where `fitAudit` reads it. While one is queued or running, a second
   * request is a 409 whose `detail.job_id` names the job to follow.
   */
  async runFitAuditCrossValidation(
    fitId: string,
    folds: number | "loo" = 5,
  ): Promise<JobCreatedOut> {
    return this.request<JobCreatedOut>(`/fits/${fitId}/audit/cross-validation`, {
      method: "POST",
      body: JSON.stringify({ folds }),
    });
  }

  /**
   * The methods text of a stored fit in three styles (plain text, Markdown, LaTeX). The
   * server refuses a fit that did not converge with a 422.
   */
  async fitMethods(fitId: string): Promise<FitMethodsOut> {
    return this.request<FitMethodsOut>(`/fits/${fitId}/methods`);
  }

  /** Every statistical tool, with the families a person meets them in. */
  async statsTools(): Promise<StatsToolsOut> {
    return this.request<StatsToolsOut>("/stats/tools");
  }

  /**
   * Run a tool on its choices; with a dataset the columns are read from it, without one a
   * sample-size tool runs from its options alone.
   */
  async runStat(
    tool: string,
    choices: Record<string, unknown>,
    datasetId: string | null,
  ): Promise<StatResultOut> {
    const path = datasetId === null ? `/stats/${tool}` : `/datasets/${datasetId}/stats/${tool}`;
    return this.request<StatResultOut>(path, {
      method: "POST",
      body: JSON.stringify(choices),
    });
  }

  /** The user's stored results, newest first; only one dataset's when an id is given. */
  async statResults(datasetId: string | null): Promise<StatResultSummaryOut[]> {
    const query = datasetId === null ? "" : `?dataset_id=${encodeURIComponent(datasetId)}`;
    return this.request<StatResultSummaryOut[]>(`/stats/results${query}`);
  }

  /** One stored result, with its full result body. */
  async statResult(id: string): Promise<StatResultOut> {
    return this.request<StatResultOut>(`/stats/results/${id}`);
  }

  /** Delete a stored result. */
  async deleteStatResult(id: string): Promise<void> {
    return this.request<void>(`/stats/results/${id}`, { method: "DELETE" });
  }
}

export const api = new ApiClient();

/** Save a Blob as a file download (used by the export buttons). */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
