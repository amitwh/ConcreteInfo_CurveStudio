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

import type { components } from "./schema";

/**
 * Where the API lives. Same origin (`/api/v1`) by default; a static-only deployment can
 * point the built SPA at another host with `VITE_API_BASE` (for example
 * `VITE_API_BASE=https://api.example.com/api/v1`), and the API then lists that origin in
 * `CURVESTUDIO_CORS_ORIGINS`.
 */
export const API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").replace(/\/+$/, "");

/** Must match the API's CURVESTUDIO_MAX_UPLOAD_BYTES default (apps/api/.env.example). */
export const MAX_UPLOAD_BYTES = 20_000_000;

const REFRESH_KEY = "curvestudio.refresh";

export type Schemas = components["schemas"];
export type TokenOut = Schemas["TokenOut"];
export type UserOut = Schemas["UserOut"];
export type ModelInfo = Schemas["ModelInfo"];
export type DatasetOut = Schemas["DatasetOut"];
export type DatasetDetailOut = Schemas["DatasetDetailOut"];
export type PointsOut = Schemas["PointsOut"];
export type IssueOut = Schemas["IssueOut"];
export type JobOut = Schemas["JobOut"];
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

export interface UploadProgress {
  loaded: number;
  total: number;
}

class ApiClient {
  /** Set by the AuthProvider; invoked when the session is definitively dead. */
  onAuthFailure: (() => void) | null = null;

  private accessToken: string | null = null;
  private refreshInFlight: Promise<boolean> | null = null;

  get hasRefreshToken(): boolean {
    return localStorage.getItem(REFRESH_KEY) !== null;
  }

  setTokens(access: string, refresh: string): void {
    this.accessToken = access;
    localStorage.setItem(REFRESH_KEY, refresh);
  }

  clearTokens(): void {
    this.accessToken = null;
    localStorage.removeItem(REFRESH_KEY);
  }

  /** Test-only: detach listeners and forget tokens between tests. */
  resetForTests(): void {
    this.clearTokens();
    this.onAuthFailure = null;
  }

  /** Exchange the stored refresh token for a fresh pair; single-flight. */
  async refreshNow(): Promise<boolean> {
    if (this.refreshInFlight !== null) return this.refreshInFlight;
    this.refreshInFlight = (async () => {
      const refreshToken = localStorage.getItem(REFRESH_KEY);
      if (refreshToken === null) return false;
      try {
        const response = await fetch(`${API_BASE}/auth/refresh`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!response.ok) {
          // A rejected refresh token means the session is over (revoked or reused).
          this.clearTokens();
          this.onAuthFailure?.();
          return false;
        }
        const tokens = (await response.json()) as TokenOut;
        this.setTokens(tokens.access_token, tokens.refresh_token);
        return true;
      } catch {
        // Network trouble is not a logout: keep the token, let the caller surface the 401.
        return false;
      } finally {
        this.refreshInFlight = null;
      }
    })();
    return this.refreshInFlight;
  }

  async request<T>(path: string, init?: RequestInit, options?: RequestOptions): Promise<T> {
    const useAuth = options?.auth !== false;
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
      if (!response.ok) throw await toApiError(response);
      if (response.status === 204) return undefined as T;
      return (await response.json()) as T;
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

  async me(): Promise<UserOut> {
    return this.request<UserOut>("/me");
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
    if (!response.ok) throw await toApiError(response);
    const disposition = response.headers.get("content-disposition");
    const match = disposition?.match(/filename="?([^";]+)"?/);
    return { blob: await response.blob(), filename: match?.[1] ?? null };
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
