/**
 * Anonymous error reporting for the web app (contract: `POST /api/v1/errors`, 204, no body).
 *
 * Two rules shape everything here.
 *
 * 1. The reporter must never report itself and must never double-count. It sends with raw
 *    `fetch`, never through the API client. Whatever it hands to the server is final: a
 *    refused or failed request is dropped, never retried, never put back. A same-origin 5xx
 *    that carries `X-Request-Id` is already the server's own record, so it is skipped.
 * 2. A report never identifies a person and never holds their data. No account id, e-mail
 *    or name, no token, no dataset content, no query string. The route is the page path
 *    without its query, any UUID shown as `{id}`, and `scrubText` blanks whatever looks like
 *    a credential in everything that leaves.
 *
 * Nothing here logs and nothing here throws: a failure of the reporter's own that reached the
 * page would be caught by the reporter's own hooks and reported as a new error.
 *
 * With consent off nothing is collected at all: no event is built and no console line is kept.
 */

/** What the API client knows about a failed answer, without the body (which may hold data). */
export interface ServerErrorLike {
  /** HTTP status of the answer. */
  status: number;
  /** The final URL the answer came from. */
  url: string;
  /** The `X-Request-Id` header the server stamps on each 5xx it produces, when it reached us. */
  requestId: string | null;
}

/** One event exactly as `POST /errors` takes it: the server refuses an unknown key. */
export interface ErrorReportEvent {
  category: "crash" | "ui" | "network";
  severity: "error" | "fatal";
  error_type: string;
  message: string;
  stack: string;
  route: string;
  user_agent: string;
  app_version: string;
  console_log: string;
  context: Record<string, unknown>;
  /** Set when the failed request carried one, so the administrator can match the server's record. */
  request_id?: string;
}

/** The three fields an error contributes, scrubbed and cut. */
interface ErrorParts {
  error_type: string;
  message: string;
  stack: string;
}

/** Everything one installation owns. It dies with the installation: a request still in flight when the reporter is removed touches this state and nothing newer. */
interface Reporter {
  endpoint: string;
  /** The console ring buffer, newest last. */
  consoleLines: string[];
  /** Every fingerprint input already queued on this page. */
  seen: Set<string>;
  queue: Queued[];
  flushTimer: number | null;
  flushing: boolean;
}

interface Queued {
  event: ErrorReportEvent;
  bytes: number;
}

const CONSENT_KEY = "curvestudio.errorReports";

/** The server's limits, as the client keeps to them. */
const CAP = {
  errorType: 200,
  message: 2_000,
  // Short on purpose: a request of five events must stay under the server's 64 KB.
  stack: 6_000,
  console: 4_000,
  route: 200,
  userAgent: 500,
  appVersion: 40,
  requestId: 64,
  description: 1_000,
  contextValue: 2_000,
  context: 8_000,
} as const;

/**
 * One request carries at most five events and at most this many bytes: comfortably under
 * the server's 64 KB whichever way it counts a kilobyte. A batch over the limit is refused
 * whole, so the count alone is not enough to keep a batch safe.
 */
const BATCH_MAX_BYTES = 56_000;
const PER_REQUEST_MAX = 5;
const QUEUE_MAX = 20;
const FLUSH_AT = 5;
const FLUSH_DELAY_MS = 10_000;
const CONSOLE_MAX_LINES = 20;
const CONSOLE_LINE_MAX = 1_000;
/** What the recovery screen shows, and what a user's note quotes: the start of the fingerprint. */
const REFERENCE_LENGTH = 12;

/*
 * The server's own scrub (``scrub_secrets`` in apps/api .../modules/errors/service.py), rule
 * for rule, in its order and with its replacement texts. What leaves here is then already
 * what the server would store: its scrub changes nothing on it, and the fingerprint computed
 * here from these fields is the one it computes there.
 *
 * Python's `re` reads `\w`, `\d` and `\s` as Unicode for text; JavaScript reads them as
 * ASCII only. They are spelled out here (under the `u` flag) so that the two agree on every
 * input, not merely on ASCII ones.
 */
const WORD = String.raw`\p{L}\p{N}_`;
const DIGIT = String.raw`\p{Nd}`;
/** Python's whitespace for text: the ASCII blanks, the four separators U+001C-U+001F, NEL, and the Unicode spaces. */
const SPACE =
  String.raw`\t\n\v\f\r \x1c-\x1f\x85\xa0\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}`;

const REDACTED = "[REDACTED]";
const REDACTED_JWT = "[REDACTED_JWT]";
const REDACTED_EMAIL = "[REDACTED_EMAIL]";

/**
 * `Authorization: <scheme> <credentials>`: the scheme stays, the credentials go. A value with
 * no scheme is a credential too. Under Python's `IGNORECASE` an `i` also reads as `İ` or `ı`
 * (in the name, and in the class that opens the scheme), which JavaScript's `iu` does not.
 */
const AUTHORIZATION = new RegExp(
  String.raw`(Author[iİı]zat[iİı]on["']?[${SPACE}]*[:=][${SPACE}]*["']?)((?:[A-Za-zİı][${WORD}.-]*[${SPACE}]+)?)([^${SPACE}"',;)}]+)`,
  "giu",
);
/** A JWT: three base64url segments, the first opening with the `eyJ` of a JSON header. */
const JWT = new RegExp(String.raw`eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+`, "gu");
/**
 * The value of anything named `password`, `secret`, `token`, `code` or `key`. The name is a
 * substring (`monkey=` counts): redacting too much is the safe side. A quoted value goes
 * whole; a bare one ends at whitespace or a separator, but a `]` may be part of it.
 */
const NAMED_SECRET = new RegExp(
  String.raw`((?:password|secret|token|code|key)["']?[${SPACE}]*[:=][${SPACE}]*)(?:"([^"]*)"|'([^']*)'|([^${SPACE}"',;)}&]+))`,
  "giu",
);
/** `Bearer <token>` with no `Authorization` before it is a credential all the same. Eight or more token characters, so "the bearer of bad news" stays. */
const BEARER = new RegExp(String.raw`(Bearer[${SPACE}]+)[A-Za-z0-9._~+/=-]{8,}`, "giu");
/** An address names an account as surely as an id does. The domain needs a dot: `user@host` stays. */
const EMAIL = new RegExp(String.raw`[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+`, "gu");

/* Wider than the server's, and applied after its rules: no id and no query string leaves. */
const UUID = /[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}/g;
/** The query or fragment of a URL: a sign-in or password-reset token travels there. */
const URL_QUERY = /(https?:\/\/[^\s<>"'`?#]+)[?#][^\s<>"'`]*/g;

/** `file.ext:line` as a JavaScript stack names it, and a Python frame. */
const JS_FRAME = new RegExp(String.raw`[${WORD}.-]+\.[A-Za-z]{1,8}:${DIGIT}+`, "gu");
/** Not global: it is read with `exec()`, which would otherwise resume after its last match. */
const PYTHON_FRAME = new RegExp(String.raw`File "([^"]+)", line (${DIGIT}+)`, "u");
/** Ends in a word character, as Python's `re.search(r"\w$", …)` sees it: its `$` also matches just before a newline that ends the string. */
const ENDS_IN_WORD = new RegExp(String.raw`[${WORD}]\n?$`, "u");

/** The installed reporter; nothing is collected or sent while this is null. */
let reporter: Reporter | null = null;
let removeCurrent: (() => void) | null = null;

/* ---- consent ---------------------------------------------------------------------------- */

let consentResolver: (() => boolean | null) | null = null;
/** Told whenever the answer of `errorReportsEnabled()` may have changed. */
const consentListeners = new Set<() => void>();

function consentChanged(): void {
  for (const listener of consentListeners) listener();
}

/**
 * To be told when consent changes (the recovery screen shows different things either way).
 * Returns the function that stops the subscription.
 */
export function subscribeConsent(listener: () => void): () => void {
  consentListeners.add(listener);
  return () => {
    consentListeners.delete(listener);
  };
}

/** Whether reports may leave this browser right now. */
export function errorReportsEnabled(): boolean {
  const resolved = consentResolver?.() ?? null;
  if (resolved !== null) return resolved;
  return localStorage.getItem(CONSENT_KEY) !== "off";
}

/** The visitor's own choice; it decides whenever no account preference does. */
export function setErrorReportsEnabled(on: boolean): void {
  if (on) localStorage.removeItem(CONSENT_KEY);
  else localStorage.setItem(CONSENT_KEY, "off");
  consentChanged();
}

/**
 * Hands in the signed-in account's preference. While a resolver is set it decides; one that
 * answers `null` (the session is not an account's) hands the decision to the visitor's setting.
 */
export function setConsentResolver(resolver: (() => boolean | null) | null): void {
  consentResolver = resolver;
  consentChanged();
}

/* ---- text that leaves the browser ------------------------------------------------------- */

/**
 * The server's scrub itself, rule for rule: the credentials of an `Authorization` value, a
 * `Bearer` token on its own, JWTs, the value of anything named like a secret, and e-mail
 * addresses. Repeated until the
 * text stops changing, because the server scrubs what it is given and fingerprints that: what
 * leaves here has to be a fixed point of its scrub, or the reference shown to the user is not
 * the one the administrator sees.
 */
export function scrubSecrets(text: string): string {
  let current = text;
  // One pass can leave work for the next: a word that holds a JWT and precedes the credentials
  // of an `Authorization` value is read as its scheme until the JWT is gone. No chain needs
  // more than two passes; four is the limit all the same.
  for (let pass = 0; pass < 4; pass++) {
    const next = scrubOnce(current);
    if (next === current) return current;
    current = next;
  }
  return current;
}

function scrubOnce(text: string): string {
  return text
    .replace(AUTHORIZATION, (_match, lead: string, scheme: string) => `${lead}${scheme}${REDACTED}`)
    .replace(JWT, REDACTED_JWT)
    .replace(NAMED_SECRET, redactNamed)
    .replace(BEARER, (_match, lead: string) => `${lead}${REDACTED}`)
    .replace(EMAIL, REDACTED_EMAIL);
}

/** The server's scrub, then a little more: ids as `{id}`, and a URL without its query. */
export function scrubText(text: string): string {
  return scrubSecrets(text)
    .replace(URL_QUERY, "$1")
    .replace(UUID, "{id}");
}

/** The value of a named secret goes; its quotes, if any, stay. */
function redactNamed(_match: string, prefix: string, double: string | undefined, single: string | undefined): string {
  if (double !== undefined) return `${prefix}"${REDACTED}"`;
  if (single !== undefined) return `${prefix}'${REDACTED}'`;
  return `${prefix}${REDACTED}`;
}

/**
 * Where a stack fails, as the server reads it: the first `file.ext:line` of a JavaScript
 * stack (a URL counts by its file name; the column is not kept), else the first Python frame
 * as `basename:line`, else nothing. Part of the fingerprint.
 */
export function topFrame(stack: string): string {
  for (const match of stack.matchAll(JS_FRAME)) {
    const before = stack.slice(0, match.index);
    // `host.tld:port` reads the same way, but after `//` it is where the app was served
    // from; and nor is the tail of a longer token a frame.
    if (before.endsWith("//") || ENDS_IN_WORD.test(before)) continue;
    return match[0];
  }
  const python = PYTHON_FRAME.exec(stack);
  if (python !== null) {
    const file = (python[1] ?? "").replaceAll("\\", "/").split("/").pop() ?? "";
    return `${file}:${python[2] ?? ""}`;
  }
  return "";
}

/** The type, message and stack of anything that was thrown or rejected. */
export function eventFromError(error: unknown): ErrorParts {
  if (error instanceof Error) return parts(error.name, error.message, error.stack ?? "");
  // Not an Error: a cross-realm one, or a rejection of a string, a number, a plain object.
  const reason = error as { name?: unknown; message?: unknown; stack?: unknown } | null;
  if (typeof reason?.message === "string") {
    return parts(
      typeof reason.name === "string" ? reason.name : "NonError",
      reason.message,
      typeof reason.stack === "string" ? reason.stack : "",
    );
  }
  // An object is named by its type alone: its contents are not the reporter's to read (it
  // may be a dataset or a response body). A primitive says itself.
  const described =
    typeof error === "object" ? Object.prototype.toString.call(error) : String(error);
  return parts("NonError", described, "");
}

function parts(errorType: string, message: string, stack: string): ErrorParts {
  return {
    error_type: cut(scrubText(errorType), CAP.errorType) || "Error",
    message: cut(scrubText(message), CAP.message) || "(no message)",
    stack: cut(scrubText(stack), CAP.stack),
  };
}

function cut(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) : text;
}

/** The last `max` characters, without the line the cut opened. */
function tail(text: string, max: number): string {
  if (text.length <= max) return text;
  const kept = text.slice(text.length - max);
  const firstWholeLine = kept.indexOf("\n");
  return firstWholeLine === -1 ? kept : kept.slice(firstWholeLine + 1);
}

/* ---- events ---------------------------------------------------------------------------- */

interface EventInput {
  category: ErrorReportEvent["category"];
  severity: ErrorReportEvent["severity"];
  parts: ErrorParts;
  /** Added to the context; a `path` of its own replaces the page path (an API path). */
  context?: Record<string, unknown>;
  requestId?: string | null;
}

/** The event as it goes out: every field scrubbed, capped, and nothing else on it. */
function buildEvent(r: Reporter, input: EventInput): ErrorReportEvent {
  const event: ErrorReportEvent = {
    category: input.category,
    severity: input.severity,
    ...input.parts,
    route: pagePath(),
    user_agent: cut(navigator.userAgent, CAP.userAgent),
    app_version: cut(__APP_VERSION__, CAP.appVersion),
    console_log: tail(r.consoleLines.map(scrubText).join("\n"), CAP.console),
    context: context(input.context ?? {}),
  };
  if (input.requestId) event.request_id = cut(scrubText(input.requestId), CAP.requestId);
  return event;
}

/** The page path: no query, no UUID. It is the `route` of every event. */
function pagePath(): string {
  return cut(scrubText(window.location.pathname), CAP.route);
}

/** `{ path, viewport, online }` plus what a caller adds, as primitives only. */
function context(extra: Record<string, unknown>): Record<string, unknown> {
  const base = {
    path: pagePath(),
    viewport: { width: window.innerWidth, height: window.innerHeight },
    online: navigator.onLine,
  };
  const additions: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(extra)) {
    if (typeof value === "string") additions[key] = cut(scrubText(value), CAP.contextValue);
    else if (value === null || typeof value === "number" || typeof value === "boolean") {
      additions[key] = value;
    }
    // anything else does not leave: an object's contents are not the reporter's to read
  }
  const merged = { ...base, ...additions };
  // The server takes 10 KB of context and refuses the whole batch beyond it.
  return JSON.stringify(merged).length <= CAP.context ? merged : base;
}

/**
 * What the server hashes into the fingerprint: the same five fields, in the same order. It
 * is compared before hashing, so two errors in one tick are still told apart correctly.
 */
function fingerprintInput(event: ErrorReportEvent): string {
  return [
    event.category,
    event.error_type,
    event.message.slice(0, 200),
    topFrame(event.stack),
    event.route,
  ].join("|");
}

/** SHA-256 hex, as the server computes it. */
async function fingerprint(input: string): Promise<string> {
  // `crypto.subtle` exists only on a secure context (https, or localhost), so it is looked up, not assumed.
  const subtle = globalThis.crypto?.subtle;
  if (subtle !== undefined) {
    try {
      const digest = await subtle.digest("SHA-256", new TextEncoder().encode(input));
      return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
        "",
      );
    } catch {
      // an environment that has the object but refuses the call: the fallback below
    }
  }
  return fallbackHash(input);
}

/**
 * The fallback where `crypto.subtle` is missing: two 32-bit FNV-1a passes, forwards and
 * backwards. It is not SHA-256. It is stable, so errors still dedupe and the recovery screen
 * still has a reference to show, but that reference is not the one the administrator sees.
 */
function fallbackHash(text: string): string {
  const fnv = (value: string, seed: number): string => {
    let hash = seed;
    for (let index = 0; index < value.length; index++) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
  };
  return fnv(text, 0x811c9dc5) + fnv(Array.from(text).reverse().join(""), 0x811c9dc5);
}

/* ---- the queue -------------------------------------------------------------------------- */

/**
 * Queues one event, unless its fingerprint was already seen on this page. A flush starts at
 * five events, or ten seconds after the first; the queue holds twenty and drops its oldest.
 */
function enqueue(r: Reporter, event: ErrorReportEvent): void {
  if (!errorReportsEnabled()) return;
  const input = fingerprintInput(event);
  if (r.seen.has(input)) return;
  r.seen.add(input);
  r.queue.push({ event, bytes: new TextEncoder().encode(JSON.stringify(event)).length });
  if (r.queue.length > QUEUE_MAX) r.queue.shift();
  if (r.queue.length >= FLUSH_AT) startFlush(r);
  else if (r.flushTimer === null) armFlushTimer(r);
}

/** Takes up to `limit` events that fit one request off the front of the queue. The first always goes. */
function fit(r: Reporter, limit: number): ErrorReportEvent[] {
  const events: ErrorReportEvent[] = [];
  let bytes = 0;
  while (r.queue.length > 0 && events.length < limit) {
    const next = r.queue[0];
    if (next === undefined || (events.length > 0 && bytes + next.bytes > BATCH_MAX_BYTES)) break;
    r.queue.shift();
    events.push(next.event);
    bytes += next.bytes + 1;
  }
  return events;
}

function armFlushTimer(r: Reporter): void {
  r.flushTimer = window.setTimeout(() => {
    r.flushTimer = null;
    startFlush(r);
  }, FLUSH_DELAY_MS);
}

function clearFlushTimer(r: Reporter): void {
  if (r.flushTimer === null) return;
  window.clearTimeout(r.flushTimer);
  r.flushTimer = null;
}

/** One flush at a time: one asked for while another runs is picked up by that loop instead. */
function startFlush(r: Reporter): void {
  clearFlushTimer(r);
  if (r.flushing) return;
  r.flushing = true;
  void flush(r);
}

async function flush(r: Reporter): Promise<void> {
  try {
    do {
      const batch = fit(r, PER_REQUEST_MAX);
      if (batch.length === 0) return;
      await send(r, batch);
    } while (r.queue.length >= FLUSH_AT);
  } catch {
    // Nothing the reporter does may end in an unhandled rejection: this reporter would report it.
  } finally {
    r.flushing = false;
    if (r.queue.length > 0 && r.flushTimer === null) armFlushTimer(r);
  }
}

/** One POST. True when the server took it; anything else drops the batch for good. */
async function send(r: Reporter, events: ErrorReportEvent[]): Promise<boolean> {
  try {
    const response = await fetch(r.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ events }),
      keepalive: true,
    });
    return response.ok;
  } catch {
    // Offline, blocked, too large for the browser to keep alive: dropped, never retried.
    return false;
  }
}

/** Run one piece of the reporter: it must not throw into the page, where its own hooks wait. */
function guard(run: () => void): void {
  try {
    run();
  } catch {
    // dropped by design (see the note at the top); nothing here may log
  }
}

/* ---- what the app and the client call --------------------------------------------------- */

/**
 * A 5xx the API client got. Below 500: nothing. A same-origin answer with `X-Request-Id` is
 * the server's own record already and is skipped; one without it, or from another origin,
 * is reported as a network error.
 */
export function reportApiError(response: ServerErrorLike, path: string): void {
  guard(() => {
    const r = reporter;
    if (r === null || response.status < 500) return;
    if (response.requestId !== null && isSameOrigin(response.url)) return;
    const apiPath = cut(scrubText(path.split("?", 1)[0] ?? ""), CAP.route);
    enqueue(
      r,
      buildEvent(r, {
        category: "network",
        severity: "error",
        parts: parts("ServerError", `the server answered ${response.status} for ${apiPath}`, ""),
        context: { path: apiPath, status: response.status },
        requestId: response.requestId,
      }),
    );
  });
}

function isSameOrigin(url: string): boolean {
  try {
    return new URL(url).origin === window.location.origin;
  } catch {
    // A URL that does not parse cannot be shown to be ours, so it is not one to skip.
    return false;
  }
}

/**
 * An error a component threw while rendering. Reports it as a fatal `ui` event with the
 * component stack in the context, and answers with the reference to show the user: the first
 * twelve characters of the fingerprint, or nothing when reports are off.
 */
export function reportUiError(
  error: unknown,
  info: { componentStack: string | null },
): Promise<string> {
  try {
    const r = reporter;
    if (r === null || !errorReportsEnabled()) return Promise.resolve("");
    const event = buildEvent(r, {
      category: "ui",
      severity: "fatal",
      parts: eventFromError(error),
      context: info.componentStack === null ? {} : { componentStack: info.componentStack },
    });
    const input = fingerprintInput(event);
    enqueue(r, event);
    return fingerprint(input).then((hex) => hex.slice(0, REFERENCE_LENGTH));
  } catch {
    // Must not throw into React's error handling, of all places: no reference, nothing sent.
    return Promise.resolve("");
  }
}

/**
 * The user's answer to "What were you doing?", for the error `reference` names. Its own
 * event, so it never changes an error's count, and not subject to the dedupe. Sent at once,
 * on its own: the user is about to reload. True when the server took it.
 */
export async function sendDescription(reference: string, text: string): Promise<boolean> {
  try {
    const note = text.trim().slice(0, CAP.description);
    const r = reporter;
    if (r === null || note === "" || !errorReportsEnabled()) return false;
    return await send(r, [
      buildEvent(r, {
        category: "ui",
        severity: "error",
        parts: parts("UserDescription", note, ""),
        context: { reference },
      }),
    ]);
  } catch {
    // Not sent, and the screen says so: it must not look as if the note had gone.
    return false;
  }
}

/* ---- installation ----------------------------------------------------------------------- */

/**
 * Hooks the page: the window's `error` event and `unhandledrejection` as `crash` events, the
 * last twenty `console.warn` / `console.error` lines, and `pagehide`, which sends what is
 * still queued with `navigator.sendBeacon`. Idempotent: a second call adds nothing and
 * answers with the same function that removes everything again.
 */
export function installErrorReporter(options: { apiBase: string }): () => void {
  if (removeCurrent !== null) return removeCurrent;

  const r: Reporter = {
    endpoint: `${options.apiBase.replace(/\/+$/, "")}/errors`,
    consoleLines: [],
    seen: new Set(),
    queue: [],
    flushTimer: null,
    flushing: false,
  };
  const originalWarn = console.warn;
  const originalError = console.error;
  const warn = function (this: unknown, ...args: unknown[]): void {
    recordConsole(r, "WARN", args);
    originalWarn.apply(console, args);
  };
  const error = function (this: unknown, ...args: unknown[]): void {
    recordConsole(r, "ERROR", args);
    originalError.apply(console, args);
  };
  console.warn = warn;
  console.error = error;
  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  window.addEventListener("pagehide", onPageHide);
  reporter = r;

  const remove = (): void => {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    window.removeEventListener("pagehide", onPageHide);
    if (console.warn === warn) console.warn = originalWarn;
    if (console.error === error) console.error = originalError;
    // A request still in flight finishes on its own; it must not find anything left to send.
    clearFlushTimer(r);
    r.queue = [];
    if (reporter === r) reporter = null;
    removeCurrent = null;
  };
  removeCurrent = remove;
  return remove;
}

function onError(event: ErrorEvent): void {
  guard(() => {
    const r = reporter;
    if (r === null) return;
    // A cross-origin script fails with no Error object; the event itself still says where.
    const thrown =
      event.error === undefined || event.error === null
        ? parts(
            "ScriptError",
            event.message,
            event.filename === "" ? "" : `at ${event.filename}:${event.lineno}:${event.colno}`,
          )
        : eventFromError(event.error);
    enqueue(
      r,
      buildEvent(r, { category: "crash", severity: "error", parts: thrown, context: {} }),
    );
  });
}

function onUnhandledRejection(event: PromiseRejectionEvent): void {
  guard(() => {
    const r = reporter;
    if (r === null) return;
    enqueue(
      r,
      buildEvent(r, {
        category: "crash",
        severity: "error",
        parts: eventFromError(event.reason),
        context: {},
      }),
    );
  });
}

function onPageHide(): void {
  guard(() => {
    const r = reporter;
    if (r === null) return;
    const events = fit(r, r.queue.length);
    // The page is going: what cannot leave now is dropped, like every failed send.
    r.queue = [];
    clearFlushTimer(r);
    if (events.length === 0 || !errorReportsEnabled()) return;
    if (typeof navigator.sendBeacon !== "function") return;
    navigator.sendBeacon(
      r.endpoint,
      new Blob([JSON.stringify({ events })], { type: "application/json" }),
    );
  });
}

/** One console line as the ring buffer keeps it: `[HH:MM:SS] [LEVEL] text`. */
function recordConsole(r: Reporter, level: "WARN" | "ERROR", args: unknown[]): void {
  guard(() => {
    if (!errorReportsEnabled()) return;
    r.consoleLines.push(
      cut(`[${istClock()}] [${level}] ${args.map(describeArg).join(" ")}`, CONSOLE_LINE_MAX),
    );
    if (r.consoleLines.length > CONSOLE_MAX_LINES) r.consoleLines.shift();
  });
}

/**
 * IST, the one clock the project shows everywhere: whoever the visitor is, the lines carry
 * the same time the administrator reads on the event itself.
 */
function istClock(): string {
  return new Date().toLocaleTimeString("en-GB", { hourCycle: "h23", timeZone: "Asia/Kolkata" });
}

/**
 * What one logged value adds to a line. An object contributes its type and nothing else: a
 * logged dataset row or response body must not leave the browser.
 */
function describeArg(value: unknown): string {
  if (typeof value === "string") return value;
  if (value instanceof Error) return `${value.name}: ${value.message}`;
  if (typeof value === "object" && value !== null) return Object.prototype.toString.call(value);
  return String(value);
}
