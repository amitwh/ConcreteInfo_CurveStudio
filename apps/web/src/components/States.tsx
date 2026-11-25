/** The loading / empty / error states every async surface uses. */

import { Suspense, type ReactElement, type ReactNode } from "react";
import { ApiError } from "../api/client";

export function Loading({ label = "Loading…" }: { label?: string }): ReactNode {
  return (
    <p className="state state--loading" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </p>
  );
}

/**
 * The Suspense boundary every lazy page loads behind: the app's own spinner state,
 * labelled with what is opening, so a page whose chunk is still downloading reads as
 * loading — never as a blank flash.
 */
export function LazyBoundary({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return <Suspense fallback={<Loading label={label} />}>{children}</Suspense>;
}

export function Empty({ children }: { children: ReactNode }): ReactNode {
  return <p className="state state--empty">{children}</p>;
}

/** Turn any thrown value into the message the user sees (network vs API vs other). */
export function errorText(error: unknown, fallback = "something went wrong"): string {
  if (error instanceof ApiError) return error.message;
  if (typeof error === "string" && error !== "") return error;
  if (error instanceof Error && error.message !== "") return error.message;
  return fallback;
}

/**
 * The message for a refused request. Every API 422 carries the same generic message
 * ("the request is not valid"); the reason is the first entry of `detail.errors`, so that
 * is what a 422 shows. Any other error reads like `errorText`.
 */
export function apiErrorText(error: unknown, fallback = "something went wrong"): string {
  if (error instanceof ApiError && error.status === 422) {
    const errors = (error.detail as { errors?: unknown } | null)?.errors;
    if (Array.isArray(errors)) {
      const first = errors[0] as { msg?: unknown } | undefined;
      if (typeof first?.msg === "string" && first.msg !== "") return first.msg;
    }
  }
  return errorText(error, fallback);
}

export function ErrorBox({ error }: { error: unknown }): ReactNode {
  return (
    <p className="alert alert--error" role="alert">
      {errorText(error)}
    </p>
  );
}
