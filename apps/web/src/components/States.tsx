/** The loading / empty / error states every async surface uses. */

import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export function Loading({ label = "Loading…" }: { label?: string }): ReactNode {
  return (
    <p className="state state--loading" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </p>
  );
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

export function ErrorBox({ error }: { error: unknown }): ReactNode {
  return (
    <p className="alert alert--error" role="alert">
      {errorText(error)}
    </p>
  );
}
