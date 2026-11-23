import type { ReactElement } from "react";

/** An error event's status, or an account's role. */
export type StatusKind = "new" | "acknowledged" | "resolved" | "ignored" | "member" | "admin";

/** One tone per kind: urgency for an error's status, standing for a role. */
const TONE: Record<StatusKind, string> = {
  new: "badge--bad",
  acknowledged: "badge--warning",
  resolved: "badge--ok",
  ignored: "badge--muted",
  member: "badge--info",
  admin: "badge--accent",
};

/** A status or role as a labelled pill. */
export function StatusBadge({ kind }: { kind: StatusKind }): ReactElement {
  return (
    <span className={`badge ${TONE[kind]}`}>
      {kind.charAt(0).toUpperCase()}
      {kind.slice(1)}
    </span>
  );
}
