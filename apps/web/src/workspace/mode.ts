/**
 * The workspace's analysis mode and its per-user persistence. The product rule is that
 * Auto-fit leads: it is the default, Manual fit appears only when the user asks for it,
 * and the choice is remembered per signed-in user (a first visit always starts Auto-fit).
 */

export type WorkspaceMode = "auto" | "manual";

const key = (userId: string): string => `curvestudio.mode.${userId}`;

/** The stored mode for this user, or "auto" when nothing valid is stored. */
export function loadMode(userId: string | null): WorkspaceMode {
  if (userId === null) return "auto";
  try {
    return localStorage.getItem(key(userId)) === "manual" ? "manual" : "auto";
  } catch {
    // Storage can be unavailable (private mode); the default is the safe answer.
    return "auto";
  }
}

/** Persist the user's mode choice. */
export function saveMode(userId: string, mode: WorkspaceMode): void {
  try {
    localStorage.setItem(key(userId), mode);
  } catch {
    // A preference that cannot be stored still applies for this visit.
  }
}
