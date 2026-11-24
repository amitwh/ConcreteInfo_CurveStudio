/**
 * Multiple workspaces: each named workspace remembers the dataset it is showing and
 * whether it is in Auto-fit or Manual mode, so a user can keep, say, "Cube tests" and
 * "Mix trial" apart — and switch without losing where they were. The list is stored per
 * signed-in account in localStorage; datasets and fits themselves live on the server.
 *
 * Everything here is pure except `load`/`save`, which tolerate storage being unavailable
 * (private mode) and a corrupted payload: invalid data falls back to one fresh workspace.
 */

import type { WorkspaceMode } from "./mode";

export interface Workspace {
  id: string;
  name: string;
  /** The dataset this workspace shows; null until the user picks one. */
  datasetId: string | null;
  mode: WorkspaceMode;
}

export interface WorkspaceSet {
  workspaces: Workspace[];
  activeId: string;
}

const KEY_PREFIX = "curvestudio.workspaces.";
const MAX_NAME = 60;

const storageKey = (userId: string): string => `${KEY_PREFIX}${userId}`;

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `ws-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/** A fresh workspace, numbered after the ones already in the set. */
export function freshWorkspace(existing: readonly Workspace[]): Workspace {
  const used = new Set(existing.map((workspace) => workspace.name));
  let number = existing.length + 1;
  while (used.has(`Workspace ${number}`)) number += 1;
  return { id: newId(), name: `Workspace ${number}`, datasetId: null, mode: "auto" };
}

export function defaultSet(): WorkspaceSet {
  const first: Workspace = { id: newId(), name: "Workspace 1", datasetId: null, mode: "auto" };
  return { workspaces: [first], activeId: first.id };
}

/** The stored set for this account, or a fresh one when nothing valid is stored. */
export function loadWorkspaces(userId: string | null): WorkspaceSet {
  if (userId === null) return defaultSet();
  try {
    const raw = localStorage.getItem(storageKey(userId));
    if (raw === null) return defaultSet();
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return defaultSet();
    const candidate = parsed as Partial<WorkspaceSet>;
    if (!Array.isArray(candidate.workspaces) || candidate.workspaces.length === 0) {
      return defaultSet();
    }
    const workspaces: Workspace[] = [];
    for (const entry of candidate.workspaces as unknown[]) {
      if (typeof entry !== "object" || entry === null) return defaultSet();
      const item = entry as Partial<Workspace>;
      if (typeof item.id !== "string" || typeof item.name !== "string") return defaultSet();
      workspaces.push({
        id: item.id,
        name: normalizeName(item.name) || "Workspace",
        datasetId: typeof item.datasetId === "string" ? item.datasetId : null,
        mode:
          item.mode === "manual" || item.mode === "stats" ? item.mode : "auto",
      });
    }
    const activeId =
      typeof candidate.activeId === "string" &&
      workspaces.some((workspace) => workspace.id === candidate.activeId)
        ? candidate.activeId
        : (workspaces[0] as Workspace).id;
    return { workspaces, activeId };
  } catch {
    return defaultSet();
  }
}

export function saveWorkspaces(userId: string | null, set: WorkspaceSet): void {
  if (userId === null) return;
  try {
    localStorage.setItem(storageKey(userId), JSON.stringify(set));
  } catch {
    // A workspace list that cannot be stored still works for this visit.
  }
}

/** Trim a workspace name; whitespace-only becomes an empty string (callers keep the old name). */
export function normalizeName(name: string): string {
  return name.trim().slice(0, MAX_NAME);
}

export function activeWorkspace(set: WorkspaceSet): Workspace {
  return set.workspaces.find((workspace) => workspace.id === set.activeId) ?? set.workspaces[0]!;
}

export function addWorkspace(set: WorkspaceSet): WorkspaceSet {
  const workspace = freshWorkspace(set.workspaces);
  return { workspaces: [...set.workspaces, workspace], activeId: workspace.id };
}

/** Remove a workspace; the last one is never removed (the app always has somewhere to work). */
export function removeWorkspace(set: WorkspaceSet, id: string): WorkspaceSet {
  if (set.workspaces.length <= 1) return set;
  const workspaces = set.workspaces.filter((workspace) => workspace.id !== id);
  if (workspaces.length === set.workspaces.length) return set;
  const index = set.workspaces.findIndex((workspace) => workspace.id === id);
  const fallback = workspaces[Math.min(index, workspaces.length - 1)] as Workspace;
  return { workspaces, activeId: set.activeId === id ? fallback.id : set.activeId };
}

export function activateWorkspace(set: WorkspaceSet, id: string): WorkspaceSet {
  if (set.activeId === id || !set.workspaces.some((workspace) => workspace.id === id)) return set;
  return { ...set, activeId: id };
}

export function renameWorkspace(set: WorkspaceSet, id: string, name: string): WorkspaceSet {
  const normalized = normalizeName(name);
  if (normalized === "") return set;
  return updateWorkspace(set, id, { name: normalized });
}

export function updateWorkspace(
  set: WorkspaceSet,
  id: string,
  patch: Partial<Omit<Workspace, "id">>,
): WorkspaceSet {
  return {
    ...set,
    workspaces: set.workspaces.map((workspace) =>
      workspace.id === id ? { ...workspace, ...patch } : workspace,
    ),
  };
}
