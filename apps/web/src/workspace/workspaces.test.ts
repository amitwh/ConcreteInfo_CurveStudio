import { describe, expect, it, beforeEach } from "vitest";
import {
  activateWorkspace,
  activeWorkspace,
  addWorkspace,
  defaultSet,
  loadWorkspaces,
  normalizeName,
  removeWorkspace,
  renameWorkspace,
  saveWorkspaces,
  updateWorkspace,
} from "./workspaces";

beforeEach(() => {
  localStorage.clear();
});

describe("workspaces", () => {
  it("starts with one workspace in Auto-fit mode", () => {
    const set = defaultSet();
    expect(set.workspaces).toHaveLength(1);
    expect(set.workspaces[0]?.name).toBe("Workspace 1");
    expect(set.workspaces[0]?.mode).toBe("auto");
    expect(set.workspaces[0]?.datasetId).toBeNull();
    expect(activeWorkspace(set).id).toBe(set.activeId);
  });

  it("adds a numbered workspace and activates it", () => {
    const first = defaultSet();
    const second = addWorkspace(first);
    expect(second.workspaces).toHaveLength(2);
    expect(second.workspaces[1]?.name).toBe("Workspace 2");
    expect(second.activeId).toBe(second.workspaces[1]?.id);
    const third = addWorkspace(second);
    expect(third.workspaces[2]?.name).toBe("Workspace 3");
  });

  it("never removes the last workspace, and activates a neighbour otherwise", () => {
    const one = defaultSet();
    expect(removeWorkspace(one, one.workspaces[0]!.id)).toBe(one);

    const two = addWorkspace(one);
    const activeId = two.activeId;
    const after = removeWorkspace(two, activeId);
    expect(after.workspaces).toHaveLength(1);
    expect(after.activeId).toBe(one.workspaces[0]?.id);
  });

  it("renames, trims and refuses an empty name", () => {
    const set = defaultSet();
    const id = set.workspaces[0]!.id;
    expect(activeWorkspace(renameWorkspace(set, id, "  Cube tests  ")).name).toBe("Cube tests");
    expect(activeWorkspace(renameWorkspace(set, id, "   ")).name).toBe("Workspace 1");
    expect(normalizeName("x".repeat(80))).toHaveLength(60);
  });

  it("remembers a dataset and a mode per workspace", () => {
    const set = defaultSet();
    const id = set.workspaces[0]!.id;
    const updated = updateWorkspace(set, id, { datasetId: "ds-1", mode: "manual" });
    expect(activeWorkspace(updated)).toMatchObject({ datasetId: "ds-1", mode: "manual" });
    // the other workspaces are untouched
    expect(activeWorkspace(updateWorkspace(updated, id, { datasetId: "ds-2" })).datasetId).toBe("ds-2");
  });

  it("persists per account and ignores another account's list", () => {
    const first = defaultSet();
    const mine = updateWorkspace(first, activeWorkspace(first).id, { datasetId: "ds-9" });
    saveWorkspaces("user-a", mine);
    expect(activeWorkspace(loadWorkspaces("user-a")).datasetId).toBe("ds-9");
    expect(activeWorkspace(loadWorkspaces("user-b")).datasetId).toBeNull();
    expect(loadWorkspaces(null).workspaces).toHaveLength(1);
  });

  it("falls back to a fresh workspace on a corrupted or empty payload", () => {
    localStorage.setItem("curvestudio.workspaces.user-a", "not json");
    expect(loadWorkspaces("user-a").workspaces).toHaveLength(1);
    localStorage.setItem(
      "curvestudio.workspaces.user-a",
      JSON.stringify({ workspaces: [], activeId: "missing" }),
    );
    expect(loadWorkspaces("user-a").workspaces).toHaveLength(1);
  });

  it("repairs a dangling activeId instead of leaving no active workspace", () => {
    const set = defaultSet();
    saveWorkspaces("user-a", { ...set, activeId: "gone" });
    expect(loadWorkspaces("user-a").activeId).toBe(set.workspaces[0]?.id);
  });

  it("does nothing when activating an unknown workspace", () => {
    const set = defaultSet();
    expect(activateWorkspace(set, "nope")).toBe(set);
  });
});
