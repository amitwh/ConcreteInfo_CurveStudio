import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import projects from "./other-projects.json";

describe("the shared project list", () => {
  it("is a byte-identical copy of the canonical file in docs/", () => {
    // new URL("...", import.meta.url) is rewritten by Vite's asset pipeline, so resolve
    // the paths on the filesystem instead.
    const here = dirname(fileURLToPath(import.meta.url));
    const canonical = readFileSync(resolve(here, "../../../../docs/other-projects.json"), "utf-8");
    const copy = readFileSync(resolve(here, "./other-projects.json"), "utf-8");
    expect(copy).toBe(canonical);
  });
  it("holds five entries with short summaries and valid kinds", () => {
    expect(projects.entries).toHaveLength(5);
    for (const entry of projects.entries) {
      expect(entry.summary.length).toBeLessThanOrEqual(90);
      expect(["open-source", "commercial"]).toContain(entry.kind);
    }
  });
});
