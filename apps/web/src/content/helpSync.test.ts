/**
 * The prebuild sync (scripts/sync-help.mjs) that carries docs/help-content.md into
 * src/content/help-content.md — the "one help source" of design item 10. These pin
 * the properties the build relies on: a fresh clone gets its copy, an in-sync copy
 * is untouched (idempotent), a diverged copy is rewritten, and a broken source
 * fails loudly instead of letting the page serve the wrong text.
 */

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const script = resolve(here, "../../scripts/sync-help.mjs");
const canonicalPath = resolve(here, "../../../../docs/help-content.md");
const generatedPath = resolve(here, "./help-content.md");

function runSync(source: string, target: string): string {
  return execFileSync("node", [script, "--source", source, "--target", target], {
    encoding: "utf8",
  });
}

/** Run the sync expecting failure; return its stderr (empty when it unexpectedly succeeds). */
function runSyncFailure(source: string, target: string): string {
  try {
    runSync(source, target);
  } catch (error) {
    return (error as { stderr: string }).stderr;
  }
  return "";
}

function withTempDir(body: (target: string) => void): void {
  const dir = mkdtempSync(join(tmpdir(), "curvestudio-help-sync-"));
  try {
    body(join(dir, "help-content.md"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

describe("the help sync script", () => {
  it("creates the copy when it is absent, from the canonical document", () => {
    withTempDir((target) => {
      const output = runSync(canonicalPath, target);
      expect(output).toMatch(/copied/);
      expect(readFileSync(target, "utf8")).toBe(readFileSync(canonicalPath, "utf8"));
    });
  });

  it("is idempotent: a second run leaves an in-sync copy untouched", () => {
    withTempDir((target) => {
      runSync(canonicalPath, target);
      const before = statSync(target).mtimeMs;
      const output = runSync(canonicalPath, target);
      expect(output).toMatch(/already in sync/);
      expect(statSync(target).mtimeMs).toBe(before);
    });
  });

  it("rewrites a diverged copy with the canonical text", () => {
    withTempDir((target) => {
      runSync(canonicalPath, target);
      writeFileSync(target, "hand-edited drift that must not survive a build\n", "utf8");
      const output = runSync(canonicalPath, target);
      expect(output).toMatch(/copied/);
      expect(readFileSync(target, "utf8")).toBe(readFileSync(canonicalPath, "utf8"));
    });
  });

  it("fails loudly when the canonical document is empty", () => {
    withTempDir((target) => {
      const empty = join(dirname(target), "empty.md");
      writeFileSync(empty, "  \n", "utf8");
      const stderr = runSyncFailure(empty, target);
      expect(stderr, "an empty canonical document must fail the sync").toMatch(
        /canonical help document is empty/,
      );
    });
  });

  it("fails loudly when the canonical document is missing", () => {
    withTempDir((target) => {
      const stderr = runSyncFailure(join(dirname(target), "not-there.md"), target);
      expect(stderr, "a missing canonical document must fail the sync").toMatch(
        /help content sync failed/,
      );
    });
  });

  it("the generated copy the app renders equals the canonical document", () => {
    // The pretest hook syncs before this runs; a mismatch means someone bypassed
    // the npm scripts with a stale checkout — say so, loudly.
    expect(
      readFileSync(generatedPath, "utf8"),
      "src/content/help-content.md is stale: run `npm run sync:help` (or npm test) before serving",
    ).toBe(readFileSync(canonicalPath, "utf8"));
  });
});
