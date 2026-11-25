#!/usr/bin/env node
/**
 * Copy the canonical help document (docs/help-content.md) into the web app as
 * src/content/help-content.md, so /help renders the one source of truth through
 * react-markdown (design item 10 of the 2026-11-23 UX fixes). The generated copy
 * is gitignored; docs/help-content.md is the only edited source.
 *
 * Wired as the `pretest` and `prebuild` hooks (and into `dev`), so a fresh clone
 * needs no manual step. Idempotent: an in-sync copy is left untouched.
 *
 * Usage: node scripts/sync-help.mjs [--source PATH] [--target PATH]
 * (the path overrides exist for the sync tests; the defaults need no arguments)
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_SOURCE = new URL("../../../docs/help-content.md", import.meta.url);
const DEFAULT_TARGET = new URL("../src/content/help-content.md", import.meta.url);

/** Parse `--source PATH` / `--target PATH` into URL objects, defaulting to the repo paths. */
function parseArgs(argv) {
  const paths = { source: DEFAULT_SOURCE, target: DEFAULT_TARGET };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = argv[index + 1];
    if ((flag === "--source" || flag === "--target") && value !== undefined) {
      paths[flag.slice(2)] = pathToFileURL(resolve(value));
      index += 1;
    } else {
      throw new Error(`unknown argument: ${flag} (usage: --source PATH --target PATH)`);
    }
  }
  return paths;
}

/**
 * Make the target hold exactly the canonical text.
 *
 * @returns {{wrote: boolean}} whether the copy had drifted and was rewritten
 * @throws if the canonical document is unreadable or empty, or the write cannot be
 *   read back — a help page that silently serves old or broken text is the failure
 *   this exists to prevent, so it fails loudly instead.
 */
export function syncHelp({ source = DEFAULT_SOURCE, target = DEFAULT_TARGET } = {}) {
  const canonical = readFileSync(source, "utf8");
  if (canonical.trim() === "") {
    throw new Error(`the canonical help document is empty: ${source.pathname}`);
  }
  let current = null;
  try {
    current = readFileSync(target, "utf8");
  } catch {
    // absent on a fresh clone: the copy is written below
  }
  if (current === canonical) return { wrote: false };
  writeFileSync(target, canonical, "utf8");
  if (readFileSync(target, "utf8") !== canonical) {
    throw new Error(`could not write the help copy verbatim: ${target.pathname}`);
  }
  return { wrote: true };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  try {
    const { wrote } = syncHelp(parseArgs(process.argv.slice(2)));
    console.log(
      wrote
        ? "help content: copied docs/help-content.md into src/content/help-content.md"
        : "help content: already in sync",
    );
  } catch (error) {
    console.error(`help content sync failed: ${error.message}`);
    process.exit(1);
  }
}
