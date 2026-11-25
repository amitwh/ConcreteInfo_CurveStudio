/**
 * The Markdown stack (react-markdown, remark-gfm) is the help route's own (design
 * item 10): /help is a lazy route, so as long as nothing but HelpPage imports the
 * stack, it is bundled into that one chunk and no other. This is the source-graph
 * twin of the App.lazy suite, which pins that the route itself stays lazy.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";

const srcRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Every non-test source file under `dir`, recursively. */
function sourceFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = `${dir}/${entry}`;
    if (statSync(path).isDirectory()) {
      files.push(...sourceFiles(path));
    } else if ((entry.endsWith(".ts") || entry.endsWith(".tsx")) && !entry.includes(".test.")) {
      files.push(path);
    }
  }
  return files;
}

describe("the Markdown stack stays in the help route's chunk", () => {
  it("no module but HelpPage imports react-markdown or remark-gfm", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles(srcRoot)) {
      if (file.endsWith("/pages/HelpPage.tsx")) continue;
      const source = readFileSync(file, "utf8");
      if (/from "(react-markdown|remark-gfm)"/.test(source)) offenders.push(file);
    }
    expect(
      offenders,
      "the Markdown stack must only be imported by pages/HelpPage.tsx, or it leaves the help chunk",
    ).toEqual([]);
  });

  it("no module but HelpPage imports the help document (the Markdown is a 55 KB string)", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles(srcRoot)) {
      if (file.endsWith("/pages/HelpPage.tsx")) continue;
      const source = readFileSync(file, "utf8");
      if (/from "\.\.\/content\/helpDocument"/.test(source)) offenders.push(file);
    }
    expect(
      offenders,
      "the help Markdown must only be imported by pages/HelpPage.tsx, or it leaves the help chunk",
    ).toEqual([]);
  });
});
