import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Read from disk rather than with a `?raw` import: vitest blanks CSS modules unless
 * `test.css` is enabled, so the import would hand this test an empty string.
 */
const css = readFileSync(
  resolve(dirname(fileURLToPath(import.meta.url)), "./styles.css"),
  "utf-8",
);

/**
 * Links styled as buttons (`<Link className="btn …">` in the public header and on the
 * home page) must not pick up the browser's underline for anchors.
 */
describe("styles.css", () => {
  it("does not underline button links: the base .btn rule sets text-decoration: none", () => {
    // Anchored to a line start so the `.btn:hover…`, `.btn--…` and `.card .btn` rules
    // do not match; the block runs to the first closing brace.
    const baseRule = /^\.btn\s*\{([^}]*)\}/m.exec(css);
    expect(baseRule, "styles.css must still have a base .btn rule").not.toBeNull();
    expect(baseRule?.[1]).toMatch(/text-decoration:\s*none/);
  });
});
