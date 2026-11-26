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

  it("stacks the manual-fit parameter grid below its breakpoint (item 14)", () => {
    // The stacking must live inside a narrow-viewport media query (360 px is covered by
    // 480), turn the table's parts into blocks, and label each stacked field from its
    // data-label attribute.
    const media = /@media \(max-width: 480px\) \{([\s\S]*?)\n\}/.exec(css);
    expect(media, "styles.css must have a max-width: 480px block").not.toBeNull();
    const block = media?.[1] ?? "";
    expect(block).toMatch(/\.params-grid,\s*\n\s*\.params-grid tbody,\s*\n\s*\.params-grid tr,\s*\n\s*\.params-grid td\s*\{\s*\n\s*display: block;/);
    expect(block).toMatch(/\.params-grid thead\s*\{\s*\n\s*display: none;/);
    expect(block).toMatch(/content: attr\(data-label\)/);
  });
});
