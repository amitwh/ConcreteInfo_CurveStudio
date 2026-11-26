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
 * The whole body of every `@media (max-width: Npx)` block: rules inside a block close
 * with an indented `}` (column > 0), so a newline followed by `}` at column 0 ends the
 * block itself.
 */
function mediaBlocks(query: string): string[] {
  const pattern = new RegExp(`@media \\(${query.replace(/[()]/g, "\\$&")}\\) \\{([\\s\\S]*?)\\n\\}`, "g");
  return [...css.matchAll(pattern)].map((match) => match[1] ?? "");
}

/** The one max-width block that holds `marker` (the rules for one surface). */
function blockWith(query: string, marker: string): string {
  const found = mediaBlocks(query).find((body) => body.includes(marker));
  expect(found, `styles.css must have a max-width: ${query} block holding ${marker}`).toBeDefined();
  return found ?? "";
}

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

  // -- item 15: the three widths the web is written and tested at --------------------------------
  // 1280 px sees the base rules; 768 px everything up to max-width 800/900; 360 px
  // everything down to max-width 480. Each test pins what keeps that surface whole at
  // that width — the widths are named so a rule moving to another bucket fails the
  // width it left.

  describe("the header", () => {
    it("at 1280: the bar is one flex row — brand fixed, navigation pushed right", () => {
      expect(css).toMatch(/^\.app-header__bar\s*\{[^}]*display: flex;/m);
      expect(css).toMatch(/^\.brand\s*\{[^}]*flex: none;/m);
      expect(css).toMatch(/^\.app-nav\s*\{[^}]*margin-left: auto;/m);
    });

    it("at 768: the bar tightens and the account shows no e-mail (max-width 900)", () => {
      const block = blockWith("max-width: 900px", ".app-header__bar");
      expect(block).toMatch(/\.app-header__bar\s*\{[^}]*gap: 0\.5rem;/s);
      expect(block).toMatch(/\.account__email\s*\{[^}]*display: none;/s);
    });

    it("at 360: the account chip collapses to its avatar (max-width 480)", () => {
      const block = blockWith("max-width: 480px", ".account__text");
      expect(block).toMatch(/\.account__text,\s*\n\s*\.account__chevron\s*\{[^}]*display: none;/s);
      const narrow = blockWith("max-width: 640px", ".nav__label");
      expect(narrow).toMatch(/\.app-nav a \.nav__label\s*\{[^}]*display: none;/s);
    });
  });

  describe("the workspace layout", () => {
    it("at 1280: three columns with the results spanning all of them", () => {
      expect(css).toMatch(
        /grid-template-columns: minmax\(12rem, 14rem\) minmax\(0, 1fr\) minmax\(16rem, 18rem\);/,
      );
      expect(css).toMatch(/\.workspace__results\s*\{[^}]*grid-column: 1 \/ -1;/s);
    });

    it("at 768: one column, the side panels below the main flow (max-width 800)", () => {
      const block = blockWith("max-width: 800px", ".workspace");
      expect(block).toMatch(/\.workspace\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\);/s);
      const collapsed = blockWith("max-width: 800px", ".workspace--rail-collapsed");
      expect(collapsed).toMatch(
        /\.workspace--rail-collapsed\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\);/s,
      );
    });

    it("at 360: the one-column layout also loses the wide gutter (max-width 900)", () => {
      const block = blockWith("max-width: 900px", ".workspace");
      expect(block).toMatch(/\.workspace\s*\{[^}]*padding: 0\.6rem;[^}]*gap: 0\.6rem;/s);
    });
  });

  describe("the tables", () => {
    it("at 1280: the ranking table keeps every column and scrolls in its wrap", () => {
      expect(css).toMatch(/^\.table-wrap\s*\{[^}]*overflow-x: auto;/m);
      expect(css).toMatch(/^\.ranking-table\s*\{[^}]*min-width: 64rem;/m);
      // Actions stays reachable at the scroll edge: pinned to the wrap's right.
      expect(css).toMatch(
        /\.ranking-table th\.ranking-row__actions,\s*\n\.ranking-table td\.ranking-row__actions\s*\{[^}]*position: sticky;[^}]*right: 0;/s,
      );
    });

    it("at 768: the columns a tablet cannot pay for are dropped (max-width 800)", () => {
      const block = blockWith("max-width: 800px", ".ranking-table");
      // equation (3), adj. R² (5), AICc (7), BIC (8), CV-RMSE (9) and k (10) go
      for (const column of [3, 5, 7, 8, 9, 10]) {
        expect(block).toMatch(new RegExp(`th:nth-child\\(${column}\\),\\s*\\n\\s*\\.ranking-table td:nth-child\\(${column}\\)`));
      }
      expect(block).toMatch(/\.ranking-table\s*\{[^}]*min-width: 42rem;/s);
      expect(block).toMatch(/display: none;/);
    });

    it("at 360: rank, model, RMSE and the row menu fit the wrap (max-width 480)", () => {
      const block = blockWith("max-width: 480px", ".ranking-table");
      // R² (4) and the status badge (11) go too, and the rank hint stops widening the
      // first column
      for (const column of [4, 11]) {
        expect(block).toMatch(new RegExp(`th:nth-child\\(${column}\\),\\s*\\n\\s*\\.ranking-table td:nth-child\\(${column}\\)`));
      }
      expect(block).toMatch(/\.ranking-row__same\s*\{[^}]*display: none;/s);
      expect(block).toMatch(/\.ranking-table\s*\{[^}]*min-width: 0;/s);
      expect(block).toMatch(/th\.ranking-row__actions,\s*\n\s*\.ranking-table td\.ranking-row__actions\s*\{[^}]*min-width: 0;/s);
    });
  });
});
