/**
 * The help document: the canonical Markdown (docs/help-content.md, copied in by
 * scripts/sync-help.mjs — never edited here) split into its `##` sections, each
 * carrying the stable anchor the shipped app links to.
 *
 * The anchors predate the Markdown rendering and are not derivable from the
 * headings (a slugger would make "9. Keyboard shortcuts (desktop)" into
 * "9-keyboard-shortcuts-desktop", not "keyboard-shortcuts"), so they are an
 * explicit map — the bridge design item 10 calls for. A heading without an entry
 * throws rather than silently shipping an unanchorable section.
 */

import document from "./help-content.md?raw";

export interface HelpSection {
  /** Stable URL fragment: /help#keyboard-shortcuts */
  id: string;
  /** The canonical `##` heading text, number included ("9. Keyboard shortcuts (desktop)"). */
  title: string;
  /** The section's Markdown: everything after its heading line, `###` subheadings included. */
  body: string;
  /** Title and body flattened and lowercased, for the index search. */
  searchText: string;
}

/** Canonical `##` heading → the anchor the app has always linked to. */
const SECTION_IDS: Readonly<Record<string, string>> = {
  "1. Getting started": "getting-started",
  "2. Importing data": "importing-data",
  "3. Auto-fit": "auto-fit",
  "4. Manual fit and custom equations": "manual-fit",
  "5. Understanding the numbers": "understanding-the-numbers",
  "6. Prediction": "prediction",
  "7. Projects (desktop)": "projects-desktop",
  "8. Exports": "exports",
  "9. Keyboard shortcuts (desktop)": "keyboard-shortcuts",
  "10. Troubleshooting": "troubleshooting",
  "11. Fit audit and methods text": "fit-audit-and-methods-text",
  "12. Tools that start from a fit": "tools-that-start-from-a-fit",
  "13. Statistics tools": "statistics-tools",
  "14. Summarising columns": "summarising-columns",
  "15. Comparing means": "comparing-means",
  "16. Rank-based comparisons": "rank-based-comparisons",
  "17. Relationships between columns": "relationships-between-columns",
  "18. Counts": "counts",
  "19. Comparing two methods": "comparing-two-methods",
  "20. Measurement and classification": "measurement-and-classification",
  "21. Sample size and power": "sample-size-and-power",
  "22. Advanced models": "advanced-models",
  "23. Evidence synthesis": "evidence-synthesis",
  "24. Bayesian tests": "bayesian-tests",
  "25. About, licence, projects and policy": "about-licence-credits",
};

const HEADING = /^##\s+(.+?)\s*$/;

/** Split the document at its `##` headings (the desktop's parse_sections, in step). */
export function parseHelpSections(markdown: string): HelpSection[] {
  const sections: { title: string; body: string[] }[] = [];
  let current: { title: string; body: string[] } | null = null;
  for (const line of markdown.split(/\r?\n/)) {
    const match = HEADING.exec(line);
    if (match !== null) {
      current = { title: match[1] ?? "", body: [] };
      sections.push(current);
      continue;
    }
    // the document title and its preamble sit above the first section
    if (current !== null) current.body.push(line);
  }
  return sections.map((section) => {
    const id = SECTION_IDS[section.title];
    if (id === undefined) {
      throw new Error(
        `the help section "${section.title}" has no anchor: add it to SECTION_IDS in helpDocument.ts`,
      );
    }
    const body = section.body.join("\n").trim();
    return { id, title: section.title, body, searchText: `${section.title}\n${body}`.toLowerCase() };
  });
}

/** The generated copy of the canonical document, for tests to pin against docs/. */
export const helpMarkdown: string = document;

export const HELP_SECTIONS: HelpSection[] = parseHelpSections(document);
