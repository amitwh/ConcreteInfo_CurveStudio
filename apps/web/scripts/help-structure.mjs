#!/usr/bin/env node
/**
 * Render docs/help-content.md with the web app's own Markdown stack (react-markdown
 * + remark-gfm, the same plugins /help uses) and print the document's section
 * structure as JSON: one {level, text} per `##`/`###` heading, in document order.
 *
 * The desktop's pytest (tests/desktop/test_desktop_help.py) feeds the same file to
 * its own renderer and compares the two lists — the conformance test of design
 * item 10. Level-1 is skipped: the document title is page chrome, which each app
 * renders its own way.
 *
 * Usage: node scripts/help-structure.mjs   (run from apps/web; needs npm ci first)
 */
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const source = new URL("../../../docs/help-content.md", import.meta.url);
const html = renderToStaticMarkup(
  createElement(Markdown, { remarkPlugins: [remarkGfm] }, readFileSync(source, "utf8")),
);

// react-dom/server escapes text (an apostrophe becomes &#x27;); the comparison wants
// the characters back. Heading text holds no markup, so these five cover it.
const ENTITIES = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'" };
const decode = (text) => text.replace(/&(?:amp|lt|gt|quot|#x27);/g, (entity) => ENTITIES[entity]);

const headings = [...html.matchAll(/<h([23])>([^<]*)<\/h\1>/g)].map((match) => ({
  level: Number(match[1]),
  text: decode(match[2]),
}));
if (headings.length === 0) {
  console.error(`no section headings rendered from ${source.pathname}`);
  process.exit(1);
}
console.log(JSON.stringify(headings));
