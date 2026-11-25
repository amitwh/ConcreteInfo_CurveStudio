/**
 * `<link rel="preload">` for the two latin woff2 files the first screen renders
 * (P6 Task 3), as a Vite build plugin.
 *
 * The fonts are self-hosted through `@fontsource-variable/*` (imported from main.tsx),
 * so their URLs exist only once the build has hashed them — an index.html written by
 * hand cannot name them. Once the bundle is complete, this plugin finds each family's
 * latin subset among the emitted assets and asks Vite to inject its preload into
 * index.html, so the browser starts both downloads in parallel with the entry script
 * instead of after the CSS has been parsed.
 *
 * This module is build tooling: nothing in the app imports it, so it never enters the
 * browser bundle; it lives under src/ only to be typechecked and unit-tested with the
 * rest of the app.
 */

import type { HtmlTagDescriptor, Plugin } from "vite";

/**
 * One pattern per family: exactly one latin woff2 is emitted for each (the fontsource
 * CSS this app imports declares only the normal-style `wght` variable font). The
 * lookahead keeps the `latin-ext` subset — also emitted, since the same CSS declares
 * every subset's @font-face — from matching.
 */
const LATIN_FONT_PATTERNS = [
  /^assets\/inter-latin-(?!ext-)[A-Za-z0-9_-]+\.woff2$/,
  /^assets\/jetbrains-mono-latin-(?!ext-)[A-Za-z0-9_-]+\.woff2$/,
] as const;

/**
 * The preload links for the latin woff2 files among a build's asset names: one per
 * family, and none for a family with no unambiguous match — a preload for a missing or
 * wrong URL costs a wasted request, more than the one it saves.
 */
export function latinFontPreloads(assetNames: readonly string[]): HtmlTagDescriptor[] {
  return LATIN_FONT_PATTERNS.flatMap((pattern) => {
    const matches = assetNames.filter((name) => pattern.test(name));
    return matches.length === 1 ? [preload(matches[0]!)] : [];
  });
}

/** crossorigin is required even same-origin: fonts are always fetched in CORS mode, and
 *  a preload without the attribute would be a second, unusable request. */
function preload(href: string): HtmlTagDescriptor {
  return {
    tag: "link",
    attrs: { rel: "preload", href: `/${href}`, as: "font", type: "font/woff2", crossorigin: "anonymous" },
    injectTo: "head",
  };
}

export function fontPreloadPlugin(): Plugin {
  return {
    name: "curvestudio:preload-latin-fonts",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler(_html, ctx) {
        // Dev has no hashed asset names to preload and no network worth optimising.
        const names = ctx.bundle === undefined ? [] : Object.keys(ctx.bundle);
        const tags = latinFontPreloads(names);
        if (tags.length === 0) {
          this.warn("no unambiguous latin woff2 asset; the latin font preloads were not injected");
          return;
        }
        return tags;
      },
    },
  };
}
