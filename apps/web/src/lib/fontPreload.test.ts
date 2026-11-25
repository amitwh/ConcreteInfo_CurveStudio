import { describe, expect, it } from "vitest";
import { latinFontPreloads } from "./fontPreload";

/**
 * The asset names a real build emits for the two imported fontsource stylesheets: the
 * latin file this app must preload, its latin-ext sibling (declared by the same CSS, so
 * emitted too), one other subset, and unrelated hashed assets of other kinds.
 */
const EMITTED_ASSETS = [
  "assets/inter-cyrillic-wght-normal-C1r1LLic.woff2",
  "assets/inter-latin-ext-wght-normal-L4T1N.woff2",
  "assets/inter-latin-wght-normal-B0aRD.woff2",
  "assets/jetbrains-mono-greek-wght-normal-6r33c.woff2",
  "assets/jetbrains-mono-latin-ext-wght-normal-3XT.woff2",
  "assets/jetbrains-mono-latin-wght-normal-M0N0.woff2",
  "assets/index-Js7xQ4mK.js",
  "assets/index-mT3pLsVq.css",
  "assets/plotly-bundle-9wQ1e5R7.js",
] as const;

const INTER_PRELOAD = {
  tag: "link",
  attrs: {
    rel: "preload",
    href: "/assets/inter-latin-wght-normal-B0aRD.woff2",
    as: "font",
    type: "font/woff2",
    crossorigin: "anonymous",
  },
  injectTo: "head",
};

const MONO_PRELOAD = {
  tag: "link",
  attrs: {
    rel: "preload",
    href: "/assets/jetbrains-mono-latin-wght-normal-M0N0.woff2",
    as: "font",
    type: "font/woff2",
    crossorigin: "anonymous",
  },
  injectTo: "head",
};

describe("latinFontPreloads", () => {
  it("preloads exactly the two latin woff2 files, with the attributes a font preload needs", () => {
    expect(latinFontPreloads(EMITTED_ASSETS)).toEqual([INTER_PRELOAD, MONO_PRELOAD]);
  });

  it("preloads nothing for a family with no unambiguous latin file", () => {
    // latin-ext alone must not be preloaded in latin's place, and two latin candidates
    // mean the naming assumption broke — no preload beats a wrong one.
    expect(latinFontPreloads(["assets/inter-latin-ext-wght-normal-L4T1N.woff2"])).toEqual([]);
    expect(
      latinFontPreloads(["assets/inter-latin-a-HASH.woff2", "assets/inter-latin-b-HASH.woff2"]),
    ).toEqual([]);
    expect(latinFontPreloads([])).toEqual([]);
  });
});
