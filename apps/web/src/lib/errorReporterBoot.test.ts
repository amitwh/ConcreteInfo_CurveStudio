import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
// Its body runs with this import, before anything else in this file: the point of the module.
import "./errorReporterBoot";
import { afterAll, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { installErrorReporter, type ErrorReportEvent } from "./errorReporter";

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = (path: string): string => readFileSync(resolve(SRC, path), "utf-8");

let beacon: Mock<(url: string | URL, data?: BodyInit | null) => boolean>;

beforeEach(() => {
  beacon = vi.fn(() => true);
  Object.defineProperty(navigator, "sendBeacon", { configurable: true, value: beacon });
});

afterAll(() => {
  // The installation is idempotent: this is the boot module's, removed now that the file is done.
  installErrorReporter({ apiBase: "/api/v1" })();
  delete (navigator as { sendBeacon?: unknown }).sendBeacon;
});

describe("errorReporterBoot", () => {
  it("has the hooks in by the time its import returns, with nothing else loaded", async () => {
    // A module of the app throws while it is being evaluated: no React, no screen, only these.
    const error = new TypeError("a module failed to load");
    window.dispatchEvent(new ErrorEvent("error", { error, message: error.message }));
    window.dispatchEvent(new Event("pagehide"));
    expect(beacon).toHaveBeenCalledTimes(1);
    expect(beacon.mock.calls[0]?.[0]).toBe("/api/v1/errors");
    const [, data] = beacon.mock.calls[0] as [string | URL, Blob];
    const { events } = JSON.parse(await data.text()) as { events: ErrorReportEvent[] };
    expect(events[0]).toMatchObject({ category: "crash", error_type: "TypeError", message: "a module failed to load" });
  });

  it("is the first import of main.tsx", () => {
    const firstImport = /^import\s(?:"([^"]+)"|[^;]*?from\s+"([^"]+)");/m.exec(source("main.tsx"));
    expect(firstImport?.[1] ?? firstImport?.[2]).toBe("./lib/errorReporterBoot");
  });

  it("depends on nothing that could fail before the hooks are in", () => {
    /** The modules a file loads when it is evaluated: plain imports, not type-only ones. */
    const runtimeImports = (path: string): string[] =>
      [...source(path).matchAll(/^import\s(?!type\b)[^;]*?from\s+"([^"]+)";$/gm)].map(
        (match) => match[1] ?? "",
      );

    expect(runtimeImports("lib/errorReporterBoot.ts")).toEqual(["../api/apiBase", "./errorReporter"]);
    expect(runtimeImports("api/apiBase.ts")).toEqual([]);
    expect(runtimeImports("lib/errorReporter.ts")).toEqual([]);
  });
});
