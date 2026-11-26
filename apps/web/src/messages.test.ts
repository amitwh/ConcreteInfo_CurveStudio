import { describe, expect, it } from "vitest";
import { ENGINE_SENTENCES, engineSentence, explainCode } from "./messages";

describe("engine messages (the core's mapping, copied for the web)", () => {
  it("ships the mapping the core test pins to curvestudio.messages", () => {
    // A spot check on both ends of the vocabulary: a fit status and a job code. The
    // exhaustive one-sentence-per-code test lives in the core suite, which can read
    // the Python sources this file cannot.
    expect(ENGINE_SENTENCES["not_converged"]).toMatch(/^The fit did not converge/);
    expect(ENGINE_SENTENCES["TIMEOUT"]).toMatch(/^The job did not finish in time/);
    expect(Object.keys(ENGINE_SENTENCES).length).toBeGreaterThan(50);
  });

  it("returns null for a code with no sentence", () => {
    expect(engineSentence("NO_SUCH_CODE")).toBeNull();
  });

  it("puts the sentence first and keeps the server's message after it", () => {
    expect(explainCode("TIMEOUT", "the fit did not finish within 30 s")).toBe(
      `${ENGINE_SENTENCES["TIMEOUT"]} — the fit did not finish within 30 s`,
    );
    expect(explainCode("TIMEOUT", null)).toBe(ENGINE_SENTENCES["TIMEOUT"]);
    expect(explainCode(null, "plain message")).toBe("plain message");
    expect(explainCode(null, null)).toBe("something went wrong");
    expect(explainCode("NO_SUCH_CODE", "plain message")).toBe("plain message");
  });
});
