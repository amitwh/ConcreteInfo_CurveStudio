/**
 * The TypeScript copy of the core's `p_text` rule. The case list is the core's, pinned in
 * `tests/unit/tools/test_stats_contract.py::test_p_text`; a change there is a change here.
 */

import { describe, expect, it } from "vitest";
import { pText } from "./pText";

describe("pText", () => {
  // One list, in ascending order of p, that pins the whole rule. None first, since it is
  // not a number.
  it.each<[number | null, string]>([
    [null, "not computed"],
    [0.0, "p < 1e-300"],
    [3.2e-300, "p < 0.0001"],
    [0.00009, "p < 0.0001"],
    // The two cut-offs themselves fall on the side that still shows the number.
    [0.0001, "p = 0.0001"],
    [0.00012, "p = 0.00012"],
    [0.0324, "p = 0.0324"],
    // Below 0.1 three significant digits are kept, and their trailing zeros dropped.
    [0.05, "p = 0.05"],
    [0.09999, "p = 0.1"],
    // From 0.1 three decimals are shown, trailing zeros and all.
    [0.1, "p = 0.100"],
    [0.3171, "p = 0.317"],
    [0.5, "p = 0.500"],
    [0.9996, "p = 1.000"],
    [1.0, "p = 1.000"],
  ])("pText(%s) is %j", (p, text) => {
    expect(pText(p)).toBe(text);
  });

  it("treats an absent p the same as None", () => {
    expect(pText(undefined)).toBe("not computed");
  });
});
