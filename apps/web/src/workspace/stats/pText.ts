/**
 * How a p-value reads, the same on every surface. This is the web copy of the core's
 * `p_text` (packages/core/src/curvestudio/tools/stats/_base.py); the pinned case list in
 * `pText.test.ts` is the core's `test_p_text`, so the two rules move together.
 */

/** A number for a sentence: at most three significant digits, trailing zeros dropped. */
function num(value: number): string {
  return value
    .toPrecision(3)
    .replace(/(\.\d*?)0+$/, "$1")
    .replace(/\.$/, "");
}

/**
 * ``None`` is ``not computed``. Below 0.0001 only a bound is shown; from there to 0.1 three
 * significant digits are kept, and above that three decimals.
 */
export function pText(p: number | null | undefined): string {
  if (p === null || p === undefined) return "not computed";
  if (p === 0) return "p < 1e-300";
  if (p < 1e-4) return "p < 0.0001";
  if (p < 0.1) return `p = ${num(p)}`;
  return `p = ${p.toFixed(3)}`;
}
