/**
 * Number and date formatting. One rule above all: a statistic that is not defined is
 * rendered as "n/a" — never 0, never NaN, never a silently substituted value.
 */

const NA = "n/a";

function isDefined(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** "3.2000e-7" → "3.2e-7": keep the mantissa free of padded zeros. */
function trimExponential(text: string): string {
  const [mantissa = "0", exponent = "0"] = text.split("e");
  const trimmed = mantissa.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return `${trimmed}e${exponent}`;
}

/** General number formatting with en-IN digit grouping, "n/a" when not defined. */
export function fmtNumber(
  value: number | null | undefined,
  options: { digits?: number; sci?: boolean } = {},
): string {
  if (!isDefined(value)) return NA;
  const { digits = 6, sci = true } = options;
  const magnitude = Math.abs(value);
  if (sci && magnitude !== 0 && (magnitude >= 1e9 || magnitude < 1e-4)) {
    return trimExponential(value.toExponential(4));
  }
  return value.toLocaleString("en-IN", { maximumFractionDigits: digits });
}

/** Integer formatting with en-IN grouping (1,00,000), "n/a" when not defined. */
export function fmtInt(value: number | null | undefined): string {
  if (!isDefined(value)) return NA;
  return value.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/**
 * A byte count in the largest unit that keeps it under 1000 ("512 KB", "20 MB", "1.5 GB"),
 * at most one decimal, "n/a" when not defined. Decimal units, as everywhere else in the
 * app: the upload limit is 20 MB = 20,000,000 bytes.
 */
export function fmtBytes(value: number | null | undefined): string {
  if (!isDefined(value)) return NA;
  let size = Math.max(value, 0);
  let unit = 0;
  // Bytes are whole numbers; from KB up, 999.95 would print as "1,000", so it is the next unit.
  while (unit < BYTE_UNITS.length - 1 && size >= (unit === 0 ? 1000 : 999.95)) {
    size /= 1000;
    unit += 1;
  }
  const digits = unit === 0 ? 0 : 1;
  const number = size.toLocaleString("en-IN", { maximumFractionDigits: digits });
  return `${number} ${BYTE_UNITS[unit] ?? "B"}`;
}

/** p-values: fixed decimals until 1e-4, exponential below; "n/a" when not defined. */
export function fmtPValue(value: number | null | undefined): string {
  if (!isDefined(value)) return NA;
  if (value === 0) return "0";
  if (value < 1e-4) return trimExponential(value.toExponential(4));
  return value.toLocaleString("en-IN", { maximumFractionDigits: 4 });
}

/** A confidence interval "[low, high]", "n/a" when either end is missing. */
export function fmtCI(
  low: number | null | undefined,
  high: number | null | undefined,
): string {
  if (!isDefined(low) || !isDefined(high)) return NA;
  return `[${fmtNumber(low)}, ${fmtNumber(high)}]`;
}

/**
 * An equation as the reader should see it: the engine stores expressions in Python's
 * operator syntax, where a power is `x**2`. Display and copy both use `^` — it is what a
 * scientist writes, what Excel evaluates, and never reads as a stray pair of asterisks.
 */
export function fmtEquation(text: string | null | undefined): string {
  if (!text) return "";
  return text.replaceAll("**", "^");
}

/** ISO-8601 timestamp → DD/MM/YYYY displayed in IST (Asia/Kolkata), "n/a" when absent. */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return NA;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return NA;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    timeZone: "Asia/Kolkata",
  }).format(date);
}
