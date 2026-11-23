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

const IST = "Asia/Kolkata";
/** IST is UTC+05:30 all year. */
const IST_OFFSET_MS = 5.5 * 60 * 60 * 1000;

/** The time of day as HH:mm on a 24-hour clock. */
const IST_TIME = new Intl.DateTimeFormat("en-IN", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: IST,
});

/** ISO-8601 timestamp → DD/MM/YYYY HH:mm in IST on a 24-hour clock, "n/a" when absent. */
export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return NA;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return NA;
  const { year, month, day } = istDayParts(date);
  const padded = (part: number): string => String(part).padStart(2, "0");
  return `${padded(day)}/${padded(month)}/${year} ${IST_TIME.format(date)}`;
}

/** A share held as a fraction, as a whole percentage ("58%" for 0.58); "–" when not defined. */
export function fmtPercent(fraction: number | null | undefined): string {
  if (!isDefined(fraction)) return "–";
  const percent = fraction * 100;
  const whole = Math.round(percent);
  // A share that exists is never shown as 0%: under half a percent keeps one decimal.
  const shown =
    whole === 0 && percent !== 0
      ? percent.toLocaleString("en-IN", { maximumFractionDigits: 1 })
      : fmtInt(whole);
  return `${shown}%`;
}

/** A duration in seconds as "3 h 12 min", or "12 min" under an hour; "n/a" when not a duration. */
export function fmtUptime(seconds: number | null | undefined): string {
  if (!isDefined(seconds) || seconds < 0) return NA;
  const totalMinutes = Math.floor(seconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${fmtInt(hours)} h ${minutes} min` : `${minutes} min`;
}

/** An IST calendar day, read part by part so the output never depends on the locale. */
const IST_DAY = new Intl.DateTimeFormat("en-CA", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  timeZone: IST,
});

/** An instant's IST calendar day, as the parts a date input's YYYY-MM-DD is made of. */
function istDayParts(date: Date): { year: number; month: number; day: number } {
  const parts = IST_DAY.formatToParts(date);
  const read = (type: Intl.DateTimeFormatPartTypes): number =>
    Number(parts.find((part) => part.type === type)?.value);
  return { year: read("year"), month: read("month"), day: read("day") };
}

/** An instant → the IST day it falls on, as the YYYY-MM-DD a date input holds. */
export function toIstDateInput(date: Date): string {
  const { year, month, day } = istDayParts(date);
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

/**
 * A date input's YYYY-MM-DD → the instant that IST day starts; null unless it is a real
 * day. Dates are IST calendar days everywhere in the app, never UTC ones.
 */
export function fromIstDateInput(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (match === null) return null;
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])];
  // Date.UTC folds an impossible day into the next month, so the day is read back.
  const noon = new Date(Date.UTC(year, month - 1, day, 12));
  if (
    noon.getUTCFullYear() !== year ||
    noon.getUTCMonth() !== month - 1 ||
    noon.getUTCDate() !== day
  ) {
    return null;
  }
  return new Date(Date.UTC(year, month - 1, day) - IST_OFFSET_MS);
}
