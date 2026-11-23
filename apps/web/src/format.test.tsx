import { describe, expect, it } from "vitest";
import {
  fmtBytes,
  fmtCI,
  fmtDate,
  fmtDateTime,
  fmtEquation,
  fmtInt,
  fmtNumber,
  fmtPercent,
  fmtPValue,
  fmtUptime,
  fromIstDateInput,
  toIstDateInput,
} from "./format";

describe("fmtNumber — 'not defined' is n/a, never 0 or NaN", () => {
  it("renders null and undefined as n/a", () => {
    expect(fmtNumber(null)).toBe("n/a");
    expect(fmtNumber(undefined)).toBe("n/a");
  });

  it("renders non-finite values as n/a", () => {
    expect(fmtNumber(Number.NaN)).toBe("n/a");
    expect(fmtNumber(Number.POSITIVE_INFINITY)).toBe("n/a");
    expect(fmtNumber(Number.NEGATIVE_INFINITY)).toBe("n/a");
  });

  it("formats finite numbers with en-IN grouping and bounded digits", () => {
    expect(fmtNumber(1234.5678)).toBe("1,234.5678");
    expect(fmtNumber(100000)).toBe("1,00,000");
    expect(fmtNumber(0.5, { digits: 4 })).toBe("0.5");
    expect(fmtNumber(-12.3456789, { digits: 3 })).toBe("-12.346");
  });

  it("uses scientific notation for very large and very small magnitudes", () => {
    expect(fmtNumber(1.23456e15)).toBe("1.2346e+15");
    expect(fmtNumber(3.2e-7)).toBe("3.2e-7");
    expect(fmtNumber(0)).toBe("0");
  });
});

describe("fmtInt", () => {
  it("renders null as n/a and groups integers en-IN", () => {
    expect(fmtInt(null)).toBe("n/a");
    expect(fmtInt(45)).toBe("45");
    expect(fmtInt(20000)).toBe("20,000");
    expect(fmtInt(10000000)).toBe("1,00,00,000");
  });
});

describe("fmtPValue", () => {
  it("renders null as n/a", () => {
    expect(fmtPValue(null)).toBe("n/a");
  });
  it("keeps small values readable", () => {
    expect(fmtPValue(0.06234)).toBe("0.0623");
    expect(fmtPValue(2.5e-9)).toBe("2.5e-9");
    expect(fmtPValue(0)).toBe("0");
  });
});

describe("fmtCI", () => {
  it("renders a bracketed interval, n/a when either end is missing", () => {
    expect(fmtCI(1.2, 3.4)).toBe("[1.2, 3.4]");
    expect(fmtCI(null, 3.4)).toBe("n/a");
    expect(fmtCI(1.2, null)).toBe("n/a");
    expect(fmtCI(null, null)).toBe("n/a");
  });
});

describe("fmtEquation — the engine's ** is shown and copied as ^", () => {
  it("turns every power operator into a caret", () => {
    expect(fmtEquation("y = 0.680918 + 2.37953*x + 0.0900383*x**2 + 1.07529e-05*x**5")).toBe(
      "y = 0.680918 + 2.37953*x + 0.0900383*x^2 + 1.07529e-05*x^5",
    );
  });

  it("leaves a parenthesised exponent and a plain product alone", () => {
    expect(fmtEquation("y = a*(1 + b*x)**(c/d)")).toBe("y = a*(1 + b*x)^(c/d)");
    expect(fmtEquation("y = a*b*x")).toBe("y = a*b*x");
  });

  it("never touches scientific notation", () => {
    expect(fmtEquation("y = 1.07529e-05*x")).toBe("y = 1.07529e-05*x");
  });

  it("renders an empty string for a missing equation", () => {
    expect(fmtEquation(null)).toBe("");
    expect(fmtEquation(undefined)).toBe("");
    expect(fmtEquation("")).toBe("");
  });
});

describe("fmtDate — IST display", () => {
  it("formats an ISO timestamp as DD/MM/YYYY in Asia/Kolkata", () => {
    expect(fmtDate("2026-11-17T06:00:00+00:00")).toBe("17/11/2026");
  });

  it("rolls to the next IST day when the offset crosses midnight", () => {
    expect(fmtDate("2026-11-17T20:00:00Z")).toBe("18/11/2026");
  });

  it("renders n/a for a missing timestamp", () => {
    expect(fmtDate(null)).toBe("n/a");
  });
});

describe("fmtBytes — a size in the largest unit below 1000", () => {
  it("uses decimal units, as the upload limit does (20 MB = 20,000,000 bytes)", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(1_500)).toBe("1.5 KB");
    expect(fmtBytes(20_000_000)).toBe("20 MB");
    expect(fmtBytes(25_000_000)).toBe("25 MB");
    expect(fmtBytes(1_000_000_000)).toBe("1 GB");
    expect(fmtBytes(1_250_000_000_000)).toBe("1.3 TB");
  });

  it("keeps one decimal at most and drops a trailing .0", () => {
    expect(fmtBytes(100_500_000)).toBe("100.5 MB");
    expect(fmtBytes(100_000_000)).toBe("100 MB");
    expect(fmtBytes(999_949_999)).toBe("999.9 MB");
    // 999.95 MB would print as "1,000 MB": it is already a gigabyte
    expect(fmtBytes(999_960_000)).toBe("1 GB");
  });

  it("renders n/a for a missing size and never a negative one", () => {
    expect(fmtBytes(null)).toBe("n/a");
    expect(fmtBytes(undefined)).toBe("n/a");
    expect(fmtBytes(Number.NaN)).toBe("n/a");
    expect(fmtBytes(-5)).toBe("0 B");
  });
});

describe("fmtDateTime — DD/MM/YYYY HH:mm in IST, 24-hour", () => {
  it("formats an ISO timestamp as date and time in Asia/Kolkata", () => {
    expect(fmtDateTime("2026-11-17T06:00:00+00:00")).toBe("17/11/2026 11:30");
    expect(fmtDateTime("2026-11-17T14:05:00+05:30")).toBe("17/11/2026 14:05");
  });

  it("never prints a 12-hour clock or a meridian", () => {
    expect(fmtDateTime("2026-11-17T00:05:00Z")).toBe("17/11/2026 05:35");
    expect(fmtDateTime("2026-11-17T12:00:00Z")).toBe("17/11/2026 17:30");
    expect(fmtDateTime("2026-11-17T18:30:00Z")).not.toMatch(/am|pm/i);
  });

  it("moves to the next IST day at IST midnight, not a second sooner", () => {
    // IST is UTC+05:30, so its midnight is 18:30 UTC
    expect(fmtDateTime("2026-11-17T18:29:59Z")).toBe("17/11/2026 23:59");
    expect(fmtDateTime("2026-11-17T18:30:00Z")).toBe("18/11/2026 00:00");
  });

  it("renders n/a for a missing or invalid timestamp", () => {
    expect(fmtDateTime(null)).toBe("n/a");
    expect(fmtDateTime(undefined)).toBe("n/a");
    expect(fmtDateTime("")).toBe("n/a");
    expect(fmtDateTime("yesterday")).toBe("n/a");
  });
});

describe("fmtPercent — a fraction as a whole percentage", () => {
  it("rounds a fraction to a whole percent", () => {
    expect(fmtPercent(0.58)).toBe("58%");
    expect(fmtPercent(0)).toBe("0%");
    expect(fmtPercent(1)).toBe("100%");
    expect(fmtPercent(0.996)).toBe("100%");
  });

  it("keeps a decimal rather than report a real share as 0%", () => {
    expect(fmtPercent(0.004)).toBe("0.4%");
    expect(fmtPercent(0.0001)).toBe("0%");
  });

  it("renders an en dash when the share is not defined", () => {
    expect(fmtPercent(null)).toBe("–");
    expect(fmtPercent(undefined)).toBe("–");
    expect(fmtPercent(Number.NaN)).toBe("–");
  });
});

describe("fmtUptime — a duration in hours and minutes", () => {
  it("renders hours and minutes, dropping the hours under one hour", () => {
    expect(fmtUptime(11520)).toBe("3 h 12 min");
    expect(fmtUptime(3600)).toBe("1 h 0 min");
    expect(fmtUptime(720)).toBe("12 min");
    expect(fmtUptime(25)).toBe("0 min");
  });

  it("counts hours past a day as hours, and rounds a float down", () => {
    expect(fmtUptime(180180)).toBe("50 h 3 min");
    expect(fmtUptime(119.9)).toBe("1 min");
  });

  it("renders n/a when the duration is missing or negative", () => {
    expect(fmtUptime(null)).toBe("n/a");
    expect(fmtUptime(undefined)).toBe("n/a");
    expect(fmtUptime(-1)).toBe("n/a");
    expect(fmtUptime(Number.POSITIVE_INFINITY)).toBe("n/a");
  });
});

describe("IST date inputs — YYYY-MM-DD of the IST calendar day, and back", () => {
  it("gives the IST day of an instant as a date input's YYYY-MM-DD", () => {
    expect(toIstDateInput(new Date("2026-11-17T06:00:00Z"))).toBe("2026-11-17");
    expect(toIstDateInput(new Date("2026-11-17T18:29:59Z"))).toBe("2026-11-17");
    // the instant IST midnight begins is already the next day
    expect(toIstDateInput(new Date("2026-11-17T18:30:00Z"))).toBe("2026-11-18");
  });

  it("reads a date input's YYYY-MM-DD as the instant that day starts in IST", () => {
    expect(fromIstDateInput("2026-11-18")?.getTime()).toBe(Date.UTC(2026, 10, 17, 18, 30));
    expect(fromIstDateInput("2026-01-01")?.getTime()).toBe(Date.UTC(2025, 11, 31, 18, 30));
  });

  it("round-trips a day", () => {
    const start = fromIstDateInput("2026-11-18") as Date;
    expect(toIstDateInput(start)).toBe("2026-11-18");
    expect(toIstDateInput(new Date(start.getTime() + 86_399_000))).toBe("2026-11-18");
    expect(toIstDateInput(new Date(start.getTime() + 86_400_000))).toBe("2026-11-19");
  });

  it("returns null for an empty, malformed or impossible date", () => {
    expect(fromIstDateInput("")).toBeNull();
    expect(fromIstDateInput("18/11/2026")).toBeNull();
    expect(fromIstDateInput("2026-11-18T00:00")).toBeNull();
    expect(fromIstDateInput("2026-02-30")).toBeNull();
  });
});
