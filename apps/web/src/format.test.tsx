import { describe, expect, it } from "vitest";
import { fmtBytes, fmtCI, fmtDate, fmtEquation, fmtInt, fmtNumber, fmtPValue } from "./format";

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
