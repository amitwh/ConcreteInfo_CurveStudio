import { describe, expect, it } from "vitest";
import {
  applyBlock,
  detectDelimiter,
  gridToImportText,
  isBlankGrid,
  parseDelimited,
  toDelimited,
  trimGrid,
} from "./spreadsheet";

describe("paste grid data rules", () => {
  it("splits a tab-separated block into a rectangle", () => {
    expect(parseDelimited("x\ty\n0\t1.5\n1\t2.5\n")).toEqual([
      ["x", "y"],
      ["0", "1.5"],
      ["1", "2.5"],
    ]);
  });

  it("keeps a quoted cell containing the delimiter, a quote or a newline in one cell", () => {
    expect(parseDelimited('name\tvalue\n"a,b"\t1\n"x\ty"\t2\n"say ""hi"""\t3\n')).toEqual([
      ["name", "value"],
      ["a,b", "1"],
      ["x\ty", "2"],
      ['say "hi"', "3"],
    ]);
    expect(parseDelimited('"two\nlines"\t1\n')[0]).toEqual(["two\nlines", "1"]);
  });

  it("pads ragged rows and detects comma / semicolon formats", () => {
    expect(parseDelimited("a,b\n1,2,3\n")).toEqual([
      ["a", "b"],
      ["1", "2", "3"],
    ].map((row) => [...row, ...Array(3 - row.length).fill("")]));
    expect(detectDelimiter("a;b\n1;2")).toBe(";");
    expect(detectDelimiter("a\tb")).toBe("\t");
    expect(parseDelimited("a;b\n1;2")).toEqual([["a", "b"], ["1", "2"]]);
  });

  it("ignores an empty paste", () => {
    expect(parseDelimited("   \n")).toEqual([]);
  });

  it("round-trips quoted cells through the tab writer", () => {
    const rows = [["a,b", "plain"], ["x\ty", 'say "hi"']];
    expect(parseDelimited(toDelimited(rows))).toEqual(rows);
  });

  it("trims trailing blank rows and columns but keeps inner blanks", () => {
    expect(
      trimGrid([
        ["x", "y", "", ""],
        ["1", "", "", ""],
        ["", "", "", ""],
      ]),
    ).toEqual([
      ["x", "y"],
      ["1", ""],
    ]);
  });

  it("is not importable while every cell is blank", () => {
    expect(gridToImportText([["", ""], ["", ""]])).toBeNull();
    expect(isBlankGrid([["", ""]])).toBe(true);
    expect(gridToImportText([["x", "y"], ["1", "2"]])).toBe("x\ty\n1\t2\n");
  });

  it("drops a pasted block at the anchor and grows the grid", () => {
    const grid = [
      ["a", "b"],
      ["1", "2"],
    ];
    expect(applyBlock(grid, { row: 1, column: 1 }, [["9", "8"]])).toEqual([
      ["a", "b", ""],
      ["1", "9", "8"],
    ]);
    // a paste into a blank grid replaces it
    expect(
      applyBlock(
        [["", "", ""], ["", "", ""], ["", "", ""]],
        { row: 2, column: 2 },
        [["x", "y"], ["1", "2"]],
        true,
      ),
    ).toEqual([
      ["x", "y"],
      ["1", "2"],
    ]);
  });
});
