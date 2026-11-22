/**
 * The paste grid's data rules: split a clipboard table into cells, serialise it back.
 *
 * Both directions mirror the desktop grid (`views/spreadsheet.py`), so a block copied
 * from Excel/LibreOffice/Google Sheets lands as the same rectangle in either app. A
 * cell may itself contain the delimiter, a newline or a double quote — spreadsheets
 * quote those (`"a,b"`), and RFC 4180 quoting is honoured in both directions.
 */

/** Tab when the text has tabs (every spreadsheet app's copy format), else comma/semicolon. */
export function detectDelimiter(text: string): string {
  if (text.includes("\t")) return "\t";
  if (text.includes(";") && !text.includes(",")) return ";";
  return ",";
}

/** Parse a pasted table into a padded rectangle of cell text. */
export function parseDelimited(text: string, delimiter?: string): string[][] {
  if (text.trim() === "") return [];
  const sep = delimiter ?? detectDelimiter(text);
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          cell += '"'; // an escaped quote inside a quoted field
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += char;
      }
      continue;
    }
    if (char === '"' && cell === "") {
      quoted = true;
    } else if (char === sep) {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (char !== "\r") {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  // a trailing newline ends the last row, it does not open an empty one
  const last = rows[rows.length - 1];
  if (last !== undefined && last.length === 1 && last[0] === "") rows.pop();
  const width = rows.reduce((widest, current) => Math.max(widest, current.length), 0);
  return rows.map((current) => [...current, ...Array(width - current.length).fill("")]);
}

function quoteCell(text: string): string {
  return /[\t\n";,]/.test(text) || text.includes('"')
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

/** Tab-separated text with minimal quoting — what the untitled import is named for. */
export function toDelimited(rows: string[][]): string {
  return rows.map((row) => row.map(quoteCell).join("\t")).join("\n") + "\n";
}

/** Drop fully blank trailing rows and columns; never returns an empty rectangle. */
export function trimGrid(rows: string[][]): string[][] {
  const kept = rows.map((row) => [...row]);
  let bottom = kept[kept.length - 1];
  while (kept.length > 1 && bottom !== undefined && bottom.every((cell) => cell.trim() === "")) {
    kept.pop();
    bottom = kept[kept.length - 1];
  }
  const width = kept.reduce((widest, row) => {
    let last = row.length;
    while (last > 0 && (row[last - 1] ?? "").trim() === "") last -= 1;
    return Math.max(widest, last);
  }, 0);
  return kept.map((row) => row.slice(0, width));
}

/** The importable text, or null while the grid holds nothing. */
export function gridToImportText(rows: string[][]): string | null {
  const trimmed = trimGrid(rows);
  if (trimmed.every((row) => row.every((cell) => cell.trim() === ""))) return null;
  return toDelimited(trimmed);
}

/** Insert a pasted block at an anchor, growing rows and columns (a blank grid is replaced). */
export function applyBlock(
  grid: string[][],
  anchor: { row: number; column: number },
  block: string[][],
  replace = false,
): string[][] {
  if (block.length === 0) return grid;
  const startRow = replace ? 0 : Math.max(0, anchor.row);
  const startColumn = replace ? 0 : Math.max(0, anchor.column);
  const blockWidth = block.reduce((widest, row) => Math.max(widest, row.length), 0);
  const height = Math.max(replace ? 0 : grid.length, startRow + block.length);
  const columns = Math.max(replace ? 0 : (grid[0]?.length ?? 0), startColumn + blockWidth);
  const next = Array.from({ length: height }, (_, row) =>
    Array.from({ length: columns }, (_, column) =>
      replace ? "" : (grid[row]?.[column] ?? ""),
    ),
  );
  block.forEach((row, offset) => {
    const target = next[startRow + offset];
    if (target === undefined) return;
    row.forEach((cell, column) => {
      target[startColumn + column] = cell;
    });
  });
  return next;
}

/** Whether every cell is empty (so a paste may replace the whole grid). */
export function isBlankGrid(rows: string[][]): boolean {
  return rows.every((row) => row.every((cell) => cell.trim() === ""));
}
