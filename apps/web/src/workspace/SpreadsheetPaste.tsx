/**
 * "Paste data": an embedded spreadsheet grid for the web workspace.
 *
 * Selecting a block in Excel/LibreOffice/Google Sheets and pressing Ctrl+V over the
 * grid drops the whole rectangle in — rows and columns grow to fit, a paste into an
 * empty grid replaces it — and the cells stay editable, so a typo is fixed in place.
 * The result is serialised as TSV with quoting and handed back as a pasted file, so
 * it travels through exactly the same upload path as a chosen file (column pickers
 * included).
 */

import { useEffect, useRef, useState, type ClipboardEvent, type ReactElement } from "react";
import { ClipboardPaste, Plus, Trash2, X } from "lucide-react";
import {
  applyBlock,
  gridToImportText,
  isBlankGrid,
  parseDelimited,
} from "./spreadsheet";

const START_ROWS = 6;
const START_COLUMNS = 3;

export interface SpreadsheetPasteProps {
  onImport: (text: string) => void;
  onClose: () => void;
  busy?: boolean;
}

function blankGrid(rows = START_ROWS, columns = START_COLUMNS): string[][] {
  return Array.from({ length: rows }, () => Array.from({ length: columns }, () => ""));
}

function columnName(column: number): string {
  let name = "";
  let index = column;
  for (;;) {
    name = String.fromCharCode(65 + (index % 26)) + name;
    index = Math.floor(index / 26) - 1;
    if (index < 0) return name;
  }
}

export function SpreadsheetPaste({ onImport, onClose, busy = false }: SpreadsheetPasteProps): ReactElement {
  const [rows, setRows] = useState<string[][]>(() => blankGrid());
  const [anchor, setAnchor] = useState({ row: 0, column: 0 });
  const [notice, setNotice] = useState<string | null>(null);
  const firstCell = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    firstCell.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const importText = gridToImportText(rows);

  const changeCell = (row: number, column: number, value: string): void => {
    setRows((current) =>
      current.map((line, r) =>
        r === row ? line.map((cell, c) => (c === column ? value : cell)) : line,
      ),
    );
  };

  const pasteText = (text: string): void => {
    const block = parseDelimited(text);
    if (block.length === 0) {
      setNotice("The clipboard is empty.");
      return;
    }
    setNotice(null);
    setRows((current) => applyBlock(current, anchor, block, isBlankGrid(current)));
  };

  const onPaste = (event: ClipboardEvent<HTMLDivElement>): void => {
    const text = event.clipboardData.getData("text/plain");
    if (text.trim() === "") return;
    // a single value pastes into the focused cell like any text; a table is ours
    if (!/[\t\n]/.test(text) && parseDelimited(text).length < 2) return;
    event.preventDefault();
    // the block lands where the paste happened — the focus state alone can lag a
    // programmatic paste, and the target input always knows its own coordinates
    const target = event.target as HTMLElement;
    const row = Number(target.dataset.row);
    const column = Number(target.dataset.column);
    if (Number.isInteger(row) && Number.isInteger(column)) {
      setAnchor({ row, column });
      setNotice(null);
      setRows((current) => {
        const at = { row, column };
        return applyBlock(current, at, parseDelimited(text), isBlankGrid(current));
      });
      return;
    }
    pasteText(text);
  };

  const pasteFromClipboard = async (): Promise<void> => {
    if (navigator.clipboard?.readText === undefined) {
      setNotice("This browser does not offer clipboard access — click a cell and press Ctrl+V.");
      return;
    }
    try {
      pasteText(await navigator.clipboard.readText());
    } catch {
      setNotice("The browser blocked clipboard access — click a cell and press Ctrl+V instead.");
    }
  };

  const addRow = (): void => {
    setRows((current) => [...current, Array.from({ length: current[0]?.length ?? START_COLUMNS }, () => "")]);
  };

  const addColumn = (): void => {
    setRows((current) => current.map((line) => [...line, ""]));
  };

  const clear = (): void => {
    setRows(() => blankGrid(rows.length, rows[0]?.length ?? START_COLUMNS));
    setAnchor({ row: 0, column: 0 });
    setNotice(null);
    firstCell.current?.focus();
  };

  return (
    <div className="modal" role="presentation" onMouseDown={onClose}>
      <div
        className="modal__panel modal__panel--wide"
        role="dialog"
        aria-modal="true"
        aria-labelledby="paste-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal__header">
          <h2 id="paste-title">Paste data</h2>
          <button type="button" className="btn btn--small" aria-label="Close" onClick={onClose}>
            <X className="icon" aria-hidden="true" focusable="false" />
          </button>
        </header>
        <p className="muted">
          Copy a block of cells from a spreadsheet and paste it here (Ctrl+V), or type the
          cells. The first row is the header.
        </p>
        <div className="spreadsheet__toolbar">
          <button type="button" className="btn btn--small btn--primary" onClick={() => void pasteFromClipboard()}>
            <ClipboardPaste className="icon" aria-hidden="true" focusable="false" />
            Paste from clipboard
          </button>
          <button type="button" className="btn btn--small" onClick={addRow}>
            <Plus className="icon" aria-hidden="true" focusable="false" />
            Row
          </button>
          <button type="button" className="btn btn--small" onClick={addColumn}>
            <Plus className="icon" aria-hidden="true" focusable="false" />
            Column
          </button>
          <button type="button" className="btn btn--small" onClick={clear}>
            <Trash2 className="icon" aria-hidden="true" focusable="false" />
            Clear
          </button>
        </div>
        <div className="spreadsheet" onPaste={onPaste}>
          <table role="grid" aria-label="Pasted data">
            <thead>
              <tr>
                <th scope="col" className="spreadsheet__gutter" aria-hidden="true" />
                {Array.from({ length: rows[0]?.length ?? 0 }, (_, column) => (
                  <th key={column} scope="col" className="spreadsheet__letter">
                    {columnName(column)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((line, row) => (
                <tr key={row}>
                  <th scope="row" className="spreadsheet__gutter">
                    {row + 1}
                  </th>
                  {line.map((cell, column) => (
                    <td key={column}>
                      <input
                        ref={row === 0 && column === 0 ? firstCell : undefined}
                        className="spreadsheet__cell"
                        data-row={row}
                        data-column={column}
                        aria-label={`${columnName(column)}${row + 1}`}
                        value={cell}
                        onChange={(event) => changeCell(row, column, event.target.value)}
                        onFocus={() => setAnchor({ row, column })}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {notice !== null && (
          <p className="alert alert--warning" role="status">
            {notice}
          </p>
        )}
        <footer className="modal__actions">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn--primary"
            disabled={busy || importText === null}
            onClick={() => {
              if (importText !== null) onImport(importText);
            }}
          >
            Import paste
          </button>
        </footer>
      </div>
    </div>
  );
}
