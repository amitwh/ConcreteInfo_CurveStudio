/**
 * The whole-workspace empty state (design item 7 of the 2026-11-23 UX fixes): an
 * account with no datasets lands here instead of the everyday workspace. It offers
 * the three next steps and leads each to the existing thing — the sidebar's upload
 * picker and paste dialog (handed in as callbacks, so no upload machinery is
 * duplicated) and the help's quick-start section. It spans everything but the
 * datasets sidebar, which stays on screen with its own upload card.
 */

import type { ReactElement } from "react";
import { Link } from "react-router-dom";
import { BookOpen, ClipboardPaste, Upload } from "lucide-react";

export interface EmptyWorkspaceProps {
  /** The sidebar's existing upload action: opens the file picker. */
  onUpload: () => void;
  /** The sidebar's existing paste action: opens the paste dialog. */
  onPaste: () => void;
}

export function EmptyWorkspace({ onUpload, onPaste }: EmptyWorkspaceProps): ReactElement {
  return (
    <section className="panel empty-workspace" aria-label="Get started">
      <h2 className="empty-workspace__title">Start with your first dataset</h2>
      <p className="empty-workspace__lead">
        Upload paired x/y measurements and every model is tried and ranked automatically.
        Get your data in one of three ways:
      </p>
      <div className="empty-workspace__steps">
        <button type="button" className="empty-step" onClick={onUpload}>
          <span className="empty-step__icon">
            <Upload className="icon" aria-hidden="true" focusable="false" />
          </span>
          <span className="empty-step__title">Upload a file</span>
          <span className="empty-step__hint">csv · tsv · txt · xlsx · json — up to 20 MB</span>
        </button>
        <button type="button" className="empty-step" onClick={onPaste}>
          <span className="empty-step__icon">
            <ClipboardPaste className="icon" aria-hidden="true" focusable="false" />
          </span>
          <span className="empty-step__title">Paste data</span>
          <span className="empty-step__hint">Copy a block straight from a spreadsheet</span>
        </button>
        <Link className="empty-step" to="/help#getting-started">
          <span className="empty-step__icon">
            <BookOpen className="icon" aria-hidden="true" focusable="false" />
          </span>
          <span className="empty-step__title">Read the quick start</span>
          <span className="empty-step__hint">A short guide in the help</span>
        </Link>
      </div>
    </section>
  );
}
