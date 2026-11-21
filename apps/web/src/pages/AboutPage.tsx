/** /about — who makes CurveStudio, its licence, credits, version and source links. */

import type { ReactElement } from "react";
import { Info } from "lucide-react";
import { Link } from "react-router-dom";

const APP_VERSION = __APP_VERSION__; // from package.json at build time
const SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio";
const AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.html";
const WEBSITE_URL = "https://www.concreteinfo.in";

/** Same credits the desktop app's About window shows. */
const CREDITS: Array<{ work: string; licence: string }> = [
  { work: "Qt / PySide6 (desktop app)", licence: "LGPL-3.0" },
  { work: "NumPy, SciPy, SymPy", licence: "BSD-3-Clause" },
  { work: "pyqtgraph", licence: "MIT" },
  { work: "Inter and JetBrains Mono typefaces", licence: "SIL Open Font License 1.1" },
  { work: "Lucide icons", licence: "ISC" },
];

export function AboutPage(): ReactElement {
  return (
    <main className="about-page">
      <section className="card">
        <img
          className="about-logo"
          src="/concreteinfo-logo.png"
          alt="ConcreteInfo"
          width={160}
          height={45}
        />
        <h1>
          <Info className="icon icon--lg" aria-hidden="true" focusable="false" />
          About CurveStudio
        </h1>
        <p>
          CurveStudio is a free, open-source curve fitting and regression tool. Fits run on
          the same engine as the desktop app, so the numbers you see here are the numbers
          the science produced.
        </p>
        <dl className="about-facts">
          <div>
            <dt>Version</dt>
            <dd>{APP_VERSION}</dd>
          </div>
          <div>
            <dt>Licence</dt>
            <dd>
              <a href={AGPL_URL} rel="noopener noreferrer" target="_blank">
                AGPL-3.0-or-later
              </a>
            </dd>
          </div>
          <div>
            <dt>Source code</dt>
            <dd>
              <a href={SOURCE_URL} rel="noopener noreferrer" target="_blank">
                {SOURCE_URL.replace("https://", "")}
              </a>
            </dd>
          </div>
          <div>
            <dt>Website</dt>
            <dd>
              <a href={WEBSITE_URL} rel="noopener noreferrer" target="_blank">
                www.concreteinfo.in
              </a>
            </dd>
          </div>
          <div>
            <dt>Help</dt>
            <dd>
              <Link to="/help">Help and documentation</Link>
            </dd>
          </div>
        </dl>

        <h2>Credits</h2>
        <ul className="credits">
          {CREDITS.map((credit) => (
            <li key={credit.work}>
              <span>{credit.work}</span>
              <span className="credits__licence">{credit.licence}</span>
            </li>
          ))}
        </ul>

        <p className="muted">
          A fit that does not converge is reported as such — never as a plausible curve.
          Statistics that are undefined show “n/a”, never 0.
        </p>
        <p className="about-links">
          <Link to="/help">Read the help</Link>
          <span aria-hidden="true"> · </span>
          <Link to="/">Back to the workspace</Link>
        </p>
      </section>
    </main>
  );
}
