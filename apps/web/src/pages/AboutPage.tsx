/** /about — licence, version and source links. No secrets, no tracking. */

import type { ReactElement } from "react";
import { Link } from "react-router-dom";

const APP_VERSION = "0.1.0";
const SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio";
const AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.html";

export function AboutPage(): ReactElement {
  return (
    <main className="about-page">
      <section className="card">
        <h1>About CurveStudio</h1>
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
        </dl>
        <p className="muted">
          A fit that does not converge is reported as such — never as a plausible curve.
          Statistics that are undefined show “n/a”, never 0.
        </p>
        <p>
          <Link to="/">Back to the workspace</Link>
        </p>
      </section>
    </main>
  );
}
