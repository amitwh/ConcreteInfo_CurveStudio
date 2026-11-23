/** /about — who makes CurveStudio, its licence, credits, version and source links. */

import type { ReactElement } from "react";
import { Info } from "lucide-react";
import { Link } from "react-router-dom";

import otherProjects from "../content/other-projects.json";

const APP_VERSION = __APP_VERSION__; // from package.json at build time
const SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio";
const AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.html";
const WEBSITE_URL = "https://www.concreteinfo.in";
const CONTACT_EMAIL = "amit@concreteinfo.in";

/** The shared list (docs/other-projects.json), split into the two About groups. */
const OPEN_SOURCE = otherProjects.entries.filter((entry) => entry.kind === "open-source");
const COMMERCIAL = otherProjects.entries.filter((entry) => entry.kind === "commercial");

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

        <h2>Other tools we build and maintain.</h2>
        <div className="about-project-groups">
          <div>
            <h3>Open source</h3>
            <ul className="about-projects">
              {OPEN_SOURCE.map((project) => (
                <li key={project.url}>
                  <a href={project.url} rel="noopener noreferrer" target="_blank">
                    {project.name}
                  </a>
                  <span className="muted"> — {project.summary}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3>Commercial software</h3>
            <ul className="about-projects">
              {COMMERCIAL.map((project) => (
                <li key={project.url}>
                  <a href={project.url} rel="noopener noreferrer" target="_blank">
                    {project.name}
                  </a>
                  <span className="muted"> — {project.summary}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <p className="muted">
          More at{" "}
          <a href={WEBSITE_URL} rel="noopener noreferrer" target="_blank">
            www.concreteinfo.in
          </a>{" "}
          ·{" "}
          <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>
        </p>

        <h2>Why it is free and open</h2>
        <p>
          CurveStudio is free software. The desktop application is licensed{" "}
          <a href={AGPL_URL} rel="noopener noreferrer" target="_blank">
            AGPL-3.0-or-later
          </a>
          : use it at no cost, read the source, change it and share it. The AGPL is
          deliberate — anyone who offers a modified CurveStudio as a hosted service must
          publish their changes too, so the fitting method stays inspectable instead of
          becoming a black box, and improvements flow back to everyone. This web
          application is the same engine behind a browser interface and is free to use
          today. If hosting costs grow, a small subscription may be introduced for the
          hosted service only: the desktop app stays free, and the numbers you can
          already compute from your own data will not move behind a paywall.
        </p>

        <h2>Disclaimer</h2>
        <p>
          <strong>CurveStudio fits models to data; it does not decide what the data mean.</strong>{" "}
          A good R², RMSE, AICc or a narrow confidence band describes agreement with the
          sample you supplied. It is not evidence that the model is physically right, that
          a mix or a member complies with a standard, or that a value is safe to use.
        </p>
        <p>
          Results are not engineering, legal or commercial advice. Check every number that
          matters against the relevant code or standard, a second method, and a competent
          person’s judgement before using it in design, construction, testing,
          certification, pricing or any decision where being wrong has consequences.
        </p>
        <p>
          The software is provided “as is”, without warranty of any kind, express or
          implied, including fitness for a particular purpose. For the desktop application
          the disclaimers in sections 15 and 16 of the AGPL-3.0 apply; this hosted service
          is provided as it is and may change or be unavailable. To the extent permitted by
          law, ConcreteInfo and Amit Haridas are not liable for indirect, incidental or
          consequential loss arising from use of the software or its results. This is a
          plain-language summary, not legal advice.
        </p>

        <h2>Privacy and fair-use policy</h2>
        <p>
          <strong>Desktop:</strong> everything stays on your computer. Projects, datasets
          and fits are files you create, there is no telemetry, and nothing is uploaded
          anywhere.
        </p>
        <p>
          <strong>Web:</strong> to run a fit for you this service has to hold your data. It
          stores your email address, a password hash (Argon2 — never the password itself),
          the datasets you upload and the fits you run, so your workspace and history are
          there when you return. Your data is used only to provide the service: it is not
          sold, and it is not shared for advertising. There are no third-party analytics or
          advertising scripts in the app. Signing in keeps a token in your browser’s local
          storage; that is the only browser state the app relies on. Deleting a dataset, a
          fit, or your account removes the corresponding records.
        </p>
        <p>
          <strong>Fair use:</strong> do not upload data you have no right to use, other
          people’s personal data without permission, or material that is unlawful, and do
          not attempt to disrupt the service or other users’ access to it. Per-account
          quotas and rate limits apply.
        </p>
        <p>
          <strong>Changes and contact:</strong> this notice may be updated as the service
          evolves; material changes are noted in the changelog. For access, correction or
          deletion requests, or any question about this policy, write to{" "}
          <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
        </p>

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
