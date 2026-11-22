/**
 * The anonymous home page at `/`.
 *
 * One long marketing page: hero with a working sign-in card, a trust strip, the three
 * pillars, how it works, the model library, the statistics that matter, an auto-fit
 * ranking mock, use cases, the desktop/web split, the FAQ and a closing CTA + footer.
 *
 * Every illustration is a hand-authored inline SVG (no Plotly, no downloaded images) so
 * the page stays still, same-origin and cheap. Equations use the mono font; the single
 * accent colour is reserved for emphasis, leaves and buttons.
 */

import { useEffect, useState, type ReactElement, type ReactNode, type SubmitEvent } from "react";
import {
  Activity,
  Cpu,
  FileText,
  FlaskConical,
  Gauge,
  Layers,
  MonitorSmartphone,
  ShieldCheck,
  Sigma,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { ApiError } from "../api/client";
import { OAuthButtons } from "../components/OAuthButtons";
import { errorText } from "../components/States";
import { clientSideAuthErrors } from "./LoginPage";

/** The API rejects an unverified sign-in with 403 + detail.code (see openapi.json). */
function isEmailNotVerified(error: unknown): error is ApiError {
  return (
    error instanceof ApiError &&
    error.status === 403 &&
    (error.detail as { code?: string } | null)?.code === "EMAIL_NOT_VERIFIED"
  );
}

function SectionHead({
  eyebrow,
  title,
  lede,
}: {
  eyebrow: string;
  title: ReactNode;
  lede?: ReactNode;
}): ReactElement {
  return (
    <div className="hp-sec-head">
      <p className="hp-eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      {lede !== undefined && <p className="hp-lede">{lede}</p>}
    </div>
  );
}

function Sketch({ kind }: { kind: "upload" | "rank" | "inspect" | "export" }): ReactElement {
  if (kind === "upload") {
    return (
      <svg className="hp-sketch" viewBox="0 0 160 96" aria-hidden="true" focusable="false">
        <rect x="20" y="18" width="62" height="62" rx="6" fill="none" stroke="var(--border-strong)" />
        <path d="M34 34h34M34 46h34M34 58h22" stroke="var(--border-strong)" strokeWidth="2" strokeLinecap="round" />
        <path d="M124 64V30M114 40l10-10 10 10" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M104 72h40" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  }
  if (kind === "rank") {
    return (
      <svg className="hp-sketch" viewBox="0 0 160 96" aria-hidden="true" focusable="false">
        {[0, 1, 2, 3, 4].map((row) => (
          <g key={row}>
            <circle cx="24" cy={20 + row * 14} r="5" fill={row === 0 ? "var(--accent)" : "var(--accent-weak)"} />
            <text x={24} y={23 + row * 14} textAnchor="middle" fontSize="6" fill={row === 0 ? "#fff" : "var(--accent)"}>
              {row + 1}
            </text>
            <rect x="38" y={16 + row * 14} width={86 - row * 12} height="8" rx="4" fill={row === 0 ? "var(--accent)" : "var(--border)"} opacity={row === 0 ? 1 : 0.9} />
          </g>
        ))}
      </svg>
    );
  }
  if (kind === "inspect") {
    return (
      <svg className="hp-sketch" viewBox="0 0 160 96" aria-hidden="true" focusable="false">
        <path d="M24 14v66h122" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
        <path d="M28 66 C56 62 74 46 96 34 C112 26 126 22 142 20" fill="none" stroke="var(--accent)" strokeWidth="2" />
        <path d="M28 74 C56 68 74 50 96 40 C112 33 126 30 142 28" fill="none" stroke="var(--accent)" strokeWidth="1" opacity="0.55" />
        <path d="M28 58 C56 56 74 42 96 28 C112 20 126 16 142 14" fill="none" stroke="var(--accent)" strokeWidth="1" opacity="0.55" />
        <g fill="var(--accent)">
          <circle cx="44" cy="64" r="2.4" />
          <circle cx="68" cy="52" r="2.4" />
          <circle cx="92" cy="38" r="2.4" />
          <circle cx="116" cy="27" r="2.4" />
          <circle cx="138" cy="21" r="2.4" />
        </g>
      </svg>
    );
  }
  return (
    <svg className="hp-sketch" viewBox="0 0 160 96" aria-hidden="true" focusable="false">
      <rect x="28" y="14" width="62" height="68" rx="6" fill="none" stroke="var(--border-strong)" />
      <path d="M40 30h38M40 42h38M40 54h26" stroke="var(--border-strong)" strokeWidth="2" strokeLinecap="round" />
      <path d="M124 26v40M114 56l10 10 10-10" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M104 80h40" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

const PILLARS = [
  {
    icon: Sparkles,
    title: "Auto-fit ranking",
    body: "Every applicable model is fitted and ranked in one run, so the best form is never an accident.",
  },
  {
    icon: Sigma,
    title: "Honest statistics",
    body: "R², adjusted R², RMSE, AICc and confidence bands are reported — never hidden behind a single score.",
  },
  {
    icon: FileText,
    title: "Reproducible reports",
    body: "Export PDF, XLSX or CSV with the data, parameters and diagnostics that produced the result.",
  },
] as const;

const STEPS = [
  {
    kind: "upload" as const,
    title: "Upload x–y data",
    body: "Drop a CSV or XLSX file, or paste columns. Map the pair you want to fit.",
  },
  {
    kind: "rank" as const,
    title: "Every model tried and ranked",
    body: "Around 250 forms are fitted and ordered by AICc or BIC, not by whichever looks best.",
  },
  {
    kind: "inspect" as const,
    title: "Inspect statistics, bands, diagnostics",
    body: "Read the parameters, confidence bands, residuals and leverage before you trust a line.",
  },
  {
    kind: "export" as const,
    title: "Export PDF / XLSX / CSV",
    body: "The report carries the inputs, the model, the statistics and the plots together.",
  },
] as const;

const EQUATIONS = [
  "y = a·exp(−b·x)+c",
  "y = L/(1+exp(−k(x−x₀)))",
  "y = Vmax·S/(Km+S)",
  "y = a+b·x+c·x²",
];

const STAT_ROWS: ReadonlyArray<[string, string]> = [
  ["R²", "0.9964"],
  ["Adjusted R²", "0.9951"],
  ["RMSE", "0.0412"],
  ["AICc", "−182.4"],
  ["95% confidence interval", "±0.083"],
];

const RANK_ROWS = [
  { rank: 1, name: "y = a·exp(−b·x)+c", r2: "0.9964", status: "best", best: true },
  { rank: 2, name: "y = a·exp(−b·x)", r2: "0.9949", status: "close", best: false },
  { rank: 3, name: "y = L/(1+exp(−k(x−x₀)))", r2: "0.9912", status: "plausible", best: false },
  { rank: 4, name: "y = c+a·x^b", r2: "0.9871", status: "plausible", best: false },
  { rank: 5, name: "y = a+b·ln(x)", r2: "0.9738", status: "review", best: false },
] as const;

const USE_CASES = [
  { icon: FlaskConical, title: "Cube strength vs age", body: "Fit the 3-, 7- and 28-day breaks and carry a defensible prediction forward." },
  { icon: Gauge, title: "Mix / water-binder optimisation", body: "Model the response surface and see which variables actually move strength." },
  { icon: TrendingUp, title: "Decay & ingress curves", body: "Carbonation, chloride ingress and drying — compare several decay forms at once." },
  { icon: Activity, title: "Calibration curves", body: "Fit instrument response to known standards, with residuals and confidence bands." },
  { icon: Cpu, title: "Sensor drift", body: "Track a slow baseline change and separate drift from real signal." },
  { icon: ShieldCheck, title: "Lab QC", body: "Fit control charts and flag the points that fall outside the fitted envelope." },
] as const;

const FAQ = [
  {
    q: "Is an R² of 1 meaningful?",
    a: "Not on its own. A perfect R² can mean the model has enough parameters to trace noise. CurveStudio reports adjusted R², AICc and residual plots so you judge the fit, not just the headline number.",
  },
  {
    q: "Does it need an internet connection?",
    a: "The desktop app runs fully offline. The web app needs a connection for the account, uploads and exports; the same fitting engine runs in both.",
  },
  {
    q: "What happens to my data?",
    a: "On the web app, datasets stay in your account on the server. The desktop app keeps them on your machine. Your data is never sold or used to train models.",
  },
  {
    q: "Which models are available?",
    a: "Around 250 forms: growth and decay, sigmoids, binding models, polynomials, plus a custom grammar for equations of your own.",
  },
  {
    q: "What can I export?",
    a: "PDF, XLSX and CSV. Each export carries the input data, fitted parameters, statistics, confidence bands and residual diagnostics in one reproducible report.",
  },
  {
    q: "Is it free?",
    a: "CurveStudio is AGPL-3.0 open source. Create an account to use the hosted web app, or build and run the desktop app yourself.",
  },
] as const;

export function HomePage(): ReactElement {
  const { login } = useAuth();
  const navigate = useNavigate();
  const { hash } = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [unverified, setUnverified] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // The header links carry a hash; scroll to it whenever it changes.
  useEffect(() => {
    if (hash === "") return;
    const target = document.getElementById(hash.slice(1));
    target?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }, [hash]);

  const onSubmit = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    setUnverified(null);
    const invalid = clientSideAuthErrors(email, password);
    if (invalid !== null) {
      setError(invalid);
      return;
    }
    setBusy(true);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      if (isEmailNotVerified(err)) {
        setUnverified(err.message);
      } else {
        setError(
          err instanceof ApiError ? err.message : errorText(err, "sign-in failed; try again"),
        );
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <main className="hp-main">
        {/* 1 — hero with the sign-in card */}
        <section className="hp-hero">
          <div className="hp-wrap hp-hero__grid">
            <div className="hp-hero__copy">
              <p className="hp-eyebrow">CURVE FITTING &amp; DATA ANALYSIS · NIST-VALIDATED</p>
              <h1>
                From paired measurements to a defensible model —{" "}
                <span className="hp-accent">automatically.</span>
              </h1>
              <p className="hp-lede">
                Auto-fit ranks every applicable model, reports honest statistics with confidence
                bands, and turns the result into a reproducible report.
              </p>
              <div className="hp-actions">
                <Link to="/register" className="btn btn--primary btn--pill">
                  Create an account
                </Link>
                <a href="#login-card" className="btn btn--accent-outline btn--pill">
                  Sign in
                </a>
              </div>
            </div>

            <div className="hp-hero__panel" id="login-card">
              <form className="card auth-card hp-login" onSubmit={(e) => void onSubmit(e)} noValidate>
                <h2>Sign in</h2>
                <p className="muted">CurveStudio — curve fitting in your browser.</p>
                {error !== null && (
                  <p className="alert alert--error" role="alert">
                    {error}
                  </p>
                )}
                {unverified !== null && (
                  <div className="alert alert--warning">
                    <p role="alert">{unverified}</p>
                    <p className="muted">
                      Verify your email, then <Link to="/login">sign in</Link>.
                    </p>
                  </div>
                )}
                <div className="field">
                  <label htmlFor="home-email">Email</label>
                  <input
                    id="home-email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="field">
                  <label htmlFor="home-password">Password</label>
                  <input
                    id="home-password"
                    name="password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <button type="submit" className="btn btn--primary btn--block" disabled={busy}>
                  {busy ? "Signing in…" : "Sign in"}
                </button>
                <p className="muted hp-login__links">
                  <Link to="/register">Create an account</Link> ·{" "}
                  <Link to="/forgot-password">Forgot password?</Link>
                </p>
              </form>
              <OAuthButtons />
            </div>
          </div>
        </section>

        {/* 2 — trust strip */}
        <section className="hp-trust" aria-label="Capabilities at a glance">
          <div className="hp-wrap hp-pills">
            {[
              "NIST StRD validated",
              "~250 models",
              "AICc / BIC ranking",
              "Confidence & prediction bands",
              "AGPL-3.0",
              "Offline desktop app",
            ].map((pill) => (
              <span className="hp-pill" key={pill}>
                {pill}
              </span>
            ))}
          </div>
        </section>

        {/* 3 — three pillars */}
        <section className="hp-sec" id="features">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="WHY CURVESTUDIO"
              title="Fit the data. Defend the result."
              lede="One workflow from raw pairs to a model you can put your name on."
            />
            <div className="hp-grid-3">
              {PILLARS.map(({ icon: Icon, title, body }) => (
                <article className="hp-card" key={title}>
                  <span className="hp-icon-tile" aria-hidden="true">
                    <Icon />
                  </span>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* 4 — how it works */}
        <section className="hp-sec hp-sec--alt" id="how-it-works">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="HOW IT WORKS"
              title="Four steps from raw data to a reported model."
              lede="No scripts, no spreadsheet gymnastics — the same sequence every time."
            />
            <div className="hp-grid-4">
              {STEPS.map((step, index) => (
                <article className="hp-card hp-step" key={step.title}>
                  <span className="hp-step__num">{String(index + 1).padStart(2, "0")}</span>
                  <Sketch kind={step.kind} />
                  <h3>{step.title}</h3>
                  <p>{step.body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* 5 — model library */}
        <section className="hp-sec" id="models">
          <div className="hp-wrap hp-models">
            <SectionHead
              eyebrow="MODEL LIBRARY"
              title="A library that covers the shapes you actually measure."
              lede="Pick a family or let Auto-fit try them all; every fit is reported with its assumptions."
            />
            <div className="hp-eq-grid">
              {EQUATIONS.map((eq) => (
                <code className="hp-eq" key={eq}>
                  {eq}
                </code>
              ))}
              <span className="hp-eq hp-eq--more">and ~250 more</span>
            </div>
            <div className="hp-categories">
              <span>Growth &amp; decay</span>
              <span>Sigmoids</span>
              <span>Binding models</span>
              <span>Polynomials</span>
              <span>Custom grammar</span>
            </div>
          </div>
        </section>

        {/* 6 — statistics */}
        <section className="hp-sec hp-sec--alt">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="STATISTICS THAT MEAN SOMETHING"
              title="Numbers you can explain to a reviewer."
              lede="R² is agreement, not proof. The report shows how the model earns that agreement."
            />
            <div className="hp-stats-grid">
              <div className="hp-card">
                <table className="hp-table">
                  <caption className="sr-only">Fit statistics for the example model</caption>
                  <thead>
                    <tr>
                      <th scope="col">Statistic</th>
                      <th scope="col">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {STAT_ROWS.map(([label, value]) => (
                      <tr key={label}>
                        <th scope="row">{label}</th>
                        <td>
                          <code>{value}</code>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="hp-note">
                  Adjusted R² penalises extra parameters; AICc compares models while accounting
                  for small samples; the confidence interval belongs to the fitted line, not to a
                  future measurement — that is the prediction band.
                </p>
              </div>
              <div className="hp-plot-stack">
                <figure className="hp-card hp-figure">
                  <svg viewBox="0 0 320 150" aria-hidden="true" focusable="false">
                    <path d="M30 12v116h272" fill="none" stroke="var(--border-strong)" />
                    <path d="M30 70h272" stroke="var(--border)" strokeDasharray="4 4" />
                    <g fill="var(--accent)">
                      <circle cx="56" cy="82" r="3" />
                      <circle cx="86" cy="60" r="3" />
                      <circle cx="118" cy="76" r="3" />
                      <circle cx="152" cy="54" r="3" />
                      <circle cx="188" cy="68" r="3" />
                      <circle cx="222" cy="62" r="3" />
                      <circle cx="260" cy="80" r="3" />
                      <circle cx="288" cy="58" r="3" />
                    </g>
                  </svg>
                  <figcaption>Residuals — no visible pattern</figcaption>
                </figure>
                <figure className="hp-card hp-figure">
                  <svg viewBox="0 0 320 150" aria-hidden="true" focusable="false">
                    <path d="M30 12v116h272" fill="none" stroke="var(--border-strong)" />
                    <path
                      d="M38 112 C92 104 128 72 168 52 C208 32 250 24 292 22 L292 54 C250 56 208 66 168 84 C128 104 92 122 38 128 Z"
                      fill="var(--accent)"
                      opacity="0.12"
                    />
                    <path
                      d="M38 120 C92 113 128 82 168 64 C208 46 250 38 292 38"
                      fill="none"
                      stroke="var(--accent)"
                      strokeWidth="2"
                    />
                    <g fill="var(--accent)">
                      <circle cx="58" cy="112" r="2.6" />
                      <circle cx="104" cy="96" r="2.6" />
                      <circle cx="150" cy="76" r="2.6" />
                      <circle cx="198" cy="58" r="2.6" />
                      <circle cx="246" cy="46" r="2.6" />
                      <circle cx="284" cy="40" r="2.6" />
                    </g>
                  </svg>
                  <figcaption>95% confidence band around the fit</figcaption>
                </figure>
              </div>
            </div>
          </div>
        </section>

        {/* 7 — auto-fit ranking */}
        <section className="hp-sec">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="AUTO-FIT RANKING"
              title="The ranking is the result."
              lede="174 candidates ranked in one run — AICc picks the order, not a hunch."
            />
            <div className="hp-card hp-rank">
              <div className="hp-rank__head" aria-hidden="true">
                <span>Rank</span>
                <span>Model</span>
                <span>R²</span>
                <span>Status</span>
              </div>
              <ol className="hp-rank__list">
                {RANK_ROWS.map((row) => (
                  <li className={row.best ? "hp-rank__row hp-rank__row--best" : "hp-rank__row"} key={row.rank}>
                    <span className="hp-rank__pill">{row.rank}</span>
                    <code className="hp-rank__name">{row.name}</code>
                    <code className="hp-rank__r2">{row.r2}</code>
                    <span className={row.best ? "hp-badge hp-badge--accent" : "hp-badge"}>{row.status}</span>
                  </li>
                ))}
              </ol>
              <p className="hp-note">
                Rows continue past rank five. The export keeps the full ranking, so a reviewer can
                see what was rejected and why.
              </p>
            </div>
          </div>
        </section>

        {/* 8 — use cases */}
        <section className="hp-sec hp-sec--alt" id="use-cases">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="USE CASES"
              title="Built for the curves engineers actually measure."
              lede="The same engine works wherever a pair of measurements needs a defensible shape."
            />
            <div className="hp-grid-3">
              {USE_CASES.map(({ icon: Icon, title, body }) => (
                <article className="hp-card hp-usecase" key={title}>
                  <span className="hp-icon-tile" aria-hidden="true">
                    <Icon />
                  </span>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* 9 — desktop + web */}
        <section className="hp-sec">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="DESKTOP + WEB"
              title="One engine. Two ways to run it."
              lede="Start in the browser; take the same fits to a machine that never sees a network."
            />
            <div className="hp-split">
              <article className="hp-card hp-split__card">
                <span className="hp-icon-tile" aria-hidden="true">
                  <MonitorSmartphone />
                </span>
                <h3>Desktop app</h3>
                <ul className="hp-list">
                  <li>Runs completely offline</li>
                  <li>AGPL-3.0 source available</li>
                  <li>Datasets stay on your machine</li>
                </ul>
              </article>
              <article className="hp-card hp-split__card">
                <span className="hp-icon-tile" aria-hidden="true">
                  <Layers />
                </span>
                <h3>Web app</h3>
                <ul className="hp-list">
                  <li>Same fitting engine</li>
                  <li>Your data stays in your account</li>
                  <li>Nothing is sold or used for training</li>
                </ul>
              </article>
            </div>
          </div>
        </section>

        {/* 10 — FAQ */}
        <section className="hp-sec hp-sec--alt" id="faq">
          <div className="hp-wrap hp-faq">
            <SectionHead eyebrow="FAQ" title="Questions, answered." />
            {FAQ.map((item) => (
              <details className="hp-faq__item" key={item.q}>
                <summary>{item.q}</summary>
                <p>{item.a}</p>
              </details>
            ))}
          </div>
        </section>

        {/* 11 — final CTA */}
        <section className="hp-cta">
          <div className="hp-wrap hp-cta__inner">
            <h2>Start fitting in minutes.</h2>
            <p>Create an account, upload a column pair, and let Auto-fit rank the models for you.</p>
            <div className="hp-actions hp-actions--center">
              <Link to="/register" className="btn btn--white btn--pill">
                Create an account
              </Link>
              <Link to="/login" className="btn btn--on-accent btn--pill">
                Sign in
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="hp-footer">
        <div className="hp-wrap hp-footer__grid">
          <div className="hp-footer__brand">
            <img
              className="hp-footer__logo"
              src="/curvestudio-logo.png"
              alt="ConcreteInfo CurveStudio"
              width={165}
              height={44}
            />
            <p>Curve fitting and model discovery for paired measurements.</p>
          </div>
          <nav aria-label="Product">
            <h3>Product</h3>
            <ul>
              <li>
                <Link to="/#features">Features</Link>
              </li>
              <li>
                <Link to="/#how-it-works">How it works</Link>
              </li>
              <li>
                <Link to="/#models">Models</Link>
              </li>
              <li>
                <Link to="/#use-cases">Use cases</Link>
              </li>
              <li>
                <Link to="/#faq">FAQ</Link>
              </li>
            </ul>
          </nav>
          <nav aria-label="Resources">
            <h3>Resources</h3>
            <ul>
              <li>
                <Link to="/help">Help</Link>
              </li>
              <li>
                <Link to="/about">About</Link>
              </li>
              <li>
                <a href="https://www.concreteinfo.in">ConcreteInfo</a>
              </li>
            </ul>
          </nav>
        </div>
        <div className="hp-wrap hp-footer__legal">
          <p>© {new Date().getFullYear()} ConcreteInfo. CurveStudio is AGPL-3.0 licensed.</p>
          <p>Built for engineers who need a defensible fit.</p>
        </div>
      </footer>
    </>
  );
}
