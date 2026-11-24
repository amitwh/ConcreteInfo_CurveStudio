/**
 * The anonymous home page at `/`.
 *
 * One long marketing page: hero with a working sign-in card, a trust strip, who it is
 * for, the research requirements table, the three pillars, how it works, the model
 * library, fits you can reproduce, the fit audit and the methods text, the statistics
 * that matter, an auto-fit ranking mock, use cases, the desktop/web split, the other
 * ConcreteInfo projects, the FAQ and a closing CTA + footer.
 *
 * Every illustration is a hand-authored inline SVG (no Plotly, no downloaded images) so
 * the page stays still, same-origin and cheap. Equations use the mono font; the single
 * accent colour is reserved for emphasis, leaves and buttons.
 */

import {
  Fragment,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
  type RefObject,
  type SubmitEvent,
} from "react";
import {
  Activity,
  Atom,
  Cog,
  FileText,
  FlaskConical,
  Gauge,
  GraduationCap,
  Layers,
  ListChecks,
  MonitorSmartphone,
  Presentation,
  ScrollText,
  Sigma,
  Sparkles,
  TestTube,
  TrendingUp,
  Waves,
} from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../api/auth";
import { api, ApiError } from "../api/client";
import { OAuthButtons } from "../components/OAuthButtons";
import { errorText } from "../components/States";
import otherProjects from "../content/other-projects.json";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const RESEND_NOTE = "if an unverified account exists, a new verification email has been sent.";

/** The API redirects failed social sign-ins back with ?oauth_error=<reason>. */
const OAUTH_ERRORS: Record<string, string> = {
  state: "your sign-in session expired or was opened in another tab — please try again",
  provider: "the provider could not complete sign-in — please try again",
};

const OAUTH_ERROR_FALLBACK = "social sign-in could not be completed; try again";

/** Client-side checks mirroring the server, so obvious mistakes never reach the API. */
function clientSideAuthErrors(email: string, password: string): string | null {
  if (!EMAIL_RE.test(email)) return "enter a valid email address";
  if (password.length < 10 || password.length > 128)
    return "the password must be between 10 and 128 characters";
  return null;
}

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
    body: "Every applicable model is fitted and ranked in one run. The order is the ranking, not a guess.",
  },
  {
    icon: Sigma,
    title: "Honest statistics",
    body: "R², adjusted R², RMSE, AICc and confidence bands are all reported, not hidden behind one score.",
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
    title: "Upload x/y data",
    body: "Drop a CSV or XLSX file, or paste columns. Map the pair you want to fit.",
  },
  {
    kind: "rank" as const,
    title: "Every model tried and ranked",
    body: "Around 200 forms are fitted and ordered by AICc or BIC, not by whichever looks best.",
  },
  {
    kind: "inspect" as const,
    title: "Inspect statistics, bands, diagnostics",
    body: "Read the parameters, confidence bands and residuals before you trust a line.",
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
  { icon: Activity, title: "Reaction kinetics", body: "Concentration against time, with rate constants and their limits." },
  { icon: FlaskConical, title: "Enzyme kinetics", body: "Rate against substrate concentration." },
  { icon: TrendingUp, title: "Growth curves", body: "Population against time." },
  { icon: Gauge, title: "Calibration and drift", body: "Instrument response against standard concentration, today and next month." },
  { icon: Waves, title: "Spectra and chromatograms", body: "Sum-of-peaks models resolve overlapping bands." },
  { icon: Sigma, title: "Dose–response", body: "Sigmoid models give a half-maximal concentration with its interval." },
  { icon: Atom, title: "Teaching-lab physics", body: "Decay, cooling and damped oscillations." },
] as const;

const PERSONAS = [
  { icon: FlaskConical, title: "Research scientists", body: "Test competing models on equal terms and report the one the data supports." },
  { icon: TestTube, title: "Laboratory analysts", body: "Calibration curves, peak fits and method comparisons, with the uncertainty attached." },
  { icon: GraduationCap, title: "Postgraduate students", body: "Learn what R², AICc and a confidence band actually say about your data." },
  { icon: Cog, title: "Engineers in research and test", body: "Fit sensor, materials and reliability data on your own machine." },
  { icon: Presentation, title: "Lecturers", body: "Show a class why a high R² is not the end of the argument." },
] as const;

/** The research requirements table: the requirement, then what the software does about it. */
const REQUIREMENTS: ReadonlyArray<readonly [string, string]> = [
  ["Choose a model on evidence", "Fits every applicable model and ranks them by AICc, BIC, adjusted R² and cross-validated error. Models that draw the same curve share a rank."],
  ["Report uncertainty honestly", "Parameter standard errors and 95 % confidence intervals; confidence and prediction bands; bootstrap; Monte Carlo propagation."],
  ["Show that the fit hides nothing", "Residual diagnostics: Shapiro–Wilk, Anderson–Darling, Durbin–Watson, Ljung–Box, Breusch–Pagan, leverage and influence."],
  ["Trust the arithmetic", "Checked against NIST StRD certified datasets. A fit that does not converge is reported as failed, never replaced by a plausible answer."],
  ["Use your own equation", "Custom models with named parameters, start values and bounds."],
  ["Account for measurement error", "Weighted least squares."],
  ["Reproduce it next year", "Desktop project files record the data, every cleaning step, the fit settings, random seeds and software versions."],
  ["Publish and archive", "Reports in CSV, JSON, XLSX and PDF; plots as PNG, SVG and PDF."],
] as const;

const FAQ = [
  {
    q: "Is an R² of 1 meaningful?",
    a: "Not on its own. A perfect R² can mean the model has enough parameters to trace noise. CurveStudio reports adjusted R², AICc and residual plots so you judge the fit, not just the headline number.",
  },
  {
    q: "Can I use my own equation?",
    a: "Yes. Custom models take your own equation with named parameters, start values and bounds, and are fitted and ranked exactly like the built-in forms.",
  },
  {
    q: "How do I report a fit in a paper?",
    a: "Export the numbers as CSV, JSON, XLSX or PDF and the plots as PNG, SVG or PDF, then report the model, the parameters with their confidence intervals and the ranking statistics. A desktop project file records the full analysis: the data, the cleaning steps, the fit settings and the software versions, so the run can be repeated as written.",
  },
  {
    q: "Does it need an internet connection?",
    a: "The desktop app runs fully offline. The web app needs a connection for the account, uploads and exports; the same fitting engine runs in both.",
  },
  {
    q: "What happens to my data?",
    a: "On the web app, datasets stay in your account on the server. The desktop app keeps them on your machine. It is not sold, and it is not shared for advertising.",
  },
  {
    q: "Which models are available?",
    a: "Around 200 forms: growth and decay, sigmoids, binding models, polynomials, plus a custom grammar for equations of your own.",
  },
  {
    q: "What can I export?",
    a: "Reports in CSV, JSON, XLSX and PDF; plots as PNG, SVG and PDF. Each export carries the input data, the fitted parameters, the statistics and the residual diagnostics in one report.",
  },
  {
    q: "Is it free?",
    a: "CurveStudio is AGPL-3.0 open source. Create an account to use the hosted web app, or build and run the desktop app yourself.",
  },
] as const;

/* ---- Statistics: the fit audit and the methods text -------------------------------- */

/** The section's three sentences. Pinned verbatim by `HomePage.test.tsx` and backed, claim
 *  by claim, in `homepage-claims.md`. The third says what a check found and where its record
 *  is: a capability not found is not proof that nobody has it, so nothing here claims one. */
const FIT_AUDIT_SENTENCE =
  "After every fit, a fit audit checks the evidence for you: each check gets a verdict, the numbers behind it and what to do next.";
const METHODS_TEXT_SENTENCE =
  "A methods text writes the fit up for you: the model, the estimator, the estimates with their standard errors and the citations, as plain text, Markdown or LaTeX.";
const PRIOR_ART_SENTENCE =
  "We looked for these two in the public documentation of CurveExpert and CurveExpert Pro, Origin, GraphPad Prism, LabPlot, SciDAVis, QtiPlot, jamovi and JASP in November 2026, and did not find either offered in its own right; those tools offer residual plots and tests, rankings, pairwise comparisons and equation export.";

/** The record of that check: a file of the repository the footer's source link names. */
const PRIOR_ART_URL =
  "https://github.com/amitwh/ConcreteInfo_CurveStudio/blob/main/docs/prior-art.md";

/* ---- "Fits you can reproduce" gallery ------------------------------------------------ */

interface FitModel {
  id: string;
  name: string;
  equation: string;
  r2: string;
  rmse: string;
  note: string;
  f: (x: number) => number;
  domain: readonly [number, number];
}

/** Six representative shapes, each with a fixed illustrative fit. Everything is
 *  deterministic, so the chart is identical on every visit without a fitting engine. */
const FIT_MODELS: readonly FitModel[] = [
  {
    id: "exp-decay",
    name: "Exponential decay",
    equation: "y = a·exp(−b·x)+c",
    r2: "0.9964",
    rmse: "1.84",
    note: "radioactive decay, first-order reaction kinetics and diffusion profiles with depth or time.",
    f: (x) => 98 * Math.exp(-0.42 * x) + 12,
    domain: [0, 10],
  },
  {
    id: "logistic",
    name: "Logistic growth",
    equation: "y = L/(1+exp(−k(x−x₀)))",
    r2: "0.9931",
    rmse: "2.37",
    note: "population growth, substrate uptake and any response that levels off.",
    f: (x) => 96 / (1 + Math.exp(-0.9 * (x - 4))),
    domain: [0, 10],
  },
  {
    id: "langmuir",
    name: "Langmuir binding",
    equation: "y = Bmax·x/(Kd+x)",
    r2: "0.9917",
    rmse: "2.05",
    note: "adsorption and binding isotherms, from inhibitor uptake to sensor saturation.",
    f: (x) => (120 * x) / (3.2 + x),
    domain: [0, 10],
  },
  {
    id: "gaussian",
    name: "Gaussian peak",
    equation: "y = A·exp(−(x−μ)²/(2σ²))",
    r2: "0.9894",
    rmse: "1.62",
    note: "peaks and drift: instrument response, particle-size modes and spectral lines.",
    f: (x) => 95 * Math.exp(-((x - 5.2) ** 2) / (2 * 1.35 ** 2)),
    domain: [0, 10],
  },
  {
    id: "power",
    name: "Power law",
    equation: "y = a·x^b",
    r2: "0.9862",
    rmse: "3.41",
    note: "scaling laws: allometric relations and fatigue or fracture growth exponents.",
    f: (x) => 14 * x ** 0.72,
    domain: [0, 10],
  },
  {
    id: "quadratic",
    name: "Quadratic",
    equation: "y = a+b·x+c·x²",
    r2: "0.9828",
    rmse: "3.07",
    note: "simple calibration and gentle curvature where a straight line is not enough.",
    f: (x) => 8 - 2.4 * x + 0.62 * x * x,
    domain: [0, 10],
  },
];

const FIT_CHART = { width: 560, height: 320, top: 18, right: 18, bottom: 38, left: 48 } as const;

/** Deterministic residuals (fractions of the data range) for the scattered points. */
const FIT_NOISE = [
  0.024, -0.018, 0.031, -0.026, 0.012, 0.038, -0.014, -0.033, 0.02, -0.007, 0.027, -0.021,
] as const;

interface FitChart {
  curve: string;
  band: string;
  points: Array<{ cx: number; cy: number }>;
  xTicks: Array<{ x: number; label: string }>;
  yTicks: Array<{ y: number; label: string }>;
  axisX: number;
  axisY: number;
}

function buildFitChart(model: FitModel): FitChart {
  const [x0, x1] = model.domain;
  const samples = 72;
  const xs = Array.from({ length: samples + 1 }, (_, i) => x0 + ((x1 - x0) * i) / samples);
  const ys = xs.map((x) => model.f(x));
  const span = Math.max(...ys) - Math.min(...ys) || 1;

  const pointXs = FIT_NOISE.map((_, i) => x0 + ((x1 - x0) * (i + 0.5)) / FIT_NOISE.length);
  const pointYs = pointXs.map((x, i) => model.f(x) + FIT_NOISE[i]! * span);
  const bandHalf = (x: number): number =>
    span * (0.028 + (0.02 * Math.abs(x - (x0 + x1) / 2)) / ((x1 - x0) / 2));
  const lower = xs.map((x, i) => ys[i]! - bandHalf(x));
  const upper = xs.map((x, i) => ys[i]! + bandHalf(x));

  const yMin = Math.min(...ys, ...pointYs, ...lower);
  const yMax = Math.max(...ys, ...pointYs, ...upper);
  const pad = (yMax - yMin) * 0.08 || 1;
  const lo = yMin - pad;
  const hi = yMax + pad;

  const plotW = FIT_CHART.width - FIT_CHART.left - FIT_CHART.right;
  const plotH = FIT_CHART.height - FIT_CHART.top - FIT_CHART.bottom;
  const px = (x: number): number => FIT_CHART.left + ((x - x0) / (x1 - x0)) * plotW;
  const py = (y: number): number =>
    FIT_CHART.height - FIT_CHART.bottom - ((y - lo) / (hi - lo)) * plotH;
  const command = (x: number, y: number, first: boolean): string =>
    `${first ? "M" : "L"}${px(x).toFixed(1)} ${py(y).toFixed(1)}`;

  const curve = xs.map((x, i) => command(x, ys[i]!, i === 0)).join(" ");
  const band = [
    ...xs.map((x, i) => command(x, upper[i]!, i === 0)),
    ...xs
      .map((x, i) => ({ x, y: lower[i]! }))
      .reverse()
      .map((point) => command(point.x, point.y, false)),
    "Z",
  ].join(" ");

  const xTicks = [0, 1, 2, 3, 4, 5].map((step) => {
    const x = x0 + ((x1 - x0) * step) / 5;
    return { x: px(x), label: String(Math.round(x)) };
  });
  const yTicks = [0, 1, 2, 3, 4].map((step) => {
    const y = lo + ((hi - lo) * step) / 4;
    return { y: py(y), label: String(Math.round(y)) };
  });

  return {
    curve,
    band,
    points: pointXs.map((x, i) => ({ cx: px(x), cy: py(pointYs[i]!) })),
    xTicks,
    yTicks,
    axisX: FIT_CHART.height - FIT_CHART.bottom,
    axisY: FIT_CHART.left,
  };
}

const AICC_ROWS = [
  { name: "Exp. decay", delta: 0, aicc: "−182.4" },
  { name: "Logistic", delta: 10.6, aicc: "−171.8" },
  { name: "Langmuir", delta: 14.2, aicc: "−168.2" },
  { name: "Gaussian", delta: 27.7, aicc: "−154.7" },
  { name: "Power law", delta: 41.1, aicc: "−141.3" },
] as const;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** `inView` starts true when there is no IntersectionObserver (tests, very old
 *  browsers), so nothing is ever left hidden without JS. */
function useInView<T extends HTMLElement>(): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(typeof IntersectionObserver === "undefined");

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const element = ref.current;
    if (element === null) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, inView];
}

/** Counts from zero to `target` once `active`; reduced motion (and environments without
 *  IntersectionObserver or rAF) shows the final value immediately. */
function useCountUp(target: number, active: boolean): number {
  const animate =
    typeof IntersectionObserver !== "undefined" &&
    typeof requestAnimationFrame === "function" &&
    !prefersReducedMotion();
  const [value, setValue] = useState(animate ? 0 : target);

  useEffect(() => {
    if (!animate) {
      setValue(target);
      return;
    }
    if (!active) return;
    const start = performance.now();
    let frame = 0;
    const step = (now: number): void => {
      const progress = Math.min(1, (now - start) / 900);
      const eased = 1 - (1 - progress) ** 3;
      setValue(Math.round(target * eased));
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, active, animate]);

  return value;
}

function CountUp({
  target,
  prefix = "",
  suffix = "",
}: {
  target: number;
  prefix?: string;
  suffix?: string;
}): ReactElement {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const value = useCountUp(target, inView);
  return (
    <span ref={ref}>
      {prefix}
      {value.toLocaleString("en-IN")}
      {suffix}
    </span>
  );
}

function FitGallery(): ReactElement {
  const fallback = FIT_MODELS[0]!;
  const [selected, setSelected] = useState(fallback.id);
  const model = FIT_MODELS.find((item) => item.id === selected) ?? fallback;
  const chart = useMemo(() => buildFitChart(model), [model]);

  return (
    <>
      <div className="hp-chips" role="group" aria-label="Model families">
        {FIT_MODELS.map((item) => {
          const active = item.id === model.id;
          return (
            <button
              key={item.id}
              type="button"
              className={active ? "hp-chip hp-chip--active" : "hp-chip"}
              aria-pressed={active}
              onClick={() => setSelected(item.id)}
            >
              {item.name}
            </button>
          );
        })}
      </div>

      <div className="hp-fits">
        <figure className="hp-card hp-fits__figure">
          <svg
            viewBox={`0 0 ${FIT_CHART.width} ${FIT_CHART.height}`}
            aria-hidden="true"
            focusable="false"
          >
            <g stroke="var(--border)" strokeWidth="1">
              {chart.yTicks.map((tick) => (
                <line
                  key={`gy-${tick.label}`}
                  x1={FIT_CHART.left}
                  x2={FIT_CHART.width - FIT_CHART.right}
                  y1={tick.y}
                  y2={tick.y}
                />
              ))}
              {chart.xTicks.map((tick) => (
                <line
                  key={`gx-${tick.label}`}
                  x1={tick.x}
                  x2={tick.x}
                  y1={FIT_CHART.top}
                  y2={chart.axisX}
                />
              ))}
            </g>
            <g fill="var(--muted)" fontSize="11">
              {chart.xTicks.map((tick) => (
                <text
                  key={`tx-${tick.label}`}
                  x={tick.x}
                  y={chart.axisX + 18}
                  textAnchor="middle"
                >
                  {tick.label}
                </text>
              ))}
              {chart.yTicks.map((tick) => (
                <text
                  key={`ty-${tick.label}`}
                  x={FIT_CHART.left - 8}
                  y={tick.y + 4}
                  textAnchor="end"
                >
                  {tick.label}
                </text>
              ))}
            </g>
            <line
              x1={FIT_CHART.left}
              x2={FIT_CHART.width - FIT_CHART.right}
              y1={chart.axisX}
              y2={chart.axisX}
              stroke="var(--border-strong)"
            />
            <line
              x1={chart.axisY}
              x2={chart.axisY}
              y1={FIT_CHART.top}
              y2={chart.axisX}
              stroke="var(--border-strong)"
            />
            <path
              key={`band-${model.id}`}
              className="hp-fits__band"
              d={chart.band}
              fill="var(--accent)"
              fillOpacity="0.1"
            />
            <g key={`points-${model.id}`} className="hp-fits__points" fill="var(--accent)">
              {chart.points.map((point) => (
                <circle
                  key={`${point.cx}-${point.cy}`}
                  cx={point.cx}
                  cy={point.cy}
                  r="3"
                  opacity="0.85"
                />
              ))}
            </g>
            <path
              key={`curve-${model.id}`}
              className="hp-fits__curve"
              d={chart.curve}
              pathLength={1}
              fill="none"
              stroke="var(--accent)"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </svg>
        </figure>

        <div className="hp-card hp-fits__detail">
          <h3 className="hp-fits__name">{model.name}</h3>
          <code className="hp-fits__equation">{model.equation}</code>
          <dl className="hp-fits__stats">
            <div>
              <dt>R²</dt>
              <dd>{model.r2}</dd>
            </div>
            <div>
              <dt>RMSE</dt>
              <dd>{model.rmse}</dd>
            </div>
          </dl>
          <p>
            <strong>Where you meet it:</strong> {model.note}
          </p>
        </div>
      </div>
    </>
  );
}

function AiccChart(): ReactElement {
  const labelW = 86;
  const maxBar = 190;
  const maxDelta = AICC_ROWS[AICC_ROWS.length - 1]?.delta ?? 1;
  return (
    <figure className="hp-card hp-figure hp-aicc">
      <svg viewBox="0 0 340 170" aria-hidden="true" focusable="false">
        {AICC_ROWS.map((row, index) => {
          const y = 14 + index * 32;
          const width = Math.max(2, (row.delta / maxDelta) * maxBar);
          return (
            <g key={row.name}>
              <text x="0" y={y + 12} fontSize="10.5" fill="var(--muted)">
                {row.name}
              </text>
              <rect
                x={labelW}
                y={y}
                width={maxBar}
                height="16"
                rx="4"
                fill="var(--border)"
                opacity="0.35"
              />
              <rect
                className="hp-aicc__bar"
                x={labelW}
                y={y}
                width={width}
                height="16"
                rx="4"
                fill="var(--accent)"
                opacity={row.delta === 0 ? 1 : 0.45}
              />
              <text
                x={labelW + maxBar + 6}
                y={y + 12}
                fontSize="10"
                fill="var(--muted)"
                fontFamily="var(--font-mono)"
              >
                {row.aicc}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>ΔAICc against the best model. Lower is better.</figcaption>
    </figure>
  );
}

export function HomePage(): ReactElement {
  const { login } = useAuth();
  const navigate = useNavigate();
  const { hash, search } = useLocation();
  const oauthError = new URLSearchParams(search).get("oauth_error");
  const oauthMessage =
    oauthError === null ? null : (OAUTH_ERRORS[oauthError] ?? OAUTH_ERROR_FALLBACK);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [unverified, setUnverified] = useState<string | null>(null);
  const [resendNote, setResendNote] = useState<string | null>(null);
  const [resendError, setResendError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const mainRef = useRef<HTMLElement>(null);

  // Reveal each top-level section once as it scrolls into view. The `hp-reveal` class is
  // only applied when IntersectionObserver exists, so tests and no-observer browsers
  // never start with hidden content. Respects prefers-reduced-motion in CSS.
  useLayoutEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const sections = Array.from(
      mainRef.current?.querySelectorAll<HTMLElement>(":scope > section") ?? [],
    );
    for (const section of sections) section.classList.add("hp-reveal");
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.08, rootMargin: "0px 0px -48px 0px" },
    );
    for (const section of sections) observer.observe(section);
    return () => observer.disconnect();
  }, []);

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
    setResendNote(null);
    setResendError(null);
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

  const onResend = async (): Promise<void> => {
    if (!EMAIL_RE.test(email)) return;
    setResendNote(null);
    setResendError(null);
    setBusy(true);
    try {
      await api.resendVerification(email);
      setResendNote(RESEND_NOTE);
    } catch (err) {
      setResendError(errorText(err, "could not resend the verification email; try again"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <main className="hp-main" ref={mainRef}>
        {/* 1 — hero with the sign-in card */}
        <section className="hp-hero">
          <div className="hp-wrap hp-hero__grid">
            <div className="hp-hero__copy">
              <p className="hp-eyebrow">CURVE FITTING AND STATISTICS · VALIDATED ON NIST STRD DATASETS</p>
              <h1>
                From paired measurements to a defensible model.{" "}
                <span className="hp-accent">automatically.</span>
              </h1>
              <p className="hp-lede">
                CurveStudio fits every applicable model to your data, ranks them on the evidence,
                and gives you the statistics to report: parameter uncertainty and residual
                diagnostics you can defend.
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
                <p className="muted">CurveStudio: curve fitting in your browser.</p>
                {oauthMessage !== null && (
                  <p className="alert alert--error" role="alert">
                    {oauthMessage}
                  </p>
                )}
                {error !== null && (
                  <p className="alert alert--error" role="alert">
                    {error}
                  </p>
                )}
                {unverified !== null && (
                  <div className="alert alert--warning">
                    <p role="alert">{unverified}</p>
                    <button
                      type="button"
                      className="btn btn--small"
                      disabled={busy || !EMAIL_RE.test(email)}
                      onClick={() => void onResend()}
                    >
                      Resend verification email
                    </button>
                    {resendNote !== null && <p role="status">{resendNote}</p>}
                    {resendError !== null && <p role="alert">{resendError}</p>}
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
          <div className="hp-wrap">
            <div className="hp-pills">
              {[
                "NIST StRD reference sets",
                "~200 models",
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
            <div className="hp-trust__stats">
              <div className="hp-trust__stat">
                <strong>
                  <CountUp target={174} />
                </strong>
                <span>candidates ranked in one run</span>
              </div>
              <div className="hp-trust__stat">
                <strong>
                  <CountUp target={200} prefix="~" />
                </strong>
                <span>models in the library</span>
              </div>
              <div className="hp-trust__stat">
                <strong>
                  <CountUp target={6} />
                </strong>
                <span>export formats</span>
              </div>
            </div>
          </div>
        </section>

        {/* 3 — who it is for */}
        <section className="hp-sec" id="who">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="WHO IT IS FOR"
              title="For the analysis behind a paper, a thesis or a lab report."
              lede="CurveStudio is a general fitting workbench. These are the people it is for."
            />
            <div className="hp-grid-3">
              {PERSONAS.map(({ icon: Icon, title, body }) => (
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

        {/* 4 — research requirements */}
        <section className="hp-sec hp-sec--alt" id="requirements">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="RESEARCH REQUIREMENTS"
              title="What an analysis has to survive, and how CurveStudio answers."
              lede="Each requirement, and what the software does about it."
            />
            <div className="hp-split">
              <p>
                <strong>Research requirement</strong>
              </p>
              <p>
                <strong>What CurveStudio does</strong>
              </p>
              {REQUIREMENTS.map(([requirement, answer]) => (
                <Fragment key={requirement}>
                  <p>
                    <strong>{requirement}</strong>
                  </p>
                  <p>{answer}</p>
                </Fragment>
              ))}
            </div>
          </div>
        </section>

        {/* 5 — three pillars */}
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

        {/* 6 — how it works */}
        <section className="hp-sec hp-sec--alt" id="how-it-works">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="HOW IT WORKS"
              title="Four steps from raw data to a reported model."
              lede="No scripts and no spreadsheet workarounds; the same sequence every time."
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

        {/* 7 — model library */}
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
              <span className="hp-eq hp-eq--more">and ~200 more</span>
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

        {/* 7b — fits you can reproduce */}
        <section className="hp-sec" id="fits">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="MODEL GALLERY"
              title="Fits you can reproduce."
              lede="Six shapes cover most measurements. Pick one to see the curve, the confidence band and the numbers CurveStudio reports. The values below are illustrative runs, not live data."
            />
            <FitGallery />
          </div>
        </section>

        {/* 7c — the fit audit and the methods text */}
        <section className="hp-sec" id="statistics">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="FIT AUDIT AND METHODS TEXT"
              title="Checked after the fit. Written up for the paper."
            />
            <div className="hp-split">
              <article className="hp-card">
                <span className="hp-icon-tile" aria-hidden="true">
                  <ListChecks />
                </span>
                <h3>Fit audit</h3>
                <p>{FIT_AUDIT_SENTENCE}</p>
              </article>
              <article className="hp-card">
                <span className="hp-icon-tile" aria-hidden="true">
                  <ScrollText />
                </span>
                <h3>Methods text</h3>
                <p>{METHODS_TEXT_SENTENCE}</p>
              </article>
            </div>
            <p className="hp-note">
              <span>{PRIOR_ART_SENTENCE}</span>{" "}
              <a href={PRIOR_ART_URL} target="_blank" rel="noopener noreferrer">
                How we checked
              </a>
              .
            </p>
          </div>
        </section>

        {/* 8 — statistics */}
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
                  future measurement; that is the prediction band.
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
                  <figcaption>Residuals: no visible pattern</figcaption>
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
                <AiccChart />
              </div>
            </div>
          </div>
        </section>

        {/* 9 — auto-fit ranking */}
        <section className="hp-sec">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="AUTO-FIT RANKING"
              title="The ranking is the result."
              lede="174 candidates ranked in one run. AICc sets the order, not a hunch."
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

        {/* 10 — use cases */}
        <section className="hp-sec hp-sec--alt" id="use-cases">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="USE CASES"
              title="Built for measured data, in every discipline."
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

        {/* 11 — desktop + web */}
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
                  <li>Nothing is sold or shared for advertising</li>
                </ul>
              </article>
            </div>
          </div>
        </section>

        {/* 12 — more from ConcreteInfo */}
        <section className="hp-sec" id="projects">
          <div className="hp-wrap">
            <SectionHead
              eyebrow="MORE FROM CONCRETEINFO"
              title="Other tools we build and maintain."
              lede="Open source you can self-host, and the commercial platforms we run."
            />
            <div className="hp-split">
              <article className="hp-card hp-split__card">
                <h3>Open source</h3>
                <ul className="hp-list">
                  {otherProjects.entries
                    .filter((entry) => entry.kind === "open-source")
                    .map((entry) => (
                      <li key={entry.url}>
                        <a href={entry.url} target="_blank" rel="noopener noreferrer">
                          {entry.name}
                        </a>
                        {". "}
                        <span>{entry.summary}</span>
                      </li>
                    ))}
                </ul>
              </article>
              <article className="hp-card hp-split__card">
                <h3>Commercial software</h3>
                <ul className="hp-list">
                  {otherProjects.entries
                    .filter((entry) => entry.kind === "commercial")
                    .map((entry) => (
                      <li key={entry.url}>
                        <a href={entry.url} target="_blank" rel="noopener noreferrer">
                          {entry.name}
                        </a>
                        {". "}
                        <span>{entry.summary}</span>
                      </li>
                    ))}
                </ul>
              </article>
            </div>
            <p className="hp-note">
              The full list of ConcreteInfo&apos;s tools lives at{" "}
              <a
                href="https://www.concreteinfo.in"
                target="_blank"
                rel="noopener noreferrer"
              >
                www.concreteinfo.in
              </a>
              .
            </p>
          </div>
        </section>

        {/* 13 — FAQ */}
        <section className="hp-sec hp-sec--alt" id="faq">
          <div className="hp-wrap hp-faq">
            <SectionHead eyebrow="FAQ" title="Questions we get asked." />
            {FAQ.map((item) => (
              <details className="hp-faq__item" key={item.q}>
                <summary>{item.q}</summary>
                <p>{item.a}</p>
              </details>
            ))}
          </div>
        </section>

        {/* 14 — final CTA */}
        <section className="hp-cta">
          <div className="hp-wrap hp-cta__inner">
            <h2>Start fitting in minutes.</h2>
            <p>Create an account, upload a column pair, and let Auto-fit rank the models for you.</p>
            <div className="hp-actions hp-actions--center">
              <Link to="/register" className="btn btn--white btn--pill">
                Create an account
              </Link>
              <Link to="/#login-card" className="btn btn--on-accent btn--pill">
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
            <a
              className="hp-footer__ci"
              href="https://www.concreteinfo.in"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src="/concreteinfo-logo.png" alt="ConcreteInfo" width={100} height={28} />
            </a>
            <p className="hp-footer__ci-note">A ConcreteInfo product.</p>
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
                <a href="mailto:amit@concreteinfo.in">Contact</a>
              </li>
              <li>
                <Link to="/about">Privacy</Link>
              </li>
              <li>
                <a href="https://www.concreteinfo.in">ConcreteInfo</a>
              </li>
            </ul>
          </nav>
          <nav aria-label="Open source">
            <h3>Open source</h3>
            <ul>
              <li>
                <a
                  href="https://github.com/amitwh/ConcreteInfo_CurveStudio"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  CurveStudio source (AGPL-3.0)
                </a>
              </li>
              {otherProjects.entries
                .filter((entry) => entry.kind === "open-source")
                .map((entry) => (
                  <li key={entry.url}>
                    <a href={entry.url} target="_blank" rel="noopener noreferrer">
                      {entry.name}
                    </a>
                  </li>
                ))}
            </ul>
          </nav>
        </div>
        <div className="hp-wrap hp-footer__legal">
          <p>© {new Date().getFullYear()} ConcreteInfo. CurveStudio is AGPL-3.0 licensed.</p>
          <p>Built for the analysis behind a paper, a thesis or a lab report.</p>
        </div>
      </footer>
    </>
  );
}
