/**
 * Fit-report data mapping and the browser export entry point.
 *
 * `reportModel()` is pure and is the single place that decides every displayed value
 * (including the "n/a, never 0" rules), so it can be unit-tested without react-pdf.
 * `exportFitPdf()` is the only importer of the renderer and the browser asset loaders,
 * and it imports all of them dynamically so the heavy PDF engine lives in its own
 * chunk instead of the initial app bundle.
 */

import { createElement, type ReactElement } from "react";
import type { DocumentProps } from "@react-pdf/renderer";
import { fmtCI, fmtEquation, fmtInt, fmtNumber } from "../../format";
import type { FitBundle } from "../../api/client";
import type { FitReportModel, ReportMetaRow, ReportStat } from "./fitReport";

const NA = "n/a";

export interface FitReportMeta {
  /** The dataset's display name; "Untitled dataset" when absent. */
  datasetName?: string | null;
  /** The chart captured at full scale (PNG data URL), or null. */
  chartImage?: string | null;
  /** App version shown in the report header. */
  version: string;
  /** One timestamp for the page and the file name (tests pass a fixed value). */
  generatedAt?: Date;
  /** Who prepared the report: printed as "Prepared by …" and stored as the PDF's author. */
  byline?: string | null;
}

function isDefined(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Full float precision for R² and adj. R²: the shortest decimal that round-trips to the
 * same double (the same value the desktop app now prints), never a rounded 1.000.
 */
export function preciseNumber(value: number | null | undefined): string {
  return isDefined(value) ? String(value) : NA;
}

function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

/** UTC instant → `DD/MM/YYYY HH:mm` in IST (Asia/Kolkata). */
export function istTimestamp(date: Date): string {
  const parts = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "Asia/Kolkata",
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes): string =>
    parts.find((candidate) => candidate.type === type)?.value ?? "";
  return `${part("day")}/${part("month")}/${part("year")} ${part("hour")}:${part("minute")}`;
}

/** UTC instant → `YYYYMMDD-HHmm` in IST, for the report file name. */
export function istFileStamp(date: Date): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "Asia/Kolkata",
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes): string =>
    parts.find((candidate) => candidate.type === type)?.value ?? "";
  return `${part("year")}${part("month")}${part("day")}-${part("hour")}${part("minute")}`;
}

/** `curvestudio-fit-<sanitised-model-id>-<YYYYMMDD-HHmm>.pdf`. */
export function reportFileName(modelId: string, date: Date): string {
  const slug =
    modelId
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^[-.]+|[-.]+$/g, "") || "fit";
  return `curvestudio-fit-${slug}-${istFileStamp(date)}.pdf`;
}

function datasetRows(bundle: FitBundle, datasetName: string): ReportMetaRow[] {
  return [
    { key: "Dataset", value: datasetName },
    {
      key: "Columns",
      value: `${bundle.data.x_name} / ${bundle.data.y_name}`,
    },
    { key: "Rows used (n)", value: fmtInt(bundle.data.n_total) },
  ];
}

function modelRows(bundle: FitBundle): ReportMetaRow[] {
  const result = bundle.result;
  // A seed is an identifier, not a quantity: print its digits (as EquationTab does)
  // rather than grouping it with en-IN separators.
  const seed = result.provenance?.seed;
  return [
    { key: "Model", value: result.model_id },
    { key: "Method", value: result.method },
    { key: "Loss", value: result.loss },
    { key: "Starts", value: fmtInt(result.n_starts) },
    { key: "Seed", value: seed === undefined ? NA : String(seed) },
  ];
}

function statusRows(bundle: FitBundle, converged: boolean): ReportMetaRow[] {
  const rows: ReportMetaRow[] = [
    { key: "Status", value: statusLabel(bundle.result.status) },
  ];
  if (!converged) {
    rows.push({
      key: "Termination",
      value: bundle.result.termination_reason || NA,
    });
  }
  return rows;
}

function statisticsRows(bundle: FitBundle): ReportStat[] {
  const stats = bundle.result.stats;
  return [
    { label: "n", value: fmtInt(stats?.n) },
    { label: "k", value: fmtInt(stats?.k) },
    { label: "df", value: fmtInt(stats?.df) },
    { label: "R²", value: preciseNumber(stats?.r2) },
    { label: "adj. R²", value: preciseNumber(stats?.adj_r2) },
    { label: "RMSE", value: fmtNumber(stats?.rmse) },
    { label: "MAE", value: fmtNumber(stats?.mae) },
    { label: "SSE", value: fmtNumber(stats?.sse) },
    { label: "Residual SE", value: fmtNumber(stats?.residual_se) },
  ];
}

/**
 * Every value the report prints, derived from the bundle alone. Undefined statistics
 * are "n/a"; a non-converged fit gets the banner, no fitted-equation claim, and a chart
 * caption that does not pretend a curve exists.
 */
export function reportModel(bundle: FitBundle, meta: FitReportMeta): FitReportModel {
  const result = bundle.result;
  const converged = result.status === "converged";
  const datasetName = meta.datasetName?.trim() || "Untitled dataset";
  const reason = result.termination_reason?.trim() ?? "";
  const warnings = [
    ...new Set(
      [...result.warnings, ...bundle.warnings]
        .map((warning) => warning.trim())
        .filter((warning) => warning !== ""),
    ),
  ];

  return {
    title: "Fit report",
    subtitle: `${datasetName} — ${result.model_id}`,
    byline: meta.byline?.trim() || null,
    statusLabel: statusLabel(result.status),
    converged,
    generatedAt: istTimestamp(meta.generatedAt ?? new Date()),
    version: meta.version,
    dataset: datasetRows(bundle, datasetName),
    model: modelRows(bundle),
    statusRows: statusRows(bundle, converged),
    failureBanner: converged
      ? null
      : `Fit did not converge — ${reason || "no reason reported"}`,
    equation: fmtEquation(result.equation) || NA,
    fittedEquation: result.equation_fitted ? fmtEquation(result.equation_fitted) : null,
    warnings,
    statistics: statisticsRows(bundle),
    parameters: result.parameters.map((parameter) => ({
      name: parameter.name,
      estimate: fmtNumber(parameter.value),
      stdErr: fmtNumber(parameter.std_err),
      ci: fmtCI(parameter.ci_low, parameter.ci_high),
    })),
    chartImage: meta.chartImage ? meta.chartImage : null,
    chartCaption: converged
      ? "Data and fitted curve (exported at full scale)."
      : "Data only — the fit did not converge, so no curve is claimed.",
  };
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Render the report in the browser and download it.
 *
 * Dependency check: `@react-pdf/renderer` 4.9.0 is the current release (npm, published
 * 2026-08-27); OSV.dev reported 0 advisories for it when this was pinned on 2026-11-22.
 * It is imported dynamically so the engine is a separate chunk and only loads when the
 * user exports a report.
 */
export async function exportFitPdf(
  bundle: FitBundle,
  meta: FitReportMeta,
): Promise<void> {
  const generatedAt = meta.generatedAt ?? new Date();
  const report = reportModel(bundle, { ...meta, generatedAt });

  const [{ pdf }, { registerReportFonts }, { FitReport }, assets] = await Promise.all([
    import("@react-pdf/renderer"),
    import("./fonts"),
    import("./fitReport"),
    import("./assets"),
  ]);

  registerReportFonts();

  const [logoDataUrl, watermarkDataUrl] = await Promise.all([
    assets.loadLogoDataUrl(),
    assets.loadWatermarkDataUrl(),
  ]);

  // `FitReport` only renders a `<Document>`; the cast supplies react-pdf's element type
  // for the component's own props.
  const document = createElement(FitReport, {
    report,
    logoDataUrl,
    watermarkDataUrl,
  }) as unknown as ReactElement<DocumentProps>;
  const blob = await pdf(document).toBlob();

  downloadBlob(blob, reportFileName(bundle.result.model_id, generatedAt));
}
