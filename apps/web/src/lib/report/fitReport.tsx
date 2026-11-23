/**
 * The fit report as a flow-laid-out react-pdf document. Pure in the rendering sense:
 * no fetching, font registration or browser APIs, so the model `reportModel()` in
 * `export.ts` decides every displayed value and this module only draws it.
 *
 * Honesty rules carried over from the results UI: a fit that did not converge gets a
 * red banner and never a fitted-curve claim, and every undefined statistic prints as
 * "n/a" rather than 0.
 */

import { StyleSheet, Document, Page, View, Text, Image } from "@react-pdf/renderer";
import type { ReactElement } from "react";
import { REPORT_FONT_FAMILIES } from "./fonts";

export interface ReportMetaRow {
  key: string;
  value: string;
}

export interface ReportStat {
  label: string;
  value: string;
}

export interface ReportParameter {
  name: string;
  estimate: string;
  stdErr: string;
  ci: string;
}

export interface FitReportModel {
  title: string;
  subtitle: string;
  /** Who prepared the report: printed as "Prepared by …" and stored as the author. */
  byline: string | null;
  statusLabel: string;
  converged: boolean;
  generatedAt: string;
  version: string;
  dataset: ReportMetaRow[];
  model: ReportMetaRow[];
  statusRows: ReportMetaRow[];
  failureBanner: string | null;
  equation: string;
  fittedEquation: string | null;
  warnings: string[];
  statistics: ReportStat[];
  parameters: ReportParameter[];
  chartImage: string | null;
  chartCaption: string;
}

export interface FitReportProps {
  report: FitReportModel;
  logoDataUrl: string | null;
  watermarkDataUrl: string | null;
}

const A4_WIDTH = 595.28;
const A4_HEIGHT = 841.89;
const PAGE_MARGIN = 36;
const CONTENT_WIDTH = A4_WIDTH - PAGE_MARGIN * 2;
/** Captured at 1000 × 620, i.e. the chart PNG's own aspect at full content width. */
const CHART_WIDTH = CONTENT_WIDTH;
const CHART_HEIGHT = 324;
/** Parameters per table block; a long list repeats the header instead of orphaning it. */
const PARAMETER_CHUNK_ROWS = 12;

const BRAND = "#e5461f";
const INK = "#1a1714";
const MUTED = "#6b625b";
const FAINT = "#8a8078";
const RULE = "#d9d0c9";
const ZEBRA = "#fafbfc";
const BOX = "#f6f7f8";
const RED = "#b91c1c";
const RED_WEAK = "#fdecec";
const GREEN = "#1a7a56";
const GREEN_WEAK = "#e8f5ee";
const AMBER = "#a15c07";

const styles = StyleSheet.create({
  page: {
    paddingTop: 64,
    paddingHorizontal: PAGE_MARGIN,
    paddingBottom: 60,
    fontFamily: REPORT_FONT_FAMILIES.body,
    fontSize: 9,
    color: INK,
    lineHeight: 1.35,
  },
  watermark: {
    position: "absolute",
    top: 0,
    left: 0,
    width: A4_WIDTH,
    height: A4_HEIGHT,
    opacity: 0.07,
  },
  headerLogo: {
    position: "absolute",
    top: 18,
    left: PAGE_MARGIN,
    height: 26,
    width: 98,
  },
  headerWordmark: {
    position: "absolute",
    top: 24,
    left: PAGE_MARGIN,
    fontSize: 11,
    fontWeight: 700,
    color: BRAND,
    letterSpacing: 1.2,
  },
  headerTitle: {
    position: "absolute",
    top: 18,
    right: PAGE_MARGIN,
    fontSize: 6.5,
    color: FAINT,
    letterSpacing: 1,
    textAlign: "right",
  },
  headerSubtitle: {
    position: "absolute",
    top: 27,
    right: PAGE_MARGIN,
    fontSize: 8,
    color: MUTED,
    textAlign: "right",
  },
  headerRule: {
    position: "absolute",
    top: 52,
    left: PAGE_MARGIN,
    right: PAGE_MARGIN,
    height: 0.7,
    backgroundColor: RULE,
  },
  footerRule: {
    position: "absolute",
    top: A4_HEIGHT - PAGE_MARGIN,
    left: PAGE_MARGIN,
    right: PAGE_MARGIN,
    height: 0.7,
    backgroundColor: RULE,
  },
  footerPage: {
    position: "absolute",
    top: A4_HEIGHT - PAGE_MARGIN + 3,
    right: PAGE_MARGIN,
    fontFamily: REPORT_FONT_FAMILIES.numeric,
    fontSize: 6.5,
    color: MUTED,
  },
  footerDisclaimer: {
    position: "absolute",
    top: A4_HEIGHT - PAGE_MARGIN + 13,
    left: PAGE_MARGIN,
    right: PAGE_MARGIN,
    fontSize: 6.5,
    color: FAINT,
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    color: INK,
    lineHeight: 1.2,
  },
  subtitle: {
    fontSize: 9.5,
    color: MUTED,
    marginTop: 5,
  },
  byline: {
    fontSize: 9.5,
    color: MUTED,
    marginTop: 4,
  },
  generated: {
    fontSize: 7.5,
    color: FAINT,
    marginTop: 3,
  },
  pill: {
    alignSelf: "flex-start",
    borderRadius: 2,
    paddingVertical: 2,
    paddingHorizontal: 6,
    marginTop: 8,
  },
  pillOk: {
    backgroundColor: GREEN_WEAK,
  },
  pillBad: {
    backgroundColor: RED_WEAK,
  },
  pillText: {
    fontSize: 7.5,
    fontWeight: 700,
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },
  pillTextOk: {
    color: GREEN,
  },
  pillTextBad: {
    color: RED,
  },
  metaGrid: {
    flexDirection: "row",
    marginTop: 10,
  },
  metaColumn: {
    width: "50%",
    paddingRight: 12,
  },
  metaRow: {
    paddingVertical: 3,
    borderBottomWidth: 0.5,
    borderBottomColor: RULE,
  },
  metaKey: {
    fontSize: 6.5,
    color: FAINT,
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  metaValue: {
    fontSize: 8.5,
    fontWeight: 600,
    color: INK,
    marginTop: 1,
  },
  section: {
    marginTop: 14,
  },
  sectionHeading: {
    fontSize: 11.5,
    fontWeight: 700,
    color: INK,
    borderTopWidth: 0.9,
    borderTopColor: BRAND,
    paddingTop: 5,
    marginBottom: 6,
    letterSpacing: 0.4,
  },
  failureBanner: {
    marginTop: 12,
    borderWidth: 1,
    borderColor: RED,
    borderRadius: 2,
    backgroundColor: RED_WEAK,
    paddingVertical: 6,
    paddingHorizontal: 8,
  },
  failureText: {
    fontSize: 9,
    fontWeight: 700,
    color: RED,
  },
  equationBox: {
    backgroundColor: BOX,
    borderWidth: 0.7,
    borderColor: RULE,
    borderRadius: 2,
    paddingVertical: 7,
    paddingHorizontal: 9,
  },
  equationKey: {
    fontSize: 6.5,
    color: FAINT,
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginBottom: 2,
  },
  equationLine: {
    fontFamily: REPORT_FONT_FAMILIES.numeric,
    fontSize: 10,
    color: INK,
  },
  equationLineSpaced: {
    marginTop: 6,
  },
  warningLine: {
    fontSize: 8,
    color: AMBER,
    marginTop: 2,
  },
  tableHeaderRow: {
    flexDirection: "row",
    backgroundColor: BRAND,
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
  },
  tableHeaderCell: {
    color: "#ffffff",
    fontSize: 7.2,
    fontWeight: 700,
    letterSpacing: 0.3,
    paddingVertical: 4,
    paddingHorizontal: 5,
  },
  tableRow: {
    flexDirection: "row",
    alignItems: "center",
    borderBottomWidth: 0.5,
    borderBottomColor: RULE,
  },
  tableRowAlt: {
    backgroundColor: ZEBRA,
  },
  tableBlockSpaced: {
    marginTop: 10,
  },
  tableCell: {
    fontSize: 8,
    color: INK,
    paddingVertical: 3.4,
    paddingHorizontal: 5,
  },
  tableCellNumeric: {
    fontFamily: REPORT_FONT_FAMILIES.numeric,
    textAlign: "right",
  },
  tableCellRight: {
    textAlign: "right",
  },
  note: {
    fontSize: 8,
    color: AMBER,
    marginTop: 5,
  },
  colStatLabel: {
    width: "58%",
  },
  colStatValue: {
    width: "42%",
  },
  colParamName: {
    width: "30%",
  },
  colParamEstimate: {
    width: "24%",
  },
  colParamStdErr: {
    width: "20%",
  },
  colParamCi: {
    width: "26%",
  },
  chartImage: {
    width: CHART_WIDTH,
    height: CHART_HEIGHT,
    objectFit: "contain",
    borderWidth: 0.7,
    borderColor: RULE,
    borderRadius: 2,
    marginTop: 2,
  },
  chartCaption: {
    fontSize: 8,
    color: MUTED,
    marginTop: 4,
  },
});

function renderMetaRows(rows: readonly ReportMetaRow[]): ReactElement[] {
  return rows.map((row) => (
    <View key={row.key} style={styles.metaRow} wrap={false}>
      <Text style={styles.metaKey}>{row.key}</Text>
      <Text style={styles.metaValue}>{row.value}</Text>
    </View>
  ));
}

export function FitReport({
  report,
  logoDataUrl,
  watermarkDataUrl,
}: FitReportProps): ReactElement {
  const pillTextStyle = report.converged
    ? [styles.pillText, styles.pillTextOk]
    : [styles.pillText, styles.pillTextBad];

  const parameterChunks: ReportParameter[][] = [];
  for (let i = 0; i < report.parameters.length; i += PARAMETER_CHUNK_ROWS) {
    parameterChunks.push(report.parameters.slice(i, i + PARAMETER_CHUNK_ROWS));
  }
  if (parameterChunks.length === 0) parameterChunks.push([]);

  return (
    <Document
      title={`${report.title} — ${report.subtitle}`}
      author={report.byline ?? "CurveStudio (ConcreteInfo)"}
      creator="CurveStudio"
      subject="Curve fit report: equation, parameters, statistics and chart"
      keywords="curve fit, regression, parameters, statistics, CurveStudio"
    >
      <Page size="A4" style={styles.page}>
        {watermarkDataUrl !== null ? (
          <Image fixed src={watermarkDataUrl} style={styles.watermark} />
        ) : null}

        {logoDataUrl !== null ? (
          <Image fixed src={logoDataUrl} style={styles.headerLogo} />
        ) : (
          <Text fixed style={styles.headerWordmark}>
            CURVESTUDIO
          </Text>
        )}
        <Text fixed style={styles.headerTitle}>
          CURVESTUDIO_FIT_REPORT
        </Text>
        <Text fixed style={styles.headerSubtitle}>
          Fit report
        </Text>
        <View fixed style={styles.headerRule} />

        <Text style={styles.title}>{report.title}</Text>
        {report.byline !== null && (
          <Text style={styles.byline}>Prepared by {report.byline}</Text>
        )}
        <Text style={styles.subtitle}>{report.subtitle}</Text>
        <Text style={styles.generated}>
          {`Generated ${report.generatedAt} IST · CurveStudio v${report.version}`}
        </Text>
        <View style={[styles.pill, report.converged ? styles.pillOk : styles.pillBad]}>
          <Text style={pillTextStyle}>{report.statusLabel}</Text>
        </View>

        <View style={styles.metaGrid} wrap={false}>
          <View style={styles.metaColumn}>{renderMetaRows(report.dataset)}</View>
          <View style={styles.metaColumn}>
            {renderMetaRows(report.model)}
            {renderMetaRows(report.statusRows)}
          </View>
        </View>

        {report.failureBanner !== null ? (
          <View style={styles.failureBanner} wrap={false}>
            <Text style={styles.failureText}>{report.failureBanner}</Text>
          </View>
        ) : null}

        <View style={styles.section} wrap={false}>
          <Text style={styles.sectionHeading}>Equation</Text>
          <View style={styles.equationBox}>
            <Text style={styles.equationKey}>Model equation</Text>
            <Text style={styles.equationLine}>{report.equation}</Text>
            {report.fittedEquation !== null ? (
              <View style={styles.equationLineSpaced}>
                <Text style={styles.equationKey}>Fitted equation</Text>
                <Text style={styles.equationLine}>{report.fittedEquation}</Text>
              </View>
            ) : null}
          </View>
          {report.warnings.map((warning, index) => (
            <Text key={`${warning}-${index}`} style={styles.warningLine}>
              {`• ${warning}`}
            </Text>
          ))}
        </View>

        <View style={styles.section} wrap={false}>
          <Text style={styles.sectionHeading}>Statistics</Text>
          <View style={styles.tableHeaderRow}>
            <Text style={[styles.tableHeaderCell, styles.colStatLabel]}>Statistic</Text>
            <Text style={[styles.tableHeaderCell, styles.colStatValue, styles.tableCellRight]}>
              Value
            </Text>
          </View>
          {report.statistics.map((stat, index) => (
            <View
              key={stat.label}
              style={[styles.tableRow, index % 2 === 1 ? styles.tableRowAlt : {}]}
              wrap={false}
            >
              <Text style={[styles.tableCell, styles.colStatLabel]}>{stat.label}</Text>
              <Text
                style={[styles.tableCell, styles.tableCellNumeric, styles.colStatValue]}
              >
                {stat.value}
              </Text>
            </View>
          ))}
        </View>

        <View style={styles.section}>
          {parameterChunks.map((chunk, chunkIndex) => (
            // Each block is kept whole and repeats the header, so a page break never
            // orphans the heading or a header row.
            <View
              key={`parameters-${chunkIndex}`}
              wrap={false}
              style={chunkIndex > 0 ? styles.tableBlockSpaced : undefined}
            >
              {chunkIndex === 0 ? (
                <Text style={styles.sectionHeading}>Parameters</Text>
              ) : null}
              <View style={styles.tableHeaderRow}>
                <Text style={[styles.tableHeaderCell, styles.colParamName]}>Parameter</Text>
                <Text
                  style={[styles.tableHeaderCell, styles.colParamEstimate, styles.tableCellRight]}
                >
                  Estimate
                </Text>
                <Text
                  style={[styles.tableHeaderCell, styles.colParamStdErr, styles.tableCellRight]}
                >
                  Std. error
                </Text>
                <Text style={[styles.tableHeaderCell, styles.colParamCi, styles.tableCellRight]}>
                  95% CI
                </Text>
              </View>
              {chunk.map((parameter, index) => (
                <View
                  key={parameter.name}
                  style={[styles.tableRow, index % 2 === 1 ? styles.tableRowAlt : {}]}
                  wrap={false}
                >
                  <Text style={[styles.tableCell, styles.colParamName]}>{parameter.name}</Text>
                  <Text
                    style={[styles.tableCell, styles.tableCellNumeric, styles.colParamEstimate]}
                  >
                    {parameter.estimate}
                  </Text>
                  <Text
                    style={[styles.tableCell, styles.tableCellNumeric, styles.colParamStdErr]}
                  >
                    {parameter.stdErr}
                  </Text>
                  <Text style={[styles.tableCell, styles.tableCellNumeric, styles.colParamCi]}>
                    {parameter.ci}
                  </Text>
                </View>
              ))}
            </View>
          ))}
          {!report.converged ? (
            <Text style={styles.note}>
              Estimates from a fit that did not converge — diagnostics only, not results.
            </Text>
          ) : null}
        </View>

        {report.chartImage !== null ? (
          <View style={styles.section} wrap={false}>
            <Text style={styles.sectionHeading}>Chart</Text>
            <Image src={report.chartImage} style={styles.chartImage} />
            <Text style={styles.chartCaption}>{report.chartCaption}</Text>
          </View>
        ) : null}

        <View fixed style={styles.footerRule} />
        <Text
          fixed
          style={styles.footerPage}
          render={({ pageNumber, totalPages }) => `Page ${pageNumber} of ${totalPages}`}
        />
        <Text fixed style={styles.footerDisclaimer}>
          Results describe agreement with the sample, not compliance or engineering advice —
          verify before use.
        </Text>
      </Page>
    </Document>
  );
}
