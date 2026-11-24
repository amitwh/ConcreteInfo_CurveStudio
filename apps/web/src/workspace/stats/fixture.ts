/**
 * One statistical result that exercises every block a renderer shows: estimates, hypothesis
 * tests (one that could not run, one adjusted for a family), an assumption of every verdict,
 * warnings, tables, all five plot kinds, the counts and a long excluded-rows list.
 *
 * The wording is the kind a tool sends: a verdict states evidence and never proof.
 */

import type { StatCheck, ToolResult } from "../../api/client";

const LEVEL = 0.95;

const ASSUMPTIONS: StatCheck[] = [
  {
    id: "independence",
    title: "The observations are independent within each group",
    verdict: "pass",
    statistic: null,
    threshold: "random assignment or independent sampling",
    evidence: "Group membership was supplied per observation, with no repeated subject id.",
    explanation: "A two-sample test compares two independent sets of observations.",
    next_step: null,
  },
  {
    id: "normality",
    title: "Group A may depart from a normal distribution",
    verdict: "caution",
    statistic: "Shapiro-Wilk p = 0.081",
    threshold: "p >= 0.1 pass; 0.01 <= p < 0.1 caution; p < 0.01 fail",
    evidence: "Shapiro-Wilk on the 12 values of group A gives p = 0.081.",
    explanation:
      "The t test relies on the group means being about normal when the groups are this small.",
    next_step:
      "Read the result with the interval in mind, or compare the medians with a rank test.",
  },
  {
    id: "equal_variance",
    title: "The groups do not share a variance",
    verdict: "fail",
    statistic: "Levene p = 0.004",
    threshold: "p >= 0.05 pass",
    evidence: "Levene's test on the two groups gives p = 0.004.",
    explanation:
      "An equal-variance t test pools one variance across both groups; here the spreads differ.",
    next_step: "Use the Welch test, which does not assume equal variances.",
  },
  {
    id: "group_sizes",
    title: "Group sizes could not be balanced",
    verdict: "not_applicable",
    statistic: null,
    threshold: null,
    evidence: "The groups were observed, not designed: 12 against 20.",
    explanation:
      "Balanced groups are a design choice, so an observational comparison has nothing to check.",
    next_step: null,
  },
];

const EXCLUDED = Array.from({ length: 200 }, (_, index) => ({
  index,
  reason: index % 2 === 0 ? "missing in y" : "not a finite number in x",
}));

export const FIXTURE_RESULT: ToolResult = {
  tool: "welch_t",
  title: "Two-sample comparison of means",
  level: LEVEL,
  estimates: [
    { label: "Mean of A", value: 24.5, se: 1.2, lower: 21.9, upper: 27.1 },
    { label: "Mean of B", value: 27.1, se: 0.9, lower: 25.2, upper: 29.0 },
    { label: "Difference in means (B - A)", value: 2.6, se: 1.126, lower: null, upper: null },
  ],
  tests: [
    {
      name: "Difference in means",
      method: "Welch two-sample t-test",
      statistic: 2.31,
      df: 26.4,
      df_denominator: null,
      p_value: 0.0324,
      p_value_text: "p = 0.0324",
      adjusted_p: null,
      adjustment: null,
      null_hypothesis: "The two population means are equal.",
      reason: null,
      effect: null,
    },
    {
      name: "A against B, adjusted for the family",
      method: "Welch two-sample t-test with Holm adjustment",
      statistic: 2.31,
      df: 26.4,
      df_denominator: null,
      p_value: 0.02,
      p_value_text: "p = 0.02",
      adjusted_p: 0.05,
      adjustment: "Holm",
      null_hypothesis: "The mean of A equals the mean of B.",
      reason: null,
      effect: null,
    },
    {
      name: "Normality of A",
      method: "Shapiro-Wilk test",
      statistic: null,
      df: null,
      df_denominator: null,
      p_value: null,
      p_value_text: "not computed",
      adjusted_p: null,
      adjustment: null,
      null_hypothesis: "The values of A come from a normal distribution.",
      reason: "Group A has fewer than three observations.",
      effect: null,
    },
  ],
  assumptions: ASSUMPTIONS,
  warnings: [
    "Group A has 12 observations and group B has 20, so the test is Welch's, which does " +
      "not assume equal variances.",
  ],
  tables: [
    {
      title: "Summary by group",
      note:
        "SD is the sample standard deviation; the interval is the mean plus or minus t " +
        "times its SE.",
      columns: [
        { key: "group", label: "Group", kind: "text" },
        { key: "n", label: "n", kind: "int" },
        { key: "mean", label: "Mean", kind: "number" },
        { key: "p", label: "p", kind: "p_value" },
      ],
      rows: [
        ["A", 12, 24.5, 0.0324],
        ["B", 20, 27.1, null],
      ],
    },
    {
      title: "Pairwise comparisons",
      note: "Adjusted p-values follow Holm's step-down rule.",
      columns: [
        { key: "comparison", label: "Comparison", kind: "text" },
        { key: "difference", label: "Difference", kind: "number" },
        { key: "p", label: "p", kind: "p_value" },
      ],
      rows: [["B - A", 2.6, 0.05]],
    },
  ],
  plots: [
    {
      kind: "scatter",
      title: "Difference in means against its interval",
      note: "Each point is one pair's difference; the band is the interval of the mean.",
      x_label: "Pair",
      y_label: "Difference",
      x: [1, 2, 3, 4, 5],
      y: [1.2, 3.4, 2.1, 4.0, 2.6],
      lines: [{ label: "Zero", x0: 1, y0: 0, x1: 5, y1: 0 }],
      band: {
        x: [1, 2, 3, 4, 5],
        lower: [1.0, 1.1, 1.2, 1.3, 1.4],
        upper: [2.0, 2.1, 2.2, 2.3, 2.4],
      },
    },
    {
      kind: "histogram",
      title: "Distribution of the differences",
      note: "Counts per bin, with the normal density at the same mean and SD.",
      x_label: "Difference",
      y_label: "Count",
      edges: [0, 1, 2, 3, 4, 5],
      counts: [2, 5, 6, 4, 3],
      overlays: [{ label: "Normal fit", values: [1.5, 3.0, 4.5, 3.0, 1.5] }],
    },
    {
      kind: "qq",
      title: "Normal Q-Q plot of the differences",
      note: "Points on the reference line would sit on a normal distribution.",
      ordered: [1.0, 1.4, 1.9, 2.2, 2.6, 3.0, 3.5],
      theoretical: [-1.4, -0.9, -0.5, 0, 0.5, 0.9, 1.4],
      line: { label: "Reference", x0: -1.4, y0: 1.0, x1: 1.4, y1: 3.5 },
    },
    {
      kind: "box",
      title: "Spread by group",
      note: "The boxes are the quartiles; the whiskers reach the last value inside 1.5 IQR.",
      y_label: "Value",
      boxes: [
        {
          label: "A",
          q1: 22.0,
          median: 24.5,
          q3: 26.5,
          lower_whisker: 20.1,
          upper_whisker: 28.8,
          outliers: [9.5],
        },
        {
          label: "B",
          q1: 25.0,
          median: 27.0,
          q3: 29.0,
          lower_whisker: 23.5,
          upper_whisker: 30.4,
          outliers: [],
        },
      ],
    },
    {
      kind: "lines",
      title: "Group means by site",
      note: "One line per group; the x values are site names, not numbers.",
      x_label: "Site",
      y_label: "Mean",
      x_labels: ["North", "South", "East"],
      series: [
        { label: "A", x: [0, 1, 2], y: [23.0, 25.1, 24.9] },
        { label: "B", x: [0, 1, 2], y: [26.0, 28.2, 27.4] },
      ],
    },
  ],
  interpretation:
    "The mean of group B is higher than the mean of group A by 2.6 units (Welch p = 0.0324). " +
    "Group A is the smaller group and its values may depart from a normal distribution, so the " +
    "interval relies on an approximation.",
  methods_fragment:
    "Group means were compared with Welch's two-sample t-test (t = 2.31, df = 26.4, p = 0.0324); " +
    "group summaries are the sample mean and n.",
  counts: { n_offered: 40, n_used: 36, n_excluded: 4 },
  excluded: EXCLUDED,
  excluded_unlisted: 50,
};
