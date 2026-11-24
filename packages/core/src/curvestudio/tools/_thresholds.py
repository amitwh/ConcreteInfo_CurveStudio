"""Every number a rule of the audit turns on.

Each is the rule it stands for; the checks read no other number of this kind. The uppercase
constants at the top are the audit's settings and are re-exported by
:mod:`curvestudio.tools.audit`; the rest are the core's own conventions, repeated here.
"""

import math
import re

NORMALITY_CAUTION_P = 0.05
"""Shapiro-Wilk p below this, but not below ``NORMALITY_FAIL_P``, reads ``caution``."""
NORMALITY_FAIL_P = 0.001
"""Shapiro-Wilk p below this reads ``fail``: the level every residual p-value fails at. One
honest fit in a thousand shows it by chance, so with four residual tests at this level
about four in a thousand honest fits fail one of them."""
SMALL_SAMPLE_N = 20
"""With fewer points than this the normality evidence says the test rarely detects anything."""
VARIANCE_CAUTION_P = 0.05
"""Breusch-Pagan p below this, but not below ``VARIANCE_FAIL_P``, reads ``caution``."""
VARIANCE_FAIL_P = 0.001
"""Breusch-Pagan p below this reads ``fail``: the level every residual p-value fails at."""
TREND_CAUTION_P = 0.05
"""Residual-trend p below this, but not below ``TREND_FAIL_P``, reads ``caution``."""
TREND_FAIL_P = 0.001
"""Residual-trend p below this reads ``fail``: the level every residual p-value fails at."""
TREND_EXTRA_TERMS = 2
"""Legendre degrees tested beyond a fitted polynomial's degree ``d``: ``d + 1`` up to
``d + TREND_EXTRA_TERMS``."""
TREND_MODEL_TERMS = 3
"""Legendre degrees 1 up to this number are tested, against an intercept, for every fit
that is not a polynomial."""
TREND_MIN_RESIDUAL_DF = 3
"""The residual-trend F test needs at least this many residual degrees of freedom in its
denominator; with fewer it reads ``not_applicable``."""
COOKS_CAUTION_REGION = 0.10
"""A Cook's distance above ``F(k, n - k)`` at this upper tail means that leaving the point
out moves the estimates outside their 90 % confidence region: ``caution``."""
COOKS_FAIL_REGION = 0.01
"""The same at this upper tail, the 99 % region: ``fail``."""
COOKS_CHANCE_SHARE = 0.05
"""The share of points that cross the core's 4/n influence line on a well-behaved fit, by
chance alone."""
CHANCE_CAUTION_Q = 0.95
"""More points than this quantile of a Binomial(n, ``COOKS_CHANCE_SHARE``) count crossing
the 4/n line is not chance: ``caution``."""
CHANCE_FAIL_Q = 0.995
"""More points than this quantile of the same count is not chance: ``fail``."""
MIN_POINTS_PER_PARAMETER_CAUTION = 10
"""Fewer points per parameter than this, but at least ``MIN_POINTS_PER_PARAMETER_FAIL``:
``caution``."""
MIN_POINTS_PER_PARAMETER_FAIL = 5
"""Fewer points per parameter than this reads ``fail``."""
CV_CAUTION_RATIO = 1.1
"""A cross-validation RMSE above this multiple of what a correct model would show (its
expected ratio to the in-sample RMSE) reads ``caution``."""
CV_FAIL_RATIO = 1.5
"""A cross-validation RMSE above this multiple of what a correct model would show reads
``fail``."""
SIMPLE_DELTA_AICC = 2.0
"""Ranked models within this AICc of the best count as fitting as well as it."""
WIDE_INTERVAL_SHARE = 0.99
"""A confidence interval covering at least this share of the parameter's allowed range reads
``caution``."""
DW_CAUTION_P = 0.025
"""Either exact Durbin-Watson p-value below this, but not below ``DW_FAIL_P``, reads
``caution`` for independence. Each p-value is one side of the test, so on a clean fit the
two spend 5 % between them. Where the exact test ran it alone decides: Ljung-Box's p is
reported beside it and changes nothing."""
DW_FAIL_P = 0.0005
"""Either exact Durbin-Watson p-value below this reads ``fail`` for independence: 0.05 % on
each side, 0.1 % for the check, the level every residual p-value fails at."""
AUTOCORRELATION_CAUTION_P = 0.025
"""Ljung-Box p below this, but not below ``AUTOCORRELATION_FAIL_P``, reads ``caution`` for
independence when Ljung-Box decides alone, as it does without the fit's settings.

The cut-off is halved on purpose. The test comes from the core's diagnostics, whose p-value
is approximate here: it ignores the parameters that were estimated, so on the residuals of
a fitted model it runs low. Measured on clean quadratic fits (2,000, 1,000 and 500 of them
at 20, 40 and 400 points, on seeds beyond the calibration's forty), 10.8 / 8.8 / 6.2 % of
the fits fell below 0.05 and 6.9 / 5.5 / 3.2 % below this 0.025."""
AUTOCORRELATION_FAIL_P = 0.0002
"""Ljung-Box p below this reads ``fail`` for independence when Ljung-Box decides alone.

A fifth of the 0.001 every other residual p-value fails at, for the reason given at
``AUTOCORRELATION_CAUTION_P``. On the same clean quadratic fits at 20 / 40 / 400 points,
0.55 / 1.0 / 0.6 % fell below 0.001 and 0.4 / 0.4 / 0.2 % below this 0.0002."""
DW_MIN_RESIDUAL_DF = 3
"""The exact Durbin-Watson test is computed only with at least this many residual degrees of
freedom: with two the statistic is confined between two values, with one it never moves."""

BOUND_RTOL = 1e-8
"""A parameter within this relative distance of a finite bound sits on it: the fitting
layer's own rule, by which it writes ``PARAMETER_AT_BOUND``."""
EXACT_FIT_RATIO = 1e-10
"""A fit whose RMSE is at most this fraction of the data's scale is exact: the diagnostics
layer's own rule, by which it withholds the residual tests."""
OFFSET_FLOOR_RATIO = 1e-4
"""Rounding in a large ``y`` offset reaches this fraction of ``max|y|``; it floors the scale."""
LISTED_MAX = 8
"""The evidence names at most this many flagged points; beyond that it gives the count of the
rest, so a sentence stays readable however many points are flagged."""

AT_BOUND = "PARAMETER_AT_BOUND: "
"""Prefix of the fitting layer's warning that a parameter stopped on a bound."""
POLYNOMIAL_ID = re.compile(r"poly(\d+)(_origin)?")
"""Model ids ``fit_polynomial`` writes (``poly2``, ``poly2_origin``); a polynomial has no
bounds."""
NOT_CONVERGED = "the fit did not converge"
"""What every check after the first says when the fit did not converge."""
NO_STATISTICS = "the fit has no statistics"
UNBOUNDED = (-math.inf, math.inf)
