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
DW_CAUTION_P = 0.05
"""Either exact Durbin-Watson p-value below this, but not below ``DW_FAIL_P``, reads
``caution`` for independence: each is the chance of a statistic that far out on its side.
Ljung-Box's p is supporting evidence there, and the test
:func:`~curvestudio.tools.audit_fit` decides by without the fit's settings."""
DW_FAIL_P = 0.001
"""Either exact Durbin-Watson p-value below this reads ``fail`` for independence: the level
every residual p-value fails at, and the one at which Ljung-Box fails when it decides
alone."""
AUTOCORRELATION_CAUTION_P = 0.05
"""Ljung-Box p below this reads ``caution``: on its own when Ljung-Box decides, or by
raising a pass of the exact Durbin-Watson test when that test decided."""
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
