"""Checks that read the core's residual diagnostics: normality, independence, variance, influence.

The trend check, which computes something of its own, lives in :mod:`curvestudio.tools._trend`.
"""

from scipy import stats as scipy_stats

from curvestudio.tools._common import (
    Check,
    Inputs,
    audit_check,
    core_test,
    is_exact,
    no_summary,
    p_verdict,
    read_test,
    skipped,
    title_for,
    where,
)
from curvestudio.tools._thresholds import (
    AUTOCORRELATION_CAUTION_P,
    CHANCE_CAUTION_Q,
    CHANCE_FAIL_Q,
    COOKS_CAUTION_REGION,
    COOKS_CHANCE_SHARE,
    COOKS_FAIL_REGION,
    DW_FAIL_P,
    NORMALITY_CAUTION_P,
    NORMALITY_FAIL_P,
    SMALL_SAMPLE_N,
    VARIANCE_CAUTION_P,
    VARIANCE_FAIL_P,
)
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import (
    INDEPENDENCE_NEXT_STEP,
    NORMALITY_NEXT_STEP,
    VARIANCE_NEXT_STEP,
    num,
    p_threshold,
    subjects,
)


def check_residual_normality(inp: Inputs) -> AuditCheck:
    found = read_test(inp, NORMALITY, "shapiro_wilk", "Shapiro-Wilk")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = p_verdict(p, NORMALITY_CAUTION_P, NORMALITY_FAIL_P)
    evidence = f"Shapiro-Wilk p = {num(p)}"
    if inp.n < SMALL_SAMPLE_N:
        evidence += "; with fewer than 20 points this test rarely detects anything"
    statistic = f"p = {num(p)}"
    if test.statistic is not None:
        statistic = f"W = {num(test.statistic)}, {statistic}"
    return audit_check(
        NORMALITY,
        verdict,
        title_for(NORMALITY_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=p_threshold(NORMALITY_CAUTION_P, NORMALITY_FAIL_P),
        next_step=None if verdict is Verdict.PASS else NORMALITY_NEXT_STEP,
    )


def check_residual_independence(inp: Inputs) -> AuditCheck:
    found = read_test(inp, INDEPENDENCE, "ljung_box", "Ljung-Box")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = p_verdict(p, AUTOCORRELATION_CAUTION_P, DW_FAIL_P)
    evidence = f"Ljung-Box p = {num(p)}, residuals taken in ascending x."
    durbin_watson = None if inp.summary is None else core_test(inp.summary, "durbin_watson")
    if durbin_watson is None:
        evidence += " The Durbin-Watson statistic is not in the summary."
    elif durbin_watson.statistic is not None:
        evidence += (
            f" Durbin-Watson statistic {num(durbin_watson.statistic)}, which is reported "
            "here without a verdict of its own."
        )
    else:
        why = durbin_watson.reason or "the test did not apply"
        evidence += f" The Durbin-Watson statistic is not available ({why})."
    statistic = f"p = {num(p)}"
    if test.statistic is not None:
        lags = "" if test.df is None else f"({num(test.df)})"
        statistic = f"Q{lags} = {num(test.statistic)}, {statistic}"
    return audit_check(
        INDEPENDENCE,
        verdict,
        title_for(INDEPENDENCE_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=p_threshold(AUTOCORRELATION_CAUTION_P, DW_FAIL_P),
        next_step=None if verdict is Verdict.PASS else INDEPENDENCE_NEXT_STEP,
    )


def check_constant_variance(inp: Inputs) -> AuditCheck:
    found = read_test(inp, CONSTANT_VARIANCE, "breusch_pagan", "Breusch-Pagan")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = p_verdict(p, VARIANCE_CAUTION_P, VARIANCE_FAIL_P)
    statistic = f"p = {num(p)}"
    if test.statistic is not None:
        statistic = f"BP = {num(test.statistic)}, {statistic}"
    return audit_check(
        CONSTANT_VARIANCE,
        verdict,
        title_for(CONSTANT_VARIANCE_TITLES, verdict),
        f"Breusch-Pagan p = {num(p)}",
        statistic=statistic,
        threshold=p_threshold(VARIANCE_CAUTION_P, VARIANCE_FAIL_P),
        next_step=None if verdict is Verdict.PASS else VARIANCE_NEXT_STEP,
    )


def check_influential_points(inp: Inputs) -> AuditCheck:
    """Does the fit stand without any one point?

    A point's Cook's distance above ``F(k, n - k)`` at upper tail ``alpha`` means that
    leaving it out moves the estimates outside the (1 - alpha) confidence region. The
    core's 4/n line is a screen that about ``COOKS_CHANCE_SHARE`` of the points cross on a
    well-behaved fit, so only a count that chance would rarely give decides anything.
    """
    if inp.summary is None:
        return skipped(INFLUENCE, no_summary(inp))
    if is_exact(inp):
        return skipped(INFLUENCE, "the fit is exact, so no point stands out from the others")
    residuals = inp.summary.residuals
    n, k = inp.n, inp.k
    if n - k < 1:
        return skipped(
            INFLUENCE,
            "the fit has no residual degrees of freedom, so there is no confidence region to leave",
        )
    distances = [
        (distance, position)
        for position, distance in enumerate(residuals.cooks_distance)
        if distance is not None
    ]
    if not distances:
        return skipped(INFLUENCE, "no Cook's distance is defined for this fit")
    points = len(residuals.x)
    largest, top = max(distances)
    flagged = list(residuals.flagged_high_influence)
    count = len(flagged)
    cut_fail = float(scipy_stats.f.isf(COOKS_FAIL_REGION, k, n - k))
    cut_caution = float(scipy_stats.f.isf(COOKS_CAUTION_REGION, k, n - k))
    chance_fail = int(scipy_stats.binom.ppf(CHANCE_FAIL_Q, n, COOKS_CHANCE_SHARE))
    chance_caution = int(scipy_stats.binom.ppf(CHANCE_CAUTION_Q, n, COOKS_CHANCE_SHARE))
    if largest > cut_fail:
        verdict, decided = Verdict.FAIL, "distance"
    elif count > chance_fail:
        verdict, decided = Verdict.FAIL, "count"
    elif largest > cut_caution:
        verdict, decided = Verdict.CAUTION, "distance"
    elif count > chance_caution:
        verdict, decided = Verdict.CAUTION, "count"
    else:
        verdict, decided = Verdict.PASS, "none"
    region_fail = num((1.0 - COOKS_FAIL_REGION) * 100)
    region_caution = num((1.0 - COOKS_CAUTION_REGION) * 100)
    chance = num(COOKS_CHANCE_SHARE * 100)
    line = num(residuals.cooks_threshold)
    where_top = where(inp, [top], points)
    outside_fail_region = verdict is Verdict.FAIL and decided == "distance"
    cut = cut_fail if outside_fail_region else cut_caution
    region = region_fail if outside_fail_region else region_caution
    statistic = (
        f"largest Cook's distance {num(largest)} ({where_top}); {region} % region at {num(cut)}"
    )
    distance_cut = f"Cook's distance above the {region} % region cut F({k}, {n - k}) = {num(cut)}"
    if verdict is Verdict.FAIL and decided == "distance":
        title = f"The fit does not stand without {where_top}"
        sentences = [
            f"Removing {where_top} would move the estimates outside their {region} % region "
            f"(Cook's distance {num(largest)} against {num(cut)}): the fit does not stand "
            "without that point."
        ]
        threshold = distance_cut
        rows = where_top
    elif decided == "distance":
        title = f"Removing {where_top} would move the estimates outside their {region} % region"
        sentences = [
            f"Removing {where_top} would move the estimates outside their {region} % region "
            f"(Cook's distance {num(largest)} against {num(cut)})."
        ]
        threshold = distance_cut
        rows = where_top
    else:
        title = "No single point has a large influence on the fit"
        sentences = [
            f"No single point moves the estimates outside their {region} % region: the largest "
            f"Cook's distance is {num(largest)} ({where_top}), against {num(cut)}."
        ]
        threshold = distance_cut
        rows = None
    chance_cut = f"more than {chance_caution} of {n} points above the 4/n line {line}"
    if decided == "count":
        limit = chance_fail if verdict is Verdict.FAIL else chance_caution
        many = "Far more" if verdict is Verdict.FAIL else "More"
        sentences.append(
            f"{count} of {n} points cross the 4/n line; about {chance} % cross it by "
            f"chance, so more than {limit} is not chance."
        )
        threshold = f"more than {limit} of {n} points above the 4/n line {line}"
        title = f"{many} points than chance cross the influence line"
        rows = where(inp, flagged, points)
    else:
        sentences.append(
            f"{count} of {n} points cross the 4/n line ({line}), about the {chance} % that "
            "cross it by chance."
        )
        if decided == "none":
            threshold = f"{distance_cut}, or {chance_cut}"
    flagged_set = set(flagged)
    leverage_only = [i for i in residuals.flagged_high_leverage if i not in flagged_set]
    if leverage_only:
        leverage = num(residuals.leverage_threshold)
        sentences.append(
            f"{subjects(len(leverage_only), 'point', 'has', 'have')} high leverage but "
            f"little influence (leverage above the threshold 2k/n = {leverage}): "
            f"{where(inp, leverage_only, points)}."
        )
    next_step = None
    if rows is not None:
        next_step = (
            f"Refit without {rows} and compare the estimates. If the conclusions differ, "
            "report both fits or justify leaving the point out. Do not delete data to "
            "improve a fit."
        )
    return audit_check(
        INFLUENCE,
        verdict,
        title,
        " ".join(sentences),
        statistic=statistic,
        threshold=threshold,
        next_step=next_step,
    )


NORMALITY_TITLES = (
    "Residuals are consistent with a normal distribution",
    "Residuals may depart from a normal distribution",
    "Residuals depart from a normal distribution",
)
INDEPENDENCE_TITLES = (
    "Residuals show no autocorrelation in x order",
    "Residuals may be autocorrelated in x order",
    "Residuals are autocorrelated in x order",
)
CONSTANT_VARIANCE_TITLES = (
    "Residual variance is consistent with being constant",
    "Residual variance may change with the fitted value",
    "Residual variance changes with the fitted value",
)

NORMALITY = Check(
    "residual_normality",
    "Normality of the residuals was not tested",
    "Intervals and p-values assume the residuals are roughly normal. The Shapiro-Wilk test "
    "looks at the shape of their distribution, not at whether the model is right.",
    check_residual_normality,
)
INDEPENDENCE = Check(
    "residual_independence",
    "Independence of the residuals was not tested",
    "Points taken in sequence often carry information from their neighbours. The Ljung-Box "
    "test asks whether residuals close together in x are also alike.",
    check_residual_independence,
)
CONSTANT_VARIANCE = Check(
    "constant_variance",
    "Constancy of the residual variance was not tested",
    "Intervals and tests assume the residuals scatter equally at every fitted value. The "
    "Breusch-Pagan test asks whether their spread grows with the fitted value.",
    check_constant_variance,
)
INFLUENCE = Check(
    "influential_points",
    "Influence of single points was not assessed",
    "One point can pull a fitted curve towards itself. Cook's distance says how far the "
    "estimates would move if the point were left out, against their own uncertainty.",
    check_influential_points,
)
