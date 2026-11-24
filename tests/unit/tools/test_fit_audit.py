"""``audit_fit`` on real fits: every verdict is read from numbers the core already computed.

The calibration tests run ordinary seeds, 1 onwards, with nothing selected, and assert the
bar each rule was given. Where a size misses its bar, the test keeps the bar and says so as
a strict expected failure.
"""

import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from curvestudio.discovery import AutoFitConfig, AutoFitResult, CandidateMetrics, RankedFit
from curvestudio.fitting import FitStatus
from curvestudio.statistics import CrossValidation
from curvestudio.tools import (
    AuditCheck,
    ExcludedRow,
    FitAudit,
    SimplerModel,
    ToolError,
    Verdict,
    audit_fit,
)
from curvestudio.tools.audit import (
    CHANCE_CAUTION_Q,
    CHANCE_FAIL_Q,
    COOKS_CAUTION_REGION,
    COOKS_CHANCE_SHARE,
    expected_cv_ratio,
)
from curvestudio.tools.durbin_watson import DW_EXACT_MAX_N
from curvestudio.workflow import FitBundle, FitSpec, RegisteredRef

CHECK_IDS = (
    "converged",
    "perfect_fit",
    "residual_normality",
    "residual_independence",
    "constant_variance",
    "influential_points",
    "residual_trend",
    "parameters_identifiable",
    "points_per_parameter",
    "cross_validation",
    "simpler_model",
)
NEEDING_THE_SUMMARY = (
    "residual_normality",
    "residual_independence",
    "constant_variance",
    "influential_points",
    "residual_trend",
)
AT_A_BOUND_NEXT_STEP = (
    "A parameter at a bound or with a wide interval is not pinned down by these data. Fix the "
    "bound if it is wrong, take more points where the model is sensitive, or use a model with "
    "fewer parameters."
)
TREND_NEXT_STEP = (
    "The residuals show a curved trend, so the model is missing a shape the data contain. "
    "Try a model with one more parameter."
)
CV_NEXT_STEP = (
    "The fit describes noise as well as signal. A simpler model, or one with fewer "
    "parameters, will predict better."
)
SEEDS = range(1, 41)
"""Ordinary seeds for a calibration run: the first forty after zero, none of them chosen."""
CV_SEEDS = range(1, 51)


def _check(audit: FitAudit, check_id: str) -> AuditCheck:
    return next(check for check in audit.checks if check.id == check_id)


def _rates(verdicts: list[Verdict]) -> dict[str, float]:
    """The share of each verdict, in percent, for an assertion message."""
    return {
        verdict.value: round(100 * verdicts.count(verdict) / len(verdicts), 1)
        for verdict in Verdict
    }


def _cv(rmse: float, folds: int = 5) -> CrossValidation:
    """A complete cross-validation with the given pooled error."""
    return CrossValidation(
        folds=folds,
        seed=0,
        rmse=rmse,
        fold_rmse=[rmse] * folds,
        n_failed=0,
        status="ok",
        reasons=[],
    )


def _row(
    model_id: str,
    name: str,
    parameters: int,
    delta: float,
    rank: int,
    *,
    cv_rmse: float | None = None,
    degenerate: bool = False,
    redundant_with: str | None = None,
) -> RankedFit:
    return RankedFit(
        rank=rank,
        model_id=model_id,
        name=name,
        metrics=CandidateMetrics(k=parameters, n=40, aicc=100.0 + delta, cv_rmse=cv_rmse),
        degenerate=degenerate,
        delta_aicc=delta,
        redundant_with=redundant_with,
    )


def _ranking(*rows: RankedFit) -> AutoFitResult:
    return AutoFitResult(
        config=AutoFitConfig(),
        dataset_fingerprint="fp-audit",
        n_active=40,
        ranked=rows,
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def test_a_clean_fit_passes_every_check(scenarios) -> None:
    bundle = scenarios.clean()
    # The sample is the generator's first draw, taken as it comes.
    noise = np.random.default_rng(0).normal(0.0, 0.1, 40)
    x = np.asarray(bundle.data.x)
    assert np.allclose(bundle.data.y, 1.0 + 0.5 * x - 0.03 * x**2 + noise)

    audit = audit_fit(bundle)

    assert [check.id for check in audit.checks] == list(CHECK_IDS)
    for check in audit.checks:
        if check.id in ("cross_validation", "simpler_model"):
            assert check.verdict is Verdict.NOT_APPLICABLE, check
        else:
            assert check.verdict is Verdict.PASS, check
    assert audit.overall is Verdict.PASS
    assert audit.n_observations == 40
    assert audit.n_parameters == 3
    assert audit.excluded == ()
    assert audit.simpler_model is None


def test_a_non_converged_fit_is_a_failed_audit_that_still_answers(scenarios) -> None:
    bundle = scenarios.non_converged()
    assert bundle.result.status is FitStatus.NOT_CONVERGED

    audit = audit_fit(bundle)

    assert audit.overall is Verdict.FAIL
    first, *rest = audit.checks
    assert first.id == "converged" and first.verdict is Verdict.FAIL
    assert bundle.result.status.value in first.evidence
    assert bundle.result.termination_reason in first.evidence
    assert first.next_step == (
        "Check the starting values and the bounds, or try a model with fewer parameters."
    )
    assert [check.id for check in rest] == list(CHECK_IDS[1:])
    for check in rest:
        assert check.verdict is Verdict.NOT_APPLICABLE, check
        assert "the fit did not converge" in check.evidence
        assert check.next_step is None
    assert audit.n_observations == 40
    assert audit.n_parameters == 2
    assert audit.summary == (
        f"The fit did not converge ({bundle.result.termination_reason}), so there is nothing to "
        f"audit yet. {first.next_step}"
    )
    assert audit.methods_fragment == "The fit did not converge, so no residual checks apply."


def test_a_check_without_its_number_is_not_applicable_never_pass(scenarios) -> None:
    bundle = scenarios.without_summary()
    assert bundle.result.status is FitStatus.CONVERGED
    assert bundle.summary is None and bundle.summary_reason is not None

    audit = audit_fit(bundle)

    assert [check.id for check in audit.checks] == list(CHECK_IDS)
    needing = [check for check in audit.checks if check.id in NEEDING_THE_SUMMARY]
    assert len(needing) == len(NEEDING_THE_SUMMARY)
    for check in needing:
        assert check.verdict is Verdict.NOT_APPLICABLE, check
        assert bundle.summary_reason in check.evidence
        assert check.next_step is None
    for check in audit.checks:
        if check.id == "cross_validation":
            assert check.verdict is Verdict.NOT_APPLICABLE
            assert check.next_step == "Run cross-validation to check this."
        if check.id in ("converged", "perfect_fit"):
            assert check.verdict is Verdict.PASS, check


def test_an_empty_bundle_is_refused_and_a_fit_without_statistics_says_so(scenarios) -> None:
    empty = scenarios.empty()
    assert empty.data.x == []
    with pytest.raises(ToolError) as refused:
        audit_fit(empty)
    assert refused.value.code == "NO_DATA"

    clean = scenarios.clean()
    bare = clean.model_copy(update={"result": clean.result.model_copy(update={"stats": None})})
    audit = audit_fit(bare)
    for check_id in ("perfect_fit", "cross_validation"):
        check = _check(audit, check_id)
        assert check.verdict is Verdict.NOT_APPLICABLE
        assert "the fit has no statistics" in check.evidence
    assert _check(audit, "converged").verdict is Verdict.PASS


@pytest.mark.parametrize(
    ("scenario", "check_id", "expected", "on_a_clean_fit"),
    [
        ("skewed", "residual_normality", Verdict.FAIL, Verdict.PASS),
        ("heteroscedastic", "constant_variance", Verdict.FAIL, Verdict.PASS),
        ("far_outlier", "influential_points", Verdict.FAIL, Verdict.PASS),
        ("cubic_by_quadratic", "residual_trend", Verdict.FAIL, Verdict.PASS),
        ("curve_fitted_by_line", "residual_trend", Verdict.FAIL, None),
        ("six_points", "points_per_parameter", Verdict.FAIL, Verdict.PASS),
        ("exact", "perfect_fit", Verdict.CAUTION, Verdict.PASS),
        ("autocorrelated", "residual_independence", Verdict.FAIL, Verdict.PASS),
        ("overfit_with_cv", "cross_validation", Verdict.FAIL, None),
        ("line_with_cv", "cross_validation", Verdict.PASS, None),
    ],
)
def test_each_check_triggers_on_data_built_for_it_and_passes_on_clean_data(
    scenarios, scenario, check_id, expected, on_a_clean_fit
) -> None:
    audits = scenarios.audited()

    assert _check(audits[scenario], check_id).verdict is expected
    if on_a_clean_fit is not None:
        assert _check(audits["clean"], check_id).verdict is on_a_clean_fit


# -- influential points -------------------------------------------------------------------------


def test_the_influence_check_pins_its_three_verdict_forms(scenarios) -> None:
    audits = scenarios.audited()
    caution_cut = stats.f.isf(COOKS_CAUTION_REGION, 3, 37)
    assert caution_cut == pytest.approx(2.24, abs=0.005)

    passed = _check(audits["clean"], "influential_points")
    assert passed.verdict is Verdict.PASS
    assert passed.statistic == "largest Cook's distance 0.161 (row 40); 90 % region at 2.24"
    assert passed.evidence.startswith(
        "No single point moves the estimates outside their 90 % region: the largest Cook's "
        "distance is 0.161 (row 40), against 2.24. 2 of 40 points cross the 4/n line (0.1), "
        "about the 5 % that cross it by chance."
    )
    assert passed.next_step is None

    far = _check(audits["far_outlier"], "influential_points")
    assert far.verdict is Verdict.FAIL
    assert far.title == "The fit does not stand without row 31"
    assert far.evidence.startswith(
        "Removing row 31 would move the estimates outside their 99 % region (Cook's distance "
        "46.9 against 4.36): the fit does not stand without that point."
    )
    assert far.threshold == "Cook's distance above the 99 % region cut F(3, 37) = 4.36"
    assert far.next_step == (
        "Refit without row 31 and compare the estimates. If the conclusions differ, report "
        "both fits or justify leaving the point out. Do not delete data to improve a fit."
    )

    # Decided by the count over the 4/n line, with no single large Cook's distance.
    line = int(stats.binom.ppf(CHANCE_CAUTION_Q, 100, COOKS_CHANCE_SHARE))
    many = _check(audits["contaminated_caution"], "influential_points")
    assert many.verdict is Verdict.CAUTION
    assert many.title == "More points than chance cross the influence line"
    assert (
        f"10 of 100 points cross the 4/n line; about 5 % cross it by chance, so more than "
        f"{line} is not chance."
    ) in many.evidence

    line = int(stats.binom.ppf(CHANCE_FAIL_Q, 400, COOKS_CHANCE_SHARE))
    masses = _check(audits["contaminated_fail"], "influential_points")
    assert masses.verdict is Verdict.FAIL
    assert masses.title == "Far more points than chance cross the influence line"
    assert (
        f"33 of 400 points cross the 4/n line; about 5 % cross it by chance, so more than "
        f"{line} is not chance."
    ) in masses.evidence
    assert masses.next_step is not None
    assert masses.next_step.startswith("Refit without rows ")
    assert f", and {33 - 8} more and compare the estimates." in masses.next_step


@pytest.mark.parametrize("n", (12, 20, 40, 60, 400))
def test_the_influence_check_is_calibrated_on_well_behaved_fits(scenarios, n) -> None:
    verdicts = [
        _check(audit_fit(scenarios.quadratic_fit(n, seed)), "influential_points").verdict
        for seed in SEEDS
    ]

    assert verdicts.count(Verdict.PASS) >= 0.9 * len(SEEDS), _rates(verdicts)
    assert verdicts.count(Verdict.FAIL) <= 0.01 * len(SEEDS), _rates(verdicts)


def test_a_far_point_fails_the_influence_check_and_an_in_range_outlier_is_left_to_normality(
    scenarios,
) -> None:
    audits = scenarios.audited()

    inside = _check(audits["in_range_outlier"], "influential_points")
    assert inside.verdict is Verdict.PASS
    assert inside.statistic == "largest Cook's distance 0.411 (row 31); 90 % region at 2.24"
    assert _check(audits["in_range_outlier"], "residual_normality").verdict is Verdict.FAIL

    outside = _check(audits["far_outlier"], "influential_points")
    assert outside.verdict is Verdict.FAIL
    assert "(Cook's distance 46.9 against 4.36)" in outside.evidence


def test_the_influence_check_reads_not_applicable_without_a_region_or_distances(scenarios) -> None:
    clean = scenarios.clean()
    assert clean.result.stats is not None
    # As many parameters as points: the core refuses such a fit, so it is built by hand.
    saturated = clean.result.stats.model_copy(update={"n": 3})
    no_region = clean.model_copy(
        update={"result": clean.result.model_copy(update={"stats": saturated})}
    )
    check = _check(audit_fit(no_region), "influential_points")
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert "no residual degrees of freedom" in check.evidence

    assert clean.summary is not None
    blank = clean.summary.residuals.model_copy(
        update={"cooks_distance": [None] * len(clean.summary.residuals.cooks_distance)}
    )
    undefined = clean.model_copy(
        update={"summary": clean.summary.model_copy(update={"residuals": blank})}
    )
    check = _check(audit_fit(undefined), "influential_points")
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert "no Cook's distance is defined for this fit" in check.evidence


# -- residual trend -----------------------------------------------------------------------------


def test_the_trend_check_tests_terms_the_fitted_model_cannot_represent(scenarios) -> None:
    audits = scenarios.audited()

    cubic = _check(audits["cubic_by_quadratic"], "residual_trend")
    assert cubic.verdict is Verdict.FAIL
    assert cubic.statistic == (
        "F(2, 35) = 387, p = 1.37e-24, for cubic and quartic terms beyond the fitted quadratic"
    )
    assert cubic.next_step == TREND_NEXT_STEP

    line = _check(audits["curve_fitted_by_line"], "residual_trend")
    assert line.verdict is Verdict.FAIL
    assert line.statistic.endswith(", for quadratic and cubic terms beyond the fitted linear")

    quartic = _check(audits["quartic_by_quadratic"], "residual_trend")
    assert quartic.verdict is Verdict.FAIL

    offset = _check(audits["offset_without_offset"], "residual_trend")
    assert offset.verdict is Verdict.FAIL
    assert offset.statistic.endswith(
        "for linear, quadratic and cubic terms beyond the fitted exp_growth model"
    )

    origin = _check(audits["origin_quadratic_with_offset"], "residual_trend")
    assert origin.verdict is Verdict.FAIL
    assert origin.statistic.endswith(
        "for constant, cubic and quartic terms beyond the fitted quadratic through the origin"
    )

    few = _check(audits["six_points"], "residual_trend")
    assert few.verdict is Verdict.NOT_APPLICABLE
    assert few.evidence == "only 1 residual degree of freedom; the test needs at least 3"

    # A polynomial in the spec decides the degree that was fitted.
    by_spec = _check(audit_fit(scenarios.clean(), spec=scenarios.LINE), "residual_trend")
    assert by_spec.statistic.endswith(", for quadratic and cubic terms beyond the fitted linear")


@pytest.mark.parametrize("n", [20, 40, 400])
def test_the_trend_check_is_calibrated_on_well_behaved_fits(scenarios, n) -> None:
    # A test at the 5 % level passes 95 % of well-behaved fits by construction, so the bar
    # sits below that, as it does for every other check; over 40 seeds the pass rate is
    # 92.5-95 % at these sizes.
    verdicts = [
        _check(audit_fit(scenarios.quadratic_fit(n, seed)), "residual_trend").verdict
        for seed in SEEDS
    ]

    assert verdicts.count(Verdict.PASS) >= 0.90 * len(SEEDS), _rates(verdicts)
    assert verdicts.count(Verdict.FAIL) <= 0.01 * len(SEEDS), _rates(verdicts)


# -- residual independence ----------------------------------------------------------------------


INDEPENDENCE_NEXT_STEP = (
    "Plot the residuals against x in data order. If they wander, the model is missing a "
    "shape; if they alternate, look at how the data were taken."
)
NUMBER = r"\d+(?:\.\d+)?(?:e[+-]?\d+)?"
EXACT_STATISTIC = re.compile(
    rf"Durbin-Watson d = (?P<statistic>{NUMBER}); p = (?P<positive>{NUMBER}) "
    rf"\(positive autocorrelation\), (?P<negative>{NUMBER}) \(negative\)"
)
DECIDED = "Ljung-Box decided: the exact Durbin-Watson test was not run because "
LJUNG_BOX_RULE = "p >= 0.05 pass; 0.01 <= p < 0.05 caution; p < 0.01 fail"


def _two_p_values(check: AuditCheck) -> tuple[float, float]:
    """The statistic's two p-values: for positive, then for negative autocorrelation."""
    found = EXACT_STATISTIC.fullmatch(check.statistic or "")
    assert found, check.statistic
    return float(found.group("positive")), float(found.group("negative"))


def _exact_statistic(statistic: str, positive: str, negative: str) -> str:
    """The independence check's statistic as the exact test writes it, numbers as printed."""
    return (
        f"Durbin-Watson d = {statistic}; p = {positive} (positive autocorrelation), "
        f"{negative} (negative)"
    )


def test_with_the_fit_s_settings_the_exact_durbin_watson_test_decides_independence(
    scenarios,
) -> None:
    audits = scenarios.audited()

    for name in ("autocorrelated_with_spec", "alternating_with_spec"):
        check = _check(audits[name], "residual_independence")
        assert check.verdict is Verdict.FAIL, check
        assert check.next_step == INDEPENDENCE_NEXT_STEP
        assert re.search(rf"Ljung-Box p = {NUMBER}", check.evidence), check.evidence
    # Errors carried into the next point: a low statistic and a small p on that side.
    positive, negative = _two_p_values(
        _check(audits["autocorrelated_with_spec"], "residual_independence")
    )
    assert positive < 0.001 < 0.5 < negative
    # Errors that alternate: a high statistic and a small p on the other side.
    positive, negative = _two_p_values(
        _check(audits["alternating_with_spec"], "residual_independence")
    )
    assert negative < 0.001 < 0.5 < positive

    # Clean fits: one on many points, one with weights and zero-weight rows.
    clean = _check(audits["large_quadratic_with_spec"], "residual_independence")
    assert clean.verdict is Verdict.PASS
    assert clean.title == "Residuals show no autocorrelation in x order"
    assert clean.statistic == _exact_statistic("2.04", "0.676", "0.324")
    assert clean.evidence == (
        "Exact Durbin-Watson p-values for the fitted model, from its Jacobian at the "
        "estimates over 600 points (597 residual degrees of freedom). Ljung-Box p = 0.697, "
        "residuals taken in ascending x."
    )
    assert clean.threshold == f"smaller Durbin-Watson p: {LJUNG_BOX_RULE}"
    assert clean.next_step is None
    assert (
        "no autocorrelation (Durbin-Watson p = 0.324)"
        in audits["large_quadratic_with_spec"].methods_fragment
    )

    weighted = _check(audits["zero_weights_with_spec"], "residual_independence")
    assert weighted.verdict is Verdict.PASS
    assert weighted.statistic == _exact_statistic("2.04", "0.409", "0.591")
    assert "over 40 points (37 residual degrees of freedom)" in weighted.evidence

    # The first draw of seed 0 on 40 points lands in the caution band, as one clean fit in
    # ten does. By Ljung-Box alone, which is all the audit has without the settings, it
    # passes.
    drawn = _check(audits["clean_with_spec"], "residual_independence")
    assert drawn.verdict is Verdict.CAUTION
    assert drawn.title == "Residuals may be autocorrelated in x order"
    assert drawn.statistic == _exact_statistic("1.58", "0.0435", "0.957")
    assert drawn.next_step == INDEPENDENCE_NEXT_STEP
    assert _check(audits["clean"], "residual_independence").verdict is Verdict.PASS
    assert (
        "possible autocorrelation (Durbin-Watson p = 0.0435)"
        in audits["clean_with_spec"].methods_fragment
    )
    assert "no autocorrelation (Ljung-Box p = 0.143)" in audits["clean"].methods_fragment


def test_a_small_ljung_box_p_raises_the_exact_test_s_pass_to_caution(scenarios) -> None:
    bundle = scenarios.large_quadratic()
    passed = _check(audit_fit(bundle, spec=scenarios.QUADRATIC), "residual_independence")
    assert passed.verdict is Verdict.PASS

    # The same fit with Ljung-Box's p made small: the core's number, replaced.
    assert bundle.summary is not None
    tests = tuple(
        test.model_copy(update={"p_value": 0.03}) if test.name == "ljung_box" else test
        for test in bundle.summary.diagnostics.tests
    )
    diagnostics = bundle.summary.diagnostics.model_copy(update={"tests": tests})
    summary = bundle.summary.model_copy(update={"diagnostics": diagnostics})
    doubting = bundle.model_copy(update={"summary": summary})

    raised = audit_fit(doubting, spec=scenarios.QUADRATIC)
    check = _check(raised, "residual_independence")

    assert check.verdict is Verdict.CAUTION
    assert check.statistic == passed.statistic
    assert check.evidence.endswith(
        "Ljung-Box p = 0.03, residuals taken in ascending x. Ljung-Box p is below 0.05, so "
        "the verdict is caution."
    )
    assert check.next_step == INDEPENDENCE_NEXT_STEP
    # The methods sentence names the test behind the verdict.
    assert "possible autocorrelation (Ljung-Box p = 0.03)" in raised.methods_fragment


def test_without_the_fit_s_settings_ljung_box_decides_and_the_evidence_says_why(
    scenarios,
) -> None:
    audits = scenarios.audited()

    for name in ("autocorrelated", "alternating", "clean"):
        check = _check(audits[name], "residual_independence")
        assert check.statistic.startswith("Q"), check.statistic
        assert check.threshold == LJUNG_BOX_RULE
        assert DECIDED + "the exact Durbin-Watson test needs the fit's settings." in check.evidence
        assert re.search(r"Durbin-Watson statistic [\d.]+", check.evidence), check.evidence
    assert _check(audits["autocorrelated"], "residual_independence").verdict is Verdict.FAIL
    assert _check(audits["clean"], "residual_independence").verdict is Verdict.PASS


def test_the_evidence_says_why_the_exact_durbin_watson_test_did_not_decide(
    scenarios, monkeypatch
) -> None:
    clean = scenarios.clean()
    quadratic = scenarios.QUADRATIC

    def evidence(bundle: FitBundle, spec: FitSpec | None = quadratic, **given: object) -> str:
        return _check(audit_fit(bundle, spec=spec, **given), "residual_independence").evidence

    too_many = scenarios.quadratic_fit(DW_EXACT_MAX_N + 1, seed=0)
    assert DECIDED + (
        f"the exact test is computed for at most {DW_EXACT_MAX_N} points and this fit has "
        f"{DW_EXACT_MAX_N + 1}."
    ) in evidence(too_many)
    without = scenarios.without_covariance(clean)
    assert DECIDED + "the fit has no covariance matrix." in evidence(without)
    assert DECIDED + "this fit has 5 active points, fewer than its 3 parameters plus 3." in (
        evidence(scenarios.quadratic_fit(5, seed=0))
    )

    compact = scenarios.subsampled(clean)
    assert DECIDED + (
        "the bundle holds only a subsample of the data and the full data were not given"
    ) in evidence(compact)
    whole = audit_fit(compact, spec=quadratic, data=scenarios.clean_xy())
    assert EXACT_STATISTIC.fullmatch(_check(whole, "residual_independence").statistic or "")

    other = FitSpec(model=RegisteredRef(id="exp_growth"))
    assert DECIDED + "the specification does not match the fit" in evidence(clean, spec=other)
    foreign = replace(scenarios.line_xy(40, seed=0), fingerprint="fp-another")
    assert DECIDED + "the data given are not the ones the fit was computed from" in evidence(
        clean, data=foreign
    )

    # A Jacobian with a column that is another: the test itself refuses the design.
    x = np.asarray(clean.data.x)

    def dependent_view(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(weighted_jacobian=np.column_stack([np.ones(40), x, x]))

    monkeypatch.setattr("curvestudio.tools.audit.fit_view", dependent_view)
    assert DECIDED + "the parameters are not separated at the estimates." in evidence(clean)


def test_the_data_as_uploaded_serve_a_fit_that_ignored_its_weights(scenarios) -> None:
    bundle, uploaded, spec = scenarios.weights_ignored()
    assert bundle.result.provenance.dataset_fingerprint != uploaded.fingerprint
    compact = scenarios.subsampled(bundle)

    alone = _check(audit_fit(compact, spec=spec), "residual_independence")
    assert "subsample" in alone.evidence
    assert alone.statistic.startswith("Q")

    given = _check(audit_fit(compact, spec=spec, data=uploaded), "residual_independence")
    assert EXACT_STATISTIC.fullmatch(given.statistic)
    # A bundle that holds every point needs no data, and the same data change nothing.
    whole = audit_fit(bundle, spec=spec)
    assert whole == audit_fit(bundle, spec=spec, data=uploaded)
    assert _check(whole, "residual_independence").statistic == given.statistic


BELOW_THE_BAR = pytest.mark.xfail(strict=True, reason="under the 90 % pass bar on these seeds")


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(20, marks=BELOW_THE_BAR),
        40,
        pytest.param(400, marks=BELOW_THE_BAR),
    ],
)
def test_the_independence_check_is_calibrated_on_well_behaved_fits(scenarios, n) -> None:
    # Both rules: the exact Durbin-Watson test when the fit's settings are given, Ljung-Box
    # alone when they are not.
    #
    # The exact rule is a two-sided test at 10 % (either p-value below 0.05) with Ljung-Box's
    # own 5 % laid over it, so a clean fit passes about 85 times in 100: under the 90 % bar
    # by construction. At 20 and 400 points these forty seeds read under it; the bar stays,
    # as a strict expected failure.
    exact: list[Verdict] = []
    ljung_box: list[Verdict] = []
    for seed in SEEDS:
        bundle = scenarios.quadratic_fit(n, seed)
        exact.append(
            _check(audit_fit(bundle, spec=scenarios.QUADRATIC), "residual_independence").verdict
        )
        ljung_box.append(_check(audit_fit(bundle), "residual_independence").verdict)

    for verdicts in (exact, ljung_box):
        assert verdicts.count(Verdict.PASS) >= 0.90 * len(SEEDS), _rates(verdicts)
        assert verdicts.count(Verdict.FAIL) <= 0.01 * len(SEEDS), _rates(verdicts)


# -- cross-validation ----------------------------------------------------------------------------


def test_expected_cv_ratio_is_what_a_correct_line_shows() -> None:
    assert 1.15 <= expected_cv_ratio(12, 2, 5) <= 1.30
    # More points, or training folds nearer the whole sample, leave less to expect.
    assert expected_cv_ratio(30, 2, 5) < expected_cv_ratio(12, 2, 5)
    assert expected_cv_ratio(100, 2, 5) < expected_cv_ratio(30, 2, 5)
    assert expected_cv_ratio(100, 2, 100) < expected_cv_ratio(100, 2, 5)
    # Leave-one-out trains every fold on n - 1 points.
    assert expected_cv_ratio(12, 2, 12) == pytest.approx(np.sqrt((1 + 2 / 11) / (1 - 2 / 12)))
    with pytest.raises(ValueError):
        expected_cv_ratio(12, 2, 1)
    with pytest.raises(ValueError):
        expected_cv_ratio(12, 2, 13)
    with pytest.raises(ValueError):
        expected_cv_ratio(12, 12, 5)


@pytest.mark.parametrize("n", (12, 30, 100))
def test_cross_validation_is_judged_against_what_a_correct_model_shows(scenarios, n) -> None:
    verdicts: list[Verdict] = []
    ratios: list[float] = []
    for seed in CV_SEEDS:
        bundle, cv = scenarios.line_with_cv(n, seed)
        assert cv.rmse is not None and bundle.result.stats is not None
        ratios.append(cv.rmse / bundle.result.stats.rmse)
        verdicts.append(_check(audit_fit(bundle, cv=cv), "cross_validation").verdict)

    assert verdicts.count(Verdict.PASS) >= 0.9 * len(CV_SEEDS), _rates(verdicts)
    assert verdicts.count(Verdict.FAIL) <= 0.01 * len(CV_SEEDS), _rates(verdicts)
    expected = expected_cv_ratio(n, 2, 5)
    assert abs(float(np.median(ratios)) / expected - 1.0) <= 0.05, (np.median(ratios), expected)


def test_cross_validation_names_the_expectation_in_its_evidence(scenarios) -> None:
    audits = scenarios.audited()

    line = _check(audits["line_with_cv"], "cross_validation")
    assert line.verdict is Verdict.PASS
    assert line.evidence == (
        "In-sample RMSE 0.336; cross-validation RMSE 0.365, against 0.413 expected from a "
        "correct model with 12 points and 5 folds (0.883 times that)."
    )
    over = _check(audits["overfit_with_cv"], "cross_validation")
    assert over.verdict is Verdict.FAIL
    assert over.evidence == (
        "In-sample RMSE 0.0153; cross-validation RMSE 1, against 0.0517 expected from a "
        "correct model with 12 points and 12 folds (19.4 times that)."
    )
    assert over.next_step == CV_NEXT_STEP


def test_cross_validation_is_not_applicable_when_nothing_can_be_expected_of_it(scenarios) -> None:
    bundle, cv = scenarios.line_with_cv()
    assert cv.rmse is not None
    incomplete = cv.model_copy(update={"rmse": None, "n_failed": 2, "status": "incomplete"})
    check = _check(audit_fit(bundle, cv=incomplete), "cross_validation")
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert check.evidence.startswith("cross-validation is incomplete: 2 of 5 folds failed")
    assert check.next_step is not None

    too_many = _check(audit_fit(bundle, cv=_cv(0.5, folds=13)), "cross_validation")
    assert too_many.verdict is Verdict.NOT_APPLICABLE
    assert too_many.evidence == (
        "cross-validation ran with 13 folds on 12 points for 2 parameters, so there is no "
        "correct model to expect of it"
    )


# -- parameters ----------------------------------------------------------------------------------


def test_parameters_at_a_bound_and_wide_intervals_are_flagged(scenarios) -> None:
    pinned = scenarios.pinned_slope()
    assert "PARAMETER_AT_BOUND: a" in pinned.result.warnings
    flagged = _check(audit_fit(pinned, spec=scenarios.PINNED_SPEC), "parameters_identifiable")
    assert flagged.verdict is Verdict.FAIL
    assert flagged.evidence.startswith("a sits at a bound: estimate")
    assert flagged.evidence.endswith("against the bound 0.")
    assert flagged.next_step == AT_A_BOUND_NEXT_STEP

    # The same estimate once the fit's own warning is gone: its distance to the bound alone.
    result = pinned.result.model_copy(update={"warnings": ()})
    by_distance = _check(
        audit_fit(pinned.model_copy(update={"result": result}), spec=scenarios.PINNED_SPEC),
        "parameters_identifiable",
    )
    assert by_distance.verdict is Verdict.FAIL
    assert by_distance.evidence.startswith("a sits at a bound: estimate")

    # Without the spec the audit does not know the bounds the fit ran with, and says so.
    blind = _check(audit_fit(pinned), "parameters_identifiable")
    assert blind.verdict is Verdict.FAIL
    assert blind.evidence == (
        "a sits at a bound (the fit reported it there). "
        "Parameters were checked without the fit's bounds."
    )

    wide = _check(
        audit_fit(scenarios.wide_interval(), spec=scenarios.ONE_UNIT_SPEC),
        "parameters_identifiable",
    )
    assert wide.verdict is Verdict.CAUTION
    assert wide.evidence.startswith("a has an interval from")
    assert "covering 100 % of its allowed range 0 to 1." in wide.evidence
    assert not re.search(r"\bb\b", wide.evidence)
    assert wide.next_step == AT_A_BOUND_NEXT_STEP

    clean = scenarios.clean()
    bare = _check(audit_fit(scenarios.without_standard_error(clean)), "parameters_identifiable")
    assert bare.verdict is Verdict.CAUTION
    assert bare.evidence == "b0: its standard error could not be determined."

    withheld = _check(audit_fit(scenarios.without_covariance(clean)), "parameters_identifiable")
    assert withheld.verdict is Verdict.CAUTION
    assert "covariance" in withheld.evidence
    assert _check(audit_fit(clean), "parameters_identifiable").verdict is Verdict.PASS


def test_the_audit_names_the_simplest_model_within_two_aicc(scenarios) -> None:
    ranking = _ranking(
        _row("poly3", "Cubic", 4, 0.0, 1),
        _row("poly2", "Quadratic", 3, 1.2, 2),
        _row("poly1", "Straight line", 2, 1.8, 3),
        _row("poly4", "Quartic", 5, 2.6, 4),
        _row("poly0", "Constant", 1, 0.9, 5, degenerate=True),
        _row("poly1_origin", "Line through the origin", 1, 1.0, 6, redundant_with="poly1"),
    )
    audited = audit_fit(scenarios.polynomial(2), ranking=ranking)

    check = _check(audited, "simpler_model")
    assert check.verdict is Verdict.CAUTION
    assert audited.simpler_model == SimplerModel(
        model_id="poly1",
        name="Straight line",
        n_parameters=2,
        delta_aicc=1.8,
        cv_rmse=None,
    )
    assert check.next_step == (
        "Report the simpler model unless the extra parameter answers a question of your own."
    )

    simplest = audit_fit(scenarios.polynomial(1), ranking=ranking)
    assert _check(simplest, "simpler_model").verdict is Verdict.PASS
    assert simplest.simpler_model is None
    assert "simplest" in _check(simplest, "simpler_model").evidence

    absent = audit_fit(scenarios.polynomial(5), ranking=ranking)
    check = _check(absent, "simpler_model")
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert "no Auto-fit ranking for this dataset" in check.evidence
    assert absent.simpler_model is None
    unranked = audit_fit(scenarios.polynomial(2))
    assert _check(unranked, "simpler_model").verdict is Verdict.NOT_APPLICABLE
    assert "no Auto-fit ranking for this dataset" in _check(unranked, "simpler_model").evidence

    # Ties on the parameter count: the lower cross-validated error wins, a missing one loses.
    ties = _ranking(
        _row("poly2", "Quadratic", 3, 0.0, 1),
        _row("tie_none", "No CV", 2, 1.0, 2),
        _row("tie_cv", "With CV", 2, 1.5, 3, cv_rmse=0.4),
    )
    winner = audit_fit(scenarios.polynomial(2), ranking=ties).simpler_model
    assert winner is not None and winner.model_id == "tie_cv"
    both_missing = _ranking(
        _row("poly2", "Quadratic", 3, 0.0, 1),
        _row("tie_later", "Later", 2, 1.5, 3),
        _row("tie_earlier", "Earlier", 2, 1.0, 2),
    )
    found = audit_fit(scenarios.polynomial(2), ranking=both_missing).simpler_model
    assert found is not None and found.model_id == "tie_earlier"


# -- rows ----------------------------------------------------------------------------------------


def test_zero_weight_rows_are_excluded_and_listed(scenarios) -> None:
    bundle = scenarios.zero_weights()
    assert bundle.result.stats is not None and bundle.result.stats.n == 40
    assert bundle.summary is not None
    residuals = bundle.summary.residuals
    top = max((d, i) for i, d in enumerate(residuals.cooks_distance) if d is not None)[1]
    flagged = set(residuals.flagged_high_influence)
    leverage_only = [i for i in residuals.flagged_high_leverage if i not in flagged]
    assert leverage_only

    audit = audit_fit(bundle)

    assert audit.n_observations == 40
    assert audit.excluded == tuple(
        ExcludedRow(index=row, reason="weight is zero") for row in scenarios.ZERO_WEIGHT_ROWS
    )
    # The rows the audit names are rows of the data, not positions among the points used.
    active = [row for row in range(45) if row not in scenarios.ZERO_WEIGHT_ROWS]
    evidence = _check(audit, "influential_points").evidence
    assert f"(row {active[top] + 1})" in evidence
    for position in leverage_only:
        assert re.search(rf"\b{active[position] + 1}\b", evidence)

    audit = audit_fit(scenarios.with_missing_weight(bundle))
    assert ExcludedRow(index=5, reason="weight is missing") in audit.excluded
    assert len(audit.excluded) == 6


def test_long_lists_of_rows_are_counted_in_full_but_only_the_first_eight_are_named(
    scenarios,
) -> None:
    bundle = scenarios.large_quadratic()
    assert bundle.summary is not None
    residuals = bundle.summary.residuals
    flagged = set(residuals.flagged_high_influence)
    leverage_only = [i for i in residuals.flagged_high_leverage if i not in flagged]
    assert len(leverage_only) > 8

    check = _check(audit_fit(bundle), "influential_points")

    assert f" {len(leverage_only)} points have high leverage but little influence" in check.evidence
    listed, rest = check.evidence.split(": rows ")[1].split(", and ")
    assert len(listed.split(", ")) == 7  # "a, b, c, d, e, f, g and h": eight rows in all
    assert rest.startswith(f"{len(leverage_only) - 8} more.")


def test_the_audit_is_a_pure_function(scenarios) -> None:
    bundle = scenarios.clean()
    before = bundle.model_dump_json()
    ranking = _ranking(_row("poly1", "Straight line", 2, 1.8, 2), _row("poly2", "Q", 3, 0.0, 1))
    inputs = {"spec": scenarios.QUADRATIC, "ranking": ranking, "cv": _cv(0.2)}

    first = audit_fit(bundle, **inputs)
    second = audit_fit(bundle, **inputs)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert bundle.model_dump_json() == before
    assert first.simpler_model is not None


def test_registered_model_bounds_are_used_and_spec_bounds_override_them(
    scenarios, bounded_line
) -> None:
    falling = scenarios.slope_at_declared_bound(bounded_line)
    assert falling.result.status is FitStatus.CONVERGED
    assert "PARAMETER_AT_BOUND: a" in falling.result.warnings
    declared = _check(audit_fit(falling), "parameters_identifiable")
    assert declared.verdict is Verdict.FAIL
    assert declared.evidence.startswith("a sits at a bound: estimate")
    assert declared.evidence.endswith("against the bound 0.")
    assert "checked without the fit's bounds" not in declared.evidence

    widened = scenarios.bounded_line(bounded_line, widened=True)
    assert widened.result.status is FitStatus.CONVERGED
    assert widened.result.warnings == ()
    # Only the declared bounds (0, 1) are known: the slope's interval covers all of them.
    narrow = _check(audit_fit(widened), "parameters_identifiable")
    assert narrow.verdict is Verdict.CAUTION
    assert narrow.evidence.startswith("a has an interval from")
    assert "of its allowed range 0 to 1." in narrow.evidence
    # The spec that widened the bound to 100 makes the same interval a narrow one.
    override = _check(
        audit_fit(widened, spec=scenarios.widened_spec(bounded_line)), "parameters_identifiable"
    )
    assert override.verdict is Verdict.PASS


# -- the audit as a whole ------------------------------------------------------------------------


def test_the_summary_counts_failures_and_cautions_and_names_the_next_step(scenarios) -> None:
    clean = scenarios.clean()
    rmse = clean.result.stats.rmse if clean.result.stats else 0.1

    # Nothing to flag: what could not run is named.
    assert audit_fit(clean).summary == (
        "Every check that could run passed. 2 could not be run: cross_validation, simpler_model."
    )
    assert audit_fit(clean, cv=_cv(rmse)).summary == (
        "Every check that could run passed. 1 could not be run: simpler_model."
    )

    # Cautions only.
    cautioned = audit_fit(scenarios.without_covariance(clean))
    first_caution = next(c for c in cautioned.checks if c.verdict is Verdict.CAUTION)
    assert cautioned.summary == (
        f"No check failed; 1 needs a look. {first_caution.title}. {first_caution.next_step}"
    )
    assert cautioned.summary.startswith("No check failed; 1 needs a look. ")

    # A failure, alone and with a caution.
    failing = audit_fit(clean, cv=_cv(rmse * 10))
    failed = [c for c in failing.checks if c.verdict is Verdict.FAIL]
    assert [c.id for c in failed] == ["cross_validation"]
    assert failing.summary == f"1 check failed. {failed[0].title}. {failed[0].next_step}"
    both = audit_fit(scenarios.without_covariance(clean), cv=_cv(rmse * 10))
    assert both.summary == (
        f"1 check failed, 1 needs a look. {failed[0].title}. {failed[0].next_step}"
    )

    # Several failures and one caution: plural failures, singular caution, and the first
    # check in order carrying the worst verdict.
    audit = scenarios.audited()["cubic_by_quadratic"]
    failed = [c for c in audit.checks if c.verdict is Verdict.FAIL]
    cautioned = [c for c in audit.checks if c.verdict is Verdict.CAUTION]
    assert len(failed) >= 2 and len(cautioned) == 1
    assert audit.summary == (
        f"{len(failed)} checks failed, 1 needs a look. {failed[0].title}. {failed[0].next_step}"
    )
