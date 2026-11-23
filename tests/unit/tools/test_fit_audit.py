"""``audit_fit`` on real fits: every verdict is read from numbers the core already computed."""

import re

import pytest

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


def _check(audit: FitAudit, check_id: str) -> AuditCheck:
    return next(check for check in audit.checks if check.id == check_id)


def _cv(rmse: float) -> CrossValidation:
    """A complete five-fold cross-validation with the given pooled error."""
    return CrossValidation(
        folds=5,
        seed=0,
        rmse=rmse,
        fold_rmse=[rmse] * 5,
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
    audit = audit_fit(scenarios.clean())

    assert [check.id for check in audit.checks] == list(CHECK_IDS)
    for check in audit.checks:
        if check.id == "residual_trend":
            assert check.verdict in (Verdict.PASS, Verdict.CAUTION)
        elif check.id in ("cross_validation", "simpler_model"):
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
        ("one_outlier", "influential_points", Verdict.CAUTION, Verdict.PASS),
        ("three_outliers", "influential_points", Verdict.FAIL, Verdict.PASS),
        ("curve_fitted_by_line", "residual_trend", Verdict.FAIL, Verdict.PASS),
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
    assert audit_fit(scenarios.polynomial(2), ranking=ties).simpler_model is not None
    assert audit_fit(scenarios.polynomial(2), ranking=ties).simpler_model.model_id == "tie_cv"
    both_missing = _ranking(
        _row("poly2", "Quadratic", 3, 0.0, 1),
        _row("tie_later", "Later", 2, 1.5, 3),
        _row("tie_earlier", "Earlier", 2, 1.0, 2),
    )
    found = audit_fit(scenarios.polynomial(2), ranking=both_missing).simpler_model
    assert found is not None and found.model_id == "tie_earlier"


def test_zero_weight_rows_are_excluded_and_listed(scenarios) -> None:
    bundle = scenarios.zero_weights()
    assert bundle.result.stats is not None and bundle.result.stats.n == 40
    flagged = bundle.summary.residuals.flagged_high_influence if bundle.summary else []
    assert flagged

    audit = audit_fit(bundle)

    assert audit.n_observations == 40
    assert audit.excluded == tuple(
        ExcludedRow(index=row, reason="weight is zero") for row in scenarios.ZERO_WEIGHT_ROWS
    )
    # The rows the audit names are rows of the data, not positions among the points used.
    active = [row for row in range(45) if row not in scenarios.ZERO_WEIGHT_ROWS]
    cook_sentence = _check(audit, "influential_points").evidence.split(". ")[0]
    assert cook_sentence.startswith("2 points have a Cook's distance")
    for position in flagged:
        assert re.search(rf"\b{active[position] + 1}\b", cook_sentence)
    assert "rows" in cook_sentence

    audit = audit_fit(scenarios.with_missing_weight(bundle))
    assert ExcludedRow(index=5, reason="weight is missing") in audit.excluded
    assert len(audit.excluded) == 6


def test_many_flagged_points_are_counted_but_only_the_first_eight_are_listed(scenarios) -> None:
    bundle = scenarios.large_quadratic()
    assert bundle.summary is not None
    flagged = bundle.summary.residuals.flagged_high_influence
    assert len(flagged) > 8

    check = _check(audit_fit(bundle), "influential_points")

    assert check.verdict is Verdict.FAIL
    assert check.evidence.startswith(f"{len(flagged)} points have a Cook's distance")
    listed, rest = check.evidence.split(": rows ")[1].split(", and ")
    assert len(listed.split(", ")) == 7  # "a, b, c, d, e, f, g and h": eight rows in all
    assert rest.startswith(f"{len(flagged) - 8} more.")
    assert check.next_step is not None
    assert f", and {len(flagged) - 8} more. Refit without them" in check.next_step


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

    # Plural forms, with the first check in order carrying the worst verdict.
    audits = scenarios.audited()
    audit = audits["three_outliers"]
    failed = [c for c in audit.checks if c.verdict is Verdict.FAIL]
    cautioned = [c for c in audit.checks if c.verdict is Verdict.CAUTION]
    assert len(failed) >= 2 or len(cautioned) >= 2
    look = f", {len(cautioned)} need a look" if cautioned else ""
    assert (
        audit.summary
        == f"{len(failed)} checks failed{look}. {failed[0].title}. {failed[0].next_step}"
    )
