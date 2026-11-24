"""The words the audit produces: evidence never proof, advice where needed, a reason where not."""

from collections.abc import Iterator

import pytest

from curvestudio.discovery import AutoFitConfig, AutoFitResult, CandidateMetrics, RankedFit
from curvestudio.statistics import CrossValidation
from curvestudio.tools import FitAudit, Verdict, audit_fit
from curvestudio.tools._wording import claims_proof


def _words(audit: FitAudit) -> Iterator[tuple[str, str]]:
    for check in audit.checks:
        for field in ("title", "evidence", "explanation", "next_step"):
            yield f"{check.id}.{field}", getattr(check, field) or ""
    yield "summary", audit.summary
    yield "methods_fragment", audit.methods_fragment


def _cv(rmse: float) -> CrossValidation:
    return CrossValidation(
        folds=5, seed=0, rmse=rmse, fold_rmse=[rmse] * 5, n_failed=0, status="ok", reasons=[]
    )


def _ranking() -> AutoFitResult:
    def row(model_id: str, name: str, k: int, delta: float, rank: int) -> RankedFit:
        return RankedFit(
            rank=rank,
            model_id=model_id,
            name=name,
            metrics=CandidateMetrics(k=k, n=40, aicc=100.0 + delta),
            delta_aicc=delta,
        )

    return AutoFitResult(
        config=AutoFitConfig(),
        dataset_fingerprint="fp-audit",
        n_active=40,
        ranked=(
            row("poly3", "Cubic", 4, 0.0, 1),
            row("poly2", "Quadratic", 3, 1.2, 2),
            row("poly1", "Straight line", 2, 1.8, 3),
        ),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


@pytest.fixture
def audits(scenarios, bounded_line) -> dict[str, FitAudit]:
    """Every audit the two modules produce, with and without the optional inputs."""
    found = dict(scenarios.audited())
    found["slope_at_declared_bound"] = audit_fit(scenarios.slope_at_declared_bound(bounded_line))
    widened = scenarios.bounded_line(bounded_line, widened=True)
    found["declared_bounds"] = audit_fit(widened)
    found["widened_bounds"] = audit_fit(widened, spec=scenarios.widened_spec(bounded_line))
    found["with_ranking"] = audit_fit(scenarios.polynomial(2), ranking=_ranking())
    found["line_with_cv_and_ranking"] = audit_fit(scenarios.polynomial(1), ranking=_ranking())
    clean = scenarios.clean()
    found["clean_with_cv"] = audit_fit(clean, cv=_cv(0.1))
    found["clean_with_failing_cv"] = audit_fit(clean, cv=_cv(1.0))
    return found


def test_no_check_claims_proof(audits) -> None:
    assert claims_proof("This proves the fit") and claims_proof("Does not Confirm")
    assert claims_proof("Cross-validation validates the model")
    assert not claims_proof(
        "Do not delete data to improve a fit. Cross-validation was not run; the "
        "cross-validated error is 0.4."
    )

    claims = [
        (name, where, text)
        for name, audit in audits.items()
        for where, text in _words(audit)
        if claims_proof(text)
    ]
    assert claims == []
    for audit in audits.values():
        assert audit.summary and audit.methods_fragment
        for check in audit.checks:
            assert check.title and check.evidence and check.explanation


def test_every_failing_or_cautioned_check_says_what_to_do_next(audits) -> None:
    flagged = [
        (name, check)
        for name, audit in audits.items()
        for check in audit.checks
        if check.verdict in (Verdict.FAIL, Verdict.CAUTION)
    ]
    assert {check.verdict for _, check in flagged} == {Verdict.FAIL, Verdict.CAUTION}
    assert all(check.next_step for _, check in flagged)


def test_every_not_applicable_check_gives_its_reason(audits) -> None:
    skipped = [
        check
        for audit in audits.values()
        for check in audit.checks
        if check.verdict is Verdict.NOT_APPLICABLE
    ]
    assert len({check.id for check in skipped}) >= 8
    assert all(check.evidence.strip() for check in skipped)
    assert all(check.statistic is None for check in skipped)


def test_every_evidence_names_its_numbers(audits) -> None:
    with_threshold = [
        check.evidence
        for audit in audits.values()
        for check in audit.checks
        if "threshold" in check.evidence.lower()
    ]
    assert with_threshold
    assert all(any(character.isdigit() for character in evidence) for evidence in with_threshold)
    # A check that ran on numbers shows them; one that could not run shows none.
    for audit in audits.values():
        for check in audit.checks:
            if check.verdict is not Verdict.NOT_APPLICABLE and check.id != "converged":
                assert check.statistic is not None, check
