"""Profile-likelihood intervals on NIST StRD problems with certified values."""

import pytest
from nist_harness import start_values, xy_from_problem
from nist_models import model_spec, problem

from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.statistics import profile_likelihood


@pytest.mark.parametrize("name", ["Misra1a", "BoxBOD"])
def test_profile_intervals_contain_the_certified_values_and_stay_near_the_wald_width(
    name: str,
) -> None:
    p = problem(name)
    config = FitConfig(initial=start_values(p, "start2"))
    data = xy_from_problem(p)
    spec = model_spec(name)
    result = fit(spec, data, config)
    assert result.status is FitStatus.CONVERGED, result.termination_reason

    profile = profile_likelihood(data, spec, result, config)

    for interval, estimate in zip(profile.intervals, result.parameters, strict=True):
        assert interval.low_status == interval.high_status == "found"
        assert interval.ci_low is not None and interval.ci_high is not None
        certified = p.certified[interval.name]
        assert interval.ci_low <= certified <= interval.ci_high
        assert estimate.ci_low is not None and estimate.ci_high is not None
        wald_width = estimate.ci_high - estimate.ci_low
        assert interval.ci_high - interval.ci_low <= 3.0 * wald_width
