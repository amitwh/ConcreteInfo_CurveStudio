"""``fit_to_json``: the full bundle plus a report header, all standard JSON."""

import json
from typing import TYPE_CHECKING, Any

from curvestudio.workflow import FitBundle
from curvestudio_reporting import fit_to_json

if TYPE_CHECKING:
    from collections.abc import Callable


def reject_constant(token: str) -> Any:
    raise AssertionError(f"non-finite JSON constant {token!r} in the report")


def test_json_round_trips_the_whole_bundle(poly_bundle: Callable[..., FitBundle]) -> None:
    bundle = poly_bundle()
    text = fit_to_json(bundle)
    assert "NaN" not in text and "Infinity" not in text
    report = json.loads(text, parse_constant=reject_constant)
    assert FitBundle.model_validate(report["fit"]) == bundle


def test_header_identifies_tool_versions_time_convention_and_assumptions(
    poly_bundle: Callable[..., FitBundle],
) -> None:
    header = json.loads(fit_to_json(poly_bundle()))["report"]
    assert header["tool"] == "CurveStudio"
    versions = header["versions"]
    assert versions["curvestudio-core"] and versions["curvestudio-core"] != "unknown"
    assert "python" in versions
    generated_at: str = header["generated_at"]
    assert generated_at.endswith("+05:30"), "the timestamp must be IST (+05:30)"
    assert isinstance(header["convention"], str) and "AIC" in header["convention"]
    assert isinstance(header["assumptions"], str) and header["assumptions"]


def test_json_for_a_non_converged_fit_says_so(non_converged_bundle: FitBundle) -> None:
    report = json.loads(fit_to_json(non_converged_bundle))
    assert report["fit"]["result"]["status"] == "not_converged"
    assert report["report"]["convention"] is None


def test_json_is_valid_for_a_failed_fit(failed_bundle: FitBundle) -> None:
    text = fit_to_json(failed_bundle)
    report = json.loads(text, parse_constant=reject_constant)
    assert report["fit"]["result"]["status"] == "failed"
    assert report["fit"]["curve"] is None
