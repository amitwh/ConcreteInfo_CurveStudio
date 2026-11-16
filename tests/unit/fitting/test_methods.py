"""Tests for method/loss compatibility checks, ``auto`` resolution and FitConfig validation."""

import math

import pytest

from curvestudio.fitting import FitConfig, check_method_compatibility
from curvestudio.fitting.methods import resolve_method


@pytest.mark.parametrize("method", ["auto", "lm", "trf", "dogbox"])
def test_plain_unbounded_least_squares_is_allowed_for_every_method(method: str) -> None:
    check_method_compatibility(method, "linear", has_bounds=False, has_jacobian=True)


@pytest.mark.parametrize("method", ["auto", "trf", "dogbox"])
@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
def test_bounds_and_robust_losses_are_allowed_for_trf_and_dogbox(method: str, loss: str) -> None:
    check_method_compatibility(method, loss, has_bounds=True, has_jacobian=True)


def test_lm_with_bounds_is_rejected_naming_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        check_method_compatibility("lm", "linear", has_bounds=True, has_jacobian=True)


def test_lm_with_robust_loss_is_rejected_naming_loss() -> None:
    with pytest.raises(ValueError, match="loss"):
        check_method_compatibility("lm", "huber", has_bounds=False, has_jacobian=True)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method 'simplex'"):
        check_method_compatibility("simplex", "linear", has_bounds=False, has_jacobian=True)


def test_unknown_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown loss 'l1'"):
        check_method_compatibility("trf", "l1", has_bounds=False, has_jacobian=True)


def test_missing_jacobian_is_rejected_naming_jacobian() -> None:
    with pytest.raises(ValueError, match="jacobian"):
        check_method_compatibility("trf", "linear", has_bounds=False, has_jacobian=False)


@pytest.mark.parametrize(
    ("method", "loss", "has_bounds", "expected"),
    [
        ("auto", "linear", False, "lm"),
        ("auto", "linear", True, "trf"),
        ("auto", "huber", False, "trf"),
        ("dogbox", "linear", False, "dogbox"),
        ("lm", "linear", False, "lm"),
    ],
)
def test_resolve_method(method: str, loss: str, has_bounds: bool, expected: str) -> None:
    assert resolve_method(method, loss, has_bounds) == expected


def test_fit_config_defaults_match_the_contract() -> None:
    cfg = FitConfig()
    assert (cfg.method, cfg.loss, cfg.n_starts, cfg.seed) == ("auto", "linear", 1, 0)
    assert cfg.initial is None and cfg.bounds is None and cfg.max_nfev is None
    assert (cfg.ftol, cfg.xtol, cfg.gtol, cfg.confidence_level) == (1e-10, 1e-10, 1e-10, 0.95)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_starts": 0}, "n_starts"),
        ({"seed": -1}, "seed"),
        ({"max_nfev": 0}, "max_nfev"),
        ({"ftol": 0.0}, "ftol"),
        ({"gtol": math.nan}, "gtol"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"initial": {"a": math.inf}}, "initial value for 'a'"),
        ({"bounds": {"a": (2.0, 1.0)}}, "lower < upper"),
        ({"bounds": {"a": (1.0, 1.0)}}, "lower < upper"),
        ({"bounds": {"a": (math.nan, 1.0)}}, "lower < upper"),
    ],
)
def test_fit_config_rejects_bad_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FitConfig(**kwargs)  # type: ignore[arg-type]
