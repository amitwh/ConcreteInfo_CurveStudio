"""Declarative initial-guess rules."""

import math

import numpy as np
import pytest

from curvestudio.models import Parameter
from curvestudio.models.guess import GuessRule, guess_from_rules, rules_guess

ALL_RULES: tuple[GuessRule, ...] = (
    "y_max",
    "y_min",
    "y_range",
    "y_mean",
    "y_first",
    "y_last",
    "x_min",
    "x_max",
    "x_median",
    "x_span",
    "inv_x_span",
    "x_at_half_y",
)

BOUNDS = {
    "free": (-math.inf, math.inf),
    "positive": (0.0, math.inf),
    "negative": (-math.inf, 0.0),
    "boxed": (-1.0, 1.0),
    "tiny_box": (0.0, 1e-300),
}

DEGENERATE: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "empty": (np.array([]), np.array([])),
    "one_point": (np.array([2.0]), np.array([5.0])),
    "constant_y": (np.linspace(0, 1, 6), np.full(6, 3.0)),
    "constant_x": (np.full(6, 2.0), np.linspace(0, 1, 6)),
    "zeros": (np.zeros(5), np.zeros(5)),
    "nan_inf": (
        np.array([0.0, np.nan, 2.0, np.inf, 4.0]),
        np.array([1.0, 2.0, np.nan, 4.0, -np.inf]),
    ),
    "all_nan": (np.full(4, np.nan), np.full(4, np.nan)),
    "huge": (np.array([-1e300, 0.0, 1e300]), np.array([-1e300, 1e300, 1e300])),
    "tiny": (np.array([0.0, 1e-300, 2e-300]), np.array([0.0, 1e-300, 3e-300])),
    "length_mismatch": (np.arange(5.0), np.arange(3.0)),
}


@pytest.mark.parametrize("case", sorted(DEGENERATE))
def test_rules_guess_is_finite_inside_bounds_for_degenerate_data(case: str) -> None:
    x, y = DEGENERATE[case]
    rules: dict[str, GuessRule] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for rule in ALL_RULES:
        for bname, bound in BOUNDS.items():
            key = f"{rule}__{bname}"
            rules[key] = rule
            bounds[key] = bound
    rules["const"] = 3.5
    guess = rules_guess(rules, bounds)(x, y)
    assert set(guess) == set(rules)
    for name, value in guess.items():
        assert math.isfinite(value), (case, name)
        low, high = bounds.get(name, (-math.inf, math.inf))
        if low < high:
            assert low < value < high or (low == high), (case, name, value)


def test_guess_does_not_modify_inputs() -> None:
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([9.0, 1.0, 4.0])
    rules_guess({"a": "y_max"})(x, y)
    np.testing.assert_array_equal(x, [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(y, [9.0, 1.0, 4.0])


def test_each_rule_on_a_known_dataset() -> None:
    # Unsorted on purpose: first/last refer to the smallest/largest x.
    x = np.array([4.0, 0.0, 2.0, 1.0, 3.0])
    y = np.array([10.0, 2.0, 6.0, 4.0, 8.0])  # y = 2 + 2x, so the half level is 6 at x = 2
    expected = {
        "y_max": 10.0,
        "y_min": 2.0,
        "y_range": 8.0,
        "y_mean": 6.0,
        "y_first": 2.0,
        "y_last": 10.0,
        "x_min": 0.0,
        "x_max": 4.0,
        "x_median": 2.0,
        "x_span": 4.0,
        "inv_x_span": 0.25,
        "x_at_half_y": 2.0,
    }
    guess = rules_guess({f"p_{r}": r for r in expected})(x, y)
    for rule, value in expected.items():
        assert guess[f"p_{rule}"] == pytest.approx(value), rule


def test_x_at_half_y_interpolates_between_points() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])  # level 5 is reached half way between x = 1 and 2
    assert rules_guess({"c": "x_at_half_y"})(x, y)["c"] == pytest.approx(1.5)


def test_x_at_half_y_falls_back_to_median_when_never_crossed() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.full(3, 4.0)
    # constant y: the level equals every y, so the first segment "crosses" at its start
    assert math.isfinite(rules_guess({"c": "x_at_half_y"})(x, y)["c"])
    assert rules_guess({"c": "x_at_half_y"})(np.array([5.0]), np.array([1.0]))["c"] == 5.0


def test_literal_numbers_and_degenerate_scale_fallbacks() -> None:
    guess = rules_guess(
        {"k": 2.5, "span": "x_span", "inv": "inv_x_span", "rng": "y_range", "m": "y_mean"}
    )
    out = guess(np.full(4, 8.0), np.full(4, 3.0))
    assert out["k"] == 2.5
    assert out["span"] == 8.0  # constant x: fall back to max |x|
    assert out["inv"] == pytest.approx(0.125)
    assert out["rng"] == 3.0  # constant y: fall back to max |y|
    assert out["m"] == 3.0
    out = guess(np.zeros(3), np.zeros(3))
    assert out["span"] == 1.0
    assert out["inv"] == 1.0
    assert out["rng"] == 1.0
    empty = guess(np.array([]), np.array([]))
    assert empty["span"] == 1.0 and empty["m"] == 0.0


def test_guess_is_clipped_strictly_inside_bounds() -> None:
    guess = rules_guess(
        {"lo": "y_min", "hi": "y_max", "free": "y_max"},
        {"lo": (0.0, 10.0), "hi": (0.0, 10.0)},
    )
    out = guess(np.array([0.0, 1.0]), np.array([-5.0, 50.0]))
    assert 0.0 < out["lo"] < 1e-3
    assert 10.0 - 1e-3 < out["hi"] < 10.0
    assert out["free"] == 50.0


@pytest.mark.parametrize(
    ("rules", "bounds", "message"),
    [
        ({"a": "y_maxx"}, None, "unknown guess rule"),
        ({"a": float("nan")}, None, "finite number"),
        ({"a": float("inf")}, None, "finite number"),
        ({"a": "y_max"}, {"b": (0.0, 1.0)}, "no guess rule"),
        ({"a": "y_max"}, {"a": (2.0, 1.0)}, "lower above upper"),
    ],
)
def test_invalid_rules_are_rejected(
    rules: dict[str, GuessRule],
    bounds: dict[str, tuple[float, float]] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rules_guess(rules, bounds)


def test_guess_from_rules_takes_bounds_from_the_parameters() -> None:
    params = (
        Parameter("lo", lower=0.0, upper=10.0),
        Parameter("pos", lower=0.0),
        Parameter("free"),
    )
    guess = guess_from_rules(params, {"lo": "y_min", "pos": "y_min", "free": "y_max"})
    for case in sorted(DEGENERATE):
        out = guess(*DEGENERATE[case])
        assert 0.0 < out["lo"] < 10.0, case
        assert out["pos"] > 0.0, case
        assert all(math.isfinite(v) for v in out.values()), case
    out = guess(np.array([0.0, 1.0]), np.array([-5.0, 50.0]))
    assert 0.0 < out["lo"] < 1e-3  # y_min = -5 is moved inside (0, 10)
    assert out["free"] == 50.0


def test_guess_from_rules_requires_exactly_the_parameters() -> None:
    params = (Parameter("a"), Parameter("b"))
    with pytest.raises(ValueError, match="exactly the parameters"):
        guess_from_rules(params, {"a": "y_max"})
    with pytest.raises(ValueError, match="exactly the parameters"):
        guess_from_rules(params, {"a": "y_max", "b": 1.0, "c": 2.0})


def test_guess_from_rules_works_inside_a_model_spec() -> None:
    from curvestudio.models import ModelSpec

    params = (Parameter("vmax", lower=0.0), Parameter("km", lower=0.0))
    spec = ModelSpec(
        id="mm_rules",
        name="MM",
        category="saturation",
        expression="vmax*x/(km+x)",
        parameters=params,
        initial_guess=guess_from_rules(params, {"vmax": "y_max", "km": "x_at_half_y"}),
    )
    x = np.linspace(0.1, 20.0, 40)
    guess = spec.initial_guess(x, spec.evaluate(x, vmax=10.0, km=2.0))
    assert 8.0 < guess["vmax"] <= 10.0
    assert 0.0 < guess["km"] < 10.0
