"""The special-function whitelist: values, analytic Jacobians, Voigt profile and arity checks."""

import time

import mpmath
import numpy as np
import pytest
import scipy.special as sp
import sympy

from curvestudio.expressions import to_sympy, validate_expression
from curvestudio.expressions.grammar import FUNCTION_ARITY
from curvestudio.models import ModelSpec, Parameter


def _symbols(*names: str) -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, real=True) for name in names}


_X = np.array([0.1, 0.35, 0.6, 0.85])
_NAMES = frozenset({"x", "a", "b", "c"})


def _spec(expression: str, parameters: tuple[str, ...] = ("a",)) -> ModelSpec:
    return ModelSpec(
        id="function_probe",
        name="Function probe",
        category="harmonic",
        expression=expression,
        parameters=tuple(Parameter(name) for name in parameters),
        initial_guess=lambda x, y: dict.fromkeys(parameters, 0.5),
    )


# (expression, parameter values, reference computed with NumPy / SciPy directly)
_VALUE_CASES = [
    ("tanh(a*x)", {"a": 1.7}, lambda x: np.tanh(1.7 * x)),
    ("asin(a*x)", {"a": 0.9}, lambda x: np.arcsin(0.9 * x)),
    ("acos(a*x)", {"a": 0.9}, lambda x: np.arccos(0.9 * x)),
    ("atan(a*x)", {"a": 3.0}, lambda x: np.arctan(3.0 * x)),
    ("erf(a*x)", {"a": 2.0}, lambda x: sp.erf(2.0 * x)),
    ("erfc(a*x)", {"a": 2.0}, lambda x: sp.erfc(2.0 * x)),
    ("abs(a - x)", {"a": 0.5}, lambda x: np.abs(0.5 - x)),
    ("Abs(a - x)", {"a": 0.5}, lambda x: np.abs(0.5 - x)),
    ("pow(x, a)", {"a": 2.5}, lambda x: x**2.5),
    ("pow(a, x)", {"a": 2.5}, lambda x: 2.5**x),
    ("min(x, a)", {"a": 0.5}, lambda x: np.minimum(x, 0.5)),
    ("max(x, a)", {"a": 0.5}, lambda x: np.maximum(x, 0.5)),
    ("atan2(a*x, 1 - x)", {"a": 2.0}, lambda x: np.arctan2(2.0 * x, 1.0 - x)),
]


@pytest.mark.parametrize(
    ("expression", "params", "reference"), _VALUE_CASES, ids=lambda v: str(v)[:24]
)
def test_each_new_function_matches_numpy_or_scipy_values(
    expression: str, params: dict[str, float], reference: object
) -> None:
    spec = _spec(expression)
    assert callable(reference)
    np.testing.assert_allclose(spec.evaluate(_X, **params), reference(_X), rtol=1e-12)


def test_voigt_value_is_the_area_normalised_scipy_profile() -> None:
    spec = _spec("a*voigt(x - b, c, 0.3)", ("a", "b", "c"))
    got = spec.evaluate(_X, a=2.0, b=0.4, c=0.2)
    np.testing.assert_allclose(got, 2.0 * sp.voigt_profile(_X - 0.4, 0.2, 0.3), rtol=1e-12)


# Parameter values chosen away from the min/max/abs kinks (x grid is 0.1..0.85).
_JACOBIAN_CASES = [
    ("tanh(a*x)", {"a": 1.7}),
    ("asin(a*x)", {"a": 0.9}),
    ("acos(a*x)", {"a": 0.9}),
    ("atan(a*x)", {"a": 3.0}),
    ("erf(a*x)", {"a": 2.0}),
    ("erfc(a*x)", {"a": 2.0}),
    ("abs(a - x)", {"a": 0.97}),
    ("pow(x, a)", {"a": 2.5}),
    ("pow(a, x)", {"a": 2.5}),
    ("min(a*x, 0.5)", {"a": 1.0}),
    ("max(a*x, 0.5)", {"a": 1.0}),
    ("atan2(a*x, 1 - x)", {"a": 2.0}),
]


@pytest.mark.parametrize(("expression", "params"), _JACOBIAN_CASES, ids=lambda v: str(v)[:24])
def test_jacobian_matches_finite_differences_for_each_function(
    expression: str, params: dict[str, float]
) -> None:
    spec = _spec(expression)
    # x = 0.1, 0.35 and 0.6 sit on one side of the 0.5 kink of min/max for a=1 (0.6 is above);
    # drop the point next to it so the central difference never straddles the kink.
    x = np.array([0.1, 0.25, 0.7, 0.85]) if "min" in expression or "max" in expression else _X
    step = 1e-6
    (name,) = params
    up = spec.evaluate(x, **{name: params[name] + step})
    down = spec.evaluate(x, **{name: params[name] - step})
    numeric = (up - down) / (2.0 * step)
    np.testing.assert_allclose(spec.jacobian(x, **params)[:, 0], numeric, rtol=1e-6, atol=1e-9)


def _voigt_reference(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    return np.asarray(sp.voigt_profile(x, sigma, gamma))


def test_voigt_limits() -> None:
    x = np.linspace(-3.0, 3.0, 25)
    tiny = 1e-9  # 1e-6 leaves a Lorentzian tail above rtol=1e-6 in the Gaussian wings
    # sigma -> 0: Lorentzian with scale gamma.
    gamma = 0.8
    lorentzian = gamma / np.pi / (x**2 + gamma**2)
    np.testing.assert_allclose(
        _spec("voigt(x, a, b)", ("a", "b")).evaluate(x, a=tiny * gamma, b=gamma),
        lorentzian,
        rtol=1e-6,
    )
    # gamma -> 0: Gaussian with standard deviation sigma.
    sigma = 0.7
    gaussian = np.exp(-(x**2) / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(
        _spec("voigt(x, a, b)", ("a", "b")).evaluate(x, a=sigma, b=tiny * sigma),
        gaussian,
        rtol=1e-6,
    )


def test_voigt_derivatives_against_finite_differences() -> None:
    spec = _spec("voigt(x - a, b, c)", ("a", "b", "c"))
    x = np.linspace(-2.0, 2.0, 21)
    point = {"a": 0.3, "b": 0.6, "c": 0.45}
    jac = spec.jacobian(x, **point)
    for column, name in enumerate(point):
        step = 1e-6
        up = spec.evaluate(x, **{**point, name: point[name] + step})
        down = spec.evaluate(x, **{**point, name: point[name] - step})
        np.testing.assert_allclose(jac[:, column], (up - down) / (2.0 * step), rtol=1e-6, atol=1e-9)


def test_voigt_derivative_with_respect_to_x_through_the_chain_rule() -> None:
    # d/da of voigt(a*x, 1, 1) = x * V_x(a*x): exercises the x-argument derivative.
    spec = _spec("voigt(a*x, 0.5, 0.4)")
    x = np.linspace(-2.0, 2.0, 21)
    step = 1e-6
    numeric = (spec.evaluate(x, a=1.5 + step) - spec.evaluate(x, a=1.5 - step)) / (2.0 * step)
    np.testing.assert_allclose(spec.jacobian(x, a=1.5)[:, 0], numeric, rtol=1e-6, atol=1e-9)


def test_voigt_with_no_width_at_its_centre_is_non_finite_not_an_exception() -> None:
    spec = _spec("voigt(x, a, b)", ("a", "b"))
    centre = np.array([0.0])  # sigma = gamma = 0 is a delta function: infinite at x = 0
    assert not np.any(np.isfinite(spec.evaluate(centre, a=0.0, b=0.0)))
    assert not np.all(np.isfinite(spec.jacobian(centre, a=0.0, b=0.0)))


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("min(x)", "exactly 2"),
        ("max(x, 1, 2)", "exactly 2"),
        ("pow(x, 2, 3)", "exactly 2"),
        ("atan2(x)", "exactly 2"),
        ("voigt(x, 1)", "exactly 3"),
        ("voigt(x, 1, 2, 3)", "exactly 3"),
        ("tanh(x, 1)", "exactly 1"),
        ("erf()", "exactly 1"),
    ],
)
def test_wrong_arity_is_rejected(text: str, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        validate_expression(text, _NAMES)


def test_bare_function_names_say_how_to_call_them() -> None:
    with pytest.raises(ValueError, match=r"voigt\(x, y, z\)"):
        validate_expression("voigt + x", _NAMES)
    with pytest.raises(ValueError, match=r"min\(x, y\)"):
        validate_expression("min + x", _NAMES)


@pytest.mark.parametrize(
    "text",
    [
        "pow(x, 1e9)",
        "pow(x, 1001)",
        "pow(x, -1001)",
        "pow(2, 9**9)",
        "pow(10, 400)",
        "pow(pow(10, 200), 2)",
    ],
)
def test_pow_exponent_cap_applies_to_pow_calls(text: str) -> None:
    with pytest.raises(ValueError):
        validate_expression(text, _NAMES)


def test_pow_calls_within_the_cap_are_accepted() -> None:
    validate_expression("pow(x, 1000) + pow(x, -1000) + pow(2, 10)", _NAMES)


def test_arity_table_covers_every_function() -> None:
    assert FUNCTION_ARITY["tanh"] == 1
    assert FUNCTION_ARITY["abs"] == 1
    assert FUNCTION_ARITY["pow"] == 2
    assert FUNCTION_ARITY["atan2"] == 2
    assert FUNCTION_ARITY["voigt"] == 3


# -- constant folding of function calls: exponent cap cannot be bypassed ---------------------


@pytest.mark.parametrize(
    "text",
    [
        "pow(9, max(30000000, 1))",
        "9**max(30000000, 1)",
        "pow(9, abs(-999999999))",
        "pow(9, Abs(-999999999))",
        "pow(9, min(30000000, 40000000))",
        "9**sqrt(9e14)",
        "x**exp(20)",
        "pow(x, sqrt(1e8))",
        "pow(x, atan2(1, 0)*1e9)",
        "9**(max(1, 2)*10**8)",
    ],
)
def test_exponent_cap_cannot_be_bypassed_through_function_calls(text: str) -> None:
    start = time.perf_counter()
    with pytest.raises(ValueError, match=r"exponent|not finite|too large"):
        validate_expression(text, _NAMES)
    with pytest.raises(ValueError):
        to_sympy(text, _symbols("x", "a", "b", "c"))
    assert time.perf_counter() - start < 1.0


def test_exponent_cap_rejection_uses_the_grammars_own_message() -> None:
    with pytest.raises(ValueError) as caught:
        to_sympy("pow(9, max(30000000, 1))", _symbols("x"))
    assert "4300" not in str(caught.value)
    assert "exponent" in str(caught.value)


@pytest.mark.parametrize("text", ["((x**1000)**1000)**1000", "(x**999)**999", "x**600*x**600"])
def test_constant_exponents_that_multiply_past_the_cap_are_rejected(text: str) -> None:
    start = time.perf_counter()
    with pytest.raises(ValueError, match="exponent"):
        to_sympy(text, _symbols("x"))
    assert time.perf_counter() - start < 1.0


def test_exponents_within_the_cap_still_build() -> None:
    expr = to_sympy("(x**30)**30 + x**1000 + pow(2, max(3, 4)) + 9**abs(-5)", _symbols("x"))
    assert expr.has(sympy.Symbol("x", real=True))


# -- domain errors of constant calls are refused at build ---------------------------------


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("asin(2)", "asin"),
        ("acos(-2)", "acos"),
        ("min(x, asin(2))", "asin"),
        ("max(x, acos(1.5)) + a", "acos"),
        ("ln(-1)", "ln"),
        ("log10(0)", "log10"),
        ("sqrt(-4) + x", "sqrt"),
        ("exp(1000)", "exp"),
        ("x*atan2(0, 0) + a*0 + log(0)", "log"),
    ],
)
def test_constant_calls_outside_their_domain_are_refused_naming_the_function(
    text: str, needle: str
) -> None:
    with pytest.raises(ValueError, match=needle):
        to_sympy(text, _symbols("x", "a"))
    with pytest.raises(ValueError, match=needle):
        validate_expression(text, _NAMES)


def test_constant_calls_inside_their_domain_are_accepted() -> None:
    for text in ("asin(0.5) + x", "acos(-1)*x", "atan2(0, 0) + x", "sqrt(2)*x", "exp(-1000) + x"):
        validate_expression(text, _NAMES)


# -- nested min / max: broadcast-safe derivatives ------------------------------------------

_XS = np.array([-1.0, 0.2, 0.9, 1.4, 2.5])


@pytest.mark.parametrize(
    ("expression", "reference"),
    [
        ("min(max(x, a), b)", lambda x, a, b: np.minimum(np.maximum(x, a), b)),
        ("max(x, min(a, b))", lambda x, a, b: np.maximum(x, np.minimum(a, b))),
        ("min(x, min(a, b))", lambda x, a, b: np.minimum(x, np.minimum(a, b))),
        (
            "max(min(x, a), min(x, b))",
            lambda x, a, b: np.maximum(np.minimum(x, a), np.minimum(x, b)),
        ),
    ],
)
def test_nested_min_max_evaluate_and_have_correct_jacobians(
    expression: str, reference: object
) -> None:
    spec = _spec(expression, ("a", "b"))
    point = {"a": 0.55, "b": 1.7}
    assert callable(reference)
    np.testing.assert_allclose(spec.evaluate(_XS, **point), reference(_XS, 0.55, 1.7))
    jac = spec.jacobian(_XS, **point)
    assert jac.shape == (5, 2)
    step = 1e-6
    for column, name in enumerate(point):
        up = spec.evaluate(_XS, **{**point, name: point[name] + step})
        down = spec.evaluate(_XS, **{**point, name: point[name] - step})
        np.testing.assert_allclose(jac[:, column], (up - down) / (2 * step), atol=1e-8)


def test_clamp_derivatives_select_the_active_branch() -> None:
    spec = _spec("min(max(x, a), b)", ("a", "b"))
    jac = spec.jacobian(np.array([-1.0, 0.8, 3.0]), a=0.0, b=2.0)
    np.testing.assert_array_equal(jac, [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


def test_min_max_ties_split_the_derivative_equally() -> None:
    low = _spec("min(a, b)", ("a", "b")).jacobian(np.array([0.0, 1.0]), a=1.5, b=1.5)
    high = _spec("max(a, b)", ("a", "b")).jacobian(np.array([0.0, 1.0]), a=1.5, b=1.5)
    np.testing.assert_array_equal(low, np.full((2, 2), 0.5))
    np.testing.assert_array_equal(high, np.full((2, 2), 0.5))
    assert _spec("min(a, b)", ("a", "b")).evaluate(np.array([0.0]), a=1.5, b=1.5)[0] == 1.5


def test_abs_derivative_at_zero_is_zero() -> None:
    spec = _spec("abs(x - a)")
    np.testing.assert_array_equal(spec.jacobian(np.array([0.5]), a=0.5), [[0.0]])
    np.testing.assert_array_equal(spec.jacobian(np.array([0.2, 0.9]), a=0.5), [[1.0], [-1.0]])


def test_min_max_of_constants_fold() -> None:
    assert to_sympy("min(2, 3) + max(2, 3)", _symbols("x")) == 5
    assert _spec("min(x, a) + max(x, a)").latex().startswith("y = ")


# -- user names can never collide with names printed into compiled code -------------------


def test_parameter_names_that_match_compiled_helper_names_are_harmless() -> None:
    names = (
        "reduce",
        "maximum",
        "minimum",
        "select",
        "greater",
        "less",
        "equal",
        "arctan2",
        "numpy",
        "scipy",
        "math",
        "sign",
    )
    # 'sign' is reserved; the others must work as ordinary parameter names.
    usable = tuple(n for n in names if n != "sign")
    spec = _spec(
        "max(x, reduce)*select + min(greater, less)*x + atan2(equal, maximum) + minimum*arctan2"
        " + numpy*scipy + math*abs(x - reduce)",
        usable,
    )
    point = dict(
        zip(usable, [0.4, 1.1, 0.7, 0.9, 1.3, 0.8, 1.6, 0.6, 0.35, 0.45, 0.25], strict=True)
    )
    got = spec.evaluate(_XS, **point)
    p = point
    expected = (
        np.maximum(_XS, p["reduce"]) * p["select"]
        + min(p["greater"], p["less"]) * _XS
        + np.arctan2(p["equal"], p["maximum"])
        + p["minimum"] * p["arctan2"]
        + p["numpy"] * p["scipy"]
        + p["math"] * np.abs(_XS - p["reduce"])
    )
    np.testing.assert_allclose(got, expected)
    jac = spec.jacobian(_XS, **point)
    assert jac.shape == (5, len(usable))
    assert np.all(np.isfinite(jac))


# -- voigt derivatives against a 50-digit mpmath reference ---------------------------------


def _mp_voigt(x: object, sigma: object, gamma: object) -> object:
    z = (x + 1j * gamma) / (sigma * mpmath.sqrt(2))
    return mpmath.re(mpmath.exp(-z * z) * mpmath.erfc(-1j * z)) / (
        sigma * mpmath.sqrt(2 * mpmath.pi)
    )


_SCALES = [1e-4, 1e-2, 1.0, 1e2, 1e4]


def _voigt_cases() -> list[tuple[float, float, float]]:
    cases = [
        (offset * gamma, sigma, gamma)
        for sigma in _SCALES
        for gamma in _SCALES
        for offset in (0.0, 3.0, -3.0, 50.0, -50.0)
    ]
    # Cases a reviewer found wrong: sigma far below gamma, and sigma at the 1e-8 extreme.
    cases += [(1.0, 1e-4, 1.0), (0.0, 1e-8, 1e-4), (3.0, 1e-8, 1.0), (-3.0, 1e-8, 1.0)]
    cases += [(0.3, 0.5, 0.4), (0.0, 1e8, 1.0), (2.0, 1e-8, 1e-8)]
    return cases


@pytest.mark.parametrize(("x", "sigma", "gamma"), _voigt_cases())
def test_voigt_value_and_partials_match_a_high_precision_reference(
    x: float, sigma: float, gamma: float
) -> None:
    mpmath.mp.dps = 50
    point = (mpmath.mpf(x), mpmath.mpf(sigma), mpmath.mpf(gamma))
    value = float(_mp_voigt(*point))
    reference = [
        float(mpmath.diff(_mp_voigt, point, order)) for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    scale = max(abs(r) for r in reference)
    spec = _spec("voigt(x, a, b)", ("a", "b"))
    xs = np.array([x])
    got_value = spec.evaluate(xs, a=sigma, b=gamma)[0]
    jac = spec.jacobian(xs, a=sigma, b=gamma)[0]
    np.testing.assert_allclose(got_value, value, rtol=1e-6, atol=1e-10 * abs(value))
    np.testing.assert_allclose(jac[0], reference[1], rtol=1e-6, atol=1e-10 * scale)
    np.testing.assert_allclose(jac[1], reference[2], rtol=1e-6, atol=1e-10 * scale)


@pytest.mark.parametrize(("x", "sigma", "gamma"), _voigt_cases())
def test_voigt_x_partial_matches_a_high_precision_reference(
    x: float, sigma: float, gamma: float
) -> None:
    mpmath.mp.dps = 50
    point = (mpmath.mpf(x), mpmath.mpf(sigma), mpmath.mpf(gamma))
    reference = [
        float(mpmath.diff(_mp_voigt, point, order)) for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    scale = max(abs(r) for r in reference)
    # d/da voigt(a, sigma, gamma) at a = x is the x-partial.
    spec = _spec("voigt(a, b, c)", ("a", "b", "c"))
    got = spec.jacobian(np.array([0.0]), a=x, b=sigma, c=gamma)[0, 0]
    np.testing.assert_allclose(got, reference[0], rtol=1e-6, atol=1e-10 * scale)
