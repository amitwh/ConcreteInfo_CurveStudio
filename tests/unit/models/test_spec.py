"""Tests for ModelSpec: parsing safety, evaluation, Jacobian, text rendering, contract."""

import threading
from dataclasses import dataclass

import numpy as np
import pytest

from curvestudio.models import ModelExample, ModelSpec, Parameter, format_equation
from curvestudio.models.starter import STARTER_MODELS, register_starter_models
from curvestudio.testing.model_contract import check_model_contract


def _guess(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"a": float(np.max(np.abs(y))) or 1.0, "b": 0.1}


def make_spec(expression: str = "a*exp(b*x)", **overrides: object) -> ModelSpec:
    kwargs: dict[str, object] = {
        "id": "exponential",
        "name": "Exponential",
        "category": "growth",
        "expression": expression,
        "parameters": (Parameter("a", lower=0.0), Parameter("b")),
        "initial_guess": _guess,
    }
    kwargs.update(overrides)
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


def test_evaluate_matches_hand_computation() -> None:
    spec = make_spec()
    x = np.array([0.0, 1.0])
    got = spec.evaluate(x, a=2.0, b=0.5)
    np.testing.assert_allclose(got, 2.0 * np.exp(0.5 * x))
    assert got.dtype == np.float64


def test_jacobian_matches_hand_derivation() -> None:
    spec = make_spec()
    x = np.array([0.0, 1.0, 2.5])
    a, b = 2.0, 0.5
    jac = spec.jacobian(x, a=a, b=b)
    assert jac.shape == (3, 2)
    np.testing.assert_allclose(jac[:, 0], np.exp(b * x))
    np.testing.assert_allclose(jac[:, 1], a * x * np.exp(b * x))


def test_constant_jacobian_column_is_broadcast_to_n_rows() -> None:
    spec = make_spec("a + b*x")
    jac = spec.jacobian(np.array([1.0, 2.0, 3.0]), a=1.0, b=2.0)
    np.testing.assert_allclose(jac[:, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(jac[:, 1], [1.0, 2.0, 3.0])


def test_constant_expression_is_broadcast() -> None:
    spec = make_spec("a", parameters=(Parameter("a"),))
    np.testing.assert_allclose(spec.evaluate(np.array([1.0, 2.0]), a=3.0), [3.0, 3.0])


def test_equation_text_substitutes_parameters() -> None:
    assert make_spec().equation_text({"a": 2.0, "b": 0.5}) == "y = 2*exp(0.5*x)"


def test_equation_text_keeps_structure_and_uses_significant_digits() -> None:
    spec = make_spec("a*(1 - exp(-b*x))")
    text = spec.equation_text({"a": 2.389423456, "b": 0.000550156789}, digits=6)
    assert text == "y = 2.38942*(1 - exp(-0.000550157*x))"


def test_equation_text_parenthesises_negative_values() -> None:
    assert make_spec("a - b*x").equation_text({"a": 1.0, "b": -2.0}) == "y = 1 - (-2)*x"


def test_equation_text_does_not_substitute_inside_substituted_numbers() -> None:
    spec = make_spec("e*x + f", parameters=(Parameter("e"), Parameter("f")))
    assert spec.equation_text({"e": 1e-7, "f": 3.0}) == "y = 1e-07*x + 3"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("b*1e-3*x + e*x", "y = 2*1e-3*x + 0.5*x"),
        ("b*2.5e3*x + e", "y = 2*2.5e3*x + 0.5"),
        ("b*.5e-2*x + e", "y = 2*.5e-2*x + 0.5"),
        ("b*1E+3*x + e", "y = 2*1E+3*x + 0.5"),
        ("b*3.*x + e", "y = 2*3.*x + 0.5"),
    ],
)
def test_equation_text_keeps_numeric_literals_when_a_parameter_is_named_e(
    expression: str, expected: str
) -> None:
    spec = make_spec(expression, parameters=(Parameter("b"), Parameter("e")))
    assert spec.equation_text({"b": 2.0, "e": 0.5}) == expected


def test_equation_text_keeps_literals_without_an_e_parameter() -> None:
    assert make_spec("a*1e-3*x + b").equation_text({"a": 2.0, "b": 1.0}) == "y = 2*1e-3*x + 1"


def test_equation_text_requires_every_parameter() -> None:
    with pytest.raises(ValueError, match="b"):
        make_spec().equation_text({"a": 1.0})


def test_format_equation_uses_caret_powers_and_keeps_numbers() -> None:
    """The user's exact fitted polynomial: every ``**`` shows as ``^``."""
    text = (
        "y = 0.680918 + 2.37953*x + 0.0900383*x**2 + 0.0127839*x**3 "
        "+ (-0.000548964)*x**4 + 1.07529e-05*x**5"
    )
    formatted = format_equation(text)
    for power in ("x^2", "x^3", "x^4", "x^5"):
        assert power in formatted
    assert "1.07529e-05" in formatted  # scientific notation is untouched
    assert "*x" in formatted  # single products survive
    assert "**" not in formatted


def test_format_equation_leaves_single_stars_alone() -> None:
    assert format_equation("y = 1e-05*x + 2*exp(-0.5*x)") == "y = 1e-05*x + 2*exp(-0.5*x)"


def test_format_equation_empty_or_none_renders_empty() -> None:
    assert format_equation(None) == ""
    assert format_equation("") == ""


def test_latex_is_non_empty() -> None:
    assert make_spec().latex() == "y = a e^{b x}"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('x')",
        "a*foo(x)",
        "a.__class__",
        "'s'",
        "(lambda: 1)()",
        "a[0]",
        "a if x else b",
        "[a for a in x]",
        "exp(a, b)",
        "exp(x=1)",
        "exp(*x)",
        "exp*x",
        "a(x)",
        "a**b**c",
        "9**9**9",
        "x**((-8)**0.5)",
        "x**(1/0)",
        "a << 2",
        "a % 2",
        "a // 2",
        "1j*x",
        "True*x",
        "not a",
        "a ^ 2",
        "a +",
        "",
        "__builtins__",
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(ValueError):
        make_spec(expression, parameters=(Parameter("a"), Parameter("b")))


def test_rejected_expression_never_executes() -> None:
    marker: list[str] = []
    import builtins

    original = builtins.__import__

    def spy(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "os":
            marker.append(name)
        return original(name, *args, **kwargs)  # type: ignore[arg-type]

    builtins.__import__ = spy  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError):
            make_spec("__import__('os').system('true')")
    finally:
        builtins.__import__ = original
    assert marker == []


@pytest.mark.parametrize("expression", ["(" * 400 + "x" + ")" * 400, "-" * 990 + "x"])
def test_deeply_nested_expressions_raise_value_error(expression: str) -> None:
    with pytest.raises(ValueError):
        make_spec(expression)


def test_unknown_name_error_names_the_offender() -> None:
    with pytest.raises(ValueError, match="zeta"):
        make_spec("a*zeta*x")


def test_huge_constant_exponent_is_rejected_quickly() -> None:
    with pytest.raises(ValueError):
        make_spec("a*x**100000")


def test_over_long_expression_is_rejected() -> None:
    with pytest.raises(ValueError, match="too long"):
        make_spec("a*" + "x+" * 600 + "x")


def test_whitelisted_functions_and_constants_work() -> None:
    spec = make_spec(
        "a*log10(x) + b*ln(x) + sqrt(x) + sin(pi*x) + cos(x) + tan(x) + sinh(x) + cosh(x)"
        " + Abs(x) + E",
    )
    x = np.array([1.5, 2.5])
    expected = (
        2 * np.log10(x)
        + 3 * np.log(x)
        + np.sqrt(x)
        + np.sin(np.pi * x)
        + np.cos(x)
        + np.tan(x)
        + np.sinh(x)
        + np.cosh(x)
        + np.abs(x)
        + np.e
    )
    np.testing.assert_allclose(spec.evaluate(x, a=2.0, b=3.0), expected)


def test_fractional_exponent_and_unary_minus_are_allowed() -> None:
    spec = make_spec("-a*x**(1/2) + +b")
    x = np.array([4.0, 9.0])
    np.testing.assert_allclose(spec.evaluate(x, a=2.0, b=1.0), -2.0 * np.sqrt(x) + 1.0)


@pytest.mark.parametrize(
    "parameters",
    [
        (Parameter("a"), Parameter("a")),
        (Parameter("1a"),),
        (Parameter("exp"),),
        (Parameter("x"),),
        (Parameter("pi"),),
        (Parameter("Integer"),),
        (Parameter("Float"),),
        (Parameter("a", lower=2.0, upper=1.0),),
    ],
)
def test_bad_parameter_definitions_are_rejected(parameters: tuple[Parameter, ...]) -> None:
    with pytest.raises(ValueError):
        make_spec("a*x", parameters=parameters)


def test_empty_id_or_no_parameters_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_spec(id="")
    with pytest.raises(ValueError):
        make_spec("x", parameters=())


def test_evaluate_rejects_missing_and_unknown_parameters() -> None:
    spec = make_spec()
    with pytest.raises(ValueError, match="b"):
        spec.evaluate(np.array([1.0]), a=1.0)
    with pytest.raises(ValueError, match="c"):
        spec.evaluate(np.array([1.0]), a=1.0, b=1.0, c=2.0)


def test_non_finite_results_are_returned_not_raised() -> None:
    spec = make_spec("a*log(x)")
    out = spec.evaluate(np.array([-1.0, 0.0]), a=1.0, b=0.0)
    assert not np.isfinite(out).any()


def test_initial_guess_is_not_called_at_construction() -> None:
    calls: list[int] = []

    def guess(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls.append(1)
        return {"a": 1.0, "b": 1.0}

    make_spec(initial_guess=guess)
    assert calls == []


def test_applicability_defaults_to_empty_and_uses_domain_check() -> None:
    x = np.array([1.0, 2.0])
    y = np.array([1.0, -1.0])
    assert make_spec().applicability(x, y) == []

    def positive_y(x: np.ndarray, y: np.ndarray) -> list[str]:
        return ["y must be positive"] if (y <= 0).any() else []

    assert make_spec(domain_check=positive_y).applicability(x, y) == ["y must be positive"]


def test_multi_variable_model_takes_columns() -> None:
    spec = make_spec("a*x + b*z", variables=("x", "z"))
    xz = np.array([[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_allclose(spec.evaluate(xz, a=1.0, b=2.0), [21.0, 42.0])
    np.testing.assert_allclose(spec.jacobian(xz, a=1.0, b=2.0), [[1.0, 10.0], [2.0, 20.0]])
    with pytest.raises(ValueError, match="column"):
        spec.evaluate(np.array([1.0, 2.0]), a=1.0, b=2.0)


def test_contract_passes_for_exponential() -> None:
    x = np.linspace(0.0, 2.0, 20)
    check_model_contract(make_spec(), x, {"a": 2.0, "b": 0.5})


def test_contract_fails_for_wrong_jacobian() -> None:
    @dataclass(frozen=True)
    class BrokenJacobian(ModelSpec):
        def jacobian(self, x: np.ndarray, **params: float) -> np.ndarray:
            return super().jacobian(x, **params) * 1.1

    broken = BrokenJacobian(
        id="broken",
        name="Broken",
        category="growth",
        expression="a*exp(b*x)",
        parameters=(Parameter("a", lower=0.0), Parameter("b")),
        initial_guess=_guess,
    )
    with pytest.raises(AssertionError, match="Jacobian"):
        check_model_contract(broken, np.linspace(0.0, 2.0, 20), {"a": 2.0, "b": 0.5})


def test_contract_fails_when_initial_guess_is_out_of_bounds() -> None:
    spec = make_spec(initial_guess=lambda x, y: {"a": -5.0, "b": 0.1})
    with pytest.raises(AssertionError, match="bounds"):
        check_model_contract(spec, np.linspace(0.0, 2.0, 20), {"a": 2.0, "b": 0.5})


def test_contract_fails_when_initial_guess_misses_a_parameter() -> None:
    spec = make_spec(initial_guess=lambda x, y: {"a": 1.0})
    with pytest.raises(AssertionError, match="initial_guess"):
        check_model_contract(spec, np.linspace(0.0, 2.0, 20), {"a": 2.0, "b": 0.5})


def test_contract_fails_when_evaluation_is_not_finite() -> None:
    spec = make_spec("a*log(x)", parameters=(Parameter("a"), Parameter("b")))
    with pytest.raises(AssertionError, match="finite"):
        check_model_contract(spec, np.linspace(-1.0, 1.0, 20), {"a": 1.0, "b": 0.0})


def test_contract_fails_when_true_params_do_not_match_parameters() -> None:
    with pytest.raises(AssertionError, match="true_params"):
        check_model_contract(make_spec(), np.linspace(0.0, 2.0, 20), {"a": 2.0})


# -- Phase 4: aliases, description, example -------------------------------------------------


def _example_spec(**extra: object) -> ModelSpec:
    return ModelSpec(
        id="demo",
        name="Demo",
        category="exponential",
        expression="a*exp(b*x)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
        **extra,  # type: ignore[arg-type]
    )


def test_new_fields_default_to_empty() -> None:
    spec = _example_spec()
    assert spec.aliases == ()
    assert spec.description == ""
    assert spec.example is None


def test_example_data_is_the_noiseless_curve() -> None:
    spec = _example_spec(
        example=ModelExample({"a": 2.0, "b": 0.5}, x_range=(0.0, 2.0), n=5),
        aliases=("growth curve",),
        description="Exponential growth.",
    )
    x, y = spec.example_data()
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(y, 2.0 * np.exp(0.5 * x))
    assert spec.aliases == ("growth curve",)


def test_example_data_without_example_raises() -> None:
    with pytest.raises(ValueError, match="no example"):
        _example_spec().example_data()


@pytest.mark.parametrize(
    "example",
    [
        {"params": {"a": 1.0}, "x_range": (0.0, 1.0)},
        {"params": {"a": 1.0, "b": 1.0, "c": 1.0}, "x_range": (0.0, 1.0)},
    ],
)
def test_example_must_name_exactly_the_parameters(example: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="exactly the parameters"):
        _example_spec(example=ModelExample(**example))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_range": (1.0, 1.0)},
        {"x_range": (2.0, 1.0)},
        {"x_range": (0.0, float("inf"))},
        {"x_range": (0.0, 1.0), "n": 1},
    ],
)
def test_model_example_rejects_bad_ranges(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelExample({"a": 1.0, "b": 1.0}, **kwargs)  # type: ignore[arg-type]


def test_model_example_rejects_non_finite_parameter() -> None:
    with pytest.raises(ValueError, match="finite"):
        ModelExample({"a": float("nan")}, x_range=(0.0, 1.0))


@pytest.mark.parametrize("aliases", [("",), ("a", "a"), ("demo",)])
def test_bad_aliases_are_rejected(aliases: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="alias"):
        _example_spec(aliases=aliases)


def _metadata_spec(**overrides: object) -> ModelSpec:
    kwargs: dict[str, object] = {
        "id": "meta",
        "name": "Meta",
        "category": "exponential",
        "expression": "a*exp(b*x)",
        "parameters": (Parameter("a", "scale"), Parameter("b", "rate")),
        "initial_guess": lambda x, y: {"a": 1.0, "b": 0.1},
        "references": ("Someone (2000) A real book",),
        "assumptions": "Constant relative growth.",
        "description": "Exponential growth.",
        "example": ModelExample({"a": 2.0, "b": 0.3}, x_range=(0.0, 2.0)),
    }
    kwargs.update(overrides)
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


def test_strict_metadata_accepts_complete_model_and_is_off_by_default() -> None:
    x = np.linspace(0.0, 2.0, 9)
    check_model_contract(_metadata_spec(), x, {"a": 2.0, "b": 0.3}, strict_metadata=True)
    bare = make_spec()
    check_model_contract(bare, x, {"a": 2.0, "b": 0.3})  # lenient by default


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": " "}, "name is empty"),
        ({"category": "nonsense"}, "vocabulary"),
        ({"category": "engineering"}, "sub-category"),
        ({"category": "engineering.nonsense"}, "sub-category"),
        ({"description": " "}, "description is empty"),
        ({"example": None}, "no example"),
        ({"references": ("Someone (2000) A real book", " ")}, "empty entry"),
        ({"parameters": (Parameter("a", "scale"), Parameter("b"))}, "parameter 'b'"),
        ({"references": ()}, "references"),
        ({"assumptions": ""}, "assumptions"),
    ],
)
def test_strict_metadata_rejects_incomplete_models(
    overrides: dict[str, object], message: str
) -> None:
    x = np.linspace(0.0, 2.0, 9)
    with pytest.raises(AssertionError, match=message):
        check_model_contract(
            _metadata_spec(**overrides), x, {"a": 2.0, "b": 0.3}, strict_metadata=True
        )


def test_strict_metadata_accepts_engineering_subcategory() -> None:
    x = np.linspace(0.0, 2.0, 9)
    spec = _metadata_spec(category="engineering.strength")
    check_model_contract(spec, x, {"a": 2.0, "b": 0.3}, strict_metadata=True)


def test_strict_metadata_allows_any_subcategory_of_a_vocabulary_head() -> None:
    x = np.linspace(0.0, 2.0, 9)
    spec = _metadata_spec(category="exponential.two_term")
    check_model_contract(spec, x, {"a": 2.0, "b": 0.3}, strict_metadata=True)


@pytest.mark.parametrize("starter", STARTER_MODELS, ids=lambda s: s.id)
def test_every_starter_passes_the_strict_contract(starter: ModelSpec) -> None:
    assert starter.example is not None
    x, _ = starter.example_data()
    check_model_contract(starter, x, dict(starter.example.params), strict_metadata=True)


def test_spec_with_example_is_hashable_and_example_is_immutable() -> None:
    spec = _metadata_spec()
    assert spec.example is not None
    assert hash(spec) == hash(spec)
    assert len({spec, spec}) == 1
    with pytest.raises(TypeError):
        spec.example.params["a"] = 1.0  # type: ignore[index]
    ModelExample({"b": 1.0, "a": 2.0}, (0.0, 1.0))
    assert hash(ModelExample({"b": 1.0, "a": 2.0}, (0.0, 1.0))) == hash(
        ModelExample({"a": 2.0, "b": 1.0}, (0.0, 1.0))
    )
    source = {"a": 2.0, "b": 0.3}
    example = ModelExample(source, (0.0, 1.0))
    source["a"] = 99.0  # the example took a copy
    assert example.params["a"] == 2.0


def test_hash_of_registered_starter_models_works() -> None:
    from curvestudio.models import get_model

    # Starters may have been cleared by another test module; re-register to be independent.
    register_starter_models()
    assert isinstance(hash(get_model("gaussian")), int)


def test_concurrent_first_use_compiles_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import sympy

    calls: list[int] = []
    real = sympy.lambdify

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sympy, "lambdify", counting)
    spec = _metadata_spec()
    barrier = threading.Barrier(8)
    results: list[float] = []

    def work() -> None:
        barrier.wait()
        results.append(float(spec.evaluate(np.array([1.0]), a=2.0, b=0.0)[0]))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [2.0] * 8
    assert len(calls) == 1 + len(spec.parameters)
