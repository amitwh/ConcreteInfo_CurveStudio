"""The custom equation builder: ``CustomModelDefinition`` -> fittable ``ModelSpec``."""

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy
from numpy.testing import assert_allclose

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import CustomModelDefinition, ParameterDefinition, build_custom_model
from curvestudio.models import custom as custom_module

CORPUS = Path(__file__).resolve().parents[2] / "data" / "expressions" / "adversarial.txt"


def _corpus() -> list[str]:
    entries: list[str] = []
    for line in CORPUS.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith("#"):
            continue
        entries.append("" if line == "<empty>" else line.replace("\\n", "\n"))
    return entries


def _xy(x: Any, y: Any) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="test-fingerprint",
    )


def test_builds_a_fittable_model() -> None:
    spec = build_custom_model(
        CustomModelDefinition(id="user_exp", name="User exponential", expression="a*exp(-b*x)+c")
    )
    assert spec.category == "custom"
    assert [p.name for p in spec.parameters] == ["a", "b", "c"]
    x = np.linspace(0.05, 3.0, 40)
    result = fit(spec, _xy(x, 2.5 * np.exp(-1.3 * x) + 0.7))
    assert result.status is FitStatus.CONVERGED
    values = {p.name: p.value for p in result.parameters}
    assert_allclose([values["a"], values["b"], values["c"]], [2.5, 1.3, 0.7], atol=1e-6)


def test_user_initial_bounds_are_respected() -> None:
    spec = build_custom_model(
        CustomModelDefinition(
            id="bounded",
            name="Bounded line",
            expression="a*x + b",
            parameters=(
                ParameterDefinition(
                    name="a", initial=2.0, lower=0.5, upper=5.0, unit="MPa", description="slope"
                ),
                ParameterDefinition(name="b", lower=2.0, upper=5.0),
            ),
        )
    )
    guess = spec.initial_guess(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert guess == {"a": 2.0, "b": 2.0}  # b: no initial, so 1.0 clipped into [2, 5]
    first = spec.parameters[0]
    assert (first.lower, first.upper, first.unit, first.description) == (0.5, 5.0, "MPa", "slope")


def test_equation_prefix_accepted() -> None:
    spec = build_custom_model(CustomModelDefinition(id="line", name="Line", expression="y = a*x+b"))
    assert spec.expression == "a*x + b"
    assert_allclose(spec.evaluate(np.array([1.0, 2.0]), a=2.0, b=1.0), [3.0, 5.0])


def test_multi_variable_spec_evaluates_but_fit_refuses() -> None:
    spec = build_custom_model(
        CustomModelDefinition(
            id="plane", name="Plane", expression="a*x + b*t", variables=("x", "t")
        )
    )
    xt = np.column_stack([np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])])
    assert_allclose(spec.evaluate(xt, a=2.0, b=10.0), [30.0, 42.0, 54.0])
    with pytest.raises(ValueError, match="single-variable"):
        fit(spec, _xy([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))


def test_unknown_parameter_name_errors_list_close_matches() -> None:
    typo = CustomModelDefinition(
        id="typo",
        name="Typo",
        expression="a*exp(-b*x)+c",
        parameters=(ParameterDefinition(name="bb"), ParameterDefinition(name="a")),
    )
    with pytest.raises(ValueError, match=r"unknown parameter 'bb'.*did you mean 'b'\?"):
        build_custom_model(typo)
    missing = CustomModelDefinition(
        id="missing",
        name="Missing",
        expression="a*exp(-b*x)+c",
        parameters=(ParameterDefinition(name="a"), ParameterDefinition(name="b")),
    )
    with pytest.raises(ValueError, match=r"uses parameter\(s\) \['c'\] that the definition"):
        build_custom_model(missing)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("E", "constant"),
        ("pi", "constant"),
        ("exp", "function"),
        ("voigt_partial", "reserved"),
    ],
)
def test_reserved_parameter_names_say_what_the_name_is(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_custom_model(
            CustomModelDefinition(
                id="reserved",
                name="Reserved",
                expression="a*x",
                parameters=(ParameterDefinition(name="a"), ParameterDefinition(name=name)),
            )
        )


def test_constant_named_as_parameter_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match=r"'E' is a constant, not a parameter"):
        build_custom_model(
            CustomModelDefinition(
                id="young",
                name="Young",
                expression="E*x",
                parameters=(ParameterDefinition(name="E"),),
            )
        )


def test_sympy_special_names_are_reserved_even_when_inferred() -> None:
    # Declared or inferred from the text, oo/nan/I/... can never be fitted parameters of a
    # custom equation. (Reserved at the builder, not in the grammar: the engineering
    # catalogues legitimately build ModelSpecs with a sorptivity parameter called 'S'.)
    for name in ("oo", "zoo", "nan", "inf", "I", "S"):
        with pytest.raises(ValueError, match="reserved"):
            build_custom_model(
                CustomModelDefinition(
                    id="special",
                    name="Special",
                    expression="a*x",
                    parameters=(ParameterDefinition(name="a"), ParameterDefinition(name=name)),
                )
            )
        with pytest.raises(ValueError, match="reserved"):
            build_custom_model(
                CustomModelDefinition(id="special", name="Special", expression=f"{name}*x")
            )


def test_expression_without_free_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="no free parameter in the expression"):
        build_custom_model(CustomModelDefinition(id="nofree", name="No free", expression="x*x"))
    with pytest.raises(ValueError, match="no free parameter in the expression"):
        build_custom_model(CustomModelDefinition(id="nofree2", name="No free", expression="E*x"))


def test_duplicate_and_colliding_names() -> None:
    with pytest.raises(ValueError, match="twice"):
        build_custom_model(
            CustomModelDefinition(
                id="dup",
                name="Duplicate",
                expression="a*x + b",
                parameters=(ParameterDefinition(name="a"), ParameterDefinition(name="a")),
            )
        )
    with pytest.raises(ValueError, match="variable"):
        build_custom_model(
            CustomModelDefinition(
                id="collide",
                name="Collision",
                expression="a*x",
                parameters=(ParameterDefinition(name="a"), ParameterDefinition(name="x")),
            )
        )


def test_too_complex_derivative_is_rejected() -> None:
    # The plan's 60-term product of sines: its 60-level multiplication chain is already
    # past the grammar's 40-level depth cap (and 479 AST nodes past the 300-node cap),
    # so the builder refuses it before any derivative is built — the same protection the
    # derivative cap adds for legal-but-heavy input.
    expression = "*".join(f"sin(a{i % 8}*x + {i})" for i in range(60))
    definition = CustomModelDefinition(id="big", name="Too big", expression=expression)
    with pytest.raises(ValueError, match=r"too deeply|too many terms"):
        build_custom_model(definition)


def test_derivative_cost_cap_rejects_expensive_but_legal_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 36 terms is the largest sin-product the grammar admits; its derivatives total
    # ~7 800 SymPy nodes, comfortably under the shipped 20 000 cap (calibrated so a
    # 60-term product, at ~21 600, would fail). Lower the cap here to prove the check
    # itself rejects rather than passing silently.
    expression = "*".join(f"sin(a{i % 8}*x + {i})" for i in range(36))
    definition = CustomModelDefinition(id="big36", name="Big product", expression=expression)
    build_custom_model(definition)
    monkeypatch.setattr(custom_module, "MAX_DERIVATIVE_NODES", 5_000)
    with pytest.raises(ValueError, match="model too complex"):
        build_custom_model(definition)


def test_flat_sum_error_names_the_term_limit() -> None:
    # 40 product terms nest one level per term past the grammar's depth cap; the builder
    # must surface the sum-shaped reason, not "nested too deeply".
    expression = "+".join(f"a{i % 4}*x" for i in range(40))
    with pytest.raises(
        ValueError, match=r"sum has too many terms \(limit \d+\); group terms or simplify"
    ):
        build_custom_model(CustomModelDefinition(id="sum40", name="Sum", expression=expression))
    # the limit itself is unchanged: 39 terms still build
    spec = build_custom_model(
        CustomModelDefinition(
            id="sum39", name="Sum", expression="+".join(f"a{i % 4}*x" for i in range(39))
        )
    )
    assert len(spec.parameters) == 4


def test_balanced_product_that_normalises_to_a_chain_still_builds() -> None:
    # Finding 3: sstr would flatten this 50-factor product into one 50-factor chain the
    # depth cap refuses; the report falls back to the author's text, so the build succeeds.
    def build(lo: int, hi: int) -> str:
        if hi - lo == 1:
            return f"sin(p{lo}*x)"
        mid = (lo + hi) // 2
        return f"({build(lo, mid)}*{build(mid, hi)})"

    text = build(0, 50)
    spec = build_custom_model(
        CustomModelDefinition(id="balanced", name="Balanced", expression=text)
    )
    assert spec.expression == text
    assert len(spec.parameters) == 50
    values = {p.name: 0.5 for p in spec.parameters}
    assert np.all(np.isfinite(spec.evaluate(np.array([0.3, 0.7]), **values)))


def test_derivatives_are_computed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The complexity cap and the spec's compilation share one round of differentiation
    # (the slow timing test checks the same on the worst-case chains; this one always runs).
    diff_calls: list[int] = []
    real_diff = sympy.diff

    def counting(*args: object, **kwargs: object) -> object:
        diff_calls.append(1)
        return real_diff(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sympy, "diff", counting)
    spec = build_custom_model(
        CustomModelDefinition(id="once", name="Once", expression="a*exp(-b*x)+c")
    )
    assert len(diff_calls) == len(spec.parameters) == 3
    jac = spec.jacobian(np.array([0.5]), a=2.0, b=1.0, c=0.0)
    assert_allclose(jac, [[math.exp(-0.5), -2.0 * 0.5 * math.exp(-0.5), 1.0]])


@pytest.mark.slow
@pytest.mark.parametrize("depth", [17, 18, 19])  # the three deepest chains the grammar admits
def test_worst_case_exp_chains_build_fast_and_differentiate_once(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "".join(f"exp(x*p{i}*" for i in range(depth)) + "x" + ")" * depth
    diff_calls: list[int] = []
    real_diff = sympy.diff

    def counting(*args: object, **kwargs: object) -> object:
        diff_calls.append(1)
        return real_diff(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sympy, "diff", counting)
    start = time.perf_counter()
    spec = build_custom_model(CustomModelDefinition(id="worst", name="Worst", expression=text))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.5, f"depth {depth} took {elapsed:.2f}s to build"
    # once for the complexity cap, reused by the spec's compilation: exactly once each
    assert len(diff_calls) == len(spec.parameters)
    # and the compiled model actually works
    starts = {p.name: 0.1 for p in spec.parameters}
    assert np.all(np.isfinite(spec.evaluate(np.array([0.01, 0.02]), **starts)))
    jac = spec.jacobian(np.array([0.01, 0.02]), **starts)
    assert jac.shape == (2, len(spec.parameters))
    assert np.all(np.isfinite(jac))


def test_json_round_trip_of_definition() -> None:
    definition = CustomModelDefinition(
        id="rt",
        name="Round trip",
        expression="y = a*x + b",
        parameters=(
            ParameterDefinition(name="a", initial=2.0, lower=0.0),
            ParameterDefinition(name="b", upper=None),
        ),
    )
    text = definition.model_dump_json()
    assert "Infinity" not in text  # JSON-safe: bounds use null, not ±Infinity
    assert '"lower":null' in text
    revived = CustomModelDefinition.model_validate_json(text)
    assert revived == definition
    assert revived.parameters[1].upper == math.inf


def test_definition_id_must_be_lowercase_snake_case() -> None:
    for bad in ("Bad", "1model", "my-model", "has space", "model/id", "café", "x" * 65, ""):
        with pytest.raises(ValueError, match="lowercase snake_case"):
            CustomModelDefinition(id=bad, name="Any", expression="a*x")
    for good in ("m", "user_exp", "m2", "a" * 64, "deeply_nested_model_42"):
        CustomModelDefinition(id=good, name="Any", expression="a*x")


def test_parameter_definition_field_limits() -> None:
    with pytest.raises(ValueError, match="32"):
        ParameterDefinition(name="a" * 33)
    with pytest.raises(ValueError, match="500"):
        ParameterDefinition(name="a", description="d" * 501)
    with pytest.raises(ValueError, match="32"):
        ParameterDefinition(name="a", unit="u" * 33)
    with pytest.raises(ValueError, match="control"):
        ParameterDefinition(name="a", unit="MPa\n")
    # the boundaries themselves, and printable non-ASCII units, are fine
    ParameterDefinition(name="a" * 32, unit="u" * 32, description="d" * 500)
    ParameterDefinition(name="a", unit="MPa·mm²")


def test_initial_and_bounds_validation() -> None:
    with pytest.raises(ValueError, match="initial"):
        build_custom_model(
            CustomModelDefinition(
                id="oob",
                name="Out of bounds",
                expression="a*x",
                parameters=(ParameterDefinition(name="a", initial=5.0, upper=2.0),),
            )
        )
    with pytest.raises(ValueError, match="lower"):
        build_custom_model(
            CustomModelDefinition(
                id="pinched",
                name="Pinched",
                expression="a*x",
                parameters=(ParameterDefinition(name="a", lower=2.0, upper=2.0),),
            )
        )


def test_multistart_recovers_a_poor_default_start() -> None:
    # The documented remedy for the fixed 1.0 default start: a seeded multistart fits
    # an equation whose optimum lies far from 1.0.
    spec = build_custom_model(
        CustomModelDefinition(
            id="wide",
            name="Wide dynamic range",
            expression="a*x**b",
            parameters=(
                ParameterDefinition(name="a"),
                ParameterDefinition(name="b", lower=0.1, upper=8.0),
            ),
        )
    )
    x = np.linspace(1.0, 3.0, 30)
    result = fit(spec, _xy(x, 0.001 * x**6.0), FitConfig(n_starts=8, seed=1))
    assert result.status is FitStatus.CONVERGED
    values = {p.name: p.value for p in result.parameters}
    assert_allclose([values["a"], values["b"]], [0.001, 6.0], rtol=1e-4)


@pytest.mark.parametrize("line", _corpus())
def test_hostile_corpus_via_builder(line: str) -> None:
    with pytest.raises(ValueError):  # as a parameter name (definition or build rejects it)
        build_custom_model(
            CustomModelDefinition(
                id="hostile",
                name="Hostile",
                expression="a*x + b",
                parameters=(ParameterDefinition(name=line),),
            )
        )
    if line == "y = a*x":
        # the one corpus line the builder accepts: an equation prefix is a contract,
        # not an attack (see test_equation_prefix_accepted)
        spec = build_custom_model(
            CustomModelDefinition(id="prefix_ok", name="Prefix ok", expression=line)
        )
        assert spec.expression == "a*x"
        return
    with pytest.raises(ValueError):  # as the expression
        build_custom_model(CustomModelDefinition(id="hostile", name="Hostile", expression=line))
