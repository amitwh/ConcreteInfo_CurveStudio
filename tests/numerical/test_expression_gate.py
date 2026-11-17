"""The expression gate: hostile input is refused on every path, and a copy of a library model fits
identically.

The adversarial corpus (``tests/data/expressions/adversarial.txt``) runs through the three doors
a user equation can come in by: ``analyze_expression``, ``build_custom_model`` and
``AutoFitConfig(custom_models=...)`` followed by ``autofit``. Each must refuse every entry with a
``ValueError`` (``AutoFitError`` is one) in bounded time and memory, and Auto-fit must not fit
anything on the way. A hand-written copy of a library model, started from the same point, must
give the same fit to 1e-9.
"""

import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitError, autofit
from curvestudio.expressions import analyze_expression
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import (
    CustomModelDefinition,
    ParameterDefinition,
    build_custom_model,
    get_model,
)
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models

CORPUS = Path(__file__).resolve().parents[1] / "data" / "expressions" / "adversarial.txt"
MAX_SECONDS = 1.0
MAX_BYTES = 50 * 1024 * 1024


def corpus() -> list[str]:
    entries: list[str] = []
    for line in CORPUS.read_text(encoding="utf-8").split("\n"):
        if line and not line.startswith("#"):
            entries.append("" if line == "<empty>" else line.replace("\\n", "\n"))
    return entries


def definition(text: str, ident: str = "hostile") -> CustomModelDefinition:
    return CustomModelDefinition(id=ident, name="hostile", expression=text)


def make_xy(x: np.ndarray, y: np.ndarray) -> XY:
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint="expression-gate")


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


ACCEPTED_AS_VALID = frozenset({"y = a*x"})
"""Corpus entries that are only invalid for the corpus's fixed name set {a, b, x, y, z}: with
``y`` the response, a leading ``y =`` is the documented way to write an equation, so every door
that infers its own names accepts it."""


def refuse_everything(
    door: str, attempt: Callable[[str], object], accepted: frozenset[str] = ACCEPTED_AS_VALID
) -> None:
    entries = [text for text in corpus() if text not in accepted]
    assert len(corpus()) >= 25, "the adversarial corpus has gone missing"
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        for text in entries:
            start = time.perf_counter()
            with pytest.raises(ValueError):
                attempt(text)
            elapsed = time.perf_counter() - start
            assert elapsed < MAX_SECONDS, f"{door}: {text[:40]!r} took {elapsed:.2f} s"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak - baseline < MAX_BYTES, f"{door}: peak {peak - baseline} bytes"


def test_corpus_is_refused_by_analyze_expression() -> None:
    # analyze_expression infers parameters from free names, so a bare identifier is a valid
    # parameter there; build_custom_model refuses the reserved ones (tested below)
    refuse_everything(
        "analyze_expression", analyze_expression, ACCEPTED_AS_VALID | {"__builtins__"}
    )


def test_a_reserved_name_is_not_a_parameter() -> None:
    with pytest.raises(ValueError):
        build_custom_model(definition("__builtins__"))


def test_the_y_equals_prefix_is_accepted_and_dropped() -> None:
    assert analyze_expression("y = a*x").expression == "a*x"
    assert build_custom_model(definition("y = a*x")).expression == "a*x"


def test_corpus_is_refused_by_build_custom_model() -> None:
    refuse_everything("build_custom_model", lambda text: build_custom_model(definition(text)))


def test_corpus_is_refused_by_autofit_before_anything_is_fitted() -> None:
    x = np.linspace(0.5, 6.0, 30)
    data = make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0 + np.random.default_rng(0).normal(0, 0.03, 30))
    fitted: list[str] = []

    def attempt(text: str) -> None:
        config = AutoFitConfig(custom_models=(definition(text),), workers=1)
        try:
            autofit(data, config, on_outcome=lambda o: fitted.append(o.candidate_id))
        except AutoFitError as exc:
            assert exc.code == "INVALID_CUSTOM_MODEL"
            assert "hostile" in str(exc)  # names the offending model
            raise

    refuse_everything("autofit", attempt)
    assert fitted == [], "a rejected equation must stop the run before any candidate is fitted"


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("a*2x", "implicit multiplication"),
        ("a*x^2", r"use \*\* for powers"),
        ("a*Exp(x)", "did you mean 'exp'"),
        ("a*sin x", "implicit multiplication"),
        ("a*x =", "stray '='"),
    ],
)
def test_user_typos_get_a_message_with_the_fix(text: str, hint: str) -> None:
    with pytest.raises(ValueError, match=hint) as raised:
        build_custom_model(definition(text))
    assert "Traceback" not in str(raised.value)


def test_a_misspelled_declared_parameter_names_the_close_match() -> None:
    bad = CustomModelDefinition(
        id="typo",
        name="typo",
        expression="vmax*x/(km + x)",
        parameters=(ParameterDefinition(name="vmax"), ParameterDefinition(name="kmm")),
    )
    with pytest.raises(ValueError, match="km"):
        build_custom_model(bad)


def test_a_hand_written_copy_of_a_library_model_gives_the_same_fit() -> None:
    library = get_model("michaelis_menten")
    x = np.linspace(0.1, 20.0, 50)
    clean = library.evaluate(x, vmax=10.0, km=2.0)
    data = make_xy(x, clean + np.random.default_rng(11).normal(0.0, 0.2, x.size))
    start = library.initial_guess(data.x, data.y)
    copy = build_custom_model(
        CustomModelDefinition(
            id="my_michaelis_menten",
            name="My Michaelis-Menten",
            expression="vmax*x/(km + x)",
            parameters=tuple(
                ParameterDefinition(
                    name=p.name, initial=start[p.name], lower=p.lower, upper=p.upper
                )
                for p in library.parameters
            ),
        )
    )
    config = FitConfig(n_starts=1)
    expected = fit(library, data, config)
    actual = fit(copy, data, config)
    assert expected.status is actual.status is FitStatus.CONVERGED
    for a, b in zip(expected.parameters, actual.parameters, strict=True):
        assert a.name == b.name
        assert abs(a.value - b.value) <= 1e-9 * max(1.0, abs(a.value))
    assert expected.stats is not None and actual.stats is not None
    assert abs(expected.stats.sse - actual.stats.sse) <= 1e-9 * max(1.0, expected.stats.sse)
    assert np.allclose(expected.fitted, actual.fitted, rtol=1e-9, atol=1e-9)


def test_autofit_ties_the_custom_copy_with_the_library_model() -> None:
    x = np.linspace(0.1, 20.0, 50)
    y = 10.0 * x / (2.0 + x) + np.random.default_rng(12).normal(0.0, 0.2, x.size)
    copy = CustomModelDefinition(
        id="my_michaelis_menten",
        name="My Michaelis-Menten",
        expression="vmax*x/(km + x)",
        parameters=(
            ParameterDefinition(name="vmax", initial=10.0, lower=0.0),
            ParameterDefinition(name="km", initial=2.0, lower=0.0),
        ),
    )
    config = AutoFitConfig(custom_models=(copy,), workers=1, n_starts=3, cv_folds=4)
    result = autofit(make_xy(x, y), config)
    rows = {row.model_id: row for row in result.ranked}
    assert "my_michaelis_menten" in rows
    assert "michaelis_menten" in rows
    assert "michaelis_menten" in rows["my_michaelis_menten"].same_curve_as
    assert rows["my_michaelis_menten"].metrics.aicc == pytest.approx(
        rows["michaelis_menten"].metrics.aicc, abs=1e-6
    )
