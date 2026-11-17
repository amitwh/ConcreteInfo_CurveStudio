"""Hostile input: rejected quickly, in bounded memory, and only ever with ValueError."""

import ast
import time
import tracemalloc
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from curvestudio.expressions import validate_expression
from curvestudio.expressions.grammar import FUNCTIONS
from curvestudio.models import CustomModelDefinition, ParameterDefinition, build_custom_model

CORPUS = Path(__file__).resolve().parents[2] / "data" / "expressions" / "adversarial.txt"
NAMES = frozenset({"a", "b", "x", "y", "z"})
WHITELIST = frozenset(FUNCTIONS)


def _corpus() -> list[str]:
    entries: list[str] = []
    for line in CORPUS.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith("#"):
            continue
        entries.append("" if line == "<empty>" else line.replace("\\n", "\n"))
    return entries


def test_corpus_is_large_enough_and_covers_the_brief() -> None:
    entries = _corpus()
    assert len(entries) >= 25
    for needle in ("__import__('os').system('x')", "9**9**9", "1e999", "", "ꞵ*x", "a" * 1001):
        assert needle in entries


def test_corpus_is_rejected_quickly() -> None:
    entries = _corpus()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        for text in entries:
            start = time.perf_counter()
            with pytest.raises(ValueError):
                validate_expression(text, NAMES)
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, f"{text[:40]!r} took {elapsed:.2f}s"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak - baseline < 50 * 1024 * 1024


@pytest.mark.parametrize("text", ["a\x00b", "\x00", "a*x\x00", "x\r\ny", "x\ty\x0b"])
def test_control_characters_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        validate_expression(text, NAMES)


def _assert_only_whitelisted_calls(text: str) -> None:
    for node in ast.walk(ast.parse(text.strip(), mode="eval")):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            assert node.func.id in WHITELIST


def _check_never_escapes(text: str) -> None:
    try:
        validate_expression(text, NAMES)
    except ValueError:
        return
    _assert_only_whitelisted_calls(text)  # accepted: must be inside the whitelist


_TOKENS = [
    *sorted(NAMES),
    *sorted(WHITELIST),
    "pi",
    "E",
    "__import__",
    "eval",
    "lambda",
    "1",
    "2",
    "0.5",
    "1e308",
    "1e999",
    "9999",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "+",
    "-",
    "*",
    "/",
    "**",
    "^",
    "//",
    "%",
    "<",
    ">",
    "=",
    ":=",
    ".",
    ",",
    ";",
    ":",
    "'",
    '"',
    "\n",
    " ",
    "@",
    "~",
    "if",
    "else",
    "for",
    "in",
    "not",
    "ꞵ",
    "\uff58",
]
_FUZZ = settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_FUZZ
@given(st.text(max_size=300))
def test_fuzz_text_never_escapes(text: str) -> None:
    _check_never_escapes(text)


@_FUZZ
@given(st.lists(st.sampled_from(_TOKENS), max_size=80).map(" ".join))
def test_fuzz_never_escapes(text: str) -> None:
    _check_never_escapes(text)


@_FUZZ
@given(st.lists(st.sampled_from(_TOKENS), max_size=80).map("".join))
def test_fuzz_unspaced_token_soup_never_escapes(text: str) -> None:
    _check_never_escapes(text)


def test_corpus_is_rejected_quickly_by_the_custom_builder() -> None:
    """The whole hostile corpus also dies fast when driven through ``build_custom_model``."""
    hostile = [entry for entry in _corpus() if entry != "y = a*x"]  # prefix is accepted
    for text in hostile:
        start = time.perf_counter()
        with pytest.raises(ValueError):
            build_custom_model(CustomModelDefinition(id="h", name="h", expression=text))
        with pytest.raises(ValueError):
            build_custom_model(
                CustomModelDefinition(
                    id="h",
                    name="h",
                    expression="a*x + b",
                    parameters=(ParameterDefinition(name=text),),
                )
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"{text[:40]!r} took {elapsed:.2f}s via the builder"
