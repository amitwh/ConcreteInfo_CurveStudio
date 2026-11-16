"""Parser for the NIST StRD reference datasets and the LRE accuracy measure.

NIST's Statistical Reference Datasets (https://www.itl.nist.gov/div898/strd/) publish data
together with certified results computed to 11-15 significant digits, so a fitting engine can be
graded against ground truth instead of against itself. The ``.dat`` files are vendored unchanged
under ``tests/data/nist_strd``; this module reads their fixed human-oriented layout.

Why parsing is strict: a benchmark that quietly mis-reads a file would grade the engine against
the wrong numbers and still look green. Every structural expectation (declared observation and
parameter counts, consecutive parameter names, certified degrees of freedom) is therefore
checked, and any mismatch raises ``ValueError`` naming the file rather than an
``IndexError``/``KeyError`` from deep inside the parser.

The nonlinear (``nls``) and linear (``lls``) layouts differ, so each has its own loader.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Difficulty = Literal["lower", "average", "higher"]

# NIST files print numbers such as ``10.07E0``, ``-0.670191154593408E-01`` or ``.5``; matching
# them with a regex (rather than ``str.split``) also copes with a value glued to its neighbour.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_DATA_HEADER_RE = re.compile(r"^Data:\s+y\s+x\s*$", re.MULTILINE)
_OBSERVATIONS_RE = re.compile(r"^\s+(\d+)\s+Observations\b", re.MULTILINE)
_PARAMETERS_RE = re.compile(r"^\s+(\d+)\s+Parameters\b.*$", re.MULTILINE)
_DIFFICULTY_RE = re.compile(r"\b(Lower|Average|Higher)\s+Level\s+of\s+Difficulty\b")
_NLS_ROW_RE = re.compile(r"^\s*(b\d+)\s*=\s*(.+?)\s*$")
_LLS_ROW_RE = re.compile(r"^\s*(B\d+)\s+(.+?)\s*$")

# The cap reflects double precision: no estimate can honestly agree with a certified value
# to more than ~15-16 significant digits.
_LRE_CAP = 15.0


@dataclass(frozen=True, eq=False)
class NistProblem:
    """One nonlinear least-squares StRD problem with its certified solution.

    ``eq=False`` keeps identity equality: the arrays make a generated ``__eq__`` ambiguous.
    ``x``/``y`` are read-only so a benchmark cannot corrupt the reference data in place.
    """

    name: str
    difficulty: Difficulty | None
    x: FloatArray
    y: FloatArray
    start1: dict[str, float]
    start2: dict[str, float]
    certified: dict[str, float]
    certified_se: dict[str, float]
    certified_sse: float
    certified_residual_sd: float
    df: int
    model_formula: str


@dataclass(frozen=True, eq=False)
class NistLinearProblem:
    """One linear least-squares StRD problem (polynomial in ``x``) with certified values.

    Parameters are named ``B0..Bk`` exactly as in the file; ``degree`` is ``k``.
    """

    name: str
    x: FloatArray
    y: FloatArray
    certified: dict[str, float]
    certified_se: dict[str, float]
    residual_sd: float
    r2: float
    degree: int


def lre(estimate: float, certified: float) -> float:
    """Log relative error: correct significant digits of ``estimate`` against ``certified``.

    ``-log10(|estimate - certified| / |certified|)``, capped at 15 (also returned when the two
    are equal). When ``certified`` is zero a relative error is undefined, so the absolute error
    is used. The result is floored at 0.0 (an estimate with no correct digit scores 0, never
    negative) and a non-finite ``estimate`` scores 0.0. A non-finite ``certified`` is a bug in
    the caller's reference data and raises ``ValueError``.
    """
    if not math.isfinite(certified):
        raise ValueError(f"certified value must be finite, got {certified!r}")
    if not math.isfinite(estimate):
        return 0.0
    error = abs(estimate - certified)
    if error == 0.0:
        return _LRE_CAP
    scale = abs(certified)
    relative = error / scale if scale != 0.0 else error
    if relative >= 1.0:
        return 0.0
    return min(_LRE_CAP, -math.log10(relative))


def load_nls(path: Path) -> NistProblem:
    """Parse a NIST nonlinear-regression ``.dat`` file (Misra1a, MGH09, ...).

    Raises ``FileNotFoundError`` if ``path`` does not exist and ``ValueError`` (naming the file)
    if the content does not have the expected layout.
    """
    lines = _read_lines(path)
    text = "\n".join(lines)
    name = path.stem
    try:
        n_obs = _declared_count(_OBSERVATIONS_RE, text, "observation count")
        n_params = _declared_count(_PARAMETERS_RE, text, "parameter count")
        difficulty = _difficulty(text)
        formula = _model_formula(lines)
        start1, start2, certified, certified_se = _nls_parameter_table(lines)
        sse = _labelled_float(text, "Residual Sum of Squares:")
        residual_sd = _labelled_float(text, "Residual Standard Deviation:")
        df = int(_labelled_float(text, "Degrees of Freedom:"))
        x, y = _data_block(lines)
        _require(
            len(certified) == n_params,
            f"declares {n_params} parameters but certifies {len(certified)}",
        )
        _require(len(y) == n_obs, f"declares {n_obs} observations but contains {len(y)}")
        _require(
            sorted(certified, key=_param_index) == [f"b{i}" for i in range(1, n_params + 1)],
            f"parameters are not b1..b{n_params}: {sorted(certified)}",
        )
    except _ParseError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return NistProblem(
        name=name,
        difficulty=difficulty,
        x=_readonly(x),
        y=_readonly(y),
        start1=start1,
        start2=start2,
        certified=certified,
        certified_se=certified_se,
        certified_sse=sse,
        certified_residual_sd=residual_sd,
        df=df,
        model_formula=formula,
    )


def load_lls(path: Path) -> NistLinearProblem:
    """Parse a NIST linear-regression ``.dat`` file (Norris, Filip, Wampler1, ...).

    The polynomial degree is inferred from the parameter count (``B0..Bk`` gives ``k``).
    Raises ``FileNotFoundError`` if ``path`` does not exist and ``ValueError`` (naming the file)
    if the content does not have the expected layout.
    """
    lines = _read_lines(path)
    text = "\n".join(lines)
    try:
        n_obs = _declared_count(_OBSERVATIONS_RE, text, "observation count")
        n_params = _declared_count(_PARAMETERS_RE, text, "parameter count")
        certified, certified_se = _lls_parameter_table(lines)
        residual_sd = _labelled_float(text, r"Residual\s+Standard Deviation")
        r2 = _labelled_float(text, "R-Squared")
        x, y = _data_block(lines)
        _require(
            len(certified) == n_params,
            f"declares {n_params} parameters but certifies {len(certified)}",
        )
        _require(len(y) == n_obs, f"declares {n_obs} observations but contains {len(y)}")
        _require(
            sorted(certified, key=_param_index) == [f"B{i}" for i in range(n_params)],
            f"parameters are not B0..B{n_params - 1}: {sorted(certified)}",
        )
    except _ParseError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return NistLinearProblem(
        name=path.stem,
        x=_readonly(x),
        y=_readonly(y),
        certified=certified,
        certified_se=certified_se,
        residual_sd=residual_sd,
        r2=r2,
        degree=n_params - 1,
    )


class _ParseError(Exception):
    """Internal: a structural problem; converted to ``ValueError`` with the file name."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _ParseError(message)


def _read_lines(path: Path) -> list[str]:
    # NIST files use CRLF; splitlines() normalises both. A missing file raises
    # FileNotFoundError and undecodable bytes raise UnicodeDecodeError (a ValueError) as-is.
    return path.read_text(encoding="ascii").splitlines()


def _readonly(values: list[float]) -> FloatArray:
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _param_index(name: str) -> int:
    return int(name[1:])


def _declared_count(pattern: re.Pattern[str], text: str, what: str) -> int:
    match = pattern.search(text)
    _require(match is not None, f"missing {what}")
    assert match is not None  # narrowed for mypy by _require
    return int(match.group(1))


def _difficulty(text: str) -> Difficulty | None:
    match = _DIFFICULTY_RE.search(text)
    if match is None:
        return None
    level = match.group(1).lower()
    if level == "lower":
        return "lower"
    if level == "average":
        return "average"
    return "higher"


def _labelled_float(text: str, label: str) -> float:
    match = re.search(rf"{label}\s*:?\s*({_NUMBER})", text)
    _require(match is not None, f"missing certified value '{label}'")
    assert match is not None
    return float(match.group(1))


def _model_formula(lines: list[str]) -> str:
    """Text between the ``N Parameters`` line and the starting-values table, whitespace-folded."""
    start = next((i for i, line in enumerate(lines) if _PARAMETERS_RE.match(line)), None)
    # NIST capitalises the table title inconsistently ("Starting values" / "Starting Values"),
    # and the file-format preamble mentions it too, so only look after the Parameters line.
    end = next(
        (
            i
            for i, line in enumerate(lines)
            if start is not None and i > start and "starting values" in line.lower()
        ),
        None,
    )
    _require(start is not None and end is not None and start < end, "missing model section")
    assert start is not None and end is not None
    formula = " ".join(" ".join(lines[start + 1 : end]).split())
    _require(bool(formula), "empty model formula")
    return formula


def _numbers(fragment: str, expected: int, context: str) -> list[float]:
    values = [float(token) for token in _NUMBER_RE.findall(fragment)]
    _require(len(values) == expected, f"{context}: expected {expected} numbers in {fragment!r}")
    return values


def _nls_parameter_table(
    lines: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    header = next((i for i, line in enumerate(lines) if "Start 1" in line), None)
    _require(header is not None, "missing 'Start 1' starting-values header")
    assert header is not None
    start1: dict[str, float] = {}
    start2: dict[str, float] = {}
    certified: dict[str, float] = {}
    certified_se: dict[str, float] = {}
    for line in lines[header + 1 :]:
        if not line.strip():
            if certified:
                break
            continue
        match = _NLS_ROW_RE.match(line)
        _require(match is not None, f"unreadable parameter row {line!r}")
        assert match is not None
        key = match.group(1)
        s1, s2, value, se = _numbers(match.group(2), 4, f"parameter {key}")
        start1[key], start2[key], certified[key], certified_se[key] = s1, s2, value, se
    _require(bool(certified), "no certified parameters found")
    return start1, start2, certified, certified_se


def _lls_parameter_table(lines: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    header = next(
        (i for i, line in enumerate(lines) if "Certified Regression Statistics" in line), None
    )
    _require(header is not None, "missing 'Certified Regression Statistics' section")
    assert header is not None
    certified: dict[str, float] = {}
    certified_se: dict[str, float] = {}
    for line in lines[header + 1 :]:
        if "Residual" in line:
            break
        match = _LLS_ROW_RE.match(line)
        if match is None:
            continue
        key = match.group(1)
        value, se = _numbers(match.group(2), 2, f"parameter {key}")
        certified[key], certified_se[key] = value, se
    _require(bool(certified), "no certified parameters found")
    return certified, certified_se


def _data_block(lines: list[str]) -> tuple[list[float], list[float]]:
    """Read the ``Data:  y  x`` block; the column order is checked, never assumed."""
    header = next((i for i, line in enumerate(lines) if _DATA_HEADER_RE.match(line)), None)
    _require(header is not None, "missing 'Data:  y  x' header (y must precede x)")
    assert header is not None
    ys: list[float] = []
    xs: list[float] = []
    for line in lines[header + 1 :]:
        if not line.strip():
            continue
        y_value, x_value = _numbers(line, 2, "data row")
        ys.append(y_value)
        xs.append(x_value)
    _require(bool(ys), "data block is empty")
    return xs, ys
