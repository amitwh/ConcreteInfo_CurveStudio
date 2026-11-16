"""NIST StRD parser, LRE measure and the fetch script (no real network access)."""

import importlib.util
import math
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from curvestudio.testing.nist import (
    NistLinearProblem,
    NistProblem,
    load_lls,
    load_nls,
    lre,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "tests" / "data" / "nist_strd"

# name -> (observations, parameters) as published by NIST.
NLS_EXPECTED = {
    "Misra1a": (14, 2),
    "Misra1b": (14, 2),
    "Misra1c": (14, 2),
    "Misra1d": (14, 2),
    "DanWood": (6, 2),
    "Chwirut2": (54, 3),
    "Lanczos3": (24, 6),
    "Gauss1": (250, 8),
    "BoxBOD": (6, 2),
    "Rat42": (9, 3),
    "MGH09": (11, 4),
    "Eckerle4": (35, 3),  # the dispatch said 4; the file declares 3 (df 32 = 35 - 3)
}
# name -> (observations, polynomial degree)
LLS_EXPECTED = {
    "Norris": (36, 1),
    "Pontius": (40, 2),
    "Wampler1": (21, 5),
    "Wampler4": (21, 5),
    "Filip": (82, 10),
}


def _load_fetch_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fetch_nist_strd", REPO_ROOT / "scripts" / "fetch_nist_strd.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_module = _load_fetch_module()


# --------------------------------------------------------------------------- brief assertions
def test_misra1a_certified_values() -> None:
    p = load_nls(DATA / "nonlinear" / "Misra1a.dat")
    assert p.name == "Misra1a"
    assert len(p.x) == len(p.y) == 14
    assert p.difficulty == "lower"
    assert p.start1 == {"b1": 500, "b2": 1e-4}
    assert p.start2 == {"b1": 250, "b2": 5e-4}
    assert p.certified["b1"] == 2.3894212918e02
    assert p.certified_se["b2"] == 7.2668688436e-06
    assert p.certified_sse == 1.2455138894e-01
    assert p.certified_residual_sd == 1.0187876330e-01
    assert p.df == 12
    assert p.model_formula == "y = b1*(1-exp[-b2*x]) + e"


def test_misra1a_data_columns_are_y_then_x() -> None:
    p = load_nls(DATA / "nonlinear" / "Misra1a.dat")
    assert (p.y[0], p.x[0]) == (10.07, 77.6)
    assert (p.y[-1], p.x[-1]) == (p.y[13], p.x[13])


def test_norris_certified_values() -> None:
    p = load_lls(DATA / "linear" / "Norris.dat")
    assert len(p.x) == len(p.y) == 36
    assert p.certified["B1"] == 1.00211681802045
    assert p.certified["B0"] == -0.262323073774029
    assert p.certified_se["B1"] == 0.429796848199937e-03
    assert p.residual_sd == 0.884796396144373
    assert p.r2 == 0.999993745883712
    assert p.degree == 1
    assert (p.y[0], p.x[0]) == (0.1, 0.2)


def test_lre_basics() -> None:
    assert lre(1.0, 1.0) == 15
    assert lre(1.001, 1.0) == pytest.approx(3, abs=1e-6)
    assert lre(0.0005, 0.0) == pytest.approx(-math.log10(0.0005))
    assert lre(float("nan"), 1.0) == 0.0
    assert lre(3.0, 1.0) == 0.0


def test_lre_edge_cases() -> None:
    assert lre(float("inf"), 1.0) == 0.0
    assert lre(0.0, 0.0) == 15
    assert lre(1.0 + 1e-14, 1.0) == pytest.approx(14, abs=0.01)
    assert lre(1.0 + 1e-17, 1.0) == 15  # rounds to 1.0 exactly
    assert lre(-1.0, 1.0) == 0.0  # error 2x the certified magnitude: floored, never negative
    assert lre(2.0, 0.0) == 0.0  # absolute error >= 1 for a zero certified value
    assert lre(-1.001, -1.0) == pytest.approx(3, abs=1e-6)
    with pytest.raises(ValueError, match="certified"):
        lre(1.0, float("nan"))


# --------------------------------------------------------------------------- all files
@pytest.mark.parametrize(("name", "expected"), list(NLS_EXPECTED.items()))
def test_every_nonlinear_file_parses(name: str, expected: tuple[int, int]) -> None:
    n_obs, n_params = expected
    p = load_nls(DATA / "nonlinear" / f"{name}.dat")
    assert isinstance(p, NistProblem)
    assert p.name == name
    assert len(p.x) == len(p.y) == n_obs
    assert np.isfinite(p.x).all()
    assert np.isfinite(p.y).all()
    assert len(p.certified) == n_params
    keys = {f"b{i}" for i in range(1, n_params + 1)}
    assert set(p.start1) == set(p.start2) == set(p.certified) == set(p.certified_se) == keys
    assert all(
        math.isfinite(v)
        for d in (p.start1, p.start2, p.certified, p.certified_se)
        for v in d.values()
    )
    assert p.certified_sse > 0
    assert p.certified_residual_sd > 0
    assert p.df == len(p.y) - len(p.certified)  # certified df agrees with n - k
    assert p.model_formula.startswith("y")
    assert "  " not in p.model_formula  # whitespace normalised
    assert p.difficulty in {"lower", "average", "higher"}
    # Residual SD is sqrt(SSE / df): ties the three certified scalars together.
    assert p.certified_residual_sd == pytest.approx(math.sqrt(p.certified_sse / p.df), rel=1e-9)
    assert not p.x.flags.writeable
    assert not p.y.flags.writeable


def test_no_nonlinear_difficulty_is_none() -> None:
    levels = {
        name: load_nls(DATA / "nonlinear" / f"{name}.dat").difficulty for name in NLS_EXPECTED
    }
    assert None not in levels.values()
    assert levels["Misra1c"] == "average"
    assert levels["MGH09"] == "higher"
    assert levels["Gauss1"] == "lower"


def test_multiline_formula_is_joined() -> None:
    p = load_nls(DATA / "nonlinear" / "Gauss1.dat")
    assert p.model_formula == (
        "y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 ) + b6*exp( -(x-b7)**2 / b8**2 ) + e"
    )


@pytest.mark.parametrize(("name", "expected"), list(LLS_EXPECTED.items()))
def test_every_linear_file_parses(name: str, expected: tuple[int, int]) -> None:
    n_obs, degree = expected
    p = load_lls(DATA / "linear" / f"{name}.dat")
    assert isinstance(p, NistLinearProblem)
    assert p.name == name
    assert len(p.x) == len(p.y) == n_obs
    assert np.isfinite(p.x).all()
    assert np.isfinite(p.y).all()
    assert p.degree == degree
    keys = {f"B{i}" for i in range(degree + 1)}
    assert set(p.certified) == set(p.certified_se) == keys
    assert all(math.isfinite(v) for v in (*p.certified.values(), *p.certified_se.values()))
    assert p.residual_sd >= 0
    assert 0 < p.r2 <= 1


def test_filip_exponent_and_sign_parsing() -> None:
    p = load_lls(DATA / "linear" / "Filip.dat")
    assert p.certified["B0"] == -1467.48961422980
    assert p.certified["B8"] == -0.670191154593408e-01
    assert p.certified_se["B10"] == 0.896632837373868e-05
    assert p.residual_sd == 0.334801051324544e-02
    assert p.x[0] == -6.860120914
    assert p.y[0] == 0.8116


def test_pontius_tiny_coefficient_and_trailing_whitespace() -> None:
    p = load_lls(DATA / "linear" / "Pontius.dat")
    assert p.certified["B2"] == -0.316081871345029e-14
    assert p.certified_se["B2"] == 0.486652849992036e-16


# --------------------------------------------------------------------------- error handling
@pytest.mark.parametrize("loader", [load_nls, load_lls])
def test_missing_file_raises_file_not_found(loader: Any, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "Nope.dat")


def _truncate(source: Path, target: Path, keep_lines: int) -> Path:
    text = "\n".join(source.read_text(encoding="ascii").splitlines()[:keep_lines])
    target.write_text(text, encoding="ascii")
    return target


@pytest.mark.parametrize("keep_lines", [0, 5, 30, 45, 60, 65])
def test_load_nls_truncated_file_raises_value_error(tmp_path: Path, keep_lines: int) -> None:
    broken = _truncate(DATA / "nonlinear" / "Misra1a.dat", tmp_path / "Misra1a.dat", keep_lines)
    with pytest.raises(ValueError, match=r"Misra1a\.dat"):
        load_nls(broken)


@pytest.mark.parametrize("keep_lines", [0, 5, 20, 31, 50, 62, 70])
def test_load_lls_truncated_file_raises_value_error(tmp_path: Path, keep_lines: int) -> None:
    broken = _truncate(DATA / "linear" / "Norris.dat", tmp_path / "Norris.dat", keep_lines)
    with pytest.raises(ValueError, match=r"Norris\.dat"):
        load_lls(broken)


def test_load_nls_garbled_number_raises_value_error(tmp_path: Path) -> None:
    text = (DATA / "nonlinear" / "Misra1a.dat").read_text(encoding="ascii")
    broken = tmp_path / "Misra1a.dat"
    broken.write_text(text.replace("2.3894212918E+02", "garbage"), encoding="ascii")
    with pytest.raises(ValueError, match="Misra1a"):
        load_nls(broken)


def test_load_nls_observation_count_mismatch_raises(tmp_path: Path) -> None:
    text = (DATA / "nonlinear" / "Misra1a.dat").read_text(encoding="ascii")
    broken = tmp_path / "Misra1a.dat"
    broken.write_text(text.replace("14 Observations", "15 Observations", 1), encoding="ascii")
    with pytest.raises(ValueError, match="15 observations"):
        load_nls(broken)


def test_load_nls_swapped_columns_header_raises(tmp_path: Path) -> None:
    text = (DATA / "nonlinear" / "Misra1a.dat").read_text(encoding="ascii")
    broken = tmp_path / "Misra1a.dat"
    broken.write_text(text.replace("Data:   y               x", "Data:   x               y"))
    with pytest.raises(ValueError, match="y must precede x"):
        load_nls(broken)


def test_load_lls_garbled_file_raises_value_error(tmp_path: Path) -> None:
    broken = tmp_path / "Junk.dat"
    broken.write_text("this is not a NIST file\n12 34 56\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"Junk\.dat"):
        load_lls(broken)


# --------------------------------------------------------------------------- fetch script
class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


@pytest.fixture
def urlopen_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        assert timeout is not None  # a timeout is mandatory
        calls.append(request.full_url)
        return _FakeResponse(b"NIST/ITL StRD\r\nfresh body\r\n")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", fake_urlopen)
    yield calls


def test_fetch_skips_existing_files_and_writes_bytes_unchanged(
    tmp_path: Path, urlopen_calls: list[str]
) -> None:
    existing = tmp_path / "linear" / "Norris.dat"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"committed original\r\n")

    written = fetch_module.download_all(tmp_path)

    assert existing.read_bytes() == b"committed original\r\n"  # untouched
    assert existing not in written
    assert len(written) == 16  # 12 nonlinear + 5 linear - 1 pre-existing
    assert not any(url.endswith("/Norris.dat") for url in urlopen_calls)
    new_file = tmp_path / "nonlinear" / "Misra1a.dat"
    assert new_file.read_bytes() == b"NIST/ITL StRD\r\nfresh body\r\n"  # CRLF preserved
    assert "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat" in urlopen_calls
    assert "https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Filip.dat" in urlopen_calls


def test_fetch_force_overwrites_existing(tmp_path: Path, urlopen_calls: list[str]) -> None:
    existing = tmp_path / "linear" / "Norris.dat"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    fetch_module.download_all(tmp_path, force=True)
    assert existing.read_bytes() == b"NIST/ITL StRD\r\nfresh body\r\n"
    assert len(urlopen_calls) == 17


def test_fetch_non_200_exits_with_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fetch_module.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"x", status=404)
    )
    with pytest.raises(SystemExit, match=r"Misra1a\.dat.*404"):
        fetch_module.download_all(tmp_path)
    assert not (tmp_path / "nonlinear" / "Misra1a.dat").exists()


def test_fetch_empty_body_exits_with_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fetch_module.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"  \r\n")
    )
    with pytest.raises(SystemExit, match=r"Misra1a\.dat.*empty"):
        fetch_module.download_all(tmp_path)


def test_fetch_network_error_exits_with_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError("u", 403, "Forbidden", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match=r"Misra1a\.dat.*403"):
        fetch_module.download_all(tmp_path)
