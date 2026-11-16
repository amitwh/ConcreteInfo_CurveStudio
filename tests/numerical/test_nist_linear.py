"""NIST StRD linear least-squares benchmarks: Norris (certified values) and Wampler1.

The Norris rows below were copied programmatically from ``Norris.dat`` (NIST/ITL StRD,
https://www.itl.nist.gov/div898/strd/lls/data/Norris.shtml); the certified values come from
the same file. Wampler1 is generated: y = 1 + x + x**2 + x**3 + x**4 + x**5 at x = 0..20.
"""

from pathlib import Path

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitStatus, fit_polynomial

NORRIS_Y = [
    0.1,
    338.8,
    118.1,
    888.0,
    9.2,
    228.1,
    668.5,
    998.5,
    449.1,
    778.9,
    559.2,
    0.3,
    0.1,
    778.1,
    668.8,
    339.3,
    448.9,
    10.8,
    557.7,
    228.3,
    998.0,
    888.8,
    119.6,
    0.3,
    0.6,
    557.6,
    339.3,
    888.0,
    998.5,
    778.9,
    10.2,
    117.6,
    228.9,
    668.4,
    449.2,
    0.2,
]
NORRIS_X = [
    0.2,
    337.4,
    118.2,
    884.6,
    10.1,
    226.5,
    666.3,
    996.3,
    448.6,
    777.0,
    558.2,
    0.4,
    0.6,
    775.5,
    666.9,
    338.0,
    447.5,
    11.6,
    556.0,
    228.1,
    995.8,
    887.6,
    120.2,
    0.3,
    0.3,
    556.8,
    339.1,
    887.2,
    999.0,
    779.0,
    11.1,
    118.3,
    229.2,
    669.1,
    448.9,
    0.5,
]

NORRIS_CERTIFIED = {
    "b0": -0.262323073774029,
    "b1": 1.00211681802045,
    "se_b0": 0.232818234301152,
    "se_b1": 0.429796848199937e-3,
    "residual_sd": 0.884796396144373,
    "r2": 0.999993745883712,
    "sse": 26.6173985294224,
}


def _xy(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="nist",
    )


def test_norris_matches_certified_values_to_9_digits() -> None:
    res = fit_polynomial(_xy(NORRIS_X, NORRIS_Y), 1)
    assert res.status is FitStatus.CONVERGED
    b0, b1 = res.parameters
    assert b0.value == pytest.approx(NORRIS_CERTIFIED["b0"], rel=1e-9)
    assert b1.value == pytest.approx(NORRIS_CERTIFIED["b1"], rel=1e-9)
    assert b0.std_err == pytest.approx(NORRIS_CERTIFIED["se_b0"], rel=1e-9)
    assert b1.std_err == pytest.approx(NORRIS_CERTIFIED["se_b1"], rel=1e-9)
    assert res.stats is not None
    assert res.stats.residual_se == pytest.approx(NORRIS_CERTIFIED["residual_sd"], rel=1e-9)
    assert res.stats.r2 == pytest.approx(NORRIS_CERTIFIED["r2"], rel=1e-9)
    assert res.stats.sse == pytest.approx(NORRIS_CERTIFIED["sse"], rel=1e-9)
    assert res.warnings == ()


def test_norris_literal_matches_downloaded_nist_file() -> None:
    path = Path(__file__).parents[1] / "data" / "nist_strd" / "linear" / "Norris.dat"
    rows = [line.split() for line in path.read_text().splitlines()[60:96]]
    assert [float(r[0]) for r in rows] == NORRIS_Y
    assert [float(r[1]) for r in rows] == NORRIS_X


def test_wampler1_recovers_all_ones() -> None:
    x = np.arange(21.0)
    y = sum(x**p for p in range(6))
    res = fit_polynomial(_xy(x, y), 5)
    assert res.status is FitStatus.CONVERGED
    assert [p.name for p in res.parameters] == ["b0", "b1", "b2", "b3", "b4", "b5"]
    assert [p.value for p in res.parameters] == pytest.approx([1.0] * 6, abs=1e-8)
    assert res.stats is not None
    assert res.stats.r2 == 1.0
    assert res.covariance is not None
    assert np.all(np.isfinite(np.array(res.covariance)))
