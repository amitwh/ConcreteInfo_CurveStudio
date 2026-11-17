"""Generate the Statsmodels cross-check fixtures (tests/data/crosscheck/statsmodels.json).

Six datasets are built from fixed seeds and their ``x``, ``y`` and ``w`` are stored inside
the JSON, so the tests never depend on a NumPy random stream. For each dataset the
reference is Statsmodels OLS/WLS on the polynomial design ``1, x, ..., x**degree``.

The fixtures are committed; run this script only when a dataset or a stored statistic is
added or Statsmodels is upgraded, and review the diff::

    uv run python scripts/make_crosscheck_fixtures.py

Output is deterministic: no timestamps, floats written with ``repr`` precision.
"""

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "crosscheck" / "statsmodels.json"
CONFIDENCE_ALPHA = 0.05


def _polynomial_truth(x: FloatArray, coefficients: tuple[float, ...]) -> FloatArray:
    return np.asarray(np.polynomial.polynomial.polyval(x, coefficients), dtype=np.float64)


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> FloatArray:
    """Stationary AR(1) noise ``e[t] = rho * e[t-1] + eps[t]`` with marginal sd ``sigma``."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - rho**2), n)
    e = np.empty(n)
    e[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + eps[t]
    return e


def build_datasets() -> dict[str, dict[str, Any]]:
    """The six datasets: x, y, optional weights, polynomial degree."""
    rng = np.random.default_rng(20261117)
    out: dict[str, dict[str, Any]] = {}

    x = np.linspace(0.0, 10.0, 30)
    y = _polynomial_truth(x, (2.0, 1.5)) + rng.normal(0.0, 1.0, x.size)
    out["linear30"] = {"degree": 1, "x": x, "y": y, "w": None}

    x = np.linspace(-3.0, 3.0, 40)
    y = _polynomial_truth(x, (1.0, -2.0, 0.8)) + rng.normal(0.0, 0.7, x.size)
    out["quad40"] = {"degree": 2, "x": x, "y": y, "w": None}

    x = np.linspace(1.0, 10.0, 35)
    y = _polynomial_truth(x, (3.0, 2.0)) + rng.normal(0.0, 1.0, x.size) * x * 0.4
    out["wls35"] = {"degree": 1, "x": x, "y": y, "w": 1.0 / x**2}

    x = np.linspace(0.0, 5.0, 50)
    y = _polynomial_truth(x, (1.0, 0.5, -0.1)) + _ar1(rng, x.size, 0.7, 0.5)
    out["ar50"] = {"degree": 2, "x": x, "y": y, "w": None}

    x = np.linspace(0.0, 6.0, 60)
    y = _polynomial_truth(x, (0.5, 1.0)) + 0.6 * rng.standard_t(3, x.size)
    out["heavy60"] = {"degree": 1, "x": x, "y": y, "w": None}

    x = np.linspace(-1.0, 1.0, 8)
    y = _polynomial_truth(x, (0.2, 1.0, -0.5, 0.8)) + rng.normal(0.0, 0.1, x.size)
    out["cubic8"] = {"degree": 3, "x": x, "y": y, "w": None}
    return out


def reference_fit(entry: dict[str, Any]) -> dict[str, Any]:
    """Statsmodels OLS (no weights) or WLS (weights) on the polynomial design."""
    x: FloatArray = entry["x"]
    design = np.vander(x, entry["degree"] + 1, increasing=True)
    if entry["w"] is None:
        res = sm.OLS(entry["y"], design).fit()
    else:
        res = sm.WLS(entry["y"], design, weights=entry["w"]).fit()
    return {
        "params": res.params.tolist(),
        "bse": res.bse.tolist(),
        "cov": np.asarray(res.cov_params()).tolist(),
        "conf_int": np.asarray(res.conf_int(alpha=CONFIDENCE_ALPHA)).tolist(),
        "df_resid": int(res.df_resid),
        "ssr": float(res.ssr),
        "scale": float(res.scale),
    }


def information_criteria_references(entry: dict[str, Any]) -> dict[str, Any]:
    """Statsmodels AIC/BIC/log-likelihood for polynomial degrees 1-4 of one dataset.

    Statsmodels charges ``df_model + k_constant`` parameters and does not count the error
    variance, so the cross-check compares differences and a known constant offset rather
    than the absolute values.
    """
    x: FloatArray = entry["x"]
    references: dict[str, Any] = {}
    for degree in (1, 2, 3, 4):
        design = np.vander(x, degree + 1, increasing=True)
        if entry["w"] is None:
            res = sm.OLS(entry["y"], design).fit()
        else:
            res = sm.WLS(entry["y"], design, weights=entry["w"]).fit()
        references[str(degree)] = {
            "aic": float(res.aic),
            "bic": float(res.bic),
            "llf": float(res.llf),
            "nobs": int(res.nobs),
            "df_model": int(res.df_model),
        }
    return references


def build_fixtures() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name, entry in build_datasets().items():
        datasets[name] = {
            "degree": entry["degree"],
            "weighted": entry["w"] is not None,
            "x": entry["x"].tolist(),
            "y": entry["y"].tolist(),
            "w": None if entry["w"] is None else entry["w"].tolist(),
            "fit": reference_fit(entry),
        }
        if name in ("linear30", "quad40", "heavy60"):
            datasets[name]["criteria"] = information_criteria_references(entry)
    return {
        "meta": {
            "generator": "scripts/make_crosscheck_fixtures.py",
            "statsmodels": metadata.version("statsmodels"),
            "numpy": metadata.version("numpy"),
            "confidence_alpha": CONFIDENCE_ALPHA,
        },
        "datasets": datasets,
    }


def main() -> int:
    fixtures = build_fixtures()
    dataset_lines = [
        f"  {json.dumps(name)}: {json.dumps(entry)}" for name, entry in fixtures["datasets"].items()
    ]
    text = (
        "{\n"
        f'"meta": {json.dumps(fixtures["meta"])},\n'
        '"datasets": {\n' + ",\n".join(dataset_lines) + "\n}\n}\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(fixtures['datasets'])} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
