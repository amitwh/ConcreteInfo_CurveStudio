# Statsmodels cross-check report

Reference: Statsmodels 0.15.0 (and SciPy for Shapiro-Wilk) on 6 stored datasets (`tests/data/crosscheck/statsmodels.json`).
Regenerate with `uv run python scripts/crosscheck_report.py`; `tests/numerical/test_crosscheck_report.py` fails when this file drifts.

Each row re-runs our implementation on every dataset that carries the quantity and reports the maximum observed deviation from the stored Statsmodels value. *Relative* deviation is `max |ours - ref| / |ref|`, with matrix and vector quantities judged as `assert_allclose(rtol, atol=1e-12 * max|ref|)` does, so that entries that are numerically zero next to the rest of the matrix are not compared by ratio; *absolute* is `max |ours - ref|` (p-values). Deviations are rounded up to the next even power of ten (nothing is shown below `1e-10`) so that floating-point round-off differences between machines cannot change this file.

## Cross-checked quantities

| Quantity | Statsmodels reference | Datasets | Kind | Max deviation | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter standard errors | OLS/WLS `bse` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Parameter covariance matrix | OLS/WLS `cov_params()` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Mean (confidence) band: fit, lower, upper | `get_prediction().summary_frame()` mean, mean_ci_lower, mean_ci_upper | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Prediction band: lower, upper | `get_prediction().summary_frame()` obs_ci_lower, obs_ci_upper | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| AIC differences between polynomial degrees 1-4 | OLS `aic` (differences only; absolute levels differ by the dropped constants) | heavy60, linear30, quad40 | relative | ≤ 1e-10 | 1e-9 | pass |
| BIC differences between polynomial degrees 1-4 | OLS `bic` (differences only; absolute levels differ by the dropped constants) | heavy60, linear30, quad40 | relative | ≤ 1e-10 | 1e-9 | pass |
| Leverage (hat-matrix diagonal) | `OLSInfluence.hat_matrix_diag` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Cook's distance | `OLSInfluence.cooks_distance` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Internally studentized (standardized) residuals | `OLSInfluence.resid_studentized_internal` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Externally studentized residuals | `OLSInfluence.resid_studentized_external` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| PRESS residuals and PRESS RMSE | `OLSInfluence.resid_press` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-9 | pass |
| Shapiro-Wilk W | `scipy.stats.shapiro` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-8 | pass |
| Shapiro-Wilk p-value | `scipy.stats.shapiro` | ar50, cubic8, heavy60, linear30, quad40, wls35 | absolute | ≤ 1e-10 | 1e-6 | pass |
| Anderson-Darling A²* (small-sample adjusted) | `statsmodels.stats.diagnostic.normal_ad` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-8 | pass |
| Anderson-Darling p-value | `statsmodels.stats.diagnostic.normal_ad` | ar50, cubic8, heavy60, linear30, quad40, wls35 | absolute | ≤ 1e-10 | 1e-6 | pass |
| Breusch-Pagan LM statistic | `statsmodels.stats.diagnostic.het_breuschpagan` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-8 | pass |
| Breusch-Pagan p-value | `statsmodels.stats.diagnostic.het_breuschpagan` | ar50, cubic8, heavy60, linear30, quad40, wls35 | absolute | ≤ 1e-10 | 1e-6 | pass |
| Ljung-Box Q (lags 1 and 5) | `statsmodels.stats.diagnostic.acorr_ljungbox` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-8 | pass |
| Ljung-Box p-value (lags 1 and 5) | `statsmodels.stats.diagnostic.acorr_ljungbox` | ar50, cubic8, heavy60, linear30, quad40, wls35 | absolute | ≤ 1e-10 | 1e-6 | pass |
| Durbin-Watson statistic | `statsmodels.stats.stattools.durbin_watson` | ar50, cubic8, heavy60, linear30, quad40, wls35 | relative | ≤ 1e-10 | 1e-8 | pass |

## Summary

- Quantities cross-checked: 20
- Within tolerance: 20/20

## Notes

- Information criteria use our convention (`k' = k + 1`, constants common to all models dropped), while Statsmodels charges `df_model + 1` parameters and keeps the constants; only criterion *differences* are comparable, and only those are checked here.
- The weighted dataset (`wls35`) is fitted with `fit_polynomial` weights; its influence measures are compared with the equivalent OLS fit of the whitened data because `OLSInfluence` mishandles WLS (see `scripts/make_crosscheck_fixtures.py`).
- Tests are run on the weighted residuals in ascending-x order; Ljung-Box is checked at lags 1 and 5, and the larger of the two deviations is reported.
