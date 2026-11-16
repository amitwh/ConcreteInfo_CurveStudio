# NIST StRD validation report

Generated: 2026-11-16T17:27:03+05:30 (IST)
Platform: Linux-7.0.0-43-generic-x86_64-with-glibc2.43
Versions: curvestudio 0.1.0.dev0 · numpy 2.5.5 · scipy 1.18.1 · lmfit 1.3.4 · sympy 1.14.0 · Python 3.14.8

Rule 1 — a CONVERGED fit must be right: minimum parameter LRE >= 5 and SSE LRE >= 7
(LRE = log relative error against the certified value, i.e. correct significant digits).

## Table 1 — nonlinear, single start

| Problem | Difficulty | Start | Status | Min param LRE | Min SE LRE | SSE LRE |
| --- | --- | --- | --- | --- | --- | --- |
| Misra1a | lower | start1 | converged | 9.8 | 9.5 | 10.5 |
| Misra1a | lower | start2 | converged | 10.1 | 9.8 | 10.5 |
| Misra1b | lower | start1 | converged | 9.5 | 9.3 | 11.3 |
| Misra1b | lower | start2 | converged | 9.3 | 9.1 | 11.3 |
| Misra1c | average | start1 | converged | 8.3 | 8.0 | 11.1 |
| Misra1c | average | start2 | converged | 10.1 | 10.0 | 11.1 |
| Misra1d | average | start1 | converged | 11.3 | 10.8 | 11.2 |
| Misra1d | average | start2 | converged | 11.6 | 10.9 | 11.2 |
| DanWood | lower | start1 | converged | 11.2 | 11.1 | 11.7 |
| DanWood | lower | start2 | converged | 10.7 | 10.7 | 11.7 |
| Chwirut2 | lower | start1 | converged | 7.5 | 7.9 | 11.2 |
| Chwirut2 | lower | start2 | converged | 8.6 | 9.0 | 11.2 |
| Lanczos3 | lower | start1 | converged | 8.2 | 8.2 | 10.6 |
| Lanczos3 | lower | start2 | converged | 7.9 | 7.9 | 10.6 |
| Gauss1 | lower | start1 | converged | 10.6 | 10.6 | 11.6 |
| Gauss1 | lower | start2 | converged | 8.6 | 8.4 | 11.6 |
| BoxBOD | higher | start1 | converged | 8.0 | 7.9 | 10.4 |
| BoxBOD | higher | start2 | converged | 8.5 | 8.4 | 10.4 |
| Rat42 | higher | start1 | converged | 9.2 | 8.8 | 11.9 |
| Rat42 | higher | start2 | converged | 8.3 | 8.0 | 11.9 |
| MGH09 | higher | start1 | not_converged | n/a | n/a | n/a |
| MGH09 | higher | start2 | converged | 7.0 | 7.1 | 11.6 |
| Eckerle4 | higher | start1 | converged | 9.7 | 9.7 | 10.7 |
| Eckerle4 | higher | start2 | converged | 9.6 | 9.6 | 10.7 |

## Table 2 — nonlinear, multistart (20 starts, seed 0, bounds ±1000·|start|)

| Problem | Difficulty | Start | Status | Min param LRE | SSE LRE | Converged starts | Matching certified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Misra1a | lower | start1 | converged | 8.3 | 10.5 | 10/20 | 10/20 |
| Misra1a | lower | start2 | converged | 9.6 | 10.5 | 11/20 | 9/20 |
| Misra1b | lower | start1 | converged | 9.4 | 11.3 | 16/20 | 7/20 |
| Misra1b | lower | start2 | converged | 10.0 | 11.3 | 18/20 | 5/20 |
| Misra1c | average | start1 | converged | 10.8 | 11.1 | 9/20 | 8/20 |
| Misra1c | average | start2 | converged | 10.8 | 11.1 | 8/20 | 8/20 |
| Misra1d | average | start1 | converged | 8.5 | 11.2 | 18/20 | 8/20 |
| Misra1d | average | start2 | converged | 9.0 | 11.2 | 19/20 | 8/20 |
| DanWood | lower | start1 | converged | 9.9 | 11.7 | 11/20 | 1/20 |
| DanWood | lower | start2 | converged | 10.7 | 11.7 | 12/20 | 2/20 |
| Chwirut2 | lower | start1 | converged | 9.1 | 11.2 | 14/20 | 3/20 |
| Chwirut2 | lower | start2 | converged | 8.9 | 11.2 | 12/20 | 3/20 |
| Lanczos3 | lower | start1 | converged | 7.2 | 10.6 | 7/20 | 1/20 |
| Lanczos3 | lower | start2 | converged | 7.4 | 10.6 | 5/20 | 1/20 |
| Gauss1 | lower | start1 | converged | 10.6 | 11.6 | 12/20 | 1/20 |
| Gauss1 | lower | start2 | converged | 8.6 | 11.6 | 11/20 | 1/20 |
| BoxBOD | higher | start1 | converged | 8.6 | 10.4 | 10/20 | 4/20 |
| BoxBOD | higher | start2 | converged | 8.5 | 10.4 | 10/20 | 5/20 |
| Rat42 | higher | start1 | converged | 9.2 | 11.9 | 7/20 | 1/20 |
| Rat42 | higher | start2 | converged | 9.5 | 11.9 | 6/20 | 1/20 |
| MGH09 | higher | start1 | converged | 6.9 | 11.6 | 8/20 | 1/20 |
| MGH09 | higher | start2 | converged | 7.0 | 11.6 | 18/20 | 2/20 |
| Eckerle4 * | higher | start1 | converged | 10.1 | 10.7 | 20/20 | 2/20 |
| Eckerle4 | higher | start2 | converged | 9.6 | 10.7 | 20/20 | 1/20 |

## Table 3 — linear (fit_polynomial)

| Problem | Status | Min coefficient LRE | Warnings |
| --- | --- | --- | --- |
| Norris | converged | 12.8 | - |
| Pontius | converged | 12.1 | ILL_CONDITIONED |
| Wampler1 | converged | 9.6 | - |
| Wampler4 | converged | 8.0 | - |
| Filip | converged | 7.4 | HIGH_ORDER_POLYNOMIAL, ILL_CONDITIONED |

## Summary

- Nonlinear single-start CONVERGED with rule 1 satisfied: 23/24
- Nonlinear single-start NOT_CONVERGED: 1
- Rule-1 violations: 0 (must be 0)
- Lower-difficulty problems converged from both starts: 6/6
- Linear problems passing: 5/5

## Notes

- Eckerle4, (b1/b2) * exp(-0.5*((x-b3)/b2)**2), is invariant under (b1, b2) -> (-b1, -b2) with b3 unchanged: both sign choices fit the data equally well, so the data cannot tell them apart. The row marked * matched that mirror minimum rather than the certified signs, and its min param LRE is measured against the certified values or their mirror, whichever is closer.
- Fits use the engine's default ftol = 1e-14 rather than SciPy's 1e-10: ftol bounds the relative SSE change and the parameters are only resolved to about its square root, so 1e-10 let ill-conditioned fits report CONVERGED with about 5 correct digits (NIST MGH09).
- Single-start caveat: a local optimiser run from one start can report CONVERGED at a stationary point that is not the certified minimum (a plateau or secondary minimum with a genuinely small gradient); the multistart runs in Table 2 inside wide bounds are the remedy.
- 'Converged starts' and 'Matching certified' are measured by replaying the multistart: curvestudio.fitting.nonlinear.generate_starts regenerates the engine's own 20 starts (same base, bounds and seed) and each is re-run as an independent single-start fit with the same defaults. A start counts as matching when it converges with param LRE >= 5 and SSE LRE >= 7 against the certified values (Eckerle4: or their sign mirror).
- Generated by scripts/nist_report.py (uv run python scripts/nist_report.py); tests/numerical/test_nist_report.py checks the renderer and a restricted build.
