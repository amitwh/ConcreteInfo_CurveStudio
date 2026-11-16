# CurveStudio examples

## decay.csv

One hundred noisy observations of an exponential decay relaxing towards an offset
asymptote: `y = 5*exp(-0.4*x) + 2 + N(0, 0.1)`, with the noise drawn once from
NumPy's `default_rng(42)` so the file is reproducible byte for byte.

Regenerate it from the repository root with:

```bash
uv run python -c "import numpy as np; x = np.linspace(0, 10, 100); y = 5*np.exp(-0.4*x) + 2 + np.random.default_rng(42).normal(0, 0.1, 100); np.savetxt('examples/decay.csv', np.column_stack([x, y]), fmt='%.12g', delimiter=',', header='x,y', comments='')"
```

## Fitting it

The vertical slice the CLI is built around — import, validate, fit, report:

```bash
uv run curvestudio fit examples/decay.csv --x x --y y --model exp_decay_offset
```

which prints the fitted equation, a parameter table (value, standard error, 95 % CI),
R², adjusted R², RMSE, the status line and any warnings.

Useful variations:

```bash
uv run curvestudio fit examples/decay.csv --x x --y y --poly 2            # polynomial least squares
uv run curvestudio fit examples/decay.csv --x x --y y --model exp_decay_offset \
    --method trf --loss huber --starts 5 --seed 7 --json out.json
uv run curvestudio models                                                 # list model ids, names, categories
uv run curvestudio fit --help                                             # full option reference
```

## Exit codes

| Code | Meaning                                                                       |
|------|-------------------------------------------------------------------------------|
| 0    | the fit converged                                                             |
| 2    | invalid input: unreadable file, missing column, unknown model, validation errors, or a fit that cannot run (too few observations for the parameters, or a model that does not apply) |
| 3    | the fit ran but did not converge or failed                                    |

## Row numbers in messages

Row numbers in validation errors and data warnings are **1-based data rows**: the
first data row after the header is row 1. The header line is not counted, and fully
blank rows that the importer drops are not counted either. (Internally the data
layer reports 0-based table positions; the CLI displays them as data-row numbers.)
