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

## Exit codes of `fit`

| Code | Meaning                                                                       |
|------|-------------------------------------------------------------------------------|
| 0    | the fit converged                                                             |
| 2    | invalid input: unreadable file, missing column, unknown model, validation errors, or a fit that cannot run (too few observations for the parameters, or a model that does not apply) |
| 3    | the fit ran but did not converge or failed                                    |

## Auditing a fit, and writing it up

Two more commands take the same data arguments as `fit`, and read the fit through the same
code the desktop and web apps use:

```bash
uv run curvestudio audit examples/decay.csv --x x --y y --model exp_decay_offset --cv 5
uv run curvestudio methods examples/decay.csv --x x --y y --model exp_decay_offset --style markdown
```

`audit` prints an overall verdict, a summary, and then each check with its verdict, the
evidence behind it and what to do next. `--cv K` (or `--cv loo`) adds the cross-validation
check. `--ranking` also runs an Auto-fit, which takes as long as the `autofit` command, to
name a simpler model that fits as well. Its exit codes are `0` for pass, `1` for caution or
for nothing that could be checked, `2` for fail, and `3` for a usage or input error.

`methods` prints the methods paragraph, the equation with each estimate and its standard
error, the parameter table, the citations and what the author must complete, as plain text,
Markdown (`--style markdown`) or LaTeX (`--style latex`). Its exit codes are `0`, and `3` for
a usage or input error or a fit that did not converge.

Both take `--json` for the result as JSON on standard output.

## Running a statistical tool

`stats` runs one of the standard statistical tools on a file and prints what it returns. The
tools and their options come from the same catalogue the desktop and web apps read, so the
command line takes the options those apps take.

```bash
uv run curvestudio stats --list                       # every tool, grouped by family
uv run curvestudio stats descriptives examples/decay.csv --values y
uv run curvestudio stats regression examples/decay.csv --outcome y --predictors x \
    --predictions "2.0;4.0"
uv run curvestudio stats power --design independent --effect-size 0.5 --power 0.8
```

A column is `--<field> COL`, several columns are `--<field> COL,COL`, and an option is a flag
of its own type: a number, an integer, a switch such as `--equal-variances`, or a choice such
as `--design`. Prediction rows are numbers comma-separated within a row and rows separated by
semicolons: `"1.2,3.4;2.0,4.1"`.

Without `--json` the report prints the estimates and the tests as tables, the assumptions
with their verdicts, the warnings, every table, the interpretation and the methods fragment,
then the counts and the rows left out. `--json` prints the result exactly as the tool returns
it, for a script to read. `power` is the one tool that needs no file.

Its exit codes are `0` when the tool ran, and `3` for a usage or input error, with the tool's
code and message on standard error.

## The fit tools: discernibility, feasibility, next measurement

Three commands read a fit and answer a question about it: whether the data can tell the
leading models apart, how well a design would recover a model's parameters, and where to
measure next. Each prints the one result the desktop and web apps render; `--json` prints it
exactly as the tool returns it.

```bash
# Can these data tell the leading models apart? Runs Auto-fit with the autofit defaults.
uv run curvestudio discern examples/decay.csv --x x --y y

# How well would this design recover the model's parameters?
uv run curvestudio simulate --model exp_growth --design 0:4:5 --noise 0.1 --runs 200 --seed 7
uv run curvestudio simulate --model exp_decay_offset --from-file examples/decay.csv --x x --y y \
    --design 0:10:5 --noise 0.1 --difference a=0.5

# Where should the next measurements be taken?
uv run curvestudio advise examples/decay.csv --x x --y y --model exp_decay_offset --points 3
```

`simulate` needs no file: without `--params` a library model's `example` supplies the true
values (the note on standard error says so), and without `--design` or `--at` it uses five
evenly spaced points over the example's range. `--cv` simulates multiplicative noise instead
of `--noise`'s constant standard deviation. `advise` fits through the same workflow `audit`
uses; `--parameter NAME` minimises that parameter's standard error instead of using
D-optimality. Their exit codes are `0` when the tool ran and `3` for a usage or input error,
with the tool's code and message on standard error.

## Row numbers in messages

Row numbers in validation errors and data warnings are **1-based data rows**: the
first data row after the header is row 1. The header line is not counted, and fully
blank rows that the importer drops are not counted either. (Internally the data
layer reports 0-based table positions; the CLI displays them as data-row numbers.)
