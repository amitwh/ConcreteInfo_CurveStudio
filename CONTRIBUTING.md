# Contributing to CurveStudio

Thanks for helping. CurveStudio is licensed under **AGPL-3.0-or-later**. By submitting a
contribution you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence of contributions

Inbound = outbound: you license your contribution under the same terms as the project,
AGPL-3.0-or-later. You confirm that you wrote it or have the right to submit it under that licence.
There is no separate contributor agreement.

## Development setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:amitwh/ConcreteInfo_CurveStudio.git
cd ConcreteInfo_CurveStudio
uv sync
```

Checks that CI runs on Linux and Windows (run them before opening a pull request):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/core/src
uv run lint-imports
uv run pytest -m "not slow"
```

Numerical benchmarks that take more than 5 seconds are marked `@pytest.mark.slow`.
Tests live in `tests/unit`, `tests/integration` and `tests/numerical`.

## Architecture rule

`curvestudio-core` (`packages/core`) must stay free of GUI and web code. It imports only NumPy,
SciPy, LMfit, Statsmodels, SymPy, Pandas, Pydantic (and openpyxl for XLSX). Importing `PySide6`,
`fastapi`, `sqlalchemy`, `starlette` or `uvicorn` fails `uv run lint-imports` and
`tests/unit/test_architecture.py`.

## Rules for models

Every model added to the library must come with all of the following (PRD section 24):

1. **Equation**: the exact functional form.
2. **Parameter definitions**: meaning, units, bounds and starting-value strategy for each parameter.
3. **References**: the standard, paper or textbook it comes from.
4. **Tests**: at least one case with a known answer (published values or synthetic data with known
   parameters).
5. **Assumptions**: valid range, known failure modes and identifiability limits.

A model that lacks any of these is not merged. Use the "Model request" issue template to propose one.

## Principles

- Never silently drop or alter observations; every cleaning step is a recorded, reversible operation.
- A fit that fails is reported as failed, never replaced by a plausible-looking result.
- Never `eval`, `exec` or unpickle user-supplied content.
- Weights `w_i` multiply squared residuals (minimise the sum of `w_i * r_i^2`).
- Timestamps are timezone-aware ISO 8601 with the IST offset (`+05:30`).

## Pull requests

Keep changes focused, write the test first, and fill in the pull request template. Commit messages
use short imperative subjects (`feat:`, `fix:`, `docs:`, `ci:`, `test:`, `chore:`).

Security issues: do not open a public issue, see [SECURITY.md](SECURITY.md).
