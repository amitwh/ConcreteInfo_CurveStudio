# curvestudio-models

The hand-written CurveStudio model catalogue: one `ModelSpec` per equation, grouped in
family modules under `curvestudio_models.families`. `curvestudio-core` never imports this
package; it is discovered through the entry-point group `curvestudio.models`
(`library = curvestudio_models.register:register_all`).

## Adding a model

1. Put it in the family module of its category (or add a module that exposes
   `MODELS: tuple[ModelSpec, ...]`; every module of `families/` is picked up automatically).
2. Fill every field: name, category (first dotted segment from the core vocabulary), a
   sentence per parameter, real references (omit a citation you cannot verify), assumptions,
   an `example` whose Jacobian is well conditioned, and an `initial_guess` that is finite and
   inside the bounds for degenerate data.
3. Add an independent NumPy reference for every model to
   `tests/unit/library/golden/golden_<family>.py` as `GOLDEN: dict[str, Callable]` (and, for
   ratios, `DENOMINATORS`); a model without one fails `test_catalogue_golden`. Write it from the
   mathematical definition, not from the expression string.
4. Run `uv run pytest tests/unit/library` (add `-m slow` for recovery and equivalence) and
   regenerate the report with `uv run python scripts/library_report.py`.

Two ids that describe the same curve must be merged with `aliases`, or, when they differ in
meaning (a rate against a time constant), declared in the family module as
`EQUIVALENT_PAIRS: tuple[tuple[str, str], ...]` (a pair naming a model that does not exist yet is
ignored until it does). `curvestudio_models.equivalence.KNOWN_EQUIVALENCES` is the union of every
family's pairs. A family module that fails to import is skipped with a warning naming it and the
catalogue tests fail until it is fixed.

Tests that need the installed entry point carry `@pytest.mark.real_plugins`; by default the test
suite hides installed model plugins.

## Generated families

`curvestudio_models.generated` (not a `families/` module) registers lazily built families with
`register_family`: rational functions `rat_{m}_{n}`, `invpoly_{d}`, `recip_{d}`, sums of
exponentials / Gaussians / powers, `fourier_{n}` and TableCurve-style `tc_*` / `tc3_*`. The
stable-id table, the listing rules (`ids(exhaustive)`) and the models deliberately left out because
an existing model is the same curve are in the module docstring. `generated_info(id)` returns
`(family, k, description)` without building the model. Tests: `tests/unit/library/test_generated*.py`.
Changing the id table means regenerating `tests/unit/library/golden/generated_ids.txt` (command in
`test_generated.py`).

Licence: AGPL-3.0-or-later.
