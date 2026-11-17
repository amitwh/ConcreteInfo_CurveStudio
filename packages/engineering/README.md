# curvestudio-engineering

The optional CurveStudio catalogue of concrete and civil engineering relationships: strength
development, concrete properties, cementitious materials, durability and particle packing.
`curvestudio-core` never imports this package; it is discovered through the entry-point group
`curvestudio.models` (`engineering = curvestudio_engineering.register:register_all`).

Every model is an empirical or theoretical relationship with documented assumptions, not an
acceptance equation of a code. `engineering_model()` appends `ENGINEERING_NOTICE` to the
model's assumptions, enforces an `engineering.<sub>` category, and refuses a model without
references and a worked example. A citation that cannot be verified is omitted, never
invented. Parameters carry their unit; `x` is always the single fitted variable (age, ratio,
content, depth, particle size, ...) so Auto-fit can use every model.

## Layout

- `curvestudio_engineering/families/strength.py` — strength against age and mix proportions
- `curvestudio_engineering/families/properties.py` — slump, UPV, modulus, shrinkage, creep,
  maturity and Arrhenius temperature response
- `curvestudio_engineering/families/cementitious.py` — cement development, fly ash and GGBS
  replacement, pozzolanic reactivity, hydration heat
- `curvestudio_engineering/families/durability.py` — chloride profile, carbonation,
  sorptivity, water absorption, gas permeability
- `curvestudio_engineering/families/packing.py` — Fuller, modified Fuller, Andreasen and the
  modified Andreasen (Dinger-Funk) distribution

## Adding a model

1. Add it to the family module of its sub-category with `engineering_model(...)`, filling
   every field: one-sentence parameter descriptions and units, real references, assumptions,
   an `example` whose Jacobian is well conditioned, and an `initial_guess` that is finite and
   inside the bounds for degenerate data.
2. Add an independent NumPy reference to `tests/unit/engineering/golden/golden_<family>.py`
   (`GOLDEN`, and `DENOMINATORS` for a model that divides by a function of `x`). Write it
   from the mathematics, not from the expression string.
3. If the model is a reparametrisation of another one, declare it in the family module as
   `EQUIVALENT_PAIRS = (("this_id", "other_id"), ...)` instead of shipping a duplicate
   equation.
4. Run `uv run pytest tests/unit/engineering` (add `-m slow` for the recovery tests).

Licence: AGPL-3.0-or-later.
