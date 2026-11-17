# CurveStudio desktop app

PySide6 shell for ConcreteInfo CurveStudio. All engine calls go through
`curvestudio.workflow` (shared with the web API), `curvestudio.project`
(`.cstudio` files) and `curvestudio.data` types — never the maths layers
directly.

## Run

```bash
uv run curvestudio-desktop            # the app
uv run curvestudio-desktop --selftest # headless smoke: window + example + one fit
```

## Tests

```bash
uv run pytest tests/desktop
```

Tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set by `tests/desktop/conftest.py`).
Licensed AGPL-3.0-or-later, like the rest of the repository.
