# CurveStudio desktop app

PySide6 shell for ConcreteInfo CurveStudio. All engine calls go through
`curvestudio.workflow` (shared with the web API), `curvestudio.project`
(`.cstudio` files), `curvestudio.data` types and `curvestudio.discovery`
(Auto-fit runs, streamed through a worker) — never the maths layers directly;
exports use `curvestudio_plotting` / `curvestudio_reporting`.

Auto-fit is the primary workflow: importing or selecting a dataset starts a run
by itself (Manual fit mode never starts anything). Runs stream a live ranking,
are cancellable, and every result is stored in the project.

## Run

```bash
uv run curvestudio-desktop            # the app
uv run curvestudio-desktop --selftest # headless smoke: window + example + fit + Auto-fit
```

## Tests

```bash
uv run pytest tests/desktop
```

Tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set by `tests/desktop/conftest.py`)
and default to Manual fit mode so background Auto-fit never races them; the
Auto-fit tests opt in explicitly.
Licensed AGPL-3.0-or-later, like the rest of the repository.
